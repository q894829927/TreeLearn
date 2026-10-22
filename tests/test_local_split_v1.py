"""Behavioral tests for a frozen-cache local edit, including failure/refusal."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from tree_learn.local_split_v1 import data, geometry
from tree_learn.local_split_v1.audit import advancement, comparison, edge_targets, seed_audit, split_table
from tree_learn.mask_refinement.evaluation import evaluate_labels
from tools.forest_abc.diagnose import scan_arrays

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/experiments/local_split_v1/full.yaml').read_text())


def stems(count=2):
    rng = np.random.default_rng(17)
    z = np.repeat(np.arange(0, 4, .1), 16)
    pieces = [np.column_stack([rng.normal(k * .9, .025, len(z)), rng.normal(0, .025, len(z)), z]) for k in range(count)]
    return np.concatenate(pieces)


class LocalSplitTests(unittest.TestCase):
    def test_vote_pair_has_hash_and_shared_height_support(self):
        xyz = stems()
        votes = xyz[:, :2]
        seeds, report = geometry.candidate(xyz, votes, np.arange(len(xyz)), CONFIG['candidate'])
        self.assertEqual(report['entrance'], 'votes')
        self.assertTrue(all(len(at) >= 20 for at in seeds))
        self.assertEqual(len(np.intersect1d(*seeds)), 0)

    def test_touching_votes_still_allow_two_lower_tracks(self):
        xyz = stems()
        votes = np.zeros((len(xyz), 2))
        seeds, report = geometry.candidate(xyz, votes, np.arange(len(xyz)), CONFIG['candidate'])
        self.assertIsNotNone(seeds)
        self.assertEqual(report['entrance'], 'geometry')

    def test_three_stable_supports_refuse_binary_split(self):
        xyz = stems(3)
        seeds, report = geometry.candidate(xyz, xyz[:, :2], np.arange(len(xyz)), CONFIG['candidate'])
        self.assertIsNone(seeds)
        self.assertEqual(report['reason'], 'third_stable_support')
        seeds, report = geometry.candidate(xyz, np.zeros((len(xyz), 2)), np.arange(len(xyz)), CONFIG['candidate'])
        self.assertIsNone(seeds)
        self.assertEqual(report['reason'], 'third_stable_support')

    def test_upper_fork_without_lower_pair_does_not_trigger(self):
        xyz = stems(1)
        upper = xyz[xyz[:, 2] >= 2].copy()
        upper[:, 0] += .9
        xyz = np.r_[xyz, upper]
        seeds, report = geometry.candidate(xyz, np.zeros((len(xyz), 2)), np.arange(len(xyz)), CONFIG['candidate'])
        self.assertIsNone(seeds)
        self.assertEqual(report['reason'], 'no_seeds')

    def test_later_third_track_overrides_earlier_accepted_pair(self):
        pair = stems()
        third = stems(3)[2 * len(pair) // 2:]
        third = third[third[:, 2] >= 2]
        xyz = np.r_[pair, third]
        seeds, report = geometry.candidate(xyz, np.zeros((len(xyz), 2)), np.arange(len(xyz)), CONFIG['candidate'])
        self.assertIsNone(seeds)
        self.assertEqual(report['reason'], 'third_stable_support')

    def test_interrupted_training_cannot_silently_restart_budget(self):
        from tree_learn.local_split_v1.learning import claim_training
        with tempfile.TemporaryDirectory() as directory:
            claim_training(directory, 500)
            with self.assertRaises(FileExistsError):
                claim_training(directory, 500)

    def test_budgets_refuse_without_truncation(self):
        xyz = stems()
        cfg = dict(CONFIG['candidate'], max_points=100)
        seeds, report = geometry.candidate(xyz, xyz[:, :2], np.arange(len(xyz)), cfg)
        self.assertIsNone(seeds)
        self.assertEqual(report['reason'], 'core_budget')
        with self.assertRaises(geometry.BudgetExceeded):
            geometry.graph_edges(xyz, dict(CONFIG['graph'], max_edges=1))

    def test_connection_scores_change_actual_ownership(self):
        xyz = np.column_stack([np.arange(5) * .1, np.zeros(5), np.zeros(5)])
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        a, _ = geometry.harmonic_partition(xyz, edges, np.array([.01, 1, 1, 1]), [np.array([0]), np.array([4])], CONFIG['graph'])
        b, _ = geometry.harmonic_partition(xyz, edges, np.array([1, 1, 1, .01]), [np.array([0]), np.array([4])], CONFIG['graph'])
        self.assertFalse(np.array_equal(a > .5, b > .5))

    def test_seedless_fragment_stays_parent(self):
        xyz = np.array([[0, 0, 0], [.1, 0, 0], [10, 0, 0]], dtype=float)
        result, report = geometry.harmonic_partition(xyz, np.array([[0, 1]]), np.ones(1),
            [np.array([0]), np.array([1])], CONFIG['graph'])
        self.assertEqual(result[2], 0.)
        self.assertEqual(report['seedless_points'], 1)

    def test_solver_failure_refuses(self):
        xyz = np.array([[0, 0, 0], [.1, 0, 0], [.2, 0, 0]], dtype=float)
        with patch.object(geometry, 'cg', return_value=(np.zeros(1), 2000)):
            result, report = geometry.harmonic_partition(xyz, np.array([[0, 1], [1, 2]]), np.ones(2),
                [np.array([0]), np.array([2])], CONFIG['graph'])
        self.assertIsNone(result)
        self.assertEqual(report['reason'], 'solver_failed')

    def test_unknown_and_background_pair_loss_mask(self):
        labels = np.array([-1, 0, 0, 1, 1, 2])
        edges = np.array([[0, 3], [1, 2], [2, 3], [3, 4], [4, 5]])
        valid, targets = edge_targets(labels, edges)
        np.testing.assert_array_equal(valid, [False, False, True, True, True])
        np.testing.assert_array_equal(targets[valid], [0, 1, 0])

    def test_seed_purity_ignores_unknown_but_counts_known_background(self):
        labels = np.r_[np.ones(20, dtype=int), np.full(80, -1), np.full(20, 2)]
        result = seed_audit(labels, [np.arange(100), np.arange(100, 120)], CONFIG['audit'])
        self.assertTrue(result['reliable'])
        labels[20:100] = 0
        self.assertFalse(seed_audit(labels, [np.arange(100), np.arange(100, 120)], CONFIG['audit'])['reliable'])

    def test_local_edits_keep_every_other_point_and_foreground(self):
        baseline = np.array([0, 7, 7, 7, 10, 10])
        result = data.apply_edits(baseline, [(7, np.array([2]))])
        np.testing.assert_array_equal(result[[0, 1, 3, 4, 5]], baseline[[0, 1, 3, 4, 5]])
        self.assertEqual(result[2], 11)
        np.testing.assert_array_equal(result > 0, baseline > 0)
        np.testing.assert_array_equal(baseline, [0, 7, 7, 7, 10, 10])
        with self.assertRaises(ValueError):
            data.apply_edits(baseline, [(7, np.array([4]))])

    def test_incremental_full_forest_counts_equal_full_point_evaluation(self):
        labels = np.array([1, 1, 1, 2, 2, 2, -1, 0, 3, 3])
        baseline = np.array([8, 8, 8, 8, 8, 8, 8, 8, 9, 9])
        coords = np.zeros((len(labels), 3), dtype=np.float32)
        geo, tables = scan_arrays(coords, labels, {'baseline': baseline}, chunk_points=3)
        edits = [(8, np.array([3, 4, 5]))]
        result = comparison(geo, tables['baseline'], split_table(geo, tables['baseline'], edits, labels))
        direct = evaluate_labels(labels, data.apply_edits(baseline, edits))
        for key in ('tp', 'fp', 'fn', 'f1', 'coverage', 'point_precision', 'point_recall'):
            self.assertAlmostEqual(result['refined'][key], direct[key])
        self.assertEqual(result['lost'], 0)
        self.assertEqual(result['recovered'], 2)

    def test_iou_equal_half_is_not_tp(self):
        labels = np.array([1, 1, 0, 0])
        metrics = evaluate_labels(labels, np.ones(4, dtype=int))
        self.assertEqual(metrics['tp'], 0)

    def test_advancement_rejects_no_edits_and_three_to_one_ratio(self):
        base = dict(tp=10, fp=2, fn=3, f1=.8, completeness=.8, commission=.1,
                    coverage=.7, point_precision=.7, point_recall=.7)
        reports = [dict(baseline=base, refined=base, recovered=0, lost=0) for _ in range(5)]
        self.assertFalse(advancement(reports)['passed'])
        reports[0] = dict(baseline=base, refined=dict(base, f1=.81), recovered=3, lost=1)
        self.assertFalse(advancement(reports)['passed'])
        reports[0]['recovered'] = 4
        self.assertTrue(advancement(reports)['passed'])

    def test_model_is_symmetric_and_has_exact_small_parameter_count(self):
        import torch
        from tree_learn.local_split_v1.model import EdgeHead
        model = EdgeHead(np.zeros(32), np.ones(32))
        raw = torch.randn(10, 72)
        swapped = torch.cat((raw[:, 32:64], raw[:, :32], raw[:, 64:]), dim=1)
        torch.testing.assert_close(model(raw), model(swapped))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5265)

    def test_prepare_does_not_open_gt_and_is_chunk_independent(self):
        from tree_learn.mask_refinement.data import export_cache
        xyz = stems().astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_cache(root / 'cache', xyz, np.zeros((len(xyz), 32)), np.zeros_like(xyz),
                np.tile([10., 0.], (len(xyz), 1)), np.ones(len(xyz), dtype=int),
                np.ones(len(xyz), dtype=int), {})
            (root / 'cache/labels.npy').unlink()
            row = dict(cache=str(root / 'cache'), prepared=str(root / 'out'), plot='synthetic', split='train')
            result = data.prepare_forest(row, CONFIG)
            self.assertEqual(result['num_candidates'], 1)
            np.save(root / 'cache/labels.npy', np.full(len(xyz), -1))
            self.assertEqual(data.load_prepared(row, CONFIG), result)
            arrays, _ = data.load_cache(row['cache'], with_labels=False)
            try:
                record = result['records'][0]
                payload = data.load_parent(row, record, arrays)
                a = geometry.split_proposal(xyz, payload['indices'], [payload['seed0'], payload['seed1']],
                    CONFIG['graph'], CONFIG['policy'], lambda edges: np.ones(len(edges)))
                np.save(root / 'cache/labels.npy', np.arange(len(xyz)) + 1)
                b = geometry.split_proposal(xyz, payload['indices'], [payload['seed0'], payload['seed1']],
                    CONFIG['graph'], CONFIG['policy'], lambda edges: np.ones(len(edges)))
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1], b[1])
            finally:
                data.close_arrays(arrays)


if __name__ == '__main__':
    unittest.main()
