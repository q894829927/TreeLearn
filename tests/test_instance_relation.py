"""CPU/tiny-head regressions for the CUDA server. Do not execute on the development PC."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tree_learn.instance_relation.data import prepare_features, pair_descriptors, candidate_pairs
from tree_learn.instance_relation.supervision import make_targets, merge_table
from tree_learn.instance_relation.policy import select_pairs, write_prediction
from tree_learn.instance_relation.model import InstanceRelationHead, relation_loss
from tree_learn.mask_refinement.evaluation import evaluate_labels
from tools.forest_abc.diagnose import scan_arrays, analyze_table
from tools.instance_relation import run as workflow


class InstanceRelationTests(unittest.TestCase):
    def settings(self):
        return dict(height_strata=4, points_per_stratum=3, vote_radius_m=4.,
                    lower_radius_m=4., vote_scale_m=4., max_pairs=1000, max_instances=100)

    def fixture(self, cache):
        cache.mkdir(parents=True)
        labels = np.array([1] * 12 + [2] * 12 + [3] * 12 + [-1] * 4 + [0] * 8)
        baseline = np.array([10] * 6 + [20] * 6 + [30] * 12 + [40] * 12 + [99] * 4 + [55] * 6 + [0] * 2)
        xyz = np.zeros((len(labels), 3), dtype=np.float32)
        xyz[:, 0] = np.repeat([0., 3., 6., 9., -3., -6.], [12, 12, 12, 4, 6, 2])
        xyz[:, 1] = np.arange(len(labels)) % 3 * .1
        xyz[:, 2] = np.arange(len(labels)) % 12 * .2
        votes = xyz[:, :2].copy()
        votes[:, 1] = 0
        probability = np.where(baseline > 0, .8, .2).astype(np.float32)
        for key, value in dict(coords=xyz, labels=labels, baseline=baseline,
                               votes_xy=votes, tree_probability=probability).items():
            np.save(cache / f'{key}.npy', value)
        (cache / 'cache.json').write_text(json.dumps(dict(signature='fixture', checkpoint='official.pth', num_points=len(labels))))
        return xyz, labels, baseline

    def prepared(self, root):
        cache = root / 'cache'
        xyz, labels, baseline = self.fixture(cache)
        with contextlib.redirect_stdout(io.StringIO()):
            values = prepare_features(cache, root / 'features', self.settings(), 42, chunk=7)
        return cache, xyz, labels, baseline, values

    def policy(self):
        return dict(merge_threshold=.5, same_threshold=.5, min_union_precision=.6,
                    min_union_recall=.5, min_union_iou=.5, min_iou_gain=.01, protect_threshold=.8)

    def test_features_never_open_gt_and_do_not_depend_on_gt_or_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            self.fixture(cache)
            loader = np.load
            def guarded(path, *args, **kwargs):
                if Path(path).name == 'labels.npy':
                    raise AssertionError('Features opened GT.')
                return loader(path, *args, **kwargs)
            with patch('numpy.load', side_effect=guarded), contextlib.redirect_stdout(io.StringIO()):
                settings = dict(self.settings(), points_per_stratum=1)
                first = prepare_features(cache, root / 'first', settings, 42, chunk=7)
                np.save(cache / 'labels.npy', np.zeros(48, dtype=np.int64))
                second = prepare_features(cache, root / 'second', settings, 42, chunk=25)
            for key in first:
                np.testing.assert_allclose(first[key], second[key], atol=1e-6, rtol=0)

    def test_candidate_pair_features_are_symmetric(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, values = self.prepared(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                swapped = pair_descriptors(values, values['pairs'][:, ::-1], self.settings())
            np.testing.assert_array_equal(values['pair_features'], swapped)

    def test_candidate_guard_rejects_dense_graph_before_pair_allocation(self):
        values = dict(vote_centers=np.zeros((50, 2)), lower_centers=np.zeros((50, 3)))
        with self.assertRaisesRegex(ValueError, 'refuse before allocating'):
            candidate_pairs(values, dict(self.settings(), max_pairs=10))

    def test_supervision_labels_beneficial_union_and_ignores_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            _, xyz, labels, baseline, values = self.prepared(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                geometry, tables = scan_arrays(xyz, labels, {'baseline': baseline}, 7)
            targets, _ = make_targets(geometry, tables['baseline'], values['pairs'],
                dict(min_known_fraction=.2, same_tree_purity=.6, min_union_precision=.65, min_iou_gain=.02))
            pair_ids = values['ids'][values['pairs']]
            positive = np.flatnonzero(np.all(pair_ids == [10, 20], axis=1))[0]
            self.assertEqual(targets['pair_merge'][positive], 1)
            self.assertEqual(targets['pair_gain'][positive], .5)
            np.testing.assert_array_equal(targets['pair_quality'][positive], [1., 1., 1.])
            unknown = (pair_ids == 99).any(1)
            self.assertTrue(unknown.any())
            self.assertFalse(targets['pair_valid'][unknown].any())
            known_background = (pair_ids == 55).any(1)
            self.assertTrue(targets['pair_valid'][known_background].any())
            self.assertFalse(targets['pair_merge'][known_background].any())

    def test_merging_counts_equals_full_point_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, xyz, labels, baseline, values = self.prepared(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                geometry, tables = scan_arrays(xyz, labels, {'baseline': baseline}, 7)
            original = tables['baseline']
            supervision = dict(gt_ids=geometry['ids'], gt_counts=geometry['counts'], z_span=geometry['z_span'],
                ignored_points=geometry['ignored_points'], num_points=geometry['num_points'], **original)
            accepted = np.array([[0, 1]], dtype=np.int64)
            g, t = merge_table(supervision, accepted)
            merged = np.where(baseline == 20, 10, baseline)
            expected = evaluate_labels(labels, merged)
            actual = analyze_table(g, t)['metrics']
            self.assertEqual(actual['tp'], 3)
            self.assertEqual(actual['fp'], 0)
            for key in actual:
                self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_same_tree_completion_can_help_a_tp_without_merging_two_true_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            _, xyz, labels, baseline, values = self.prepared(Path(directory))
            baseline[:8] = 10
            with contextlib.redirect_stdout(io.StringIO()):
                geometry, tables = scan_arrays(xyz, labels, {'baseline': baseline}, 7)
            targets, original = make_targets(geometry, tables['baseline'], values['pairs'],
                dict(min_known_fraction=.2, same_tree_purity=.6, min_union_precision=.65, min_iou_gain=.02))
            pair_ids = values['ids'][values['pairs']]
            completing = np.flatnonzero(np.all(pair_ids == [10, 20], axis=1))[0]
            self.assertTrue(targets['original_tp'][0])
            self.assertEqual(targets['pair_merge'][completing], 1)
            # Different GT trees at 30/40 are close in space, but must remain keep labels.
            separate = np.flatnonzero(np.all(pair_ids == [30, 40], axis=1))[0]
            self.assertEqual(targets['pair_merge'][separate], 0)

    def test_policy_is_mutual_single_round_and_respects_protection(self):
        pairs = np.array([[0, 1], [1, 2], [2, 3]])
        scores = dict(merge=np.array([.95, .9, .85]), same=np.ones(3), quality=np.ones((3, 3)) * .9, gain=np.ones(3) * .8)
        quality = np.ones(4) * .1
        np.testing.assert_array_equal(select_pairs(pairs, scores, quality, self.policy()), [[0, 1]])
        protected = quality.copy(); protected[0] = .9
        np.testing.assert_array_equal(select_pairs(pairs, scores, protected, self.policy()), [[1, 2]])
        self.assertEqual(len(select_pairs(pairs, scores, quality, dict(self.policy(), merge_threshold=1.1))), 0)

    def test_writer_preserves_foreground_and_unselected_complete_point_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, _, _, baseline, values = self.prepared(root)
            before = (cache / 'baseline.npy').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                report = write_prediction(cache / 'baseline.npy', root / 'merged.npy', values['ids'], [[0, 1]], chunk=7)
            actual = np.load(root / 'merged.npy')
            np.testing.assert_array_equal(actual, np.where(baseline == 20, 10, baseline))
            np.testing.assert_array_equal(actual > 0, baseline > 0)
            self.assertEqual(report['changed_unselected_points'], 0)
            self.assertEqual(report['relabeled_points'], 6)
            self.assertEqual(before, (cache / 'baseline.npy').read_bytes())
            with self.assertRaises(ValueError):
                write_prediction(cache / 'baseline.npy', root / 'bad.npy', values['ids'], [[0, 1], [1, 2]])

    def test_keep_all_exactly_reproduces_original_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, _, _, baseline, values = self.prepared(root)
            with contextlib.redirect_stdout(io.StringIO()):
                result = write_prediction(cache / 'baseline.npy', root / 'unchanged.npy', values['ids'], [])
            np.testing.assert_array_equal(np.load(root / 'unchanged.npy'), baseline)
            self.assertEqual(result['relabeled_points'], 0)

    def test_relation_network_is_symmetric_and_checkpoint_preserves_normalization(self):
        torch.manual_seed(42)
        model = InstanceRelationHead(6, 4, 8).eval()
        model.node_mean.copy_(torch.arange(6, dtype=torch.float32))
        model.node_scale.fill_(2)
        left, right, features = torch.randn(5, 6), torch.randn(5, 6), torch.randn(5, 4)
        first, reverse = model(left, right, features), model(right, left, features)
        for key in first:
            self.assertTrue(torch.equal(first[key], reverse[key]))
        loss, _ = relation_loss(first, torch.tensor([1., 0., 1., 0., 0.]), torch.ones(5), torch.ones((5, 3)), torch.ones(5) * .2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        replica = InstanceRelationHead(6, 4, 8).eval()
        replica.load_state_dict(model.state_dict(), strict=True)
        for key, value in replica(left, right, features).items():
            self.assertTrue(torch.equal(value, first[key]))

    def test_small_complete_workflow_and_prediction_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = yaml.safe_load(Path('configs/experiments/instance_relation/full.yaml').read_text())
            config.update(cache_root=str(root / 'source'), external_cache_root=str(root / 'external'),
                          prepared_root=str(root / 'prepared'), work_dir=str(root / 'work'), checkpoint='official.pth',
                          geometry=self.settings(), splits=dict(train=['train'], validation=['validation']))
            config['training'].update(device='cpu', hidden_dim=8, steps=4, validation_every=2,
                                      save_every=2, log_every=1, batch_size=8, node_batch_size=8)
            config['expected_validation_baseline_f1'] = 4 / 7
            for external in (False, True):
                base = Path(config['external_cache_root' if external else 'cache_root'])
                records = []
                for plot in (['outside'] if external else ['train', 'validation']):
                    self.fixture(base / plot)
                    records.append(dict(plot=plot, split='external' if external else plot, cache=plot, signature='fixture'))
                (base / 'manifest.json').write_text(json.dumps(dict(forests=records)))
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.prepare(config)
                workflow.check(config)
                workflow.train(config)
                workflow.evaluate(config)
                # Actual deployment command has no GT dependency, even without labels.npy.
                (Path(config['external_cache_root']) / 'outside' / 'labels.npy').unlink()
                workflow.predict(config)
            path = Path(config['work_dir']) / 'external_evaluation' / 'outside'
            marker = json.loads((path / 'prediction.json').read_text())
            self.assertFalse(marker['gt_opened'])
            self.assertEqual(marker['changed_foreground_points'], 0)
            self.assertEqual(marker['changed_unselected_points'], 0)
            summary = json.loads((Path(config['work_dir']) / 'evaluation' / 'brief.json').read_text())
            self.assertEqual(summary['lost_gt'], 0)


if __name__ == '__main__':
    unittest.main()
