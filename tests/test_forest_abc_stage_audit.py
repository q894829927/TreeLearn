"""Run these CPU regressions on the server; no production model is executed."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.forest_abc.stage_audit_core import (
    best_for_gt, cell_counts, classify_target, nms_overlap, proposal_metrics, select_targets)
from tools.forest_abc.stage_audit import capture_heads, global_witnesses, replay_block


class StageAuditStatisticsTests(unittest.TestCase):
    def test_sparse_statistics_equal_expanded_masks_and_use_full_tree_denominators(self):
        nearest = np.array([0, 0, 1, 1, 2, 3, 3, 3])
        labels = np.array([1, -1, 1, 0, 2, 2, 0, -1])
        masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool)
        cells = cell_counts(nearest, labels, 4, np.array([1, 2]))
        full_sizes = np.array([4, 3])  # Some GT points are outside this crop.
        for chunk in (1, 3, 16):
            metrics = proposal_metrics(masks, cells, full_sizes, query_chunk=chunk)
            for query, voxel_mask in enumerate(masks):
                point_mask = voxel_mask[nearest]
                known = int(np.count_nonzero(point_mask & (labels >= 0)))
                self.assertEqual(metrics['total'][query], point_mask.sum())
                self.assertEqual(metrics['known'][query], known)
                for col, gt in enumerate((1, 2)):
                    common = int(np.count_nonzero(point_mask & (labels == gt)))
                    self.assertAlmostEqual(metrics['iou'][query, col], common / (known + full_sizes[col] - common))
                    self.assertAlmostEqual(metrics['recall'][query, col], common / full_sizes[col])
        self.assertEqual(metrics['iou'][1, 1], .5)
        self.assertEqual(best_for_gt(metrics, [1], 1, np.ones(4))['good_queries'], [])
        empty = best_for_gt(metrics, [], 0, np.ones(4))
        self.assertEqual(set(empty), set(best_for_gt(metrics, [0], 0, np.ones(4))))
        with self.assertRaisesRegex(ValueError, 'GT IDs'):
            cell_counts(nearest, labels, 4, np.array([1]))

    def test_stage_attribution_counts_good_partial_crop_and_preserves_controls(self):
        stages = ('raw', 'quality', 'size', 'local_nms')

        def row(present, full=True):
            return dict(full_context=full, stages={stage: dict(good_queries=[0] if stage in present else [])
                                                   for stage in stages})

        cases = (((), 'no_good_raw_mask_in_sampled_blocks'),
                 (('raw',), 'good_raw_masks_lost_at_quality_filter'),
                 (('raw', 'quality'), 'good_masks_lost_at_minimum_size_filter'),
                 (('raw', 'quality', 'size'), 'good_masks_lost_at_local_nms'),
                 (stages, 'good_local_masks_but_saved_final_bad'))
        for present, expected in cases:
            self.assertEqual(classify_target([row(present)], False, .4), expected)
        self.assertEqual(classify_target([row((), full=False)], False, .4), 'no_sampled_full_context_block')
        # A >0.5 full-tree IoU is real evidence even when the crop contains <95% of GT.
        self.assertEqual(classify_target([row(stages, full=False)], False, .4), 'good_local_masks_but_saved_final_bad')
        self.assertEqual(classify_target([], True, .8), 'detected_control')

    def test_sampling_is_deterministic_unique_and_keeps_failed_and_successful_cases(self):
        rows = []
        for span in (6, 15, 25):
            for detected in (False, True):
                for index in range(8):
                    rows.append(dict(gt_id=str(len(rows) + 1), z_span_m=str(span),
                        refined_detected=str(detected), transition='both_detected' if detected else 'lost',
                        refined_category='detected' if detected else 'high_recall_low_purity'))
        selected = select_targets(rows)
        self.assertEqual(selected, select_targets(list(reversed(rows))))
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row['gt_id'] for row in selected}), 12)
        self.assertEqual(sum(row['refined_detected'] == 'True' for row in selected), 4)
        self.assertFalse(any('stratum' in row for row in rows))

    def test_local_replay_reproduces_suppression_of_a_good_mask_by_a_broad_mask(self):
        labels = np.array([1] * 4 + [2] * 4 + [0] * 2 + [-1] * 2)
        coords = np.zeros((12, 3), dtype=np.float32)
        coords[:, 0] = np.arange(12)
        block = dict(indices=np.arange(12), source_xyz=coords, xyz=coords,
                     available_voxels=12, raw_points=12, core=np.ones(12, dtype=bool))

        class Forest:
            centers = np.zeros((1, 2))
            arrays = dict(coords=coords)
            metadata = dict(num_points=12)

            def make_block(self, *args):
                return block

        masks = np.zeros((3, 12), dtype=bool)
        masks[0, :4], masks[1, :8], masks[2, 4:8] = True, True, True
        heads = [dict(masks=masks, scores=np.array([.4, .9, .5], dtype=np.float32))]
        auxiliary = dict(anchors=np.array([0, 6, 4]), semantic_foreground=np.ones(12, dtype=bool),
                         query_candidates=np.arange(12))
        config = dict(seed=42, blocks={}, model=dict(use_mfa=False), inference=dict(
            quality_threshold=.3, minimum_instance_points=1, max_block_memberships=8,
            nms_iou=.5, nms_containment=.8))
        saved = {(0, 1): dict(instance_id=1, count=8, score=.9, bbox=[0, 0, 0, 7, 0, 0])}
        with tempfile.TemporaryDirectory() as temporary:
            with patch('tools.forest_abc.stage_audit.capture_heads', return_value=(heads, auxiliary)):
                report, rows, _, records, remaining = replay_block(None, Forest(), labels, config, 0,
                    np.array([1, 2]), np.array([4, 4]), [dict(gt_id='1'), dict(gt_id='2')],
                    saved, Path(temporary), None, 100)
            self.assertEqual(report['local_kept'], 1)
            self.assertEqual(remaining, 92)
            self.assertTrue(records[(0, 1)]['replay_matches_saved_metadata'])
            self.assertEqual(rows[0]['stages']['raw']['iou'], 1)
            self.assertEqual(rows[0]['stages']['local_nms']['iou'], .5)
            self.assertEqual(classify_target([rows[0]], False, .5), 'good_masks_lost_at_local_nms')
            np.testing.assert_array_equal(np.load(records[(0, 1)]['path']), np.arange(8))

    def test_global_conflict_witness_and_allocation_degradation_are_separate(self):
        target = dict(gt_id='1', gt_points='4')
        block_rows = [dict(block=0, stages=dict(local_nms=dict(good_queries=[0])))]
        points = {1: dict(indices=np.arange(4))}
        config = dict(inference=dict(nms_iou=.5, nms_containment=.8))
        with tempfile.TemporaryDirectory() as temporary:
            first, second = [Path(temporary) / name for name in ('first.npy', 'second.npy')]
            np.save(first, np.arange(4))
            np.save(second, np.arange(8))
            candidate = dict(path=str(first), global_id=0, score=.4, bbox=[0, 0, 0, 3, 0, 0],
                             target_iou={1: 1.}, replay_matches_saved_metadata=False)
            winner = dict(path=str(second), global_id=1, score=.9, bbox=[0, 0, 0, 7, 0, 0],
                          target_iou={1: .5}, replay_matches_saved_metadata=True)
            report = global_witnesses(target, block_rows, {(0, 0): candidate, (1, 1): winner},
                                      np.ones(8), {1: 8}, points, config)
            self.assertEqual(len(report['sampled_higher_score_nms_conflicts']), 1)
            self.assertEqual(report['accepted_good_proposals'], [])
            candidate = dict(candidate, global_id=2, replay_matches_saved_metadata=True)
            report = global_witnesses(target, block_rows, {(0, 0): candidate}, np.array([1, 1, 2, 2]),
                                      {1: 2, 2: 2}, points, config)
            self.assertEqual(report['sampled_higher_score_nms_conflicts'], [])
            self.assertEqual(report['accepted_good_proposals'][0]['allocated_iou'], .5)
            self.assertTrue(nms_overlap(np.arange(4), np.arange(100), .5, .8)['conflicts'])

    def test_eval_head_observation_preserves_results_and_restores_method_on_error(self):
        import torch

        class Decoder(torch.nn.Module):
            def _forward_head(self, phase):
                scores = torch.full((4, 1), phase / 10)
                masks = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 4 - 3 + phase
                return [scores], [masks], None

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = Decoder()
                self.config = dict(layers=2, num_queries=2, isa_candidates=4)
                self.audit_mask_threshold = .5
                self.fail = False

            def forward(self, batch):
                for phase in range(3):
                    self.last = self.decoder._forward_head(phase)
                    if self.fail:
                        raise RuntimeError('injected failure')
                return dict(semantic=torch.zeros((8, 2)), anchors=torch.tensor([0, 2]))

        model = Model().eval()
        original_function = model.decoder._forward_head.__func__
        batch = dict(core=torch.ones(8, dtype=torch.bool))
        captured, auxiliary = capture_heads(model, batch, torch.device('cpu'), tensorize=lambda b, d: b)
        self.assertEqual(len(captured), 3)
        np.testing.assert_array_equal(captured[-1]['masks'], (model.last[1][0][:-2].sigmoid() >= .5).numpy())
        np.testing.assert_array_equal(auxiliary['anchors'], [0, 2])
        self.assertIs(model.decoder._forward_head.__func__, original_function)
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            capture_heads(model, batch, torch.device('cpu'), tensorize=lambda b, d: b)
        self.assertIs(model.decoder._forward_head.__func__, original_function)


if __name__ == '__main__':
    unittest.main()
