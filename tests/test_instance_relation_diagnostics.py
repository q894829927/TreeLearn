"""Saved-count audit regressions for execution on the experiment server."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from tools.forest_abc.diagnose import scan_arrays
from tools.instance_relation.diagnose import audit
from tree_learn.instance_relation.supervision import make_targets
from tree_learn.mask_refinement.evaluation import evaluate_labels


class InstanceRelationDiagnosticsTests(unittest.TestCase):
    def inputs(self, gt=None, prediction=None, pairs=None):
        if gt is None:
            gt = np.asarray([1] * 12 + [2] * 8 + [3] * 8 + [0] * 4)
            prediction = np.asarray([10] * 6 + [20] * 6 + [30] * 8 + [40] * 8 + [50] * 4)
            pairs = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        with contextlib.redirect_stdout(io.StringIO()):
            geometry, tables = scan_arrays(np.zeros((len(gt), 3)), gt, {'baseline': prediction}, 7)
        table = tables['baseline']
        targets, _ = make_targets(geometry, table, pairs,
            dict(min_known_fraction=.2, same_tree_purity=.6, min_union_precision=.65, min_iou_gain=.02))
        targets.update(gt_ids=geometry['ids'], gt_counts=geometry['counts'], z_span=geometry['z_span'],
                       ignored_points=geometry['ignored_points'], num_points=geometry['num_points'], **table)
        values = dict(ids=table['ids'], pairs=pairs)
        count = len(pairs)
        scores = dict(merge=np.ones(count) * .95, same=np.ones(count) * .9,
                      quality=np.ones((count, 3)) * .9, gain=np.ones(count) * .2,
                      node_quality=np.ones(len(table['ids'])) * .1, accepted=pairs.copy())
        policy = dict(merge_threshold=.6, same_threshold=.5, min_union_precision=.65,
                      min_union_recall=.5, min_union_iou=.5, min_iou_gain=.02, protect_threshold=.65)
        return values, targets, scores, policy, gt, prediction

    def call_audit(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit(*args)

    def test_exact_actions_identify_recovery_and_destruction_of_two_true_instances(self):
        values, target, scores, policy, gt, original = self.inputs()
        scores['merge'][1] = .1
        scores['accepted'] = values['pairs'][[0, 2]]
        brief, actions, _ = self.call_audit(values, target, scores, policy)
        merged = np.where(original == 20, 10, np.where(original == 40, 30, original))
        expected = evaluate_labels(gt, merged)
        for key in ('tp', 'fp', 'fn', 'f1', 'completeness', 'commission', 'coverage'):
            self.assertAlmostEqual(brief['refined'][key], expected[key], places=12)
        self.assertEqual(brief['lost_gt_ids'], [2, 3])
        self.assertEqual(brief['recovered_gt_ids'], [1])
        self.assertEqual(brief['accepted_harmful_pairs_in_isolation'], 1)
        self.assertEqual(brief['accepted_pairs_covering_two_original_tp'], 1)
        self.assertTrue(brief['isolated_pair_sum_matches_combined_counts'])
        self.assertEqual(brief['selected_tp_actual_at_or_above_protection'], 2)
        self.assertEqual(actions[1]['isolated_lost_gt_ids'], [2, 3])
        self.assertEqual(brief['stages']['mutual_selection']['potential_recovery_gt_ids'], [1])

    def test_gate_trace_distinguishes_protection_from_no_candidate(self):
        values, target, scores, policy, _, _ = self.inputs()
        scores['merge'][1:] = .1
        scores['node_quality'][0] = .9
        scores['accepted'] = np.empty((0, 2), dtype=np.int64)
        brief, _, _ = self.call_audit(values, target, scores, policy)
        self.assertEqual(brief['potential_recovery_candidates']['potential_recovery_pairs'], 1)
        self.assertEqual(brief['stages']['union_gain']['potential_recovery_pairs'], 1)
        self.assertEqual(brief['stages']['original_protection']['potential_recovery_pairs'], 0)
        self.assertEqual(brief['potential_recovery_first_rejection'], {'original_protection': 1})

    def test_watched_pair_trace_and_its_exact_isolated_result(self):
        values, target, scores, policy, _, _ = self.inputs()
        scores['merge'][1:] = .1
        scores['accepted'] = values['pairs'][[0]]
        with patch.dict('tools.instance_relation.diagnose.WATCHLIST', {1: (10, 20)}, clear=True):
            _, _, watched = self.call_audit(values, target, scores, policy)
        self.assertEqual(len(watched), 1)
        self.assertTrue(watched[0]['case_pair_in_candidates'])
        self.assertEqual(watched[0]['case_pair']['actual_iou_for_watched_gt'], 1.)
        self.assertEqual(watched[0]['case_pair']['isolated_recovered_gt_ids'], [1])

    def test_iou_feasibility_does_not_imply_hungarian_detection(self):
        gt = np.asarray([1] * 34 + [2] * 16 + [1] * 34 + [2] * 16 + [1] * 32)
        original = np.asarray([10] * 50 + [20] * 50 + [30] * 32)
        pairs = np.asarray([[0, 1]], dtype=np.int64)
        values, target, scores, policy, _, _ = self.inputs(gt, original, pairs)
        brief, actions, _ = self.call_audit(values, target, scores, policy)
        self.assertGreater(actions[0]['actual_union_quality'][2], .5)
        self.assertEqual(brief['potential_recovery_candidates']['potential_recovery_gt_ids'], [1])
        self.assertEqual(brief['recovered_gt_ids'], [])
        self.assertEqual(actions[0]['isolated_recovered_gt_ids'], [])
        self.assertEqual(brief['refined']['tp'], 0)

    def test_mismatched_or_nonfinite_scores_are_rejected(self):
        values, target, scores, policy, _, _ = self.inputs()
        scores['accepted'] = np.empty((0, 2), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'Saved actions'):
            self.call_audit(values, target, scores, policy)
        scores['quality'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid saved score'):
            self.call_audit(values, target, scores, policy)

    def test_empty_candidates_keep_all_and_empty_diagnostics(self):
        values, target, scores, policy, gt, prediction = self.inputs()
        values['pairs'] = np.empty((0, 2), dtype=np.int64)
        scores.update(merge=np.empty(0), same=np.empty(0), quality=np.empty((0, 3)), gain=np.empty(0),
                      accepted=np.empty((0, 2), dtype=np.int64))
        # Rebuild empty pair targets while retaining the exact contingency table.
        for key in ('pair_merge', 'pair_same', 'pair_gain', 'pair_target_gt', 'pair_valid'):
            target[key] = target[key][:0]
        target['pair_quality'] = target['pair_quality'][:0]
        brief, actions, _ = self.call_audit(values, target, scores, policy)
        self.assertEqual(brief['refined']['f1'], evaluate_labels(gt, prediction)['f1'])
        self.assertEqual(actions, [])
        self.assertEqual(brief['potential_recovery_first_rejection'], {})


if __name__ == '__main__':
    unittest.main()
