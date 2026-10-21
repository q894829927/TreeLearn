"""CPU-only audit regressions. Execute on the server, not as local model tests."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools.diagnostics.extract_wytham_baseline_cases import (
    bounded_append, edit_bounds, prioritize, select_cases, supplemental_signals)


class BaselineRepairCaseTests(unittest.TestCase):
    def test_strict_half_requires_one_foreign_removal(self):
        value = edit_bounds(100, 200, 100)
        self.assertIsNone(value['min_target_additions'])
        self.assertEqual(value['min_foreign_removals'], 1)

    def test_additions_leave_union_unchanged(self):
        # IoU starts at 45/105; eight added target points make it 53/105.
        value = edit_bounds(100, 50, 45)
        self.assertEqual(value['min_target_additions'], 8)
        self.assertIsNone(value['min_foreign_removals'])

    def test_no_prediction_needs_more_than_half_of_target(self):
        self.assertEqual(edit_bounds(100, 0, 0)['min_target_additions'], 51)

    def test_already_valid_mask_needs_no_edit(self):
        self.assertEqual(edit_bounds(100, 100, 90), dict(min_target_additions=0, min_foreign_removals=0))

    def test_invalid_counts_are_rejected(self):
        for counts in ((-1, 1, 0), (5, 1, 2), (1, 5, 2)):
            with self.assertRaises(ValueError):
                edit_bounds(*counts)

    def test_bounds_equal_exhaustive_integer_edits(self):
        for g in range(1, 12):
            for p in range(12):
                for common in range(min(g, p) + 1):
                    value = edit_bounds(g, p, common)
                    add = next((a for a in range(g - common + 1)
                                if 2 * (common + a) > g + p - common), None)
                    remove = next((r for r in range(p - common + 1)
                                   if 2 * common > g + p - common - r), None)
                    self.assertEqual(value['min_target_additions'], add)
                    self.assertEqual(value['min_foreign_removals'], remove)

    def test_locked_correct_masks_limit_the_available_edit_space(self):
        geometry = dict(counts=np.array([100, 100]), z_span=np.array([20., 5.]))
        table = dict(ids=np.array([10, 20]), known=np.array([100, 100]),
                     intersections=np.array([[80, 20], [20, 40]]))
        trees = [dict(gt_id=1, detected=True, best_pred_id=10, best_iou=2/3, category='detected'),
                 dict(gt_id=2, detected=False, best_pred_id=20, best_iou=.25, category='mostly_unassigned')]
        baseline = dict(trees=trees, predictions=[dict(pred_id=10, category='true_positive'),
                                                 dict(pred_id=20, category='false_positive')])
        rows = prioritize(geometry, table, baseline)
        self.assertEqual(rows[1]['target_points_outside_locked_predictions'], 80)
        self.assertTrue(rows[1]['clean_gt_oracle_possible_without_touching_locked_predictions'])
        self.assertFalse(rows[1]['clean_gt_oracle_add_only_original_zero'])
        self.assertFalse(rows[1]['addition_oracle_possible_to_existing_unlocked_candidate'])

    def test_a_locked_best_candidate_is_not_marked_as_safe_to_modify(self):
        geometry = dict(counts=np.array([100]), z_span=np.array([5.]))
        table = dict(ids=np.array([10]), known=np.array([200]), intersections=np.array([[60]]))
        baseline = dict(trees=[dict(gt_id=2, detected=False, best_pred_id=10,
            best_iou=.25, category='other_below_iou_threshold')],
            predictions=[dict(pred_id=10, category='true_positive')])
        row = prioritize(geometry, table, baseline)[0]
        self.assertTrue(row['best_pred_is_locked_tp'])
        self.assertFalse(row['clean_gt_oracle_possible_without_touching_locked_predictions'])
        self.assertFalse(row['addition_oracle_possible_to_existing_unlocked_candidate'])

    def test_successful_controls_span_height_groups(self):
        rows = [dict(gt_id=i + 1, detected=True, z_span_m=span, best_iou=.7,
                     diagnostic_group='detected_control') for i, span in enumerate([3., 7., 15., 25.])]
        selected = select_cases(rows, 2)
        self.assertEqual([row['gt_id'] for row in selected], [1, 2, 3, 4])

    def test_saved_probability_bands_separate_background_and_unknown(self):
        values = dict(labels=np.array([1, 1, 1, 0, -1]), baseline=np.array([10, 0, 0, 0, 0]),
            tree_probability=np.array([.9, .35, .05, .4, .8]))
        rows = [dict(gt_id=1)]
        report = supplemental_signals(values, dict(ids=np.array([1]), counts=np.array([3])), rows, 2)
        self.assertEqual(report['additional_candidate_exposure']['0.3'],
                         dict(tree_gt_points=1, known_background_points=1, ignored_points=1))
        self.assertAlmostEqual(rows[0]['max_clean_iou_add_only_p_ge_0_3'], 1/3)
        self.assertEqual(report['saved_probability_threshold_disagreements'], 1)

    def test_saved_vote_dispersion_is_translation_invariant(self):
        geometry = dict(ids=np.array([1]), counts=np.array([3]))
        values = dict(labels=np.array([1, 1, 1]), baseline=np.array([10, 10, 10]),
                      votes_xy=np.array([[0., 0.], [1., 0.], [2., 0.]]))
        first, second = [dict(gt_id=1)], [dict(gt_id=1)]
        supplemental_signals(values, geometry, first, 2)
        supplemental_signals(dict(values, votes_xy=values['votes_xy'] + 10), geometry, second, 2)
        self.assertAlmostEqual(first[0]['saved_vote_xy_rms_spread_m'], np.sqrt(2/3))
        self.assertAlmostEqual(first[0]['saved_vote_xy_rms_spread_m'], second[0]['saved_vote_xy_rms_spread_m'])

    def test_bounded_crop_sampling_keeps_every_array_aligned(self):
        state, rng = {}, np.random.default_rng(42)
        for start in (0, 10):
            indices = np.arange(start, start + 10)
            bounded_append(state, dict(source_indices=indices, labels=2 * indices,
                tree_probability=indices / 100, votes_xy=np.column_stack((indices, indices + 1))), 4, rng)
        self.assertEqual(len(state['source_indices']), 4)
        np.testing.assert_array_equal(state['labels'], 2 * state['source_indices'])
        np.testing.assert_array_equal(state['votes_xy'][:, 0], state['source_indices'])
        np.testing.assert_allclose(state['tree_probability'], state['source_indices'] / 100)


if __name__ == '__main__':
    unittest.main()
