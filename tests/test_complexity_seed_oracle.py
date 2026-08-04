import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_complexity_seed_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'complexity_seed_oracle_standalone', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class ComplexitySeedOracleTests(unittest.TestCase):

    def test_candidate_mask_matches_fp32_semantic_threshold_definition(self):
        logits = np.asarray([
            [2.0, 0.0], [0.0, 2.0], [1.0, 1.0],
        ], dtype=np.float64)
        verticality = np.asarray([0.7, 0.7, 0.5])
        offsets = np.asarray([
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        selected = ORACLE.candidate_base_seed_mask(
            logits, verticality, offsets,
            tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0)
        np.testing.assert_array_equal(selected, [True, False, False])

    def test_legacy_base_anchor_and_offsets_match_expected_geometry(self):
        coords = np.asarray([
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.2], [10.0, 0.0, 1.0],
            [4.0, 4.0, 0.0], [6.0, 4.0, 0.4], [9.0, 4.0, 1.0],
        ])
        labels = np.asarray([1, 1, 1, 2, 2, 2])
        tree_ids, anchors = ORACLE.legacy_base_anchors(
            coords, labels, base_anchor_height=0.5)
        np.testing.assert_array_equal(tree_ids, [1, 2])
        np.testing.assert_allclose(
            anchors, [[1.0, 0.0, 0.1], [5.0, 4.0, 0.2]])
        offsets = ORACLE.offsets_from_anchors(
            coords, labels, tree_ids, anchors)
        np.testing.assert_allclose(offsets[0], [1.0, 0.0, 0.1])
        np.testing.assert_allclose(offsets[4], [-1.0, 0.0, -0.2])

    def test_baseline_reference_requires_exact_detection_counts(self):
        reference = {'tp': 156, 'fp': 5, 'fn': 0}
        self.assertTrue(ORACLE.baseline_matches_reference(
            {'tp': 156, 'fp': 5, 'fn': 0, 'f1': 0.9}, reference))
        self.assertFalse(ORACLE.baseline_matches_reference(
            {'tp': 155, 'fp': 5, 'fn': 1, 'f1': 0.9}, reference))

    def test_vote_cell_purity_separates_mixed_and_pure_cells(self):
        votes = np.asarray([
            [0.1, 0.1], [0.2, 0.1], [0.3, 0.2], [1.2, 0.1],
        ])
        labels = np.asarray([1, 1, 2, 3])
        purity = ORACLE.vote_cell_purity(votes, labels, 1.0)
        np.testing.assert_allclose(purity[:2], 2 / 3)
        self.assertAlmostEqual(float(purity[2]), 1 / 3)
        self.assertAlmostEqual(float(purity[3]), 1.0)

    def test_seed_utility_combines_vote_error_purity_and_gt_validity(self):
        coords = np.zeros((4, 3), dtype=np.float32)
        predictions = np.asarray([
            [0.0, 0.0, 0.0], [0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        targets = np.zeros((4, 3), dtype=np.float32)
        labels = np.asarray([1, 1, 2, 0])
        result = ORACLE.compute_seed_utility(
            coords, predictions, targets, labels,
            np.ones(4, dtype=bool), sigma_m=0.3,
            purity_voxel_size=1.0, purity_power=1.0)
        self.assertGreater(result['utility'][0], result['utility'][1])
        self.assertGreater(result['utility'][0], result['utility'][2])
        self.assertEqual(float(result['utility'][3]), 0.0)

    def test_balanced_scores_give_each_tree_the_same_rank_range(self):
        utility = np.asarray([0.1, 0.9, 0.2, 0.8, 0.7, 0.6])
        labels = np.asarray([1, 1, 2, 2, 2, 2])
        mask = np.ones(6, dtype=bool)
        scores = ORACLE.tree_balanced_oracle_scores(
            utility, labels, mask)
        np.testing.assert_allclose(np.sort(scores[:2]), [0.5, 1.0])
        np.testing.assert_allclose(
            np.sort(scores[2:]), [0.25, 0.5, 0.75, 1.0])

    def test_exact_top_ratio_is_deterministic_for_ties(self):
        mask = np.ones(5, dtype=bool)
        selected = ORACLE.exact_top_ratio_mask(
            mask, np.ones(5), keep_ratio=0.4)
        np.testing.assert_array_equal(
            np.flatnonzero(selected), np.asarray([0, 1]))

    def test_gate_requires_oracle_effect_and_random_advantage(self):
        modes = {
            'baseline': {
                'f1': 0.70, 'commission': 0.20, 'completeness': 0.65},
            'oracle_balanced': {
                'f1': 0.73, 'commission': 0.15, 'completeness': 0.648},
        }
        random = [
            {'f1': 0.71}, {'f1': 0.712}, {'f1': 0.708},
        ]
        gate, effects = ORACLE.assess_gate(modes, random, {
            'min_f1_gain_pp': 1.0,
            'min_commission_reduction_pp': 2.0,
            'max_completeness_drop_pp': 0.5,
            'min_gain_over_random_pp': 0.5,
        })
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(effects['f1_gain_pp'], 3.0)


if __name__ == '__main__':
    unittest.main()
