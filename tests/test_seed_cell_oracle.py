import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_seed_cell_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'seed_cell_oracle_standalone', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class SeedCellOracleTests(unittest.TestCase):

    def test_vote_cells_are_deterministic(self):
        votes = np.asarray([
            [0.05, 0.05], [0.55, 0.10], [0.61, 0.10], [1.25, -0.10],
        ])
        cells, inverse, counts = ORACLE.vote_cells(votes, 0.6)
        self.assertEqual(int(counts.sum()), 4)
        self.assertEqual(len(cells), 3)
        np.testing.assert_array_equal(counts, np.asarray([2, 1, 1]))
        np.testing.assert_array_equal(inverse, np.asarray([0, 0, 1, 2]))

    def test_weighted_quotas_are_exact_and_suppress_dense_cells(self):
        counts = np.asarray([100, 25, 4])
        mandatory = np.asarray([5, 2, 1])
        quotas = ORACLE.allocate_weighted_cell_quotas(
            counts, keep_count=100, mandatory_counts=mandatory,
            density_exponent=0.5, preserve_nonempty=True)
        self.assertEqual(int(quotas.sum()), 100)
        self.assertTrue(np.all(quotas >= mandatory))
        self.assertTrue(np.all(quotas <= counts))
        # Sparse cells saturate while the dense cell absorbs the reduction.
        self.assertEqual(int(quotas[1]), 25)
        self.assertEqual(int(quotas[2]), 4)
        self.assertEqual(int(quotas[0]), 71)

    def test_cell_selection_keeps_mandatory_and_exact_quotas(self):
        inverse = np.asarray([0, 0, 0, 1, 1, 1])
        quotas = np.asarray([2, 2])
        scores = np.asarray([0.9, 0.8, 0.1, 0.9, 0.8, 0.1])
        mandatory = np.asarray([False, False, True, False, False, True])
        selected = ORACLE.select_with_cell_quotas(
            inverse, quotas, scores, mandatory)
        np.testing.assert_array_equal(
            np.flatnonzero(selected), np.asarray([0, 2, 3, 5]))
        self.assertTrue(np.all(selected[mandatory]))

    def test_global_selection_prioritizes_mandatory(self):
        scores = np.asarray([0.9, 0.8, 0.7, 0.0])
        mandatory = np.asarray([False, False, False, True])
        selected = ORACLE.select_global(scores, 2, mandatory)
        np.testing.assert_array_equal(
            np.flatnonzero(selected), np.asarray([0, 3]))

    def test_structure_metrics_detect_impurity_and_tree_collision(self):
        inverse = np.asarray([0, 0, 0, 0, 1, 1])
        counts = np.asarray([4, 2])
        labels = np.asarray([1, 1, 2, 0, 3, 3])
        utility = np.asarray([0.9, 0.8, 0.7, 0.0, 0.6, 0.5])
        result = ORACLE.cell_structure_metrics(
            inverse, counts, labels, utility)
        self.assertAlmostEqual(result['impure_cell_rate_below_0_90'], 0.5)
        self.assertAlmostEqual(result['multi_tree_collision_cell_rate'], 0.5)
        self.assertGreaterEqual(
            result['between_cell_utility_variance_ratio'], 0.0)
        self.assertLessEqual(
            result['between_cell_utility_variance_ratio'], 1.0)

    def test_selection_metrics_preserve_critical_trees(self):
        selected = np.asarray([1, 0, 1, 1], dtype=bool)
        inverse = np.asarray([0, 0, 1, 1])
        labels = np.asarray([1, 1, 2, 0])
        utility = np.asarray([0.9, 0.2, 0.8, 0.0])
        vote_error = np.asarray([0.1, 0.5, 0.2, 0.0])
        critical = np.asarray([1, 0, 1, 0], dtype=bool)
        result = ORACLE.selection_metrics(
            selected, inverse, labels, utility, vote_error, critical)
        self.assertEqual(result['selected_count'], 3)
        self.assertEqual(result['critical_recall'], 1.0)
        self.assertEqual(result['critical_tree_coverage'], 1.0)
        self.assertAlmostEqual(result['selected_non_tree_rate'], 1 / 3)

    def test_infeasible_nonempty_cell_budget_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot preserve'):
            ORACLE.allocate_weighted_cell_quotas(
                np.ones(4, dtype=np.int64), keep_count=3,
                density_exponent=0.5, preserve_nonempty=True)


if __name__ == '__main__':
    unittest.main()
