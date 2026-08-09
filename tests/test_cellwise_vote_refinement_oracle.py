import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_cellwise_vote_refinement_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'cellwise_vote_refinement_oracle', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class CellwiseVoteRefinementOracleTests(unittest.TestCase):

    def test_cellwise_residual_is_shared_and_uses_tree_median(self):
        votes = np.asarray([
            [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [1.1, 0.1],
        ])
        residual = np.asarray([
            [1.0, 0.0], [3.0, 0.0], [99.0, 99.0], [-2.0, 1.0],
        ])
        tree = np.asarray([True, True, False, True])
        corrections, stats = ORACLE.cellwise_shared_residual(
            votes, residual, tree, cell_size=1.0)
        np.testing.assert_allclose(
            corrections[:3], np.asarray([[2.0, 0.0]] * 3))
        np.testing.assert_allclose(corrections[3], [-2.0, 1.0])
        self.assertEqual(stats['num_cells'], 2)
        self.assertEqual(stats['num_tree_supported_cells'], 2)

    def test_global_translation_preserves_pairwise_distances(self):
        votes = np.asarray([[0.0, 0.0], [1.0, 2.0], [-2.0, 4.0]])
        translated = votes + np.asarray([4.2, -3.1])
        before = np.linalg.norm(votes[:, None] - votes[None], axis=2)
        after = np.linalg.norm(
            translated[:, None] - translated[None], axis=2)
        np.testing.assert_allclose(before, after)

    def test_vote_error_metrics_reach_exact_pointwise_ceiling(self):
        base = np.asarray([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        target = np.asarray([[1.0, 0.0], [1.0, 2.0], [0.0, 0.0]])
        tree = np.asarray([True, True, False])
        corrected = base.copy()
        corrected[tree] = target[tree]
        metrics = ORACLE.vote_error_metrics(corrected, target, tree)
        self.assertEqual(metrics['mean_error_xy'], 0.0)
        self.assertEqual(metrics['p90_error_xy'], 0.0)

    def test_topology_detects_multi_tree_collision(self):
        votes = np.asarray([
            [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [2.1, 0.1],
        ])
        labels = np.asarray([1, 1, 2, 2])
        metrics = ORACLE.vote_topology_metrics(votes, labels, 1.0)
        self.assertAlmostEqual(
            metrics['multi_tree_collision_cell_rate'], 0.5)
        self.assertLess(metrics['tree_point_cell_purity_mean'], 1.0)

    def test_seed_cluster_proxy_counts_unmatched_cluster(self):
        labels = np.asarray([1, 1, 2, 2, 0, 0])
        predictions = np.asarray([0, 0, 1, 1, 2, -1])
        metrics = ORACLE.seed_cluster_detection_metrics(
            labels, predictions, min_iou=0.5)
        self.assertEqual(metrics['tp'], 2)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 0)
        self.assertAlmostEqual(metrics['f1'], 0.8)

    def test_aggregate_gate_can_pass_without_hdbscan_for_unit_test(self):
        def mode(mean, p90, purity, collision):
            return {
                'error': {
                    'mean_error_xy': mean,
                    'median_error_xy': mean,
                    'p90_error_xy': p90,
                },
                'topology': {
                    'tree_point_cell_purity_mean': purity,
                    'multi_tree_collision_cell_rate': collision,
                },
            }

        report = {
            'num_candidates': 100,
            'num_tree_candidates': 80,
            'modes': {
                'base': mode(1.0, 2.0, 0.8, 0.2),
                'global_shared': mode(1.0, 2.0, 0.8, 0.2),
                'cellwise_shared': mode(0.2, 0.4, 0.9, 0.1),
                'pointwise_oracle': mode(0.0, 0.0, 1.0, 0.0),
            },
            'effects': {'plot_win': True},
        }
        settings = {
            'run_hdbscan': False,
            'gate': {
                'expected_validation_plots': 1,
                'max_pointwise_oracle_mean_error': 1e-5,
                'min_mean_error_reduction_relative': 0.1,
                'min_p90_error_reduction_relative': 0.05,
                'min_pointwise_improvement_retained': 0.7,
                'max_cell_purity_drop': 0.0,
                'max_multi_tree_collision_increase': 0.0,
                'min_plot_wins': 1,
            },
        }
        _, effects, gate = ORACLE.aggregate_reports([report], settings)
        self.assertAlmostEqual(
            effects['pointwise_improvement_retained'], 0.8)
        self.assertTrue(gate['passed'])


if __name__ == '__main__':
    unittest.main()
