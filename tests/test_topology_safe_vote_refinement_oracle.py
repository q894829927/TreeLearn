import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_topology_safe_vote_refinement_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'topology_safe_vote_refinement_oracle', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class TopologySafeVoteRefinementOracleTests(unittest.TestCase):

    def test_safe_gate_accepts_pure_improving_destination(self):
        base = np.asarray([
            [0.10, 0.10], [0.20, 0.10],
            [2.10, 0.10], [2.20, 0.10],
        ])
        labels = np.asarray([1, 1, 2, 2])
        residual = np.asarray([
            [0.30, 0.00], [0.30, 0.00],
            [-0.30, 0.00], [-0.30, 0.00],
        ])
        gate, stats = ORACLE.topology_safe_cell_gate(
            base, residual, residual, labels, cell_size=1.0)
        np.testing.assert_array_equal(gate, np.ones(4))
        self.assertEqual(stats['num_destination_safe_cells'], 2)
        self.assertEqual(stats['rejected_destination_cells'], 0)

    def test_safe_gate_rejects_mixed_source_and_foreign_destination(self):
        base = np.asarray([
            [0.10, 0.10], [0.20, 0.10],
            [2.10, 0.10], [2.20, 0.10],
            [4.10, 0.10], [4.20, 0.10],
        ])
        labels = np.asarray([1, 0, 2, 2, 3, 3])
        residual = np.asarray([
            [0.20, 0.00], [0.20, 0.00],
            [2.00, 0.00], [2.00, 0.00],
            [0.10, 0.00], [0.10, 0.00],
        ])
        gate, stats = ORACLE.topology_safe_cell_gate(
            base, residual, residual, labels, cell_size=1.0)
        np.testing.assert_array_equal(gate, np.zeros(6))
        self.assertGreater(stats['rejected_mixed_or_non_tree_cells'], 0)
        self.assertGreater(stats['rejected_destination_cells'], 0)

    def test_safe_gate_rejects_cross_tree_proposal_conflict(self):
        base = np.asarray([
            [0.10, 0.10], [0.20, 0.10],
            [2.10, 0.10], [2.20, 0.10],
        ])
        labels = np.asarray([1, 1, 2, 2])
        residual = np.asarray([
            [1.00, 0.00], [1.00, 0.00],
            [-1.00, 0.00], [-1.00, 0.00],
        ])
        gate, stats = ORACLE.topology_safe_cell_gate(
            base, residual, residual, labels, cell_size=1.0)
        np.testing.assert_array_equal(gate, np.zeros(4))
        self.assertEqual(stats['rejected_destination_cells'], 2)

    def test_safe_gate_does_not_degrade_cell_topology(self):
        base = np.asarray([
            [0.10, 0.10], [0.20, 0.10],
            [2.10, 0.10], [2.20, 0.10],
            [4.10, 0.10], [4.20, 0.10],
        ])
        labels = np.asarray([1, 1, 2, 2, 0, 0])
        residual = np.asarray([
            [0.30, 0.00], [0.30, 0.00],
            [-0.30, 0.00], [-0.30, 0.00],
            [0.00, 0.00], [0.00, 0.00],
        ])
        gate, _ = ORACLE.topology_safe_cell_gate(
            base, residual, residual, labels, cell_size=1.0)
        refined = ORACLE.apply_gated_correction(base, residual, gate)
        before = ORACLE.vote_topology_metrics(base, labels, 1.0)
        after = ORACLE.vote_topology_metrics(refined, labels, 1.0)
        self.assertGreaterEqual(
            after['tree_point_cell_purity_mean'],
            before['tree_point_cell_purity_mean'])
        self.assertLessEqual(
            after['multi_tree_collision_cell_rate'],
            before['multi_tree_collision_cell_rate'])

    def test_collision_rate_denominator_triggers_strict_fallback(self):
        base = np.asarray([
            [0.10, 0.10], [0.20, 0.10],
            [2.10, 0.10], [2.20, 0.10],
            [3.10, 0.10], [3.20, 0.10],
        ])
        labels = np.asarray([1, 2, 3, 3, 3, 3])
        residual = np.asarray([
            [0.00, 0.00], [0.00, 0.00],
            [1.00, 0.00], [1.00, 0.00],
            [0.00, 0.00], [0.00, 0.00],
        ])
        gate, stats = ORACLE.topology_safe_cell_gate(
            base, residual, residual, labels, cell_size=1.0)
        self.assertTrue(stats['strict_fallback_applied'])
        self.assertGreater(stats['rejected_topology_fallback_cells'], 0)
        np.testing.assert_array_equal(gate, np.zeros(6))

    def test_aggregate_gate_uses_preregistered_effects(self):
        def mode(mean, p90, purity, collision, f1, commission, recall):
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
                'cluster': {
                    'f1': f1,
                    'commission': commission,
                    'tree_recall': recall,
                },
            }

        base = mode(1.0, 2.0, 0.98, 0.04, 0.80, 0.10, 0.95)
        safe = mode(0.3, 0.8, 0.98, 0.04, 0.82, 0.08, 0.95)
        point = mode(0.0, 0.0, 1.0, 0.0, 0.79, 0.12, 0.98)
        report = {
            'num_candidates': 100,
            'num_tree_candidates': 80,
            'gate_stats': {
                'safe_cell_rate': 0.7,
                'corrected_tree_candidate_rate': 0.8,
            },
            'modes': {
                'base': base,
                'damped_0p25': safe,
                'damped_0p50': safe,
                'damped_0p75': safe,
                'damped_1p00': safe,
                'topology_safe': safe,
                'pointwise_oracle': point,
            },
            'effects': {'cluster_f1_gain_pp': 2.0},
        }
        settings = {
            'damping_factors': [0.25, 0.5, 0.75, 1.0],
            'run_hdbscan': True,
            'gate': {
                'expected_validation_plots': 1,
                'max_pointwise_oracle_mean_error': 1e-5,
                'min_mean_error_reduction_relative': 0.6,
                'min_p90_error_reduction_relative': 0.5,
                'max_cell_purity_drop': 0.001,
                'max_multi_tree_collision_increase': 0.0025,
                'min_cluster_f1_gain_pp': 1.0,
                'min_cluster_commission_reduction_pp': 1.0,
                'max_cluster_tree_recall_drop_pp': 0.5,
                'min_nonnegative_f1_plots': 1,
                'max_single_plot_f1_drop_pp': 0.5,
            },
        }
        _, effects, gate = ORACLE.aggregate_reports([report], settings)
        self.assertAlmostEqual(effects['cluster_f1_gain_pp'], 2.0)
        self.assertTrue(gate['passed'])


if __name__ == '__main__':
    unittest.main()
