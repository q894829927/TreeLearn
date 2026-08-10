import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_multiscale_grouping_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'multiscale_grouping_oracle', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class MultiscaleGroupingOracleTests(unittest.TestCase):

    def test_iou_table_counts_non_tree_points_in_proposal_size(self):
        gt = np.asarray([1, 1, 0, 2, 2, 0])
        pred = np.asarray([0, 0, 0, 1, 1, -1])
        _, gt_ids, iou = ORACLE.proposal_iou_table(gt, pred)
        np.testing.assert_array_equal(gt_ids, [1, 2])
        self.assertAlmostEqual(iou[0, 0], 2 / 3)
        self.assertAlmostEqual(iou[1, 1], 1.0)

    def test_oracle_recovers_tree_from_non_base_scale(self):
        gt = np.asarray([1, 1, 0, 2, 2, 0])
        base = np.asarray([0, 0, 0, -1, -1, -1])
        alternative = np.asarray([0, 0, -1, 1, 1, -1])
        result = ORACLE.multiscale_proposal_oracle(
            gt, {50: base, 25: alternative}, 50, 0.5)
        self.assertEqual(result['scale_metrics']['50']['tp'], 1)
        self.assertEqual(result['oracle']['tp'], 2)
        self.assertEqual(result['num_newly_recovered_trees'], 1)
        self.assertEqual(result['recovered_scale_contribution']['25'], 1)

    def test_tie_is_credited_to_original_scale(self):
        gt = np.asarray([1, 1, 0, 2, 2, 0])
        labels = np.asarray([0, 0, -1, 1, 1, -1])
        result = ORACLE.multiscale_proposal_oracle(
            gt, {25: labels, 50: labels.copy()}, 50, 0.5)
        self.assertEqual(result['selected_scale_contribution']['50'], 2)
        self.assertEqual(result['selected_scale_contribution']['25'], 0)
        self.assertEqual(result['non_base_best_tree_rate'], 0.0)

    def test_gate_uses_recall_and_diversity_not_trivial_oracle_precision(self):
        def row(name, recovered, recall, non_base_rate, inflation):
            return {
                'source_plot': name,
                'num_candidates': 100,
                'num_gt_trees': 100,
                'baseline_reproduced': True,
                'proposal_inflation_vs_base': inflation,
                'num_newly_recovered_trees': recovered,
                'non_base_best_tree_rate': non_base_rate,
                'scale_metrics': {
                    str(size): {
                        'tree_recall': 0.90,
                        'commission': 0.20,
                        'f1': 0.85,
                    }
                    for size in ORACLE.LOCKED_CLUSTER_SIZES
                },
                'oracle': {
                    'tree_recall': recall,
                    'commission': 0.0,
                    'f1': 2 * recall / (1 + recall),
                },
                'effects': {'strict_recovery': recovered > 0},
            }

        settings = {
            'cluster_sizes': ORACLE.LOCKED_CLUSTER_SIZES,
            'base_cluster_size': 50,
            'gate': {
                'expected_validation_plots': 5,
                'min_oracle_tree_recall_gain_pp': 1.0,
                'min_newly_recovered_trees': 15,
                'min_strict_recovery_plots': 3,
                'min_non_base_best_tree_rate': 0.05,
                'max_mean_proposal_inflation': 5.5,
            },
        }
        reports = [row(f'P{i}', 4, 0.94, 0.10, 5.0) for i in range(5)]
        _, effects, gate = ORACLE.aggregate_reports(reports, settings)
        self.assertEqual(effects['total_newly_recovered_trees'], 20)
        self.assertTrue(gate['passed'])

        for report in reports:
            report['oracle']['tree_recall'] = 0.90
            report['num_newly_recovered_trees'] = 0
            report['effects']['strict_recovery'] = False
        _, _, failed = ORACLE.aggregate_reports(reports, settings)
        self.assertFalse(failed['passed'])
        self.assertFalse(failed['oracle_recall_gain_passed'])


if __name__ == '__main__':
    unittest.main()
