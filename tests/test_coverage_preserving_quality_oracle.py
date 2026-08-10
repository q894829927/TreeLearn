import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_coverage_preserving_quality_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'coverage_preserving_quality_oracle', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class CoveragePreservingQualityOracleTests(unittest.TestCase):

    def test_dual_targets_choose_highest_iou_and_deterministic_tie(self):
        result = ORACLE.derive_dual_risk_targets(
            instance_ids=[5, 2, 7, 9],
            max_iou=[0.8, 0.8, 0.7, 0.1],
            best_gt_ids=[11, 11, 12, 0],
            classification_valid=[1, 1, 1, 1],
            target_is_true_tree=[1, 1, 1, 0])
        # ID 2 wins the IoU tie for GT 11; ID 7 represents GT 12.
        np.testing.assert_array_equal(
            result['coverage_critical'], [False, True, True, False])
        np.testing.assert_array_equal(
            result['safe_reject'], [True, False, False, True])
        np.testing.assert_array_equal(
            result['redundant_positive'], [True, False, False, False])

    def test_unknown_instances_are_protected_and_not_supervised(self):
        result = ORACLE.derive_dual_risk_targets(
            instance_ids=[1, 2, 3],
            max_iou=[0.9, 0.3, 0.0],
            best_gt_ids=[1, 1, 0],
            classification_valid=[1, 0, 1],
            target_is_true_tree=[1, 0, 0])
        np.testing.assert_array_equal(
            result['protected_unknown'], [False, True, False])
        self.assertFalse(result['safe_reject'][1])
        self.assertFalse(result['coverage_critical'][1])

    def test_oracle_uses_budget_without_rejecting_critical(self):
        ids = np.arange(1, 11)
        targets = {
            'safe_reject': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'coverage_critical': np.asarray(
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
            'counted_negative': np.asarray(
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            'redundant_positive': np.asarray(
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        }
        result = ORACLE.rejection_oracle(ids, targets, reject_ratio=0.2)
        self.assertEqual(result['reject_budget'], 2)
        self.assertEqual(result['num_rejected'], 2)
        self.assertEqual(result['rejected_counted_negative'], 2)
        self.assertEqual(result['critical_rejected'], 0)
        self.assertEqual(result['filtered']['tp'], result['baseline']['tp'])
        self.assertGreater(result['f1_gain_pp'], 0)

    def test_gate_requires_all_validation_plot_wins(self):
        def row(plot, split, f1_gain):
            baseline = ORACLE.detection_proxy(100, 50)
            filtered = ORACLE.detection_proxy(100, 10)
            return {
                'source_plot': plot,
                'split': split,
                'num_instances': 200,
                'num_safe_reject': 50,
                'num_coverage_critical': 100,
                'num_counted_negative': 50,
                'num_redundant_positive': 0,
                'num_protected_unknown': 50,
                'num_classification_valid': 150,
                'oracle': {
                    'baseline': baseline,
                    'filtered': filtered,
                    'f1_gain_pp': f1_gain,
                    'commission_reduction_pp': 20.0,
                    'completeness_drop_pp': 0.0,
                },
            }
        reports = [row(f'T{i}', 'train', 10.0) for i in range(13)]
        reports += [row(f'V{i}', 'validation', 10.0) for i in range(5)]
        settings = {'gate': {
            'expected_train_plots': 13,
            'expected_validation_plots': 5,
            'min_train_safe_reject': 500,
            'min_train_coverage_critical': 1000,
            'min_validation_safe_reject': 200,
            'min_validation_coverage_critical': 400,
            'min_validation_f1_gain_pp': 5.0,
            'min_validation_commission_reduction_pp': 5.0,
            'max_validation_completeness_drop_pp': 0.0,
            'min_validation_plot_wins': 5,
        }}
        _, gate = ORACLE.aggregate_reports(reports, settings)
        self.assertTrue(gate['passed'])
        reports[-1]['oracle']['f1_gain_pp'] = 0.0
        _, failed = ORACLE.aggregate_reports(reports, settings)
        self.assertFalse(failed['per_plot_consistency'])
        self.assertFalse(failed['passed'])


if __name__ == '__main__':
    unittest.main()
