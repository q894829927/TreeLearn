import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / 'tools' / 'training' /
    'train_coverage_preserving_quality.py')
SPEC = importlib.util.spec_from_file_location(
    'coverage_preserving_quality_training_test', SCRIPT_PATH)
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


class CoveragePreservingMetricTests(unittest.TestCase):

    def test_fixed_ratio_rejects_high_risk_safe_instances_per_plot(self):
        ids = np.arange(1, 21)
        plots = np.asarray(['A'] * 10 + ['B'] * 10)
        safe = np.zeros(20, dtype=bool)
        safe[[0, 1, 10, 11]] = True
        critical = ~safe
        unknown = np.zeros(20, dtype=bool)
        risk = np.zeros(20, dtype=np.float64)
        risk[safe] = 0.9
        result = TRAINING.fixed_ratio_metrics(
            risk, ids, plots, safe, critical, unknown, 0.2)
        self.assertEqual(result['num_rejected'], 4)
        self.assertEqual(result['safe_rejected'], 4)
        self.assertEqual(result['critical_rejected'], 0)
        self.assertEqual(result['filtered']['completeness'], 1.0)
        self.assertGreater(result['f1_gain_pp'], 0.0)

    def test_ties_use_lower_instance_id(self):
        result = TRAINING.fixed_ratio_metrics(
            np.ones(10), np.arange(10, 0, -1), np.asarray(['A'] * 10),
            np.ones(10, dtype=bool), np.zeros(10, dtype=bool),
            np.zeros(10, dtype=bool), 0.2)
        rejected_ids = np.arange(10, 0, -1)[result['rejected_mask']]
        np.testing.assert_array_equal(np.sort(rejected_ids), [1, 2])


    def test_metric_rejects_overlapping_target_masks(self):
        with self.assertRaisesRegex(ValueError, 'target masks'):
            TRAINING.fixed_ratio_metrics(
                np.zeros(2), np.asarray([1, 2]), np.asarray(['A', 'A']),
                np.asarray([True, False]), np.asarray([False, True]),
                np.asarray([True, False]), 0.15)


class CoveragePreservingLabelTests(unittest.TestCase):

    def test_q1_labels_align_by_plot_and_instance(self):
        dataset = {
            'source_plot': np.asarray(['P', 'P', 'Q']),
            'instance_id': np.asarray([2, 1, 1]),
            'split': np.asarray(['train', 'train', 'validation']),
            'classification_valid': np.asarray([True, True, False]),
        }
        rows = [
            ('P', 'train', 1, False, True, False, False, False),
            ('Q', 'validation', 1, False, False, False, False, True),
            ('P', 'train', 2, True, False, True, False, False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.csv'
            fields = [
                'source_plot', 'split', 'instance_id', 'safe_reject',
                'coverage_critical', 'counted_negative',
                'redundant_positive', 'protected_unknown']
            with path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerows(rows)
            labels = TRAINING.load_coverage_risk_labels(path, dataset)
        np.testing.assert_array_equal(
            labels['safe_reject'], [True, False, False])
        np.testing.assert_array_equal(
            labels['coverage_critical'], [False, True, False])
        np.testing.assert_array_equal(
            labels['protected_unknown'], [False, False, True])


class CoveragePreservingGateTests(unittest.TestCase):

    @staticmethod
    def _settings():
        return {'seeds': [42, 43, 44], 'reject_ratio': 0.15, 'gate': {
            'max_coverage_completeness_drop_pp': 1.0,
            'min_coverage_f1_gain_over_unfiltered_pp': 1.0,
            'min_coverage_commission_reduction_pp': 2.0,
            'min_coverage_f1_gain_over_control_pp': 0.5,
            'max_coverage_completeness_loss_vs_control_pp': 0.5,
            'min_seed_wins': 2,
            'min_validation_plot_wins': 4,
            'max_coverage_f1_sample_std_pp': 1.0,
        }}

    @staticmethod
    def _row(model, seed):
        coverage = model == 'coverage_cvar'
        return {
            'model': model,
            'seed': seed,
            'parameter_count': 100,
            'safe_roc_auc': 0.9,
            'safe_average_precision': 0.9 if coverage else 0.85,
            'critical_roc_auc': 0.9,
            'critical_average_precision': 0.95,
            'baseline_f1': 0.79,
            'filtered_f1': 0.84 if coverage else 0.82,
            'filtered_completeness': 0.995 if coverage else 0.996,
            'filtered_commission': 0.1 if coverage else 0.12,
            'f1_gain_pp': 5.0 if coverage else 3.0,
            'commission_reduction_pp': 8.0 if coverage else 6.0,
            'completeness_drop_pp': 0.5 if coverage else 0.4,
            'reject_precision': 0.9,
        }

    def test_gate_requires_parameter_match_and_coverage_wins(self):
        rows = [
            self._row(model, seed)
            for model in (
                'quality_iou_control', 'safe_iou_control', 'coverage_cvar')
            for seed in (42, 43, 44)]
        plot_rows = []
        for model in (
                'quality_iou_control', 'safe_iou_control', 'coverage_cvar'):
            for seed in (42, 43, 44):
                for plot in ('A', 'B', 'C', 'D', 'E'):
                    plot_rows.append({
                        'model': model, 'seed': seed, 'source_plot': plot,
                        'filtered_f1': (
                            0.84 if model == 'coverage_cvar' else 0.82)})
        q1 = {'gate': {'passed': True}, 'aggregate': {
            split: {
                'num_instances': 100,
                'num_safe_reject': 30,
                'num_coverage_critical': 60,
                'num_protected_unknown': 10,
            } for split in ('train', 'validation')}}
        summary = TRAINING.summarize(
            rows, plot_rows, self._settings(), q1,
            {'quality_iou_control': 100, 'safe_iou_control': 100,
             'coverage_cvar': 100})
        self.assertTrue(summary['gate']['passed'])
        failed = TRAINING.summarize(
            rows, plot_rows, self._settings(), q1,
            {'quality_iou_control': 100, 'safe_iou_control': 100,
             'coverage_cvar': 101})
        self.assertFalse(failed['gate']['parameter_count_matched'])
        self.assertFalse(failed['gate']['passed'])


if __name__ == '__main__':
    unittest.main()
