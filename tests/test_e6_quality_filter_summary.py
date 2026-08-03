import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'diagnostics' / 'summarize_e6_quality_filter.py')
SPEC = importlib.util.spec_from_file_location(
    'e6_quality_summary_standalone', MODULE_PATH)
E6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E6)


class E6QualityFilterSummaryTests(unittest.TestCase):

    @staticmethod
    def make_results(false_predictions=1):
        return {
            'detection_results': {
                'matched_gts': np.asarray([1, 2, 3]),
                'non_matched_gts': np.asarray([4]),
                'matched_preds': np.asarray([10, 20, 30]),
                'non_matched_preds_filtered': np.arange(false_predictions),
            },
            'segmentation_results': {
                'no_partition': {
                    'prec': np.asarray([0.8, 1.0]),
                    'rec': np.asarray([0.6, 0.8]),
                    'iou': np.asarray([0.5, 0.7]),
                },
            },
        }

    def test_exact_metrics_recomputes_unrounded_values(self):
        metrics = E6.exact_metrics(self.make_results())
        self.assertAlmostEqual(metrics['completeness'], 75.0)
        self.assertAlmostEqual(metrics['commission_error_rate'], 25.0)
        self.assertAlmostEqual(metrics['f1_score'], 75.0)
        self.assertAlmostEqual(metrics['precision'], 90.0)
        self.assertAlmostEqual(metrics['recall'], 70.0)
        self.assertAlmostEqual(metrics['coverage'], 60.0)

    def test_control_identity_requires_counts_and_metrics(self):
        first = E6.exact_metrics(self.make_results())
        second = dict(first)
        second['counts'] = dict(first['counts'])
        self.assertTrue(E6.metrics_identical(first, second, 1e-12))
        second['counts']['false_pred'] += 1
        self.assertFalse(E6.metrics_identical(first, second, 1e-12))


if __name__ == '__main__':
    unittest.main()
