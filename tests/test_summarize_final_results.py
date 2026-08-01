import unittest

import numpy as np

from tools.diagnostics.summarize_final_results import (
    calculate_exact_metrics, summarize_group)


class ExactMetricTests(unittest.TestCase):

    def test_recomputes_unrounded_detection_and_segmentation_metrics(self):
        results = {
            'detection_results': {
                'matched_gts': np.arange(8),
                'non_matched_gts': np.arange(2),
                'matched_preds': np.arange(8),
                'non_matched_preds_filtered': np.arange(2),
            },
            'segmentation_results': {
                'no_partition': {
                    'prec': np.array([0.8, 1.0]),
                    'rec': np.array([0.7, 0.9]),
                    'iou': np.array([0.6, 0.8]),
                },
            },
        }
        metrics = calculate_exact_metrics(results)
        self.assertAlmostEqual(metrics['completeness'], 80.0)
        self.assertAlmostEqual(metrics['commission_error_rate'], 20.0)
        self.assertAlmostEqual(metrics['f1_score'], 80.0)
        self.assertAlmostEqual(metrics['precision'], 90.0)
        self.assertAlmostEqual(metrics['recall'], 80.0)
        self.assertAlmostEqual(metrics['coverage'], 70.0)

    def test_group_uses_sample_standard_deviation(self):
        runs = {
            'a': {'f1_score': 98.0},
            'b': {'f1_score': 99.0},
            'c': {'f1_score': 100.0},
        }
        for run in runs.values():
            run.update({
                'completeness': run['f1_score'],
                'commission_error_rate': 0.0,
                'precision': run['f1_score'],
                'recall': run['f1_score'],
                'coverage': run['f1_score'],
            })
        summary = summarize_group(runs, ['a', 'b', 'c'])
        self.assertAlmostEqual(summary['f1_score']['mean'], 99.0)
        self.assertAlmostEqual(summary['f1_score']['sample_std'], 1.0)


if __name__ == '__main__':
    unittest.main()
