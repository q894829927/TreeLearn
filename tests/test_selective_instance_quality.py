import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'diagnostics' / 'evaluate_selective_instance_quality.py')
SPEC = importlib.util.spec_from_file_location(
    'selective_instance_quality_standalone', MODULE_PATH)
SELECTIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTIVE)


class SelectiveInstanceQualityTests(unittest.TestCase):

    @staticmethod
    def evaluation():
        return {
            'detection_results': {
                'matched_preds': np.asarray([10, 20]),
                'matched_gts': np.asarray([1, 2]),
                'non_matched_preds': np.asarray([30, 40]),
                'non_matched_preds_filtered': np.asarray([30]),
                'non_matched_gts': np.asarray([3]),
            },
        }

    def test_detection_status_aligns_scores_by_instance_id(self):
        result = SELECTIVE.build_detection_status(
            self.evaluation(),
            np.asarray([40, 30, 20, 10]),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
        )
        np.testing.assert_array_equal(
            result['instance_ids'], [10, 20, 30, 40])
        np.testing.assert_allclose(result['scores'], [0.9, 0.8, 0.2, 0.1])
        np.testing.assert_array_equal(result['status'], [1, 1, -1, 0])
        self.assertEqual(result['num_gt'], 3)

    def test_detection_status_rejects_missing_quality_id(self):
        with self.assertRaisesRegex(ValueError, 'do not match'):
            SELECTIVE.build_detection_status(
                self.evaluation(),
                np.asarray([10, 20, 30, 99]),
                np.asarray([0.9, 0.8, 0.2, 0.1]),
            )

    def test_score_curve_uses_instance_id_for_score_ties(self):
        rows = SELECTIVE.score_curve(
            instance_ids=np.asarray([20, 10, 30]),
            scores=np.asarray([0.9, 0.9, 0.1]),
            status=np.asarray([-1, 1, 1]),
            ratios=[1 / 3, 2 / 3, 1.0],
            num_gt=2,
        )
        self.assertEqual(rows[0]['tp'], 1)
        self.assertEqual(rows[0]['fp'], 0)
        self.assertEqual(rows[1]['fp'], 1)

    def test_random_control_is_reproducible(self):
        first = SELECTIVE.random_curves(
            np.asarray([1, 1, -1, 0]), [0.5, 1.0], 3, 20, 42)
        second = SELECTIVE.random_curves(
            np.asarray([1, 1, -1, 0]), [0.5, 1.0], 3, 20, 42)
        self.assertEqual(first, second)

    def test_good_ranking_improves_both_curve_areas(self):
        status = np.asarray([1, 1, 1, -1, -1, -1])
        score_rows = SELECTIVE.curve_for_order(
            status, np.arange(6), [0.5, 2 / 3, 5 / 6, 1.0], 3)
        random_rows = SELECTIVE.random_curves(
            status, [0.5, 2 / 3, 5 / 6, 1.0], 3, 500, 1)
        summary = SELECTIVE.summarize_curve(score_rows, random_rows)
        self.assertGreater(
            summary['relative_commission_area_reduction'], 0.0)
        self.assertGreater(summary['f1_area_gain_pp'], 0.0)


if __name__ == '__main__':
    unittest.main()
