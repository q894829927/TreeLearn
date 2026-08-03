import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / 'tools' / 'diagnostics'
sys.path.insert(0, str(DIAGNOSTICS))
MODULE_PATH = DIAGNOSTICS / 'evaluate_quality_guided_merge_oracle.py'
SPEC = importlib.util.spec_from_file_location(
    'quality_guided_merge_oracle_standalone', MODULE_PATH)
E8A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E8A)


class QualityGuidedMergeOracleTests(unittest.TestCase):

    @staticmethod
    def make_tables():
        intersections = np.asarray([
            [60, 0],
            [20, 0],
            [0, 70],
        ], dtype=np.int64)
        pred_sizes = np.asarray([70, 30, 80], dtype=np.int64)
        gt_sizes = np.asarray([100, 100], dtype=np.int64)
        precision = intersections / pred_sizes[:, None]
        recall = intersections / gt_sizes[None, :]
        unions = pred_sizes[:, None] + gt_sizes[None, :] - intersections
        iou = np.divide(
            intersections, unions,
            out=np.zeros_like(intersections, dtype=np.float64),
            where=unions > 0)
        return {
            'pred_ids': np.asarray([10, 20, 30]),
            'gt_ids': np.asarray([101, 102]),
            'pred_sizes': pred_sizes,
            'gt_sizes': gt_sizes,
            'intersections': intersections,
            'precision': precision,
            'recall': recall,
            'iou': iou,
        }

    def test_low_fragment_merges_but_distinct_tree_is_retained(self):
        tables = self.make_tables()
        scores = np.asarray([0.9, 0.1, 0.8])
        merge_map, high_mask, _ = E8A.build_oracle_merge_map(
            tables, scores, keep_ratio=0.66)
        self.assertEqual(merge_map, {1: 0})
        np.testing.assert_array_equal(high_mask, [True, False, True])

        merged = E8A.merge_contingency_rows(tables, merge_map)
        np.testing.assert_array_equal(merged['pred_ids'], [10, 30])
        np.testing.assert_array_equal(
            merged['intersections'], [[80, 0], [0, 70]])
        np.testing.assert_array_equal(merged['pred_sizes'], [100, 80])

    def test_top_ratio_ties_use_smaller_instance_id(self):
        high, low = E8A.top_ratio_masks(
            np.asarray([0.5, 0.5, 0.5]),
            np.asarray([30, 10, 20]),
            keep_ratio=1 / 3)
        np.testing.assert_array_equal(high, [False, True, False])
        np.testing.assert_array_equal(low, [True, False, True])

    def test_gate_requires_all_three_effect_conditions(self):
        settings = {
            'min_f1_gain_pp': 0.5,
            'min_commission_reduction_pp': 2.0,
            'max_completeness_drop_pp': 0.5,
        }
        baseline = {
            'f1_score': 72.0,
            'commission_error_rate': 18.0,
            'completeness': 65.0,
        }
        oracle = {
            'f1_score': 73.0,
            'commission_error_rate': 14.0,
            'completeness': 64.8,
        }
        self.assertTrue(E8A.build_gate(
            baseline, oracle, settings)['passed'])
        oracle['completeness'] = 64.0
        self.assertFalse(E8A.build_gate(
            baseline, oracle, settings)['passed'])


    def test_score_only_metadata_must_prove_unchanged_labels(self):
        metadata = {
            'filter_enabled': False,
            'score_only_labels_identical': True,
            'num_instances': 3,
        }
        checks = E8A.validate_score_only_metadata(metadata, 3)
        self.assertTrue(all(checks.values()))
        metadata['filter_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'score-only control'):
            E8A.validate_score_only_metadata(metadata, 3)


    def test_config_validation_checks_every_run_before_execution(self):
        config = {
            'runs': [
                {
                    'name': 'complete',
                    'dataset_role': 'validation',
                    'ground_truth': 'gt.laz',
                    'evaluation': 'evaluation.pt',
                    'quality_scores': 'scores.csv',
                    'quality_metadata': 'metadata.json',
                    'output_dir': 'output',
                    'primary_gate': True,
                },
                {'name': 'incomplete'},
            ],
        }
        with self.assertRaisesRegex(ValueError, 'runs\\[1\\]'):
            E8A.validate_config_runs(config)


if __name__ == '__main__':
    unittest.main()