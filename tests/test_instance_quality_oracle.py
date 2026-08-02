import importlib.util
import pathlib
import unittest

import numpy as np


repo_root = pathlib.Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location(
    'instance_quality_oracle_standalone',
    repo_root / 'tools' / 'diagnostics' /
    'evaluate_instance_quality_oracle.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)


SCIPY_AVAILABLE = importlib.util.find_spec('scipy') is not None


class InstanceQualityOracleTests(unittest.TestCase):

    def setUp(self):
        self.gt_labels = np.asarray([1, 1, 1, 2, 2, 2, 0, 0])
        self.pred_labels = np.asarray([10, 10, 10, 20, 20, 30, 30, 0])
        self.tables = module.contingency_from_arrays(
            self.gt_labels, self.pred_labels)
        self.qualities = self.tables['iou'].max(axis=1)

    def test_contingency_uses_all_prediction_points_in_iou(self):
        np.testing.assert_array_equal(
            self.tables['gt_ids'], np.asarray([1, 2]))
        np.testing.assert_array_equal(
            self.tables['pred_ids'], np.asarray([10, 20, 30]))
        np.testing.assert_allclose(
            self.qualities, np.asarray([1.0, 2 / 3, 0.25]))
        pred_30 = int(np.flatnonzero(self.tables['pred_ids'] == 30)[0])
        gt_2 = int(np.flatnonzero(self.tables['gt_ids'] == 2)[0])
        self.assertAlmostEqual(
            self.tables['precision'][pred_30, gt_2], 0.5)

    @unittest.skipUnless(SCIPY_AVAILABLE, 'SciPy is required for matching.')
    def test_oracle_filter_removes_counted_false_without_losing_gt(self):
        baseline = module.evaluate_threshold(
            self.tables, self.qualities, 0.0)
        filtered = module.evaluate_threshold(
            self.tables, self.qualities, 0.5)

        self.assertEqual(
            (baseline['tp'], baseline['fp'], baseline['fn']),
            (2, 1, 0))
        self.assertAlmostEqual(baseline['f1_score'], 80.0)
        self.assertEqual(
            (filtered['tp'], filtered['fp'], filtered['fn']),
            (2, 0, 0))
        self.assertAlmostEqual(filtered['completeness'], 100.0)
        self.assertAlmostEqual(filtered['f1_score'], 100.0)

    @unittest.skipUnless(SCIPY_AVAILABLE, 'SciPy is required for matching.')
    def test_baseline_validation_and_completeness_constrained_selection(self):
        rows = [
            module.evaluate_threshold(
                self.tables, self.qualities, threshold)
            for threshold in [0.0, 0.5, 0.75]]
        module.validate_baseline(rows[0], {
            'tp': 2, 'fp': 1, 'fn': 0, 'predictions': 3})

        best = module.select_oracle(rows, max_completeness_drop_pp=1.0)

        self.assertAlmostEqual(best['threshold'], 0.5)
        self.assertAlmostEqual(best['f1_score'], 100.0)

    def test_chunk_accumulation_matches_single_pass(self):
        accumulator = module.new_accumulator()
        module.accumulate_label_chunk(
            accumulator, self.gt_labels[:4], self.pred_labels[:4])
        module.accumulate_label_chunk(
            accumulator, self.gt_labels[4:], self.pred_labels[4:])
        chunked = module.contingency_from_accumulator(accumulator)

        np.testing.assert_array_equal(
            chunked['intersections'], self.tables['intersections'])
        np.testing.assert_array_equal(
            chunked['pred_sizes'], self.tables['pred_sizes'])
        np.testing.assert_array_equal(
            chunked['gt_sizes'], self.tables['gt_sizes'])


    def test_config_validation_checks_every_run_before_execution(self):
        complete = {
            key: key for key in module.REQUIRED_RUN_FIELDS}
        complete['primary_gate'] = True
        incomplete = {'name': 'broken'}

        with self.assertRaisesRegex(ValueError, 'runs\\[1\\] missing'):
            module.validate_config_runs({
                'runs': [complete, incomplete]})

if __name__ == '__main__':
    unittest.main()
