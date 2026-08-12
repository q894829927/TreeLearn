import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'q5b0_oracle_test',
    ROOT / 'tools' / 'diagnostics' /
    'diagnose_seed_completion_proposal_oracle.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SeedCompletionProposalOracleTests(unittest.TestCase):
    @staticmethod
    def reference_gate():
        return {
            'max_reference_tp_drift_per_plot': 1,
            'max_reference_fp_drift_per_plot': 1,
        }

    def test_load_settings_rejects_wytham(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(
                'validation_plots: [Wytham]\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Wytham'):
                MODULE.load_settings(path)

    def test_aggregate_detection_sums_counts(self):
        reports = [
            {'baseline': {'tp': 2, 'fp': 1, 'fn': 1}},
            {'baseline': {'tp': 3, 'fp': 2, 'fn': 0}},
        ]
        result = MODULE.aggregate_detection(reports, 'baseline')
        self.assertEqual((result['tp'], result['fp'], result['fn']),
                         (5, 3, 1))


    def test_reference_audit_accepts_one_border_count_drift(self):
        audit = MODULE.audit_baseline_reference(
            {'tp': 9, 'fp': 6, 'fn': 2},
            {'tp': 10, 'fp': 5, 'fn': 1},
            self.reference_gate())
        self.assertTrue(audit['gt_count_preserved'])
        self.assertTrue(audit['passed'])
        self.assertFalse(audit['exact'])
        self.assertEqual(audit['drift'], {'tp': -1, 'fp': 1, 'fn': 1})

    def test_reference_audit_rejects_changed_gt_count(self):
        audit = MODULE.audit_baseline_reference(
            {'tp': 10, 'fp': 5, 'fn': 2},
            {'tp': 10, 'fp': 5, 'fn': 1},
            self.reference_gate())
        self.assertFalse(audit['gt_count_preserved'])
        self.assertFalse(audit['passed'])

    def test_reference_audit_rejects_excess_drift(self):
        audit = MODULE.audit_baseline_reference(
            {'tp': 8, 'fp': 7, 'fn': 3},
            {'tp': 10, 'fp': 5, 'fn': 1},
            self.reference_gate())
        self.assertFalse(audit['tp_drift_passed'])
        self.assertFalse(audit['fp_drift_passed'])
        self.assertFalse(audit['passed'])


if __name__ == '__main__':
    unittest.main()
