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


if __name__ == '__main__':
    unittest.main()
