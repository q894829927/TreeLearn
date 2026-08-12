import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'diagnose_seed_completion_oracle_test',
    ROOT / 'tools' / 'diagnostics' /
    'diagnose_seed_completion_oracle.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SeedCompletionOracleTests(unittest.TestCase):
    def test_mode_gate_requires_effect_and_safety(self):
        rule = {
            'min_recovered_target_trees': 8,
            'min_f1_gain_pp': 0.25,
            'max_lost_baseline_trees': 2,
            'max_commission_increase_pp': 0.5,
            'min_nonnegative_plots': 4,
        }
        passing = {
            'recovered_target_trees': 8,
            'f1_gain_pp': 0.25,
            'lost_baseline_trees': 2,
            'commission_increase_pp': 0.5,
            'nonnegative_plots': 4,
        }
        self.assertTrue(MODULE.mode_gate(passing, rule)['passed'])
        failing = dict(passing, lost_baseline_trees=3)
        self.assertFalse(MODULE.mode_gate(failing, rule)['passed'])

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
        result = MODULE.aggregate_detection(reports)
        self.assertEqual((result['tp'], result['fp'], result['fn']),
                         (5, 3, 1))

    def test_artifact_requires_exact_tau_min_topup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q3_plot = root / 'q3' / 'validation' / 'V1'
            q3_plot.mkdir(parents=True)
            (q3_plot / 'gt_trees.csv').write_text(
                'gt_tree_id,category\n'
                '7,base_seed_support_failure\n',
                encoding='utf-8')
            (q3_plot / 'summary.json').write_text(
                json.dumps({
                    'baseline': {'tp': 1, 'fp': 0, 'fn': 1},
                    'category_counts': {
                        'base_seed_support_failure': 1,
                    },
                }),
                encoding='utf-8')
            artifact = root / 'summary.json'
            artifact.write_text(json.dumps({
                'baseline': {'tp': 1, 'fp': 0, 'fn': 1},
                'num_target_trees': 1,
                'expected_target_tree_ids': [7],
                'num_baseline_seeds': 49,
                'per_tree_topup': [{
                    'gt_tree_id': 7,
                    'baseline_seed_count': 49,
                    'added_seed_count': 0,
                }],
                'modes': {
                    mode: {} for mode in MODULE.SEED_COMPLETION_MODES
                },
                'source_plot': 'V1',
                'split': 'validation',
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exactly to 50'):
                MODULE.validate_artifact(artifact, 'V1', root / 'q3')

if __name__ == '__main__':
    unittest.main()
