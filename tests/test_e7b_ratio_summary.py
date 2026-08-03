import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / 'tools' / 'diagnostics'
sys.path.insert(0, str(DIAGNOSTICS))
MODULE_PATH = DIAGNOSTICS / 'summarize_e7b_ratio_quality.py'
SPEC = importlib.util.spec_from_file_location(
    'e7b_ratio_summary_standalone', MODULE_PATH)
E7B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E7B)


class E7BRatioSummaryTests(unittest.TestCase):

    @staticmethod
    def make_config(effect_mode='strong'):
        return SimpleNamespace(
            locked=SimpleNamespace(
                checkpoint_seed=42,
                filter_mode='top_ratio',
                keep_ratio=0.85,
            ),
            gate=SimpleNamespace(
                effect_mode=effect_mode,
                max_completeness_drop_pp=1.0,
                min_f1_gain_pp=0.5 if effect_mode == 'strong' else 0.0,
                min_commission_reduction_pp=2.0,
                max_quality_runtime_fraction=0.10,
            ),
        )

    @staticmethod
    def make_metadata(num_kept=316):
        return {
            'checkpoint_seed': 42,
            'filter_enabled': True,
            'filter_mode': 'top_ratio',
            'keep_ratio': 0.85,
            'num_instances': 371,
            'num_kept': num_kept,
        }

    def test_strong_gate_accepts_commission_gain_with_stable_completeness(self):
        baseline = {
            'f1_score': 72.0,
            'commission_error_rate': 18.0,
            'completeness': 64.8,
        }
        filtered = {
            'f1_score': 72.2,
            'commission_error_rate': 14.0,
            'completeness': 64.2,
        }
        gate, delta = E7B.build_ratio_gate(
            baseline, filtered, self.make_metadata(),
            self.make_config(), runtime_fraction=0.05)
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(delta['commission_reduction_pp'], 4.0)

    def test_gate_rejects_incorrect_ceiling_keep_count(self):
        baseline = {
            'f1_score': 98.4,
            'commission_error_rate': 3.1,
            'completeness': 100.0,
        }
        filtered = {
            'f1_score': 98.5,
            'commission_error_rate': 3.0,
            'completeness': 100.0,
        }
        gate, _ = E7B.build_ratio_gate(
            baseline, filtered, self.make_metadata(num_kept=315),
            self.make_config('f1_not_below'), runtime_fraction=0.05)
        self.assertFalse(gate['kept_count_correct'])
        self.assertFalse(gate['passed'])


if __name__ == '__main__':
    unittest.main()