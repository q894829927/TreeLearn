import copy
import importlib.util
import unittest
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / 'tools' / 'training' /
    'train_coverage_preserving_quality.py')
CONFIG_PATH = (
    ROOT / 'configs' / 'experiments' / 'coverage_preserving_quality' /
    'q2_coverage_cvar_mlp.yaml')
SPEC = importlib.util.spec_from_file_location(
    'coverage_preserving_quality_config_test', SCRIPT_PATH)
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


class CoveragePreservingQualityConfigTests(unittest.TestCase):

    def setUp(self):
        self.settings = {
            'data_root': 'data/instance_quality',
            'manifest_path': 'data/instance_quality/manifest.csv',
            'q1_summary_path': 'logs/q1/summary.json',
            'q1_label_path': 'logs/q1/dual_risk_labels.csv',
            'output_dir': 'logs/q2',
            'seeds': [42, 43, 44],
            'models': [
                'quality_iou_control', 'safe_iou_control', 'coverage_cvar'],
            'reject_ratio': 0.15,
            'model': {
                'log_interval': 5,
                'control_iou_loss_weight': 0.5,
                'coverage_cvar_weight': 0.5,
                'coverage_tail_fraction': 0.2,
                'max_completeness_drop_pp': 1.0,
            },
            'gate': {
                'max_coverage_completeness_drop_pp': 1.0,
            },
        }

    def test_preregistered_config_is_valid(self):
        TRAINING.validate_settings(self.settings)
        self.assertEqual(self.settings['seeds'], [42, 43, 44])
        self.assertEqual(self.settings['reject_ratio'], 0.15)
        text = CONFIG_PATH.read_text(encoding='utf-8').lower()
        self.assertNotIn('data/wytham', text)

    def test_wytham_is_forbidden_before_data_loading(self):
        invalid = copy.deepcopy(self.settings)
        invalid['data_root'] = 'data/wytham'
        with self.assertRaisesRegex(ValueError, 'Wytham'):
            TRAINING.validate_settings(invalid)

    def test_reject_ratio_and_selection_gate_are_locked(self):
        invalid = copy.deepcopy(self.settings)
        invalid['reject_ratio'] = 0.2
        with self.assertRaisesRegex(ValueError, '0.15'):
            TRAINING.validate_settings(invalid)
        invalid = copy.deepcopy(self.settings)
        invalid['model']['coverage_tail_fraction'] = 0.0
        with self.assertRaisesRegex(ValueError, 'tail_fraction'):
            TRAINING.validate_settings(invalid)
        invalid = copy.deepcopy(self.settings)
        invalid['model']['coverage_cvar_weight'] = 0.0
        with self.assertRaisesRegex(ValueError, 'cvar_weight'):
            TRAINING.validate_settings(invalid)
        invalid = copy.deepcopy(self.settings)
        invalid['model']['max_completeness_drop_pp'] = 2.0
        with self.assertRaisesRegex(ValueError, 'constraints differ'):
            TRAINING.validate_settings(invalid)


if __name__ == '__main__':
    unittest.main()
