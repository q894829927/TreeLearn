import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'tree_learn' / 'util' / 'coverage_quality.py'
SPEC = importlib.util.spec_from_file_location(
    'coverage_quality_model_test', MODULE_PATH)
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)

TRAIN_PATH = (
    ROOT / 'tools' / 'training' /
    'train_coverage_preserving_quality.py')
TRAIN_SPEC = importlib.util.spec_from_file_location(
    'coverage_quality_training_loss_test', TRAIN_PATH)
TRAINING = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAINING)


class CoverageQualityModelTests(unittest.TestCase):

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is required for model tests.')
        self.torch = torch

    def test_two_modes_have_identical_parameter_count_and_shapes(self):
        first = QUALITY.build_two_head_instance_mlp(35, [64, 32], 0.1)
        second = QUALITY.build_two_head_instance_mlp(35, [64, 32], 0.1)
        self.assertEqual(
            QUALITY.count_trainable_parameters(first),
            QUALITY.count_trainable_parameters(second))
        values = self.torch.randn(7, 35)
        first_output, second_output = first(values)
        self.assertEqual(tuple(first_output.shape), (7,))
        self.assertEqual(tuple(second_output.shape), (7,))

    def test_rejection_risks_are_finite_and_bounded(self):
        first = self.torch.tensor([-10.0, 0.0, 10.0])
        second = self.torch.tensor([10.0, 0.0, -10.0])
        for mode in (
                'quality_iou_control', 'safe_iou_control', 'coverage_cvar'):
            risk = QUALITY.coverage_rejection_risk(first, second, mode)
            self.assertTrue(self.torch.isfinite(risk).all())
            self.assertTrue((risk >= 0).all())
            self.assertTrue((risk <= 1).all())


    def test_cvar_adds_positive_tail_penalty_without_changing_risk_api(self):
        first = self.torch.tensor([2.0, -2.0, 0.0])
        second = self.torch.zeros(3)
        true = self.torch.tensor([1.0, 1.0, 0.0])
        target_iou = self.torch.tensor([0.8, 0.7, 0.1])
        safe = self.torch.tensor([0.0, 0.0, 1.0])
        critical = self.torch.tensor([1.0, 1.0, 0.0])
        class_mask = self.torch.ones(3, dtype=self.torch.bool)
        valid_mask = self.torch.ones(3, dtype=self.torch.bool)
        weights = {
            name: self.torch.ones(2)
            for name in ('true', 'safe', 'critical')}
        settings = {
            'control_iou_loss_weight': 0.5,
            'coverage_cvar_weight': 0.5,
            'coverage_tail_fraction': 0.5,
        }
        control = TRAINING._training_loss(
            'safe_iou_control', first, second, true, target_iou, safe,
            critical, class_mask, valid_mask, weights, settings)
        coverage = TRAINING._training_loss(
            'coverage_cvar', first, second, true, target_iou, safe,
            critical, class_mask, valid_mask, weights, settings)
        self.assertGreater(coverage.item(), control.item())
        control_risk = QUALITY.coverage_rejection_risk(
            first, second, 'safe_iou_control')
        coverage_risk = QUALITY.coverage_rejection_risk(
            first, second, 'coverage_cvar')
        self.torch.testing.assert_close(control_risk, coverage_risk)


if __name__ == '__main__':
    unittest.main()
