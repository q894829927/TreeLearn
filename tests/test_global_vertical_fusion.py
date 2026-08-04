import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'training' / 'train_global_vertical_fusion.py')
SPEC = importlib.util.spec_from_file_location(
    'global_vertical_fusion_standalone', MODULE_PATH)
FUSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FUSION)


MODEL_CONFIG = {
    'token_hidden_dim': 32,
    'token_mlp_layers': 1,
    'global_projection_dim': 32,
    'fusion_hidden_dims': [64, 32],
    'global_wide_hidden_dims': [112, 64],
    'dropout': 0.0,
}


class GlobalVerticalFusionTests(unittest.TestCase):

    def test_parameter_matched_models_are_within_ten_percent(self):
        global_model = FUSION.build_quality_model(
            35, 72, 'global_wide', MODEL_CONFIG)
        fusion_model = FUSION.build_quality_model(
            35, 72, 'global_vertical_fusion', MODEL_CONFIG)
        global_count = sum(value.numel() for value in global_model.parameters())
        fusion_count = sum(value.numel() for value in fusion_model.parameters())
        difference = abs(global_count - fusion_count) / max(
            global_count, fusion_count)
        self.assertLessEqual(difference, 0.10)

    def test_fusion_shapes_are_finite_and_padding_is_masked(self):
        torch.manual_seed(1)
        model = FUSION.build_quality_model(
            4, 3, 'global_vertical_fusion', MODEL_CONFIG).eval()
        global_values = torch.randn(2, 4)
        tokens = torch.randn(2, 3, 3)
        mask = torch.tensor([
            [True, True, False],
            [True, True, True],
        ])
        first = model(global_values, tokens, mask)
        changed = tokens.clone()
        changed[0, 2] = 10000.0
        second = model(global_values, changed, mask)
        self.assertEqual(tuple(first[0].shape), (2,))
        self.assertTrue(torch.isfinite(first[0]).all())
        self.assertTrue(torch.isfinite(first[1]).all())
        torch.testing.assert_close(first[0][0], second[0][0])
        torch.testing.assert_close(first[1][0], second[1][0])

    def test_global_wide_does_not_depend_on_vertical_inputs(self):
        torch.manual_seed(2)
        model = FUSION.build_quality_model(
            4, 3, 'global_wide', MODEL_CONFIG).eval()
        global_values = torch.randn(2, 4)
        first = model(
            global_values, torch.randn(2, 3, 3),
            torch.ones(2, 3, dtype=torch.bool))
        second = model(
            global_values, torch.randn(2, 5, 3) * 100,
            torch.zeros(2, 5, dtype=torch.bool))
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn unavailable')
    def test_tiny_fusion_training_completes(self):
        rng = np.random.default_rng(7)

        def values(size):
            true = np.arange(size) % 2 == 0
            return {
                'global': rng.normal(size=(size, 4)).astype(np.float32),
                'tokens': rng.normal(size=(size, 3, 3)).astype(np.float32),
                'mask': np.ones((size, 3), dtype=bool),
                'true': true,
                'iou': np.where(true, 0.8, 0.1).astype(np.float32),
                'class_mask': np.ones(size, dtype=bool),
                'valid_mask': np.ones(size, dtype=bool),
            }

        train = values(16)
        validation = values(8)
        result = FUSION.train_quality_model(
            train['global'], train['tokens'], train['mask'],
            train['true'], train['iou'], train['class_mask'],
            train['valid_mask'], validation['global'],
            validation['tokens'], validation['mask'], validation['true'],
            validation['iou'], validation['class_mask'],
            validation['valid_mask'], 'global_vertical_fusion',
            MODEL_CONFIG, {
                'batch_size': 8,
                'epochs': 2,
                'patience': 2,
                'learning_rate': 0.001,
                'weight_decay': 0.001,
                'iou_loss_weight': 0.5,
            }, 42)
        self.assertEqual(result['validity_probability'].shape, (8,))
        self.assertEqual(result['predicted_iou'].shape, (8,))
        self.assertTrue(np.isfinite(result['validity_probability']).all())
        self.assertTrue(np.isfinite(result['predicted_iou']).all())
        self.assertGreater(result['parameter_count'], 0)

if __name__ == '__main__':
    unittest.main()
