import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'training' / 'train_vertical_attention.py')
SPEC = importlib.util.spec_from_file_location(
    'vertical_attention_standalone', MODULE_PATH)
ATTENTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTENTION)


class VerticalAttentionTests(unittest.TestCase):

    def test_height_encoding_is_finite_and_position_dependent(self):
        encoding = ATTENTION.sinusoidal_height_encoding(8, 64)
        self.assertEqual(encoding.shape, (8, 64))
        self.assertTrue(np.isfinite(encoding).all())
        self.assertFalse(np.allclose(encoding[0], encoding[-1]))

    @unittest.skipUnless(
        importlib.util.find_spec('torch') is not None,
        'PyTorch is unavailable in this test environment.')
    def test_attention_masks_layers_and_isolates_batches(self):
        import torch

        torch.manual_seed(7)
        model = ATTENTION.build_vertical_attention(6, {
            'hidden_dim': 8,
            'num_heads': 2,
            'num_layers': 1,
            'token_mlp_layers': 2,
            'feedforward_dim': 16,
            'instance_hidden_dims': [8],
            'dropout': 0.0,
            'num_vertical_layers': 4,
        }).eval()
        tokens = torch.randn(2, 4, 6)
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False]])
        first = model(tokens, mask)
        changed = tokens.clone()
        changed[0, 2:] = 1e6
        changed[1] = -1e6
        second = model(changed, mask)
        self.assertEqual(first[0].shape, (2,))
        self.assertEqual(first[1].shape, (2,))
        self.assertTrue(torch.allclose(first[0][0], second[0][0], atol=1e-6))
        self.assertTrue(torch.allclose(first[1][0], second[1][0], atol=1e-6))
        self.assertTrue(torch.all((first[1] >= 0) & (first[1] <= 1)))

    @staticmethod
    def metric(mean):
        return {'mean': mean, 'sample_std': 0.0, 'min': mean, 'max': mean}

    @classmethod
    def e4_summary(cls):
        return {
            'vertical_mlp': {
                'roc_auc': cls.metric(0.99),
                'fp_ap': cls.metric(0.98),
                'iou_mae': cls.metric(0.09),
                'spearman': cls.metric(0.81),
                'filtered_f1': cls.metric(0.90),
                'filtered_completeness': cls.metric(0.99),
                'filtered_commission': cls.metric(0.08),
            },
            'gate': {'passed': True},
        }

    @staticmethod
    def e4_metrics():
        return {
            seed: {
                'filtered_f1': 0.90,
                'filtered_completeness': 0.99,
                'filtered_commission': 0.08,
                'fp_ap': 0.98,
            }
            for seed in (42, 43, 44)
        }

    @staticmethod
    def settings():
        return {
            'seeds': [42, 43, 44],
            'gate': {
                'min_f1_improvement': 0.005,
                'min_commission_reduction': 0.02,
                'max_completeness_drop': 0.01,
                'fp_ap_tolerance': 0.0001,
                'min_paired_seed_wins': 2,
                'paired_win_tolerance': 0.000001,
            },
        }

    @staticmethod
    def dataset():
        return {
            'instance_id': np.arange(4),
            'vertical_tokens': np.zeros((4, 8, 72), dtype=np.float32),
        }

    @staticmethod
    def rows(f1_values=(0.906, 0.907, 0.908), commission=0.07):
        return [{
            'model': 'vertical_attention',
            'seed': seed,
            'roc_auc': 0.992,
            'positive_ap': 0.995,
            'fp_ap': 0.981,
            'iou_mae': 0.085,
            'spearman': 0.82,
            'filtered_f1': f1,
            'filtered_completeness': 0.985,
            'filtered_commission': commission,
            'unfiltered_f1': 0.80,
            'inference_ms_per_instance': 0.01,
        } for seed, f1 in zip((42, 43, 44), f1_values)]

    def test_gate_passes_mean_gain_and_paired_seed_rules(self):
        summary = ATTENTION.summarize_attention(
            self.rows(), self.e4_summary(), self.e4_metrics(),
            self.settings(), self.dataset(), parameter_count=1000)
        self.assertTrue(summary['gate']['mean_f1_gain_passed'])
        self.assertEqual(summary['paired_seed_wins'], 3)
        self.assertTrue(summary['gate']['passed'])

    def test_gate_rejects_small_effect_size(self):
        summary = ATTENTION.summarize_attention(
            self.rows((0.902, 0.903, 0.904), commission=0.07),
            self.e4_summary(), self.e4_metrics(),
            self.settings(), self.dataset(), parameter_count=1000)
        self.assertFalse(summary['gate']['effect_size_passed'])
        self.assertFalse(summary['gate']['passed'])

    def test_gate_requires_two_paired_seed_wins(self):
        summary = ATTENTION.summarize_attention(
            self.rows((0.93, 0.89, 0.89), commission=0.05),
            self.e4_summary(), self.e4_metrics(),
            self.settings(), self.dataset(), parameter_count=1000)
        self.assertTrue(summary['gate']['commission_tradeoff_passed'])
        self.assertEqual(summary['paired_seed_wins'], 1)
        self.assertFalse(summary['gate']['passed'])


    def test_layer_count_uses_model_config_without_duplicate_top_level_key(self):
        dataset = {
            'vertical_tokens': np.zeros((2, 8, 4), dtype=np.float32)}
        self.assertEqual(
            ATTENTION.validate_vertical_layer_count(
                dataset, {'num_vertical_layers': 8}),
            8)
        with self.assertRaises(ValueError):
            ATTENTION.validate_vertical_layer_count(
                dataset, {'num_vertical_layers': 6})


if __name__ == '__main__':
    unittest.main()
