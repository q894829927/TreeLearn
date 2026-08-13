import importlib.util
import pathlib
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / 'tree_learn' / 'model' / 'height_context_attention.py')
SPEC = importlib.util.spec_from_file_location(
    'height_context_attention_standalone', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HeightStratifiedContextAdapter = MODULE.HeightStratifiedContextAdapter
HeightTokenAttentionMixer = MODULE.HeightTokenAttentionMixer
count_trainable_parameters = MODULE.count_trainable_parameters
normalize_heights_from_offsets = MODULE.normalize_heights_from_offsets
pool_height_tokens = MODULE.pool_height_tokens


class HeightContextAttentionTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(13)
        self.features = torch.randn(18, 8)
        self.offsets = torch.randn(18, 3)
        self.offsets[:, 2] = -torch.linspace(0, 12, 18)
        self.input_features = torch.rand(18, 1)
        self.batch_ids = torch.tensor([0] * 9 + [1] * 9)

    def make_adapter(self, adapter_type='attention'):
        return HeightStratifiedContextAdapter(
            in_channels=8,
            hidden_dim=8,
            num_height_bins=4,
            adapter_type=adapter_type,
            num_heads=2,
            dropout=0.0)

    def test_output_shapes_finite_and_exact_zero_initial_residual(self):
        adapter = self.make_adapter().eval()
        output = adapter(
            self.features,
            self.offsets,
            self.input_features,
            self.batch_ids)
        self.assertEqual(output['semantic_residual'].shape, (18, 2))
        self.assertEqual(output['offset_residual'].shape, (18, 3))
        self.assertEqual(output['height_token_valid_mask'].shape, (2, 4))
        self.assertTrue(torch.isfinite(output['semantic_residual']).all())
        self.assertTrue(torch.isfinite(output['offset_residual']).all())
        self.assertTrue(torch.equal(
            output['semantic_residual'],
            torch.zeros_like(output['semantic_residual'])))
        self.assertTrue(torch.equal(
            output['offset_residual'],
            torch.zeros_like(output['offset_residual'])))

    def test_batches_are_isolated(self):
        adapter = self.make_adapter().eval()
        first = adapter(
            self.features,
            self.offsets,
            self.input_features,
            self.batch_ids)

        changed_features = self.features.clone()
        changed_offsets = self.offsets.clone()
        changed_inputs = self.input_features.clone()
        changed_features[9:] += 100
        changed_offsets[9:, 2] -= 100
        changed_inputs[9:] = 0
        second = adapter(
            changed_features,
            changed_offsets,
            changed_inputs,
            self.batch_ids)
        torch.testing.assert_close(
            first['normalized_heights'][:9],
            second['normalized_heights'][:9])
        torch.testing.assert_close(
            first['semantic_residual'][:9],
            second['semantic_residual'][:9])

    def test_empty_bins_are_masked_and_outputs_stay_finite(self):
        features = torch.randn(4, 8)
        heights = torch.zeros(4)
        batch_ids = torch.zeros(4, dtype=torch.long)
        tokens, valid_mask, batch_inverse, bin_indices = pool_height_tokens(
            features, heights, batch_ids, num_height_bins=4)
        self.assertEqual(tokens.shape, (1, 4, 16))
        self.assertEqual(valid_mask.tolist(), [[True, False, False, False]])
        self.assertTrue(torch.equal(batch_inverse, torch.zeros(4).long()))
        self.assertTrue(torch.equal(bin_indices, torch.zeros(4).long()))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_height_normalization_is_batch_local(self):
        offsets = torch.tensor([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -100.0],
            [0.0, 0.0, -200.0],
        ])
        batch_ids = torch.tensor([0, 0, 1, 1])
        first = normalize_heights_from_offsets(offsets, batch_ids)
        changed = offsets.clone()
        changed[2:, 2] *= 10
        second = normalize_heights_from_offsets(changed, batch_ids)
        torch.testing.assert_close(first[:2], second[:2])
        self.assertTrue(torch.all(first >= 0))
        self.assertTrue(torch.all(first <= 1))

    def test_attention_is_real_qkv_and_parameter_matched(self):
        mlp = HeightStratifiedContextAdapter(
            in_channels=32,
            hidden_dim=32,
            num_height_bins=8,
            adapter_type='mlp',
            num_heads=4,
            dropout=0.0)
        attention = HeightStratifiedContextAdapter(
            in_channels=32,
            hidden_dim=32,
            num_height_bins=8,
            adapter_type='attention',
            num_heads=4,
            dropout=0.0)
        self.assertIsInstance(
            attention.token_mixer, HeightTokenAttentionMixer)
        self.assertIsInstance(
            attention.token_mixer.attention, torch.nn.MultiheadAttention)
        mlp_count = count_trainable_parameters(mlp)
        attention_count = count_trainable_parameters(attention)
        relative_difference = abs(mlp_count - attention_count) / mlp_count
        self.assertLess(relative_difference, 0.01)

    def test_residual_heads_receive_gradients_at_initialization(self):
        adapter = self.make_adapter().train()
        output = adapter(
            self.features,
            self.offsets,
            self.input_features,
            self.batch_ids)
        target_semantic = torch.randn_like(output['semantic_residual'])
        target_offset = torch.randn_like(output['offset_residual'])
        loss = (
            (output['semantic_residual'] - target_semantic).pow(2).mean() +
            (output['offset_residual'] - target_offset).pow(2).mean())
        loss.backward()
        self.assertIsNotNone(adapter.semantic_residual_head.weight.grad)
        self.assertIsNotNone(adapter.offset_residual_head.weight.grad)
        self.assertGreater(
            adapter.semantic_residual_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(
            adapter.offset_residual_head.weight.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for FP16')
    def test_fp16_cuda_is_finite(self):
        adapter = self.make_adapter().cuda().half().eval()
        output = adapter(
            self.features.cuda().half(),
            self.offsets.cuda().half(),
            self.input_features.cuda().half(),
            self.batch_ids.cuda())
        self.assertEqual(output['semantic_residual'].dtype, torch.float16)
        self.assertTrue(torch.isfinite(output['semantic_residual']).all())
        self.assertTrue(torch.isfinite(output['offset_residual']).all())


if __name__ == '__main__':
    unittest.main()