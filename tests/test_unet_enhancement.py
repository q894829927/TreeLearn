import copy
import unittest

import torch

from tree_learn.model.unet_enhancement import (
    HCAGSkipFusion,
    ResidualSkipAdapter,
    SparseSEEnhancement,
    SparseWindowAttentionEnhancement,
    batch_global_mean,
    normalize_sparse_heights,
    selective_kernel_weights,
)


class FakeSparseTensor:

    def __init__(self, features, indices, spatial_shape=(16, 16, 16),
                 batch_size=1):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = None

    def replace_feature(self, features):
        result = copy.copy(self)
        result.features = features
        return result


class UNetEnhancementDenseTests(unittest.TestCase):

    def test_batch_pooling_is_isolated(self):
        features = torch.tensor([[1., 3.], [3., 5.], [100., 200.]])
        batch_ids = torch.tensor([0, 0, 1])
        pooled = batch_global_mean(features, batch_ids, batch_size=2)
        torch.testing.assert_close(
            pooled, torch.tensor([[2., 4.], [100., 200.]]))

    def test_height_normalization_is_per_batch(self):
        indices = torch.tensor([
            [0, 10, 0, 0], [0, 20, 0, 0],
            [1, 100, 0, 0], [1, 300, 0, 0],
        ])
        normalized = normalize_sparse_heights(indices)
        torch.testing.assert_close(
            normalized, torch.tensor([0., 1., 0., 1.]))

    def test_zero_gamma_adapter_is_exact_identity_and_has_gradient(self):
        module = ResidualSkipAdapter(8, 4, gamma_init=0.0)
        skip = torch.randn(11, 8, requires_grad=True)
        decoder = torch.randn(11, 8)
        output = module(skip, decoder)
        torch.testing.assert_close(output, skip)
        output.square().mean().backward()
        self.assertIsNotNone(module.gamma.grad)
        self.assertTrue(torch.isfinite(module.gamma.grad))

    def test_hcag_attention_is_bounded_and_nonconstant(self):
        module = HCAGSkipFusion(
            8, 4, gamma_init=1.0, return_stats=True)
        skip = torch.randn(12, 8)
        decoder = torch.randn(12, 8)
        indices = torch.tensor([
            [0, z, 0, 0] for z in range(6)
        ] + [
            [1, z * 10, 0, 0] for z in range(6)
        ])
        output = module(skip, decoder, indices)
        self.assertEqual(output.shape, skip.shape)
        self.assertGreaterEqual(module.last_stats['attention_mean'], 0.0)
        self.assertLessEqual(module.last_stats['attention_mean'], 1.0)
        self.assertGreater(module.last_stats['attention_std'], 0.0)

    def test_sparse_se_preserves_sparse_metadata(self):
        module = SparseSEEnhancement(8, 4, gamma_init=0.0)
        tensor = FakeSparseTensor(
            torch.randn(7, 8),
            torch.tensor([
                [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0],
                [1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0],
                [1, 3, 0, 0],
            ]),
            batch_size=2)
        output = module(tensor)
        torch.testing.assert_close(output.features, tensor.features)
        self.assertIs(output.indices, tensor.indices)
        self.assertEqual(output.spatial_shape, tensor.spatial_shape)
        self.assertEqual(output.batch_size, tensor.batch_size)

    def test_selective_kernel_weights_sum_to_one(self):
        weights = selective_kernel_weights(torch.randn(3, 2, 16))
        torch.testing.assert_close(
            weights.sum(dim=1), torch.ones(3, 16))
        self.assertTrue(((weights >= 0) & (weights <= 1)).all())

    def test_window_attention_never_crosses_batch_or_window(self):
        module = SparseWindowAttentionEnhancement(
            8, window_size=(4, 4, 4), num_heads=2,
            max_windows_per_chunk=2, gamma_init=1.0,
            return_stats=True)
        module.eval()
        indices = torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 8, 0, 0],
            [1, 0, 0, 0], [1, 1, 0, 0],
        ])
        features = torch.randn(5, 8)
        tensor = FakeSparseTensor(features, indices, batch_size=2)
        first = module(tensor).features.detach()
        changed = features.clone()
        changed[3:] += 1000
        second = module(
            FakeSparseTensor(changed, indices, batch_size=2)
        ).features.detach()
        torch.testing.assert_close(first[:3], second[:3])
        changed_window = features.clone()
        changed_window[2] += 1000
        third = module(
            FakeSparseTensor(changed_window, indices, batch_size=2)
        ).features.detach()
        torch.testing.assert_close(first[:2], third[:2])
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreaterEqual(module.last_stats['attention_mean'], 0.0)
        self.assertLessEqual(module.last_stats['attention_mean'], 1.0)

    def test_window_attention_accepts_empty_tensor(self):
        module = SparseWindowAttentionEnhancement(
            8, window_size=(4, 4, 4), num_heads=2)
        tensor = FakeSparseTensor(
            torch.empty(0, 8),
            torch.empty(0, 4, dtype=torch.long),
            batch_size=2)
        self.assertIs(module(tensor), tensor)

    def test_fp16_modules_are_finite_when_cuda_is_available(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for FP16 attention smoke.')
        module = HCAGSkipFusion(8, 4, gamma_init=1.0).cuda().half()
        skip = torch.randn(9, 8, device='cuda', dtype=torch.float16)
        decoder = torch.randn_like(skip)
        indices = torch.tensor(
            [[0, z, 0, 0] for z in range(9)], device='cuda')
        output = module(skip, decoder, indices)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
