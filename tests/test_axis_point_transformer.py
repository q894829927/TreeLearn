import importlib.util
import pathlib
import unittest

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'tree_learn' / 'model' / 'point_transformer.py'
SPEC = importlib.util.spec_from_file_location(
    'axis_point_transformer_standalone', MODULE_PATH)
POINT_TRANSFORMER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POINT_TRANSFORMER_MODULE)
LocalPointTransformerLayer = \
    POINT_TRANSFORMER_MODULE.LocalPointTransformerLayer
get_axis_branch_target_xy = \
    POINT_TRANSFORMER_MODULE.get_axis_branch_target_xy
AXIS_MODULE_PATH = REPO_ROOT / 'tree_learn' / 'util' / 'axis.py'
AXIS_SPEC = importlib.util.spec_from_file_location(
    'axis_fusion_standalone', AXIS_MODULE_PATH)
AXIS_MODULE = importlib.util.module_from_spec(AXIS_SPEC)
AXIS_SPEC.loader.exec_module(AXIS_MODULE)
get_axis_fused_features = AXIS_MODULE.get_axis_fused_features
filter_base_seeds_by_confidence = \
    AXIS_MODULE.filter_base_seeds_by_confidence


class LocalPointTransformerTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.layer = LocalPointTransformerLayer(
            channels=8,
            num_neighbors=8,
            support_voxel_size=0.4,
            max_support_points=64).eval()

    def test_output_shape_and_finite_values(self):
        features = torch.randn(12, 8)
        coords = torch.randn(12, 3)
        batch_ids = torch.tensor([0] * 6 + [1] * 6)
        output = self.layer(features, coords, batch_ids)
        self.assertEqual(output.shape, features.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_batches_do_not_share_neighbours(self):
        features = torch.randn(10, 8)
        coords = torch.randn(10, 3)
        batch_ids = torch.tensor([0] * 5 + [1] * 5)
        first = self.layer(features, coords, batch_ids)

        changed_features = features.clone()
        changed_coords = coords.clone()
        changed_features[5:] += 100
        changed_coords[5:] += 1000
        second = self.layer(changed_features, changed_coords, batch_ids)
        torch.testing.assert_close(first[:5], second[:5])

    def test_fewer_than_k_neighbours_and_empty_input(self):
        features = torch.randn(2, 8)
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
        ])
        batch_ids = torch.zeros(2, dtype=torch.long)
        output = self.layer(features, coords, batch_ids)
        self.assertTrue(torch.isfinite(output).all())

        empty_output = self.layer(
            torch.empty(0, 8),
            torch.empty(0, 3),
            torch.empty(0, dtype=torch.long))
        self.assertEqual(empty_output.shape, (0, 8))

    def test_gradients_reach_attention_parameters(self):
        self.layer.train()
        features = torch.randn(16, 8)
        coords = torch.randn(16, 3)
        batch_ids = torch.tensor([0] * 8 + [1] * 8)
        self.layer(features, coords, batch_ids).pow(2).mean().backward()
        trainable_gradients = [
            parameter.grad for parameter in self.layer.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in trainable_gradients))
        self.assertTrue(all(
            torch.isfinite(gradient).all()
            for gradient in trainable_gradients))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for FP16')
    def test_fp16_cuda(self):
        layer = LocalPointTransformerLayer(
            channels=8,
            num_neighbors=8,
            support_voxel_size=0.4,
            max_support_points=64).cuda().half().eval()
        features = torch.randn(32, 8, device='cuda', dtype=torch.float16)
        coords = torch.randn(32, 3, device='cuda', dtype=torch.float32)
        batch_ids = torch.zeros(32, device='cuda', dtype=torch.long)
        output = layer(features, coords, batch_ids)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())


class AxisFusionTests(unittest.TestCase):

    def test_zero_weight_is_exact_base_only(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        offset = np.array([[0.5, -0.5, 0.0], [-1.0, 2.0, 0.0]])
        axis = np.array([[100.0, 100.0], [100.0, 100.0]])
        confidence = np.array([[1.0], [0.5]])
        base = coords[:, :2] + offset[:, :2]
        fused = get_axis_fused_features(
            coords, offset, axis, confidence, axis_fusion_weight=0.0)
        self.assertTrue(np.array_equal(base, fused))

    def test_confidence_is_bounded(self):
        log_variance = torch.linspace(-4, 4, 100)
        confidence = torch.sigmoid(-log_variance)
        self.assertGreaterEqual(confidence.min().item(), 0.0)
        self.assertLessEqual(confidence.max().item(), 1.0)

    def test_disabling_confidence_uses_full_axis(self):
        coords = np.zeros((2, 3), dtype=np.float32)
        offset = np.zeros((2, 3), dtype=np.float32)
        axis = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        confidence = np.zeros((2, 1), dtype=np.float32)
        fused = get_axis_fused_features(
            coords,
            offset,
            axis,
            confidence,
            axis_fusion_weight=0.25,
            use_axis_confidence=False)
        np.testing.assert_array_equal(fused, axis * 0.25)

    def test_disabled_seed_filter_is_exact_noop(self):
        seed_mask = np.array([True, False, True, True])
        filtered = filter_base_seeds_by_confidence(
            seed_mask, axis_confidence=None, enabled=False, threshold=0.75)
        np.testing.assert_array_equal(filtered, seed_mask)

    def test_seed_filter_applies_confidence_threshold(self):
        seed_mask = np.array([True, False, True, True])
        confidence = np.array([[0.9], [0.9], [0.65], [0.75]])
        filtered = filter_base_seeds_by_confidence(
            seed_mask,
            axis_confidence=confidence,
            enabled=True,
            threshold=0.75)
        np.testing.assert_array_equal(
            filtered, np.array([True, False, False, True]))

    def test_zero_seed_threshold_is_exact_baseline(self):
        seed_mask = np.array([True, False, True, True])
        confidence = np.array([0.01, 0.25, 0.5, 1.0])
        filtered = filter_base_seeds_by_confidence(
            seed_mask,
            axis_confidence=confidence,
            enabled=True,
            threshold=0.0)
        np.testing.assert_array_equal(filtered, seed_mask)

    def test_seed_filter_requires_confidence(self):
        with self.assertRaisesRegex(ValueError, 'axis_confidence'):
            filter_base_seeds_by_confidence(
                np.array([True]), enabled=True, threshold=0.5)

    def test_top_ratio_retains_exact_highest_confidence_count(self):
        seed_mask = np.array([True, False, True, True, True, True])
        confidence = np.array([0.2, 1.0, 0.9, 0.5, 0.8, 0.1])
        filtered = filter_base_seeds_by_confidence(
            seed_mask,
            axis_confidence=confidence,
            enabled=True,
            mode='top_ratio',
            keep_ratio=0.4)
        np.testing.assert_array_equal(
            filtered,
            np.array([False, False, True, False, True, False]))

    def test_top_ratio_one_is_exact_baseline(self):
        seed_mask = np.array([True, False, True])
        confidence = np.array([0.1, 0.9, 0.2])
        filtered = filter_base_seeds_by_confidence(
            seed_mask,
            axis_confidence=confidence,
            enabled=True,
            mode='top_ratio',
            keep_ratio=1.0)
        np.testing.assert_array_equal(filtered, seed_mask)


class AxisTargetTests(unittest.TestCase):

    def test_base_residual_target_corrects_frozen_vote(self):
        prediction = torch.tensor([
            [0.5, -0.5, 1.0],
            [1.0, 2.0, 3.0],
        ], requires_grad=True)
        label = torch.tensor([
            [0.75, -0.25, 1.0],
            [0.5, 2.5, 3.0],
        ])
        residual = get_axis_branch_target_xy(
            prediction, label, target_mode='base_residual')
        torch.testing.assert_close(
            prediction[:, :2] + residual,
            label[:, :2])
        self.assertFalse(residual.requires_grad)

    def test_upper_axis_target_is_unchanged(self):
        base = torch.tensor([[1.0, 2.0, 3.0]])
        upper = torch.tensor([[4.0, 6.0, 8.0]])
        prediction = torch.zeros_like(base)
        target = get_axis_branch_target_xy(
            prediction, base, upper, target_mode='upper_axis')
        torch.testing.assert_close(
            target, torch.tensor([[3.0, 4.0]]))


if __name__ == '__main__':
    unittest.main()
