"""Run on the server; pure tensor tests need no spconv installation."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('residual_instance_unit',
    ROOT / 'tree_learn/model/residual_instance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from tree_learn.residual_instance.data import legacy_base_offsets, PairedCropSampler


SETTINGS = dict(max_trees=8, points_per_tree=8, boundary_neighbors=4,
    max_boundary_pairs=16, boundary_radius=2., pull_margin=.5,
    push_margin=1.5, boundary_weight=.5)


class SparseFixture:
    def __init__(self, features, indices):
        self.features, self.indices = features, indices

    def replace_feature(self, features):
        return SparseFixture(features, self.indices)


class ResidualInstanceTests(unittest.TestCase):
    def test_coordinate_alignment_is_order_and_batch_safe(self):
        fine = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 3, 0, 0], [0, 7, 0, 0]])
        coarse = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        features = torch.tensor([[30.], [20.], [10.]], requires_grad=True)
        actual, valid = module.align_parent_features(fine, coarse, features, 1)
        torch.testing.assert_close(actual, torch.tensor([[10.], [20.], [30.], [0.]]))
        self.assertEqual(valid.tolist(), [True, True, True, False])
        actual.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features))

    def test_duplicate_coarse_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            module.align_parent_features(torch.zeros((1, 4), dtype=torch.long),
                torch.zeros((2, 4), dtype=torch.long), torch.ones((2, 2)), 1)

    def fixture(self):
        torch.manual_seed(42)
        fine_indices = torch.tensor([[0, value, 0, 0] for value in range(8)])
        fine = SparseFixture(torch.randn(8, 4), fine_indices)
        taps = {}
        for level in (1, 2):
            indices = torch.tensor([[0, value, 0, 0] for value in range(8 // 2 ** level)])
            taps[level] = SparseFixture(torch.randn(len(indices), 4 * (level + 1)), indices)
        adapter = module.ResidualInstanceAdapter(channels=4, projection_dim=3,
                                                  hidden_dim=8, embedding_dim=3)
        return fine, taps, adapter

    def test_zero_residual_is_exact_identity_and_can_open(self):
        fine, taps, adapter = self.fixture()
        actual = adapter(fine, taps)
        self.assertTrue(torch.equal(actual.features, fine.features))
        self.assertTrue(torch.equal(actual.indices, fine.indices))
        actual.features.square().sum().backward()
        self.assertGreater(abs(float(adapter.gamma.grad)), 0)
        self.assertEqual(float(adapter.fusion[-1].weight.grad.abs().sum()), 0.)
        with torch.no_grad():
            adapter.gamma.fill_(.1)
        adapter.zero_grad()
        adapter(fine, taps).features.square().sum().backward()
        self.assertGreater(float(adapter.fusion[-1].weight.grad.abs().sum()), 0)
        for projection in adapter.projections.values():
            self.assertGreater(float(projection[1].weight.grad.abs().sum()), 0)

    def test_auxiliary_loss_reaches_predictive_features(self):
        fine, taps, adapter = self.fixture()
        with torch.no_grad():
            adapter.gamma.fill_(.1)
        fine.features.requires_grad_(True)
        fused = adapter(fine, taps).features
        batch = dict(coords=torch.arange(24).view(8, 3).float() / 10,
            instance_labels=torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
            batch_ids=torch.zeros(8, dtype=torch.long), masks_sem=torch.ones(8, dtype=torch.bool))
        loss, _ = module.instance_auxiliary_loss(adapter, fused, batch, SETTINGS, 42)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(fine.features.grad.abs().sum()), 0)
        self.assertGreater(float(adapter.embedding[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(adapter.fusion[-1].weight.grad.abs().sum()), 0)

    def test_sampled_negatives_exclude_unknown_background_and_cross_batch(self):
        xyz = torch.arange(36).view(12, 3).float() / 100
        ids = torch.tensor([1, 1, 2, 2, 0, -1, 1, 1, 2, 2, 0, -1])
        batches = torch.tensor([0] * 6 + [1] * 6)
        selected, groups, pairs = module.sample_instance_points(xyz, ids, batches,
            torch.ones(12, dtype=torch.bool), SETTINGS, 42)
        self.assertTrue(np.all(ids[selected].numpy() > 0))
        self.assertEqual(len(np.unique(groups)), 4)
        self.assertTrue(len(pairs) > 0)
        for first, second in selected[pairs]:
            self.assertEqual(int(batches[first]), int(batches[second]))
            self.assertNotEqual(int(ids[first]), int(ids[second]))

    def test_height_strata_cover_low_and_high_points(self):
        xyz = torch.zeros((100, 3))
        xyz[:, 2] = torch.arange(100)
        settings = dict(SETTINGS, points_per_tree=4)
        indices, _, _ = module.sample_instance_points(xyz, torch.ones(100, dtype=torch.long),
            torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.bool), settings, 3)
        self.assertEqual(len(indices), 4)
        self.assertLess(indices.min(), 25)
        self.assertGreaterEqual(indices.max(), 75)

    def test_unknown_features_never_affect_the_auxiliary_objective(self):
        torch.manual_seed(3)
        adapter = module.ResidualInstanceAdapter(channels=4)
        features = torch.randn(6, 4, requires_grad=True)
        batch = dict(coords=torch.zeros(6, 3), instance_labels=torch.tensor([1, 1, 2, 2, -1, 0]),
            batch_ids=torch.zeros(6, dtype=torch.long), masks_sem=torch.ones(6, dtype=torch.bool))
        loss, _ = module.instance_auxiliary_loss(adapter, features, batch, SETTINGS, 3)
        modified = features.detach().clone()
        modified[4:] = 10000
        other, _ = module.instance_auxiliary_loss(adapter, modified, batch, SETTINGS, 3)
        torch.testing.assert_close(loss, other, rtol=0, atol=0)
        loss.backward()
        self.assertEqual(float(features.grad[4:].abs().sum()), 0.)

    def test_centroid_push_is_isolated_by_batch(self):
        torch.manual_seed(7)
        adapter = module.ResidualInstanceAdapter(channels=4)
        features = torch.randn(8, 4)
        batch = dict(coords=torch.zeros(8, 3), instance_labels=torch.tensor([1, 1, 2, 2] * 2),
            batch_ids=torch.tensor([0] * 4 + [1] * 4), masks_sem=torch.ones(8, dtype=torch.bool))
        whole, _ = module.instance_auxiliary_loss(adapter, features, batch, SETTINGS, 7)
        separate = []
        for rows in (slice(0, 4), slice(4, 8)):
            half = {name: value[rows] for name, value in batch.items()}
            value, _ = module.instance_auxiliary_loss(adapter, features[rows], half, SETTINGS, 7)
            separate.append(value)
        torch.testing.assert_close(whole, torch.stack(separate).mean())

    def test_empty_and_single_tree_losses_are_finite(self):
        settings = SETTINGS
        for count in (0, 1, 8):
            features = torch.randn(count, 8, requires_grad=True)
            loss, _ = module.discriminative_terms(features, torch.zeros(count, dtype=torch.long),
                torch.empty((0, 2), dtype=torch.long), settings)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(features.grad).all())

    def test_original_legacy_offset_targets(self):
        from tree_learn.dataset.dataset import TreeDataset
        rng = np.random.default_rng(42)
        xyz = rng.normal(size=(71, 3)).astype(np.float32)
        labels = np.array([1] * 40 + [2] * 10 + [0] * 10 + [-1] * 11)
        dataset = TreeDataset.__new__(TreeDataset)
        dataset.base_anchor_mode = 'legacy'
        dataset.base_anchor_height = .5
        dataset.upper_anchor_mode = 'crown_median'
        dataset.upper_anchor_lower_ratio, dataset.upper_anchor_upper_ratio = .55, .75
        dataset.upper_anchor_min_points = 3
        semantic = np.where(labels == 0, 1, 0)
        expected, _, expected_valid, _ = dataset.getOffset(xyz, labels, semantic)
        actual, valid = legacy_base_offsets(xyz, labels)
        np.testing.assert_array_equal(actual[valid], expected[valid])
        np.testing.assert_array_equal(valid, expected_valid)

    def test_paired_crop_plan_does_not_depend_on_global_rng(self):
        class FakeForest:
            def __init__(self, *args, **kwargs):
                rng = np.random.default_rng(99)
                self.arrays = {'coords': rng.uniform([-2, -2, 0], [2, 2, 20], size=(400, 3)).astype(np.float32)}
                self.labels = np.arange(400) % 3 + 1
                self.centers = np.array([[0., 0.]])

            def block_indices(self, *args):
                return np.arange(400)

        settings = dict(max_open_forests=1, outer_half_extent=17.5, inner_half_extent=4.,
                        max_points=123, minimum_points=16)
        rows = [dict(plot='synthetic', cache='unused', index='unused')]
        with patch('tree_learn.residual_instance.data.RawForest', FakeForest):
            first, plan_a = PairedCropSampler(rows, settings, 42).draw(100)
            np.random.seed(777)
            torch.manual_seed(777)
            second, plan_b = PairedCropSampler(rows, settings, 42).draw(100)
        self.assertEqual(plan_a, plan_b)
        self.assertEqual(len(first['coords']), 123)
        for name in first:
            if torch.is_tensor(first[name]):
                self.assertTrue(torch.equal(first[name], second[name]), name)


if __name__ == '__main__':
    unittest.main()
