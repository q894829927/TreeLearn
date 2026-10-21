"""Run on the server; pure tensor tests need no spconv installation."""

import importlib.util
from contextlib import nullcontext
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
from tree_learn.residual_instance.optimization import optimizer_update


SETTINGS = dict(max_trees=8, points_per_tree=8, boundary_neighbors=4,
    max_boundary_pairs=16, boundary_radius=2., pull_margin=.5,
    push_margin=1.5, boundary_weight=.5)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=.1)
        self.updates = 0

    def step(self, *args, **kwargs):
        self.updates += 1
        return super().step(*args, **kwargs)


class SimulatedScaler:
    """CPU tests of retry bookkeeping; the CUDA test also uses real GradScaler."""

    def __init__(self, enabled=True):
        self.enabled, self.factor, self.bad = enabled, 1024., False

    def is_enabled(self):
        return self.enabled

    def get_scale(self):
        return self.factor if self.enabled else 1.

    def scale(self, loss):
        return loss * self.factor

    def unscale_(self, optimizer):
        self.bad = False
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self.factor)
                    self.bad |= not bool(torch.isfinite(parameter.grad).all())

    def step(self, optimizer):
        if not self.bad:
            optimizer.step()

    def update(self):
        if self.bad:
            self.factor *= .5
        self.bad = False


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

    def test_l2_exact_and_near_zero_errors_have_finite_gradients(self):
        from tree_learn.util.train import masked_offset_loss
        error = torch.tensor([[0., 0., 0.], [1e-12, -1e-12, 0.], [3., 4., 0.]],
                             requires_grad=True)
        value = masked_offset_loss(error, torch.zeros_like(error),
                                   torch.ones(3, dtype=torch.bool), loss_type='l2')
        torch.testing.assert_close(value.detach(), torch.tensor(5. / 3.))
        value.backward()
        self.assertTrue(torch.isfinite(error.grad).all())
        torch.testing.assert_close(error.grad[0], torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(error.grad[-1], torch.tensor([.2, 4. / 15., 0.]))

    def test_l2_masks_and_empty_supervision_preserve_finite_gradients(self):
        from tree_learn.util.train import masked_offset_loss
        for mask in (torch.tensor([True, False]), torch.tensor([False, False])):
            predictions = torch.tensor([[0., 0., 0.], [1., 2., 3.]], requires_grad=True)
            value = masked_offset_loss(predictions, torch.zeros_like(predictions), mask, loss_type='l2')
            self.assertEqual(float(value), 0.)
            value.backward()
            torch.testing.assert_close(predictions.grad, torch.zeros_like(predictions), rtol=0, atol=0)

    def exercise_update(self, failure, *, enabled=True, retries=2):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        optimizer = CountingSGD(model.parameters())
        scaler = SimulatedScaler(enabled=enabled)
        calls, events = [], []

        def corrupt_gradient(gradient):
            if failure == 'always_bad' or (failure == 'first_bad' and len(calls) == 1):
                return torch.full_like(gradient, float('inf'))
            return gradient

        model.weight.register_hook(corrupt_gradient)

        def closure():
            calls.append(float(torch.rand(())))
            # Failed attempts must never update weights or the optimizer.
            self.assertEqual(optimizer.updates, 0)
            self.assertEqual(float(model.weight.detach()), 1.)
            if failure == 'bad_forward' and len(calls) == 1:
                raise FloatingPointError('Injected non-finite objective.')
            loss = model(torch.ones(1, 1)).square().mean()
            return loss, dict(loss=float(loss.detach()))

        with patch('tree_learn.residual_instance.optimization.torch.cuda.amp.autocast',
                   side_effect=lambda **kwargs: nullcontext()):
            if failure == 'always_bad':
                with self.assertRaisesRegex(FloatingPointError, 'FP32 optimizer update rejected'):
                    optimizer_update(model, optimizer, scaler, closure, 100.,
                                     fp16_retries=retries, on_event=events.append)
                update = None
            else:
                _, update = optimizer_update(model, optimizer, scaler, closure, 100.,
                    fp16_retries=retries, on_event=events.append)
        self.assertTrue(all(value == calls[0] for value in calls))
        return model, optimizer, scaler, calls, events, update

    def test_amp_overflow_retries_same_crop_and_updates_exactly_once(self):
        model, optimizer, scaler, calls, events, update = self.exercise_update('first_bad')
        self.assertEqual(len(calls), 2)
        self.assertEqual(optimizer.updates, 1)
        self.assertEqual(scaler.get_scale(), 512.)
        self.assertEqual(events[0]['parameters'][0]['parameter'], 'weight')
        self.assertEqual(events[0]['parameters'][0]['inf'], 1)
        self.assertEqual(events[-1]['event'], 'numerical_recovered')
        self.assertEqual(update['update_precision'], 'fp16')
        self.assertFalse(update['fp32_fallback'])
        torch.testing.assert_close(model.weight.detach(), torch.tensor([[.8]]))

    def test_amp_retry_budget_falls_back_to_fp32(self):
        # An AMP failure with no additional FP16 retries must get an FP32 attempt.
        _, optimizer, _, calls, _, update = self.exercise_update('first_bad', retries=0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(optimizer.updates, 1)
        self.assertEqual(update['update_precision'], 'fp32')
        self.assertTrue(update['fp32_fallback'])

    def test_nonfinite_fp32_gradient_never_updates_parameters(self):
        model, optimizer, _, calls, events, update = self.exercise_update('always_bad')
        self.assertEqual(len(calls), 4)
        self.assertEqual(optimizer.updates, 0)
        self.assertEqual(float(model.weight.detach()), 1.)
        self.assertEqual(events[-1]['event'], 'numerical_failure')
        self.assertEqual(events[-1]['precision'], 'fp32')
        self.assertIsNone(update)

    def test_nonfinite_amp_forward_retries_directly_in_fp32(self):
        _, optimizer, scaler, calls, _, update = self.exercise_update('bad_forward')
        self.assertEqual(len(calls), 2)
        self.assertEqual(optimizer.updates, 1)
        self.assertEqual(scaler.get_scale(), 1024.)
        self.assertTrue(update['fp32_fallback'])

    def test_fp32_mode_performs_one_unscaled_update(self):
        _, optimizer, _, calls, _, update = self.exercise_update('none', enabled=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(optimizer.updates, 1)
        self.assertEqual(update['update_precision'], 'fp32')
        self.assertFalse(update['fp32_fallback'])

    @unittest.skipUnless(torch.cuda.is_available(), 'Real GradScaler regression requires server CUDA.')
    def test_real_cuda_scaler_rejects_overflow_and_retries(self):
        model = torch.nn.Linear(1, 1, bias=False).cuda()
        with torch.no_grad():
            model.weight.fill_(1.)
        optimizer = CountingSGD(model.parameters())
        scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
        calls = []
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf'))
                                   if len(calls) == 1 else gradient)

        def closure():
            calls.append(float(torch.rand((), device='cuda')))
            self.assertEqual(optimizer.updates, 0)
            self.assertEqual(float(model.weight.detach()), 1.)
            loss = model(torch.ones(1, 1, device='cuda')).float().square().mean()
            return loss, dict(loss=float(loss.detach()))

        _, update = optimizer_update(model, optimizer, scaler, closure, 100.)
        self.assertEqual(calls, [calls[0], calls[0]])
        self.assertEqual(optimizer.updates, 1)
        self.assertEqual(scaler.get_scale(), 512.)
        self.assertEqual(update['update_attempts'], 2)
        torch.testing.assert_close(model.weight.detach().cpu(), torch.tensor([[.8]]))


if __name__ == '__main__':
    unittest.main()
