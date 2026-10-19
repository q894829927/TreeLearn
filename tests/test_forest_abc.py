"""Server-side checks. Do not execute these tests on the local editing machine."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tree_learn.forest_abc.data import RawForest, entries, neighborhood, prepare_index, validate_cache_files
from tree_learn.forest_abc.loss import instance_loss
from tree_learn.forest_abc.modules import RegionGatedQueryUpdate, farthest_points
from tree_learn.forest_abc.runtime import (
    MaskGradientAudit, resolve_cuda_device, require_resume_compatible,
    restore_training_state, require_trainable_gradients, selection_policy)


class ForestABCWorkflowTests(unittest.TestCase):
    def test_cache_preflight_rejects_missing_or_unaligned_evaluation_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            np.save(cache / 'coords.npy', np.zeros((4, 3), dtype=np.float32))
            np.save(cache / 'labels.npy', np.ones(4, dtype=np.int32))
            validate_cache_files(cache, 'train')  # No baseline needed for training.
            with self.assertRaises(FileNotFoundError):
                validate_cache_files(cache, 'validation')
            np.save(cache / 'baseline.npy', np.ones(3, dtype=np.int32))
            with self.assertRaisesRegex(ValueError, 'aligned integer baseline'):
                validate_cache_files(cache, 'external')
            np.save(cache / 'baseline.npy', np.ones(4, dtype=np.int32))
            validate_cache_files(cache, 'validation')

    def test_external_manifest_rejects_duplicate_plot_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'one'
            cache.mkdir()
            np.save(cache / 'coords.npy', np.zeros((4, 3), dtype=np.float32))
            for name in ('labels', 'baseline'):
                np.save(cache / f'{name}.npy', np.ones(4, dtype=np.int32))
            (cache / 'cache.json').write_text('{}')
            (root / 'manifest.json').write_text(json.dumps(dict(forests=[
                dict(plot='same', cache='one'), dict(plot='same', cache='one')])))
            with self.assertRaisesRegex(ValueError, 'Duplicate forest'):
                entries(dict(external_cache_root=str(root), index_root=str(root / 'index')), external=True)

    def test_resume_rejects_validation_policy_changes_but_allows_log_changes(self):
        config = dict(inference=dict(mask_threshold=.5, quality_threshold=.3, nms_iou=.5,
                      nms_containment=.8, minimum_instance_points=20, minimum_retention=.5,
                      allocation_score_threshold=.2, log_every_blocks=5))
        saved = dict(signature='same', check_only=False, config=copy.deepcopy(config),
                     selection_policy=selection_policy(config))
        require_resume_compatible(saved, 'same', config)
        config['inference']['log_every_blocks'] = 10
        require_resume_compatible(saved, 'same', config)
        config['inference']['minimum_retention'] = .7
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            require_resume_compatible(saved, 'same', config)
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            require_resume_compatible(dict(saved, selection_policy=dict(revision=1)), 'same', saved['config'])
        with self.assertRaisesRegex(ValueError, 'full-training checkpoint'):
            require_resume_compatible(dict(saved, check_only=True), 'same', saved['config'])
        self.assertNotIn('log_every_blocks', selection_policy(config))

    def test_validation_label_and_baseline_changes_invalidate_data_signature(self):
        from tools.forest_abc.run import signature

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'index.json').write_text('{}')
            np.save(root / 'labels.npy', np.ones(4, dtype=np.int32))
            np.save(root / 'baseline.npy', np.ones(4, dtype=np.int32))
            row = dict(plot='validation', split='validation', index=str(root), cache=str(root))
            config = dict(model={}, blocks={}, loss={}, training={}, seed=42)
            original = signature(config, [row])
            np.save(root / 'baseline.npy', np.ones(4, dtype=np.int64))
            changed_baseline = signature(config, [row])
            self.assertNotEqual(original, changed_baseline)
            np.save(root / 'labels.npy', np.ones(4, dtype=np.int64))
            self.assertNotEqual(changed_baseline, signature(config, [row]))

    def test_optimizer_and_sampling_continue_identically_after_serialization(self):
        torch.manual_seed(22)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        rng = np.random.default_rng(22)

        def step():
            values = torch.from_numpy(rng.normal(size=(3, 2)).astype(np.float32))
            target = torch.randn(3, 1)
            optimizer.zero_grad(set_to_none=True)
            (model(values) - target).square().mean().backward()
            require_trainable_gradients(model)
            optimizer.step()

        step()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'check_only.pth'
            torch.save(dict(check_only=True, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            sampler_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=[]), path)
            step()
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            saved = torch.load(path, map_location='cpu', weights_only=False)
            with patch('torch.cuda.set_rng_state_all') as restore_cuda:
                restore_training_state(saved, model, optimizer, rng)
                restore_cuda.assert_called_once_with([])
            step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_total_loss_check_reports_unused_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        model[0](torch.ones(3, 2)).sum().backward()
        with self.assertRaisesRegex(RuntimeError, '1.weight'):
            require_trainable_gradients(model)

    def test_config_rejects_fractional_counts_and_nonfinite_values_before_cuda(self):
        from tools.forest_abc.run import configuration
        import yaml

        path = Path(__file__).resolve().parents[1] / 'configs/experiments/forest_abc/full.yaml'
        original = yaml.safe_load(path.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'config.yaml'
            for group, name, value in [('model', 'heads', 0), ('blocks', 'neighbors', 1.5),
                                       ('training', 'learning_rate', float('nan'))]:
                with self.subTest(field=f'{group}.{name}'):
                    config = copy.deepcopy(original)
                    config[group][name] = value
                    target.write_text(yaml.safe_dump(config))
                    with self.assertRaises(ValueError):
                        configuration(target)

    def test_failed_evaluation_rerun_removes_old_success_summaries(self):
        from tools.forest_abc.run import evaluate_model

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ('summary.json', 'brief.json'):
                (output / name).write_text('{"old_success": true}')
            row = dict(plot='example', cache='unused', index='unused')
            with patch('tools.forest_abc.run.RawForest', side_effect=RuntimeError('new run failed')):
                with self.assertRaisesRegex(RuntimeError, 'new run failed'):
                    evaluate_model({}, None, [row], output, torch.device('cpu'), 5000)
            self.assertFalse((output / 'summary.json').exists())
            self.assertFalse((output / 'brief.json').exists())


class ForestABCAllocationTests(unittest.TestCase):
    def test_rejected_global_fragment_does_not_suppress_later_complete_mask(self):
        from tree_learn.forest_abc.inference import predict_forest
        from tree_learn.query_segmentation.inference import suppress_masks

        # C takes 6 of A's 10 points without satisfying either NMS threshold.
        # A would suppress B by 80% containment, but allocation rejects A at
        # 40% retention. B's five points are all free and should remain usable.
        proposals = [np.r_[0:6, 20:28], np.arange(10), np.arange(6, 11)]
        records = [dict(score=3 - row, count=len(mask), bbox=[0, 0, 0, 30, 1, 1])
                   for row, mask in enumerate(proposals)]
        self.assertEqual(suppress_masks(records, lambda row: proposals[row]), [0, 1])
        self.assertEqual(suppress_masks(records, lambda row: proposals[row], accept=lambda row: row != 1),
                         [0, 2])

        class FakeForest:
            metadata = dict(num_points=28)
            centers = np.zeros((3, 2))
            arrays = dict(coords=np.column_stack((np.arange(28), np.zeros((28, 2)))).astype(np.float32))

            def make_block(self, center, blocks, seed):
                xyz = self.arrays['coords'].copy()
                return dict(xyz=xyz, source_xyz=xyz, indices=np.arange(28),
                            grid=xyz.astype(np.int32), neighbors=np.arange(28)[:, None],
                            core=np.ones(28, dtype=bool), raw_points=28, available_voxels=28)

        class ProposalModel(torch.nn.Module):
            def __init__(self, nonfinite=None):
                super().__init__()
                self.block = 0
                self.nonfinite = nonfinite

            def forward(self, block):
                logits = torch.full((3, 28), -20.)
                logits[0, proposals[self.block]] = 20.
                if self.nonfinite == 'mask':
                    logits[0, 0] = float('nan')
                scores = torch.tensor([3. - self.block, 0., 0.])
                if self.nonfinite == 'score':
                    scores[0] = float('inf')
                self.block += 1
                return dict(layers=[dict(masks=logits, scores=scores)])

        config = dict(seed=42, blocks={}, inference=dict(mask_threshold=.5, quality_threshold=.3,
                      nms_iou=.5, nms_containment=.8, minimum_instance_points=1, minimum_retention=.5,
                      allocation_score_threshold=.2, chunk_size=7, max_block_memberships=4,
                      max_forest_memberships=1000, keep_proposals=False, log_every_blocks=5))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result, stats = predict_forest(ProposalModel(), FakeForest(), config, output / 'valid',
                                           torch.device('cpu'))
            try:
                self.assertEqual(stats['allocation_rejected'], 1)
                self.assertEqual(stats['final_instances'], 2)
                np.testing.assert_array_equal(result[proposals[2]], np.full(5, 2))
                np.testing.assert_array_equal(result[proposals[0]], np.ones(14))
            finally:
                result._mmap.close()
            for kind in ('mask', 'score'):
                with self.subTest(nonfinite=kind):
                    invalid = output / kind
                    invalid.mkdir()
                    (invalid / 'prediction.json').write_text('{"old_success": true}')
                    with self.assertRaisesRegex(FloatingPointError, 'Non-finite prediction'):
                        predict_forest(ProposalModel(nonfinite=kind), FakeForest(), config, invalid,
                                       torch.device('cpu'))
                    self.assertFalse((invalid / 'prediction.json').exists())


class ForestABCDeviceTests(unittest.TestCase):
    def test_unindexed_cuda_resolves_current_logical_device(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.current_device', return_value=2), \
                patch('torch.cuda.set_device') as selected:
            device = resolve_cuda_device('cuda')
        self.assertEqual(device, torch.device('cuda:2'))
        selected.assert_called_once_with(2)

    def test_explicit_cuda_index_is_selected_and_preserved(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.current_device') as current, \
                patch('torch.cuda.set_device') as selected:
            device = resolve_cuda_device('cuda:1')
        self.assertEqual(device, torch.device('cuda:1'))
        current.assert_not_called()
        selected.assert_called_once_with(1)

    def test_cpu_and_unavailable_cuda_fail_before_device_selection(self):
        for specification in ('cpu', 'cuda'):
            with self.subTest(device=specification), \
                    patch('torch.cuda.is_available', return_value=False), \
                    patch('torch.cuda.set_device') as selected:
                with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
                    resolve_cuda_device(specification)
            selected.assert_not_called()


class ForestABCGradientAuditTests(unittest.TestCase):
    def test_later_zero_gradient_does_not_erase_previous_connection_evidence(self):
        audit = MaskGradientAudit(['backbone', 'geometry', 'mfa.layer_1', 'mfa.layer_2'])
        audit.observe([torch.ones(1), torch.ones(1), torch.ones(1), torch.zeros(1)])
        groups, probes = audit.observe([torch.ones(1), torch.ones(1), torch.zeros(1), torch.zeros(1)])
        self.assertEqual(groups['mfa'], 0.)
        self.assertEqual(probes['mfa.layer_1'], 0.)
        report = audit.finish()
        self.assertEqual(report['steps'], 2)
        self.assertGreater(report['group_gradient_l1']['mfa'], 0.)
        self.assertIn('mfa.layer_2', report['zero_gradient_probes'])

    def test_group_inactive_for_the_entire_window_still_fails(self):
        audit = MaskGradientAudit(['backbone', 'mfa.layer_1'])
        for _ in range(3):
            audit.observe([torch.ones(1), torch.zeros(1)])
        with self.assertRaisesRegex(RuntimeError, 'No nonzero.*mfa'):
            audit.finish()

    def test_disconnected_layer_cannot_hide_behind_another_active_layer(self):
        audit = MaskGradientAudit(['backbone', 'mfa.layer_1', 'mfa.layer_2'])
        with self.assertRaisesRegex(RuntimeError, 'disconnected.*mfa.layer_2'):
            audit.observe([torch.ones(1), torch.ones(1), None])

    def test_nonfinite_gradients_always_fail(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value):
                audit = MaskGradientAudit(['mfa.layer_1'])
                with self.assertRaisesRegex(FloatingPointError, 'Non-finite.*mfa.layer_1'):
                    audit.observe([torch.tensor([value])])


class ForestABCModuleTests(unittest.TestCase):
    def test_large_training_crop_keeps_sampling_weights_float32(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, index = root / 'cache', root / 'index'
            cache.mkdir()
            # 41^3 > 65535: NumPy 1.x integer scalar promotion is triggered by
            # this realistic crop size in the old weight expression.
            xyz = np.indices((41, 41, 41)).reshape(3, -1).T.astype(np.float32)
            xyz = (xyz - 20) * np.float32(.4)
            np.save(cache / 'coords.npy', xyz)
            np.save(cache / 'labels.npy', np.ones(len(xyz), dtype=np.int64))
            (cache / 'cache.json').write_text('{}')
            prepare_index(cache, index, cell_size=8., with_labels=True)
            forest = RawForest(cache, index, with_labels=True)
            config = dict(radius=16., core_radius=12., voxel_size=.2, minimum_points=8,
                          max_train_voxels=512, max_inference_voxels=512, neighbors=8)
            block = forest.make_block(np.zeros(2), config, 42, training=True)
            self.assertIsNotNone(block)
            self.assertGreater(block['available_voxels'], 65535)
            self.assertEqual(block['weights'].dtype, np.dtype('float32'))
            self.assertEqual(block['whole_sizes'].dtype, np.dtype('float32'))
            self.assertEqual(block['xyz'].dtype, np.dtype('float32'))
            self.assertTrue(np.isfinite(block['weights']).all())

    def test_tensor_boundary_casts_float_supervision_and_preserves_index_types(self):
        from tree_learn.forest_abc.model import tensor_block

        block = dict(xyz=np.zeros((4, 3), dtype=np.float64),
                     grid=np.zeros((4, 3), dtype=np.int64),
                     neighbors=np.zeros((4, 2), dtype=np.int32),
                     core=np.ones(4, dtype=np.int8), labels=np.ones(4, dtype=np.int32),
                     weights=np.ones(4, dtype=np.float64), whole_sizes=np.ones(4, dtype=np.float64))
        batch = tensor_block(block, torch.device('cpu'), supervision=True)
        for name in ('xyz', 'weights', 'whole_sizes'):
            self.assertEqual(batch[name].dtype, torch.float32)
        self.assertEqual(batch['grid'].dtype, torch.int32)
        self.assertEqual(batch['neighbors'].dtype, torch.int64)
        self.assertEqual(batch['labels'].dtype, torch.int64)
        self.assertEqual(batch['core'].dtype, torch.bool)

    def test_double_supervision_keeps_quality_loss_and_backward_float32(self):
        torch.manual_seed(3)
        block = dict(labels=torch.tensor([1, 1, 0, -1]),
                     weights=torch.tensor([1., 1., 1., 0.], dtype=torch.float64),
                     whole_sizes=torch.tensor([2., 2., 0., 0.], dtype=torch.float64))
        prediction = dict(anchors=torch.tensor([0, 2, 3]),
                          semantic=torch.randn(4, 2, requires_grad=True),
                          embedding=torch.randn(4, 5, requires_grad=True),
                          layers=[dict(masks=torch.randn(5, 4, requires_grad=True),
                                       scores=torch.randn(5, requires_grad=True))])
        config = dict(dice_weight=2., quality_weight=.5, semantic_query_weight=.5, embedding_weight=.1)
        loss, _, components = instance_loss(prediction, block, config, return_components=True)
        self.assertEqual(loss.dtype, torch.float32)
        for component in components.values():
            self.assertEqual(component.dtype, torch.float32)
        loss.backward()
        for value in (prediction['layers'][0]['scores'], prediction['layers'][0]['masks'],
                      prediction['semantic'], prediction['embedding']):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_fps_is_unique_even_for_equal_embeddings(self):
        selected = farthest_points(torch.zeros(20, 5), 12)
        self.assertEqual(len(selected.unique()), 12)

    def test_empty_region_preserves_query_and_nonempty_region_has_gradient(self):
        torch.manual_seed(7)
        module = RegionGatedQueryUpdate(16)
        query = torch.randn(3, 16, requires_grad=True)
        features = torch.randn(8, 16, requires_grad=True)
        masks = torch.full((3, 8), -20.)
        masks[1:, :4] = 20.
        result = module(query, features, masks)
        torch.testing.assert_close(result[0], query[0])
        result[1:].square().sum().backward()
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertGreater(float(module.output[0].weight.grad.abs().sum()), 0)

    def test_all_empty_regions_have_connected_zero_update_gradient(self):
        torch.manual_seed(11)
        module = RegionGatedQueryUpdate(16)
        query = torch.randn(3, 16, requires_grad=True)
        features = torch.randn(8, 16, requires_grad=True)
        result = module(query, features, torch.full((3, 8), -20.))
        torch.testing.assert_close(result, query)
        gradient, = torch.autograd.grad(result.square().sum(), [module.output[0].weight], allow_unused=True)
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(float(gradient.abs().sum()), 0.)

    def test_unknown_points_do_not_change_mask_or_quality_targets(self):
        torch.manual_seed(1)
        block = dict(labels=torch.tensor([1, 1, 0, -1]), weights=torch.tensor([1., 1., 1., 0.]),
                     whole_sizes=torch.tensor([2., 2., 0., 0.]))
        prediction = dict(anchors=torch.tensor([0, 2, 3]), semantic=torch.randn(4, 2),
                          embedding=torch.randn(4, 5),
                          layers=[dict(masks=torch.randn(5, 4), scores=torch.randn(5))])
        weights = dict(dice_weight=2., quality_weight=.5, semantic_query_weight=.5, embedding_weight=.1)
        old, _ = instance_loss(prediction, block, weights)
        changed = copy.deepcopy(prediction)
        changed['layers'][0]['masks'][:, 3] = 1000.
        changed['semantic'][3] = -1000.
        changed['embedding'][3] = 1000.
        new, _ = instance_loss(changed, block, weights)
        torch.testing.assert_close(old, new)

    def test_raw_prediction_blocks_never_open_teacher_or_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, index = root / 'cache', root / 'index'
            cache.mkdir()
            coords = np.random.default_rng(4).uniform(-8, 8, (200, 3)).astype(np.float32)
            np.save(cache / 'coords.npy', coords)
            (cache / 'cache.json').write_text(json.dumps(dict(schema=1)))
            prepare_index(cache, index, cell_size=8., with_labels=False)
            original_load = np.load

            def guarded_load(path, *args, **kwargs):
                self.assertNotIn(Path(path).name, {'labels.npy', 'features.npy', 'votes_xy.npy',
                                                  'tree_probability.npy', 'baseline.npy'})
                return original_load(path, *args, **kwargs)

            with patch('numpy.load', side_effect=guarded_load):
                forest = RawForest(cache, index)
                config = dict(radius=16., core_radius=12., voxel_size=.2, minimum_points=8,
                              max_inference_voxels=128, max_train_voxels=128, neighbors=8)
                block = forest.make_block(np.zeros(2), config, 42)
            self.assertIsNotNone(block)
            self.assertNotIn('labels', block)
            np.testing.assert_array_equal(block['neighbors'][:, 0], np.arange(len(block['xyz'])))

    def test_whole_forest_merge_runs_without_any_gt_file(self):
        from tree_learn.forest_abc.inference import predict_forest

        class CoordinateModel(torch.nn.Module):
            def forward(self, block):
                side = block['xyz'][:, 0] > 0
                positive = torch.where(side, 10., -10.)
                masks = torch.stack((positive, -positive, positive * 0, positive * 0))
                return dict(layers=[dict(masks=masks, scores=torch.full((4,), 8.))])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, index = root / 'cache', root / 'index'
            cache.mkdir()
            np.save(cache / 'coords.npy', np.random.default_rng(5).uniform(-6, 6, (200, 3)).astype(np.float32))
            (cache / 'cache.json').write_text('{}')
            prepare_index(cache, index, cell_size=8., with_labels=False)
            forest = RawForest(cache, index)
            config = dict(seed=42, blocks=dict(radius=16., core_radius=12., voxel_size=.2,
                          minimum_points=8, max_train_voxels=256, max_inference_voxels=256, neighbors=8),
                          inference=dict(mask_threshold=.5, quality_threshold=.3, minimum_instance_points=2,
                          nms_iou=.5, nms_containment=.8, minimum_retention=.5, allocation_score_threshold=.2,
                          chunk_size=37, max_block_memberships=20, max_forest_memberships=100000,
                          log_every_blocks=5, keep_proposals=False))
            result, statistics = predict_forest(CoordinateModel(), forest, config, root / 'prediction',
                                                torch.device('cpu'))
            self.assertEqual(len(result), 200)
            self.assertGreater(statistics['final_instances'], 0)
            self.assertGreater(int((result > 0).sum()), 0)
            result._mmap.close()
            self.assertFalse((root / 'prediction/proposal_members.bin').exists())


if __name__ == '__main__':
    unittest.main()
