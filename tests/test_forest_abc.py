"""Server-side checks. Do not execute these tests on the local editing machine."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tree_learn.forest_abc.data import RawForest, neighborhood, prepare_index
from tree_learn.forest_abc.loss import instance_loss
from tree_learn.forest_abc.modules import RegionGatedQueryUpdate, farthest_points
from tree_learn.forest_abc.runtime import resolve_cuda_device


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


class ForestABCModuleTests(unittest.TestCase):
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
