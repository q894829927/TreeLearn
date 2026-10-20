"""Server CPU regressions; full sparse GPU paths are covered by the V2 check stage."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tree_learn.forest_abc_v2.data import CenterForest, prepare_centers
from tree_learn.forest_abc_v2.loss import balanced_mass, instance_loss, match_instances
from tree_learn.forest_abc_v2.merge import compete, group_proposals
from tree_learn.query_segmentation.data import prepare_index


class CenterDataTests(unittest.TestCase):
    def test_full_tree_centers_follow_the_same_crop_transform_and_keep_float32(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, index = root / 'cache', root / 'index'
            cache.mkdir()
            xyz = np.random.default_rng(4).uniform(-6, 6, (500, 3)).astype(np.float32)
            labels = np.where(xyz[:, 0] > 0, 8, 3).astype(np.int64)
            labels[:20] = 0
            labels[20:40] = -1
            np.save(cache / 'coords.npy', xyz)
            np.save(cache / 'labels.npy', labels)
            (cache / 'cache.json').write_text('{}')
            prepare_index(cache, index, cell_size=8., with_labels=True)
            row = dict(cache=str(cache), index=str(index), plot='synthetic', split='train')
            prepare_centers(row)
            with np.load(index / 'v2_centers.npz', allow_pickle=False) as targets:
                for i, identifier in enumerate(targets['ids']):
                    np.testing.assert_allclose(targets['centers'][i], xyz[labels == identifier, :2].mean(0), atol=1e-6)
            forest = CenterForest(cache, index, with_labels=True)
            config = dict(radius=16., core_radius=12., voxel_size=.2, minimum_points=8, neighbors=4,
                          max_train_voxels=128, max_inference_voxels=128, scale_min=.5, scale_max=1.25, density_min=.4)
            block = forest.make_block(np.zeros(2), config, 17, training=True)
            for name in ('xyz', 'weights', 'whole_sizes', 'center_offsets'):
                self.assertEqual(block[name].dtype, np.float32)
                self.assertTrue(np.isfinite(block[name]).all())
            centers = []
            for identifier in (3, 8):
                positive = block['labels'] == identifier
                votes = block['xyz'][positive, :2] + block['center_offsets'][positive]
                self.assertGreater(len(votes), 0)
                np.testing.assert_allclose(votes, np.repeat(votes[:1], len(votes), axis=0), atol=1e-6)
                centers.append(votes[0])
            original = np.linalg.norm(forest.gt_centers[0] - forest.gt_centers[1])
            self.assertAlmostEqual(float(np.linalg.norm(centers[0] - centers[1])),
                                   original * block['augmentation_scale'], places=5)
            original_load = np.load

            def guarded(path, *args, **kwargs):
                self.assertNotIn(Path(path).name, {'labels.npy', 'baseline.npy', 'features.npy',
                                                  'votes_xy.npy', 'v2_centers.npz'})
                return original_load(path, *args, **kwargs)

            with patch('numpy.load', side_effect=guarded):
                prediction_forest = CenterForest(cache, index)
                prediction_block = prediction_forest.make_block(np.zeros(2), config, 17)
            self.assertNotIn('center_offsets', prediction_block)
            self.assertNotIn('labels', prediction_block)
            with self.assertRaisesRegex(ValueError, 'only prepared for training'):
                prepare_centers(dict(row, split='external'))


class OwnershipLossTests(unittest.TestCase):
    def synthetic(self):
        torch.manual_seed(5)
        labels = torch.tensor([1, 1, 1, 2, 2, 0, -1])
        block = dict(labels=labels, weights=torch.tensor([1., 2., 1., 1., 1., 1., 0.], dtype=torch.float64),
                     whole_sizes=torch.tensor([4., 4., 4., 2., 2., 0., 0.], dtype=torch.float64),
                     center_offsets=torch.randn(7, 2))
        prediction = dict(anchors=torch.tensor([0, 1, 3, 5, 6]),
            semantic=torch.randn(7, 2, requires_grad=True), embedding=torch.randn(7, 5, requires_grad=True),
            offsets=torch.randn(7, 2, requires_grad=True),
            layers=[dict(masks=torch.randn(7, 7, requires_grad=True), scores=torch.randn(7, requires_grad=True))
                    for _ in range(3)])
        config = dict(dice_weight=2., quality_weight=.5, semantic_query_weight=.5, embedding_weight=.1,
                      auxiliary_weight=.5, ownership_weight=1., center_weight=1., center_scale=5.)
        return prediction, block, config

    def test_loss_backward_dtype_and_ignored_point_invariance(self):
        prediction, block, config = self.synthetic()
        loss, _, components = instance_loss(prediction, block, config, return_components=True)
        changed = copy.deepcopy(prediction)
        for layer in changed['layers']:
            with torch.no_grad():
                layer['masks'][:, -1] = 1000.
        for name in ('semantic', 'embedding', 'offsets'):
            with torch.no_grad():
                changed[name][-1] = -1000.
        other, _ = instance_loss(changed, block, config)
        torch.testing.assert_close(loss, other)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(all(value.dtype == torch.float32 for value in components.values()))
        loss.backward()
        for value in [prediction['semantic'], prediction['embedding'], prediction['offsets']] + [
                value for layer in prediction['layers'] for value in layer.values()]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        torch.testing.assert_close(prediction['layers'][-1]['masks'].grad[:, -1], torch.zeros(7))

    def test_tree_mass_is_equal_despite_unequal_point_counts(self):
        labels = torch.tensor([1, 1, 1, 2, 0])
        weights = torch.tensor([100., 1., 1., 2., 500.])
        mass = balanced_mass(labels, weights)
        self.assertAlmostEqual(float(mass[labels == 1].sum()), .375)
        self.assertAlmostEqual(float(mass[labels == 2].sum()), .375)
        self.assertAlmostEqual(float(mass[labels == 0].sum()), .25)

    def test_matching_finds_distinct_queries_and_rejects_gt_truncation(self):
        logits = torch.tensor([[8., 8., -8., -8.], [7., 7., -7., -7.], [-8., -8., 8., 8.]])
        targets = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
        selected = match_instances(logits, targets, torch.full((4,), .25), 2.)
        self.assertEqual(selected.tolist(), [0, 2])
        with self.assertRaisesRegex(ValueError, 'More GT trees'):
            match_instances(logits[:1], targets, torch.full((4,), .25), 2.)

    def test_unmatched_duplicates_receive_ownership_competition_gradients(self):
        prediction, block, config = self.synthetic()
        _, _, components = instance_loss(prediction, block, config, return_components=True)
        components['ownership'].backward()
        gradients = prediction['layers'][-1]['masks'].grad[:-2, :-1]
        self.assertTrue((gradients.abs().sum(-1) > 0).all())

    def test_unknown_unmatched_query_is_not_a_quality_negative(self):
        prediction, block, config = self.synthetic()
        with torch.no_grad():
            prediction['layers'][-1]['masks'][4, :-1] = -20.
        original, _ = instance_loss(prediction, block, config)
        changed = copy.deepcopy(prediction)
        with torch.no_grad():
            for layer in changed['layers']:
                layer['scores'][4] = 100.
        other, _ = instance_loss(changed, block, config)
        torch.testing.assert_close(original, other)


class MergeTests(unittest.TestCase):
    def record(self, mask, center, score=.8):
        return dict(count=len(mask), center_xy=[center, 0.], score=score, bbox=[0., 0., 0., 20., 2., 2.],
                    block=0, query=0)

    def test_containment_and_center_chains_do_not_merge_distinct_instances(self):
        masks = [np.arange(10), np.arange(3), np.arange(10), np.arange(10)]
        records = [self.record(mask, center, 1 - i * .1) for i, (mask, center) in
                   enumerate(zip(masks, [0., 0., 1., 2.]))]
        groups, assignment = group_proposals(records, lambda i: (masks[i], None), 1.5, .6)
        self.assertEqual(assignment.tolist(), [0, 1, 0, 2])
        self.assertEqual(len(groups), 3)

    def test_point_evidence_beats_a_higher_whole_mask_score_and_reclaims_fragments(self):
        cases = [([np.arange(12), np.arange(6, 12)], [.95, .7], [.55, .95], 2),
                 ([np.arange(3), np.arange(8)], [.95, .8], [.95, .7], 4)]
        for masks, scores, evidence, minimum in cases:
            with self.subTest(minimum=minimum):
                records = [self.record(mask, i * 10., scores[i]) for i, mask in enumerate(masks)]
                values = [np.full(len(mask), evidence[i], dtype=np.float32) for i, mask in enumerate(masks)]
                read = lambda i: (masks[i], values[i])
                groups, assignment = group_proposals(records, read, 1.5, .6)
                labels = np.zeros(12, dtype=np.int32)
                confidence = np.zeros(12, dtype=np.float32)
                stamp = np.zeros(12, dtype=np.int32)
                config = dict(chunk_size=2, minimum_instance_points=minimum,
                              minimum_retention=.25, allocation_score_threshold=.1)
                instances, rejected = compete(records, read, groups, assignment, labels, confidence, stamp, config)
                if minimum == 2:
                    self.assertEqual(len(instances), 2)
                    np.testing.assert_array_equal(labels, np.r_[np.ones(6), np.full(6, 2)])
                else:
                    self.assertEqual(rejected, 1)
                    self.assertEqual(instances[0]['final_count'], 8)
                    np.testing.assert_array_equal(labels[:8], np.ones(8))
                    np.testing.assert_array_equal(labels[8:], np.zeros(4))


class DecoderTests(unittest.TestCase):
    def test_initialization_reuses_old_tensors_and_preserves_new_paths(self):
        from tools.forest_abc_v2.run import initialize
        from tree_learn.forest_abc.modules import RegionGatedQueryUpdate

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(4, 4)
                self.offset = torch.nn.Linear(4, 2)
                self.decoder = torch.nn.Module()
                self.decoder.region_updates = torch.nn.ModuleDict({'1': RegionGatedQueryUpdate(4)})
                self.decoder.cross_attn_layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])

        model = TinyModel()
        new_offset = model.offset.weight.detach().clone()
        new_cross = model.decoder.cross_attn_layers[1].weight.detach().clone()
        state = {key: value.detach().clone() for key, value in model.state_dict().items()
                 if not key.startswith(('offset.', 'decoder.cross_attn_layers.1.'))}
        state['backbone.weight'].fill_(.123)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'v1.pth'
            config = dict(initialization=dict(checkpoint=str(path), step=20000),
                          splits=dict(train=['train'], validation=['val']))
            torch.save(dict(schema=1, check_only=False, step=20000, config=dict(splits=config['splits']), model=state), path)
            initialize(model, config)
            torch.testing.assert_close(model.backbone.weight, state['backbone.weight'])
            torch.testing.assert_close(model.offset.weight, new_offset)
            torch.testing.assert_close(model.decoder.cross_attn_layers[1].weight, new_cross)
            self.assertEqual(float(model.decoder.region_updates['1'].strength), -4.)

    def test_all_cross_attention_and_mfa_paths_backpropagate_with_checkpointing(self):
        from tree_learn.forest_abc_v2.model import CenterDecoder
        torch.manual_seed(8)
        config = dict(channels=8, dim=16, layers=3, heads=4, feedforward=32,
                      center_bandwidth=4., position_scale=16., checkpoint_decoder=True)
        decoder = CenterDecoder(config).train()
        features = torch.randn(12, 8, requires_grad=True)
        xyz = torch.randn(12, 3)
        votes = xyz[:, :2].clone().requires_grad_()
        layers = decoder(features, xyz, votes, torch.tensor([0, 4, 8]))
        self.assertEqual(len(layers), 4)
        objective = sum(layer['masks'].square().mean() + layer['scores'].square().mean() for layer in layers)
        objective.backward()
        for parameter in decoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(votes.grad.abs().sum()), 0.)
        decoder.eval()
        with torch.no_grad():
            result = decoder(features, xyz, votes, torch.tensor([0, 4, 8]))
        self.assertEqual(len(result), 1)
        torch.testing.assert_close(result[0]['masks'], layers[-1]['masks'])


class WholeForestTests(unittest.TestCase):
    def test_full_fusion_without_gt_is_chunk_invariant_and_clears_failure_markers(self):
        from tree_learn.forest_abc_v2.inference import predict_forest

        class Forest:
            metadata = dict(num_points=12)
            centers = np.zeros((2, 2))
            arrays = dict(coords=np.column_stack((np.arange(12), np.zeros((12, 2)))).astype(np.float32))

            def make_block(self, center, blocks, seed):
                xyz = self.arrays['coords'].copy()
                return dict(indices=np.arange(12), xyz=xyz, source_xyz=xyz, grid=xyz.astype(np.int32),
                            neighbors=np.arange(12)[:, None], core=np.ones(12, dtype=bool),
                            center=np.zeros(3, dtype=np.float32), raw_points=12, available_voxels=12)

        class Model(torch.nn.Module):
            def forward(self, block):
                side = block['xyz'][:, 0] < 6
                first = torch.where(side, 8., -8.)
                masks = torch.stack((first, -first, first * 0 - 8, first * 0 + 8))
                return dict(layers=[dict(masks=masks, scores=torch.full((4,), 8.))],
                            centers=torch.tensor([[2., 0.], [8., 0.]]))

        config = dict(seed=42, blocks={}, inference=dict(mask_threshold=.5, quality_threshold=.3,
            ownership_threshold=.5, quality_log_weight=.25, max_proposals_per_block=2,
            center_merge_distance=1.5, mutual_overlap=.6, minimum_instance_points=2,
            minimum_retention=.25, allocation_score_threshold=.1, chunk_size=3,
            max_block_memberships=100, max_forest_memberships=1000, log_every_blocks=5, keep_proposals=False))
        original_load = np.load

        def guarded(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, {'labels.npy', 'baseline.npy', 'features.npy', 'v2_centers.npz'})
            return original_load(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch('numpy.load', side_effect=guarded):
            output = Path(temporary)
            results = []
            for chunk in (3, 7):
                settings = copy.deepcopy(config)
                settings['inference']['chunk_size'] = chunk
                labels, stats = predict_forest(Model(), Forest(), settings, output / str(chunk), torch.device('cpu'))
                try:
                    self.assertEqual(stats['duplicate_groups'], 2)
                    self.assertEqual(stats['final_instances'], 2)
                    results.append(labels.copy())
                finally:
                    labels._mmap.close()
                self.assertFalse((output / str(chunk) / 'proposal_evidence.bin').exists())
            np.testing.assert_array_equal(results[0], results[1])
            np.testing.assert_array_equal(results[0], np.r_[np.ones(6), np.full(6, 2)])
            settings = copy.deepcopy(config)
            settings['inference']['max_block_memberships'] = 1
            failed = output / 'failed'
            failed.mkdir()
            (failed / 'prediction.json').write_text('{"old_success": true}')
            with self.assertRaisesRegex(RuntimeError, 'disk budget exceeded'):
                predict_forest(Model(), Forest(), settings, failed, torch.device('cpu'))
            self.assertFalse((failed / 'prediction.json').exists())

    def test_configuration_rejects_invalid_augments_and_fractional_budgets(self):
        from tools.forest_abc_v2.run import configuration
        import yaml
        path = Path(__file__).resolve().parents[1] / 'configs/experiments/forest_abc_v2/full.yaml'
        config = yaml.safe_load(path.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'config.yaml'
            for group, name, value in [('blocks', 'density_min', 1.1), ('model', 'heads', 1.5),
                                      ('inference', 'max_proposals_per_block', 3000),
                                      ('loss', 'center_weight', float('nan'))]:
                changed = copy.deepcopy(config)
                changed[group][name] = value
                target.write_text(yaml.safe_dump(changed))
                with self.subTest(field=name), self.assertRaises(ValueError):
                    configuration(target)


if __name__ == '__main__':
    unittest.main()
