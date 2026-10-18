"""Run on the server; small synthetic forests need no spconv or real data."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from tree_learn.query_segmentation.data import Forest, population_sample, prepare_index
from tree_learn.query_segmentation.inference import predict_forest, suppress_masks
from tree_learn.query_segmentation.loss import instance_loss, query_weights
from tree_learn.query_segmentation.model import QueryInstanceDecoder


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('query_workflow', ROOT / 'tools/query_segmentation/run.py')
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def synthetic(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    xyz = np.column_stack([np.repeat([0., 2., 4.], 8), np.tile([0., .2], 12), np.tile(np.arange(8), 3)])
    labels = np.repeat([0, 1, 2], 8).astype(np.int64)
    for name, values in dict(coords=xyz.astype(np.float32),
                             features=rng.normal(size=(24, 32)).astype(np.float32),
                             votes_xy=xyz[:, :2].astype(np.float32),
                             tree_probability=np.where(labels > 0, .9, .1).astype(np.float32),
                             labels=labels, baseline=labels).items():
        np.save(cache / f'{name}.npy', values, allow_pickle=False)
    (cache / 'cache.json').write_text(json.dumps(dict(schema=1, num_points=24, feature_dim=32)))


def settings():
    cfg = yaml.safe_load((ROOT / 'configs/experiments/query_instance_decoder/v1.yaml').read_text())
    cfg['model'].update(dim=16, layers=1, heads=4, feedforward=32, num_queries=8, coverage_queries=2)
    cfg['blocks'].update(max_tokens=16, train_points=32)
    cfg['training'].update(device='cpu', steps=2, smoke_steps=1, lr_warmup=1,
                           sampler_warmup=0, sampler_transition=1, statistics_points_per_forest=8,
                           evaluate_every=2, save_every=1, log_every=1)
    cfg['inference'].update(chunk_size=5, log_every_blocks=1)
    return cfg


class QueryDataTests(unittest.TestCase):
    def test_population_weights_recover_every_known_label_mass(self):
        labels = np.array([0, 0, 0, 1, 1, 2, -1])
        sampled, weights = population_sample(np.arange(7), labels, 30, np.random.default_rng(1))
        self.assertTrue(np.all(labels[sampled] >= 0))
        for identity in (0, 1, 2):
            self.assertAlmostEqual(float(weights[labels[sampled] == identity].sum()),
                                   float(np.count_nonzero(labels == identity)), places=5)

    def test_spatial_index_and_block_membership_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, index = Path(tmp) / 'cache', Path(tmp) / 'index'
            synthetic(cache)
            prepare_index(cache, index)
            forest = Forest(cache, index, with_labels=True)
            center = np.array([1., .1])
            indices = forest.block_indices(center, 1.2)
            xyz = forest.arrays['coords']
            expected = np.flatnonzero(np.square(xyz[:, :2] - center).sum(-1) <= 1.2 ** 2)
            np.testing.assert_array_equal(np.sort(indices), expected)
            block = forest.make_block(forest.centers[0], settings()['blocks'], 42, training=True)
            np.testing.assert_array_equal(block['token_labels'], forest.labels[block['token_indices']])

    def test_inference_never_opens_gt_or_original_instance_labels_and_is_chunk_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic(root / 'cache')
            prepare_index(root / 'cache', root / 'index', with_labels=False)
            cfg = settings()
            cfg['inference'].update(quality_threshold=0., mask_threshold=0.)
            model = QueryInstanceDecoder(**cfg['model']).eval()
            original_load = np.load

            def guarded(path, *args, **kwargs):
                if Path(path).name in ('labels.npy', 'baseline.npy', 'gt_counts.npz', 'supervision.npz'):
                    raise AssertionError(f'GT/old instance IDs opened during prediction: {path}')
                return original_load(path, *args, **kwargs)

            with mock.patch('numpy.load', side_effect=guarded):
                forest = Forest(root / 'cache', root / 'index')
                first, _ = predict_forest(model, forest, cfg, root / 'first', torch.device('cpu'))
                first = first.copy()
                cfg['inference']['chunk_size'] = 3
                second, _ = predict_forest(model, forest, cfg, root / 'second', torch.device('cpu'))
                np.testing.assert_array_equal(first, second)
                self.assertTrue(np.all(second > 0))


class QueryModelTests(unittest.TestCase):
    def test_vote_prior_prefers_nearby_base_votes_and_can_learn_geometry(self):
        model = QueryInstanceDecoder(**settings()['model'])
        inputs = torch.zeros(2, 38)
        inputs[1, 35] = 3 / model.coordinate_scale
        geometry = dict(center=torch.zeros(1, 2, requires_grad=True),
                        sigma=torch.ones(1, requires_grad=True),
                        strength=torch.ones(1, requires_grad=True))
        logits = model.mask_logits(torch.zeros(1, model.dim), torch.zeros(2, model.dim), inputs, geometry)
        torch.testing.assert_close(logits, torch.tensor([[2., -2.5]]))
        logits.sum().backward()
        for value in geometry.values():
            self.assertGreater(float(value.grad.abs().sum()), 0.)

    def test_joint_loss_updates_attention_embedding_foreground_and_quality(self):
        torch.manual_seed(3)
        model = QueryInstanceDecoder(dim=16, layers=2, heads=4, feedforward=32, num_queries=8, coverage_queries=2)
        tokens, points = torch.randn(8, 38), torch.randn(12, 38)
        tokens[:, -1] = torch.rand(8)
        token_labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, -1])
        point_labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1])
        output = model(tokens, torch.ones(8, dtype=torch.bool), points, progress=1.)
        targets = token_labels[output['query_indices']]
        counts = torch.where(targets > 0, 10., 0.)
        loss, _ = instance_loss(output, targets, point_labels, torch.ones(12), counts)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (model.layers[0].cross.in_proj_weight, model.embedding[-1].weight,
                          model.foreground.weight, model.quality[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_empty_targets_and_unknown_queries_remain_finite(self):
        model = QueryInstanceDecoder(**settings()['model'])
        inputs = torch.randn(8, 38)
        inputs[:, -1] = .1
        output = model(inputs, torch.ones(8, dtype=torch.bool), inputs, 0.)
        labels = torch.tensor([0, 0, 0, 0, -1, -1, -1, -1])
        targets = labels[output['query_indices']]
        loss, _ = instance_loss(output, targets, labels, torch.ones(8), torch.zeros(8))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_quality_uses_population_mass_and_full_tree_size(self):
        output = dict(masks=[torch.full((1, 3), 10., requires_grad=True)],
                      quality_logits=torch.zeros(1, requires_grad=True),
                      foreground=torch.zeros(3, requires_grad=True),
                      embedding=torch.randn(3, 5, requires_grad=True))
        loss, logs = instance_loss(output, torch.tensor([1]), torch.tensor([1, 1, 0]),
                                   torch.tensor([2., 2., 1.]), torch.tensor([8.]))
        self.assertAlmostEqual(logs['mean_target_quality'], 4 / 9, places=6)
        loss.backward()
        np.testing.assert_allclose(query_weights(torch.tensor([1, 1, 2, 0, -1])).numpy(),
                                   [.25, .25, .5, .25, 0.])

    def test_normalization_and_decoding_survive_checkpoint_round_trip(self):
        model = QueryInstanceDecoder(**settings()['model']).eval()
        model.feature_mean.copy_(torch.arange(32) / 10)
        model.feature_scale.copy_(torch.arange(32) / 20 + 1)
        inputs = torch.randn(10, 38)
        inputs[:, -1] = .9
        core = torch.ones(10, dtype=torch.bool)
        with torch.no_grad():
            expected = model(inputs, core, inputs, .7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pth'
            torch.save(model.state_dict(), path)
            restored = QueryInstanceDecoder(**settings()['model']).eval()
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
            with torch.no_grad():
                actual = restored(inputs, core, inputs, .7)
            torch.testing.assert_close(actual['quality_logits'], expected['quality_logits'])
            torch.testing.assert_close(actual['masks'][-1], expected['masks'][-1])
            torch.testing.assert_close(actual['query_indices'], expected['query_indices'])
            with torch.no_grad():
                chunks = [restored.point_predictions(part, actual['queries'], actual['geometry'])['masks'][-1]
                          for part in inputs.split(3)]
            torch.testing.assert_close(torch.cat(chunks, dim=1), actual['masks'][-1], rtol=1e-4, atol=1e-5)


class QueryWorkflowTests(unittest.TestCase):
    def test_nms_removes_duplicates_and_contained_fragments_but_keeps_separate_masks(self):
        masks = [np.arange(10), np.arange(8), np.arange(20, 25)]
        records = [dict(score=score, count=len(mask), bbox=[0, 0, 0, 3, 3, 3])
                   for score, mask in zip((.9, .8, .7), masks)]
        self.assertEqual(suppress_masks(records, lambda i: masks[i]), [0, 2])

    def test_prepare_train_interrupt_resume_and_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = settings()
            cfg.update(index_root=str(root / 'index'), work_dir=str(root / 'work'),
                       source_config=str(root / 'source.yaml'))
            source = dict(output_root=str(root / 'cache'), splits=dict(train=['A1N'], validation=['G4N']))
            (root / 'source.yaml').write_text(yaml.safe_dump(source))
            records = []
            for split, plots in source['splits'].items():
                for plot in plots:
                    synthetic(root / 'cache' / split / plot)
                    records.append(dict(split=split, plot=plot, cache=f'{split}/{plot}'))
            (root / 'cache/manifest.json').write_text(json.dumps(dict(forests=records)))
            WORKFLOW.prepare(cfg)
            original_save = WORKFLOW.save_checkpoint

            def interrupt_at_second_step(path, checkpoint):
                if checkpoint['step'] == 2:
                    raise RuntimeError('synthetic interruption')
                return original_save(path, checkpoint)

            with mock.patch.object(WORKFLOW, 'save_checkpoint', side_effect=interrupt_at_second_step):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    WORKFLOW.train(cfg)
            WORKFLOW.train(cfg, resume=True)
            checkpoint = torch.load(root / 'work/best.pth', map_location='cpu', weights_only=False)
            model = QueryInstanceDecoder(**cfg['model'])
            model.load_state_dict(checkpoint['model'])
            validation = [r for r in WORKFLOW.entries(cfg) if r['split'] == 'validation']
            report = WORKFLOW.evaluate(cfg, model, validation, root / 'report', torch.device('cpu'))
            self.assertFalse(report['partial'])
            self.assertEqual(report['forests'][0]['baseline']['num_gt'], 2)
            summary = json.loads((root / 'work/training_summary.json').read_text())
            self.assertEqual(summary['step'], 2)


if __name__ == '__main__':
    unittest.main()
