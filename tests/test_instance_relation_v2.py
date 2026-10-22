"""Server regressions. Development PC must only parse/compile these sources."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tools.forest_abc.diagnose import analyze_table, compare_models, scan_arrays
from tools.instance_relation_v2 import run as workflow
from tree_learn.instance_relation import data as base_data
from tree_learn.instance_relation.supervision import prepare_supervision, merge_table
from tree_learn.instance_relation_v2 import data, supervision, augmentation, policy
from tree_learn.instance_relation_v2.model import ConnectionRelationHead, loss


class InstanceRelationV2Tests(unittest.TestCase):
    def config(self, root):
        config = yaml.safe_load(Path('configs/experiments/instance_relation_v2/full.yaml').read_text())
        config.update(cache_root=str(root / 'source'), external_cache_root=str(root / 'external'),
            base_prepared_root=str(root / 'base'), prepared_root=str(root / 'prepared'),
            work_dir=str(root / 'work'), checkpoint='official.pth', chunk_points=17,
            splits=dict(train=['train'], validation=['validation']), expected_validation_baseline_f1=4 / 7)
        config['geometry'].update(points_per_stratum=2, max_pairs=1000, max_instances=100)
        config['augmentation'].update(bank_points=32, min_bank_points=4)
        config['model'].update(hidden_dim=8, geometric_neighbors=2, vote_neighbors=1)
        config['training'].update(device='cpu', steps=4, batch_size=8, validation_every=2,
            save_every=2, log_every=1, hard_mining_every=2, hard_pool_size=4)
        return config

    def fixture(self, cache):
        cache.mkdir(parents=True)
        gt = np.asarray([1] * 48 + [2] * 48 + [3] * 48 + [-1] * 4 + [0] * 8)
        baseline = np.asarray([10] * 24 + [20] * 24 + [30] * 48 + [40] * 48 + [99] * 4 + [55] * 6 + [0] * 2)
        n = len(gt)
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 0] = np.repeat([0., 3., 6., 9., -3., -6.], [48, 48, 48, 4, 6, 2])
        xyz[:, 1] = np.sin(np.arange(n)) * .1
        xyz[:, 2] = np.arange(n) % 48 * .25
        votes = xyz[:, :2].copy()
        votes[:, 1] = 0
        features = np.sin(np.arange(n)[:, None] * .1 + np.arange(32)[None] * .03).astype(np.float32)
        for name, value in dict(coords=xyz, baseline=baseline, labels=gt, features=features,
                                tree_probability=np.where(baseline > 0, .8, .2).astype(np.float32), votes_xy=votes).items():
            np.save(cache / f'{name}.npy', value)
        (cache / 'cache.json').write_text(json.dumps(dict(signature='fixture', checkpoint='official.pth', num_points=n)))
        return gt, baseline, features

    def prepared(self, root, external=False):
        config = self.config(root)
        cache = root / 'cache'
        gt, baseline, frozen = self.fixture(cache)
        row = dict(plot='train', split='external' if external else 'train', cache=str(cache),
                   base_prepared=str(root / 'base' / 'train'), prepared=str(root / 'prepared' / 'train'))
        with contextlib.redirect_stdout(io.StringIO()):
            base = base_data.prepare_features(cache, row['base_prepared'], config['geometry'], 42, 17)
            prepare_supervision(cache, row['base_prepared'], base, config['supervision'], 17)
        return config, row, gt, baseline, frozen

    def quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def model_batch(self):
        torch.manual_seed(42)
        model = ConnectionRelationHead(8, 2, 1).eval()
        points = torch.randn(3, 2, 8, 38)
        points[..., 5] = points[..., 5].sigmoid()
        valid = torch.ones(3, 2, 8, dtype=torch.bool)
        valid[:, :, -1] = False
        weights = valid.float()
        return model, points, valid, weights

    def test_prediction_points_never_open_gt_and_frozen_channels_align(self):
        with tempfile.TemporaryDirectory() as directory:
            config, row, _, _, frozen = self.prepared(Path(directory))
            loader = np.load
            def guarded(path, *args, **kwargs):
                if Path(path).name == 'labels.npy':
                    raise AssertionError('Point extraction opened GT.')
                return loader(path, *args, **kwargs)
            with patch('numpy.load', side_effect=guarded):
                values = self.quiet(data.prepare, row, config)
            np.testing.assert_array_equal(values['points'][values['valid'], 6:], frozen[values['indices'][values['valid']]])
            original = base_data.load_features(row['base_prepared'])
            np.testing.assert_array_equal(values['pairs'], original['pairs'])
            np.testing.assert_array_equal(values['ids'], original['ids'])
            np.testing.assert_allclose(values['weights'].sum(1), original['population'].sum(1), atol=1e-5)

    def test_frozen_feature_change_rejects_stale_samples(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            config, row, _, _, _ = self.prepared(Path(directory))
            self.quiet(data.prepare, row, config)
            path = Path(row['cache']) / 'features.npy'
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            with self.assertRaisesRegex(ValueError, 'Stale V2'):
                data.load(row, config)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            baseline = Path(row['cache']) / 'baseline.npy'
            original = baseline.stat()
            os.utime(baseline, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
            with self.assertRaisesRegex(ValueError, 'Official cache inputs'):
                data.load(row, config)

    def test_nonempty_v2_union_preserves_unselected_point_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, row, _, baseline, _ = self.prepared(root)
            values = self.quiet(data.prepare, row, config)
            result = self.quiet(policy.write_prediction, Path(row['cache']) / 'baseline.npy',
                                root / 'merged.npy', values['ids'], np.asarray([[0, 1]]), 17)
            actual = np.load(root / 'merged.npy')
            np.testing.assert_array_equal(actual, np.where(baseline == 20, 10, baseline))
            np.testing.assert_array_equal(actual > 0, baseline > 0)
            self.assertEqual(result['changed_unselected_points'], 0)
            np.testing.assert_array_equal(np.load(Path(row['cache']) / 'baseline.npy'), baseline)

    def test_counterfactual_targets_equal_complete_matcher_including_unknown_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            config, row, _, _, _ = self.prepared(Path(directory))
            original = base_data.load_features(row['base_prepared'])
            from tree_learn.instance_relation.supervision import load_supervision
            target = load_supervision(row['base_prepared'], config['supervision'])
            pairs = np.asarray([[0, 1], [2, 3], [2, 5]], dtype=np.int64)
            observed = self.quiet(supervision.action_targets, target['intersections'], target['known'], target['gt_counts'], pairs)
            geometry, table = merge_table(target, [])
            baseline = analyze_table(geometry, table)
            for i, pair in enumerate(pairs):
                _, changed = merge_table(target, [pair])
                refined = analyze_table(geometry, changed)
                _, changes = compare_models(geometry, dict(baseline=baseline, refined=refined))
                self.assertEqual(observed['delta_tp'][i], refined['metrics']['tp'] - baseline['metrics']['tp'])
                self.assertEqual(observed['delta_fp'][i], refined['metrics']['fp'] - baseline['metrics']['fp'])
                self.assertEqual(observed['lost_count'][i], changes['transitions'].get('lost', 0))
                self.assertEqual(observed['recovered_count'][i], changes['transitions'].get('recovered', 0))
            self.assertEqual(observed['recovered_count'][0], 1)
            self.assertEqual(observed['lost_count'][1], 2)

    def test_counterfactual_does_not_replace_hungarian_by_local_iou(self):
        common = np.asarray([[34, 16], [34, 16], [32, 0]])
        result = self.quiet(supervision.action_targets, common, np.asarray([50, 50, 32]),
                            np.asarray([100, 32]), np.asarray([[0, 1]]))
        self.assertGreater(result['quality'][0, 2], .5)
        self.assertEqual(result['recovered_count'][0], 0)

    def test_bank_is_training_only_and_synthetic_labels_follow_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            config, row, _, _, _ = self.prepared(Path(directory))
            values = self.quiet(data.prepare, row, config)
            target = self.quiet(supervision.prepare, row, values, config)
            bank = augmentation.prepare_bank(row, values, target, config)
            with self.assertRaisesRegex(ValueError, 'training forests'):
                augmentation.prepare_bank(dict(row, split='validation'), values, target, config)
            for desired in (0, 1, 2):
                first = augmentation.synthetic(bank, np.random.default_rng(42), 8, config['augmentation'], desired)
                repeat = augmentation.synthetic(bank, np.random.default_rng(42), 8, config['augmentation'], desired)
                for key in ('points', 'valid', 'weights', 'point_labels'):
                    np.testing.assert_array_equal(first[key], repeat[key])
                self.assertEqual(first['action'], desired)
                self.assertTrue(np.isfinite(first['points']).all())
                self.assertTrue((first['weights'][~first['valid']] == 0).all())
                self.assertEqual(first['risk'], float(desired == 0))

    def test_density_thinning_preserves_population_mass(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(80, 38)).astype(np.float32)
        weights = np.linspace(1., 3., 80)
        labels = np.ones(80, dtype=np.int64)
        _, valid, sampled, _ = augmentation.resample(points, weights, labels, 16, rng, .45)
        self.assertAlmostEqual(float(sampled.sum()), float(weights.sum()), places=4)
        self.assertTrue((sampled[~valid] == 0).all())

    def test_real_density_augmentation_never_uses_gt_or_changes_mass(self):
        _, points, valid, weights = self.model_batch()
        _, observed, sampled = augmentation.augment_real(points[0].numpy(), valid[0].numpy(), weights[0].numpy(),
                                                         np.random.default_rng(42), .45)
        np.testing.assert_allclose(sampled.sum(1), weights[0].numpy().sum(1), atol=1e-6)
        self.assertTrue((sampled[~observed] == 0).all())

    def test_point_model_is_symmetric_and_translation_invariant(self):
        model, points, valid, weights = self.model_batch()
        first = model(points, valid, weights)
        reverse = model(points.flip(1), valid.flip(1), weights.flip(1))
        shifted = points.clone()
        shifted[..., :3] += torch.tensor([100., -60., 40.])
        shifted[..., 3:5] += torch.tensor([100., -60.])
        moved = model(shifted, valid, weights)
        for key in ('action_logits', 'risk_logit', 'utility', 'quality'):
            self.assertTrue(torch.allclose(first[key], reverse[key], atol=2e-6, rtol=1e-5))
            self.assertTrue(torch.allclose(first[key], moved[key], atol=2e-5, rtol=1e-4))

    def test_unknown_point_labels_do_not_become_connection_negatives(self):
        model, points, valid, weights = self.model_batch()
        output = model(points, valid, weights)
        target = dict(action=torch.zeros(3, dtype=torch.long), risk=torch.zeros(3), utility=torch.zeros(3),
                      quality=torch.zeros(3, 3), point_labels=torch.full(valid.shape, -1, dtype=torch.long))
        total, components = loss(output, target)
        self.assertEqual(components['connection_loss'], 0.)
        total.backward()
        for module in (model.point_encoder, model.connection_encoder, model.relation):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_frozen_feature_channels_reach_trainable_head_and_buffers_round_trip(self):
        model, points, valid, weights = self.model_batch()
        model.feature_mean.copy_(torch.arange(32, dtype=torch.float32) / 32)
        model.feature_scale.fill_(2.)
        points.requires_grad_()
        first = model(points, valid, weights)
        first['action_logits'].square().sum().backward()
        self.assertGreater(float(points.grad[..., 6:].abs().sum()), 0.)
        clone = ConnectionRelationHead(8, 2, 1).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(first['action_logits'], clone(points.detach(), valid, weights)['action_logits']))

    def test_policy_uses_one_score_and_auxiliary_quality_cannot_veto(self):
        pairs = np.asarray([[0, 1], [1, 2], [2, 3]])
        scores = dict(action=np.asarray([[.05, .95, 0.], [.1, .9, 0.], [.15, .85, 0.]]),
                      utility=np.zeros(3), risk=np.zeros(3), quality=np.zeros((3, 3)))
        settings = dict(completion_credit=.25, utility_weight=.5, risk_weight=2.)
        np.testing.assert_array_equal(policy.select_pairs(pairs, scores, settings, .5), [[0, 1]])
        scores['quality'].fill(1)
        np.testing.assert_array_equal(policy.select_pairs(pairs, scores, settings, .5), [[0, 1]])
        self.assertEqual(len(policy.select_pairs(pairs, scores, settings, 2.5)), 0)

    def test_complete_workflow_resume_and_gt_free_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            for external in (False, True):
                cache_root = Path(config['external_cache_root' if external else 'cache_root'])
                records = []
                for plot in (['outside'] if external else ['train', 'validation']):
                    cache = cache_root / plot
                    self.fixture(cache)
                    split = 'external' if external else plot
                    base_root = Path(config['base_prepared_root']) / split / plot
                    original = self.quiet(base_data.prepare_features, cache, base_root, config['geometry'], 42, 17)
                    self.quiet(prepare_supervision, cache, base_root, original, config['supervision'], 17)
                    records.append(dict(plot=plot, split=split, cache=plot, signature='fixture'))
                (cache_root / 'manifest.json').write_text(json.dumps(dict(forests=records)))
            self.quiet(workflow.prepare, config)
            self.quiet(workflow.check, config)
            self.quiet(workflow.train, config)
            self.quiet(workflow.train, config, resume=True)
            self.quiet(workflow.evaluate, config)
            (Path(config['external_cache_root']) / 'outside' / 'labels.npy').unlink()
            self.quiet(workflow.predict, config)
            marker = json.loads((Path(config['work_dir']) / 'external_evaluation/outside/prediction.json').read_text())
            self.assertFalse(marker['gt_opened'])
            self.assertEqual(marker['changed_foreground_points'], 0)
            self.assertEqual(marker['changed_unselected_points'], 0)
            report = json.loads((Path(config['work_dir']) / 'evaluation/brief.json').read_text())
            self.assertEqual(report['lost_gt'], 0)


if __name__ == '__main__':
    unittest.main()
