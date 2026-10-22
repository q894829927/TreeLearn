"""Server-only regressions; do not execute models/tests on the development PC."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from tools.instance_relation_v3 import run as workflow
from tree_learn.instance_relation import data as original_data
from tree_learn.instance_relation.supervision import prepare_supervision
from tree_learn.instance_relation_v2 import data as old_data, supervision as old_supervision
from tree_learn.instance_relation_v3 import data, supervision, augmentation, policy, calibration
from tree_learn.instance_relation_v3.model import StructureRelationHead, loss
from tree_learn.mask_refinement.data import file_digest


class InstanceRelationV3Tests(unittest.TestCase):
    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def config(self, root):
        config = yaml.safe_load(Path('configs/experiments/instance_relation_v3/full.yaml').read_text())
        config.update(cache_root=str(root / 'source'), external_cache_root=str(root / 'external'),
            base_prepared_root=str(root / 'base'), point_prepared_root=str(root / 'reused'),
            prepared_root=str(root / 'new'), work_dir=str(root / 'work'), checkpoint='official.pth',
            chunk_points=17, splits=dict(train=['train'], validation=['validation']),
            expected_validation_baseline_f1=4 / 7)
        config['geometry'].update(points_per_stratum=4, max_pairs=1000, max_instances=100)
        config['augmentation'].update(bank_points=32, min_bank_points=4)
        config['model'].update(hidden_dim=8, geometric_neighbors=2, vote_neighbors=1, height_bands=4, minimum_layer_points=1)
        config['training'].update(device='cpu', steps=2, batch_size=12, validation_every=1,
            save_every=1, log_every=1, hard_mining_every=1, hard_pool_size=4)
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
        for name, value in dict(coords=xyz, baseline=baseline, labels=gt, votes_xy=votes,
            tree_probability=np.where(baseline > 0, .8, .2).astype(np.float32),
            features=np.sin(np.arange(n)[:, None] * .1 + np.arange(32)[None] * .03).astype(np.float32)).items():
            np.save(cache / f'{name}.npy', value)
        (cache / 'cache.json').write_text(json.dumps(dict(signature='fixture', checkpoint='official.pth', num_points=n)))

    def setup_files(self, root):
        config = self.config(root)
        for external in (False, True):
            cache_root = Path(config['external_cache_root' if external else 'cache_root'])
            manifest = []
            for plot in (['outside'] if external else ['train', 'validation']):
                cache = cache_root / plot
                self.fixture(cache)
                split = 'external' if external else plot
                base = Path(config['base_prepared_root']) / split / plot
                features = self.quiet(original_data.prepare_features, cache, base, config['geometry'], 42, 17)
                self.quiet(prepare_supervision, cache, base, features, config['supervision'], 17)
                row = dict(plot=plot, split=split, cache=str(cache), base_prepared=str(base),
                           prepared=str(Path(config['point_prepared_root']) / split / plot))
                points = self.quiet(old_data.prepare, row, config)
                if not external:
                    self.quiet(old_supervision.prepare, row, points, config)
                manifest.append(dict(plot=plot, split=split, cache=plot, signature='fixture'))
            (cache_root / 'manifest.json').write_text(json.dumps(dict(forests=manifest)))
        return config

    def model_batch(self):
        torch.manual_seed(3)
        model = StructureRelationHead(8, 2, 1, 4, 1).eval()
        points = torch.randn(3, 2, 16, 38)
        points[..., 2] = torch.arange(16).float() / 2
        points[..., 3:5] = points[..., :2] + .1
        valid = torch.ones(3, 2, 16, dtype=torch.bool)
        weights = torch.ones(3, 2, 16)
        return model, points, valid, weights

    def test_outcome_priority_and_safe_recovery_ignore_old_quality_cutoffs(self):
        np.testing.assert_array_equal(supervision.outcomes([0, 0, 1, 1], [0, 1, 0, 1]), [0, 1, 2, 2])
        # Union precision is below .65, yet this is an exact safe recovery.
        labels = np.asarray([[1] * 50 + [0] * 33] * 2)
        targets = augmentation.virtual_targets(labels, np.ones_like(labels, dtype=np.float32), {1: 100, 2: 100})
        self.assertLess(targets['quality'][0], .65)
        self.assertGreater(targets['quality'][2], .5)
        self.assertEqual((targets['action'], targets['risk'], targets['utility']), (1, 0., 1.))

    def test_synthetic_complete_neighbors_destroy_and_cleanup_has_no_recovery_reward(self):
        labels = np.asarray([[1] * 100, [2] * 100])
        target = augmentation.virtual_targets(labels, np.ones_like(labels, dtype=np.float32), {1: 100, 2: 100})
        self.assertEqual((target['action'], target['risk'], target['utility']), (2, 1., -1.))
        labels = np.ones((2, 80), dtype=np.int64)
        weights = np.r_[np.ones((1, 80)), np.full((1, 80), .25)]
        target = augmentation.virtual_targets(labels, weights, {1: 100, 2: 100})
        self.assertEqual((target['action'], target['risk'], target['utility']), (0, 0., 0.))

    def test_reused_samples_and_real_outcomes_are_separate_from_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.setup_files(root)
            self.quiet(workflow.prepare, config)
            row = data.rows(config)[0]
            current, old = data.load(row, config), old_data.load(data.point_row(row), config)
            for key in current:
                np.testing.assert_array_equal(current[key], old[key])
            target = supervision.load(row, config)
            np.testing.assert_array_equal(target['action'][target['valid']], supervision.outcomes(
                target['lost_count'][target['valid']], target['recovered_count'][target['valid']]))
            self.assertFalse((Path(row['prepared']) / 'points.npz').exists())
            source = Path(row['point_prepared']) / 'points.json'
            source.write_text(source.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'reference changed'):
                data.load(row, config)

    def test_training_is_real_majority_with_all_real_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.setup_files(Path(directory))
            self.quiet(workflow.prepare, config)
            values = workflow.pool(config)
            batch, real, synthetic = workflow.training_batch(values, config, np.random.default_rng(42))
            self.assertEqual(sum(real), 9)
            self.assertEqual(sum(synthetic), 3)
            self.assertTrue(all(count > 0 for count in real))
            self.assertEqual(int(batch['is_real'].sum()), 9)
            self.assertTrue(all(np.isin(group, values['groups'][i][category]).all()
                for i, pool in enumerate(values['hard']) for category, group in enumerate(pool)))

    def test_real_recovery_is_relabelled_when_legacy_action_says_keep(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.setup_files(Path(directory))
            row = data.rows(config)[0]
            old = old_supervision.load(data.point_row(row), config)
            recovered = old['valid'] & (old['lost_count'] == 0) & (old['recovered_count'] > 0)
            self.assertTrue(recovered.any())
            old['action'][recovered] = 0
            path = Path(row['point_prepared'])
            np.savez(path / 'targets.npz', **old)
            meta = json.loads((path / 'targets.json').read_text())
            meta['sha256'] = file_digest(path / 'targets.npz')
            meta['action_counts'] = {name: int(np.count_nonzero(old['valid'] & (old['action'] == i)))
                                    for i, name in enumerate(('keep', 'recover', 'complete'))}
            (path / 'targets.json').write_text(json.dumps(meta))
            points = data.prepare(row, config)
            target = supervision.prepare(row, points, config)
            self.assertTrue((target['action'][recovered] == 1).all())
            meta = json.loads((Path(row['prepared']) / 'targets.json').read_text())
            self.assertEqual(meta['newly_recovery_labeled'], int(recovered.sum()))

    def test_symmetry_translation_and_padding_are_safe(self):
        model, points, valid, weights = self.model_batch()
        valid[:, :, -2:] = False
        first = model(points, valid, weights)
        reverse = model(points.flip(1), valid.flip(1), weights.flip(1))
        shifted = points.clone()
        shifted[..., :3] += torch.tensor([60., -40., 20.])
        shifted[..., 3:5] += torch.tensor([60., -40.])
        moved = model(shifted, valid, weights)
        padded = points.clone()
        padded[~valid] = 10000
        padding = model(padded, valid, weights)
        for key in ('action_logits', 'risk_logit', 'utility', 'quality', 'height_evidence'):
            self.assertTrue(torch.allclose(first[key], reverse[key], rtol=1e-5, atol=2e-6), key)
            self.assertTrue(torch.allclose(first[key], moved[key], rtol=2e-4, atol=2e-5), key)
            self.assertTrue(torch.equal(first[key], padding[key]), key)

    def test_height_tracks_distinguish_two_lower_supports_and_vertical_continuation(self):
        model = StructureRelationHead(8, 2, 1, 4, 2).eval()
        points = torch.zeros(1, 2, 16, 38)
        points[..., 2] = torch.arange(16).float() / 2
        points[:, 1, :, 0] = 3
        points[..., 3:5] = points[..., :2]
        valid = torch.ones(1, 2, 16, dtype=torch.bool)
        independent = model(points, valid, valid.float())['height_evidence']
        self.assertGreater(float(independent[..., 18].sum()), 0)
        points[:, 1, :, 0] = 0
        points[:, 0, :, 2] /= 2
        points[:, 1, :, 2] = points[:, 0, :, 2] + 4
        points[..., 3:5] = points[..., :2]
        continuation = model(points, valid, valid.float())['height_evidence']
        self.assertEqual(float(continuation[..., 16].sum()), 0.)
        self.assertGreater(float(continuation[..., 11].sum()), 0.)

    def test_main_classification_reaches_structure_and_frozen_point_features(self):
        model, points, valid, weights = self.model_batch()
        points.requires_grad_()
        output = model(points, valid, weights)
        torch.nn.functional.cross_entropy(output['action_logits'], torch.tensor([0, 1, 2])).backward()
        for name in ('point_encoder', 'connection_encoder', 'structure_encoder', 'track_encoder', 'relation'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(model, name).parameters() if p.grad is not None), 0., name)
        self.assertGreater(float(points.grad[..., 6:].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(points.grad).all())
        p = output['action_logits'].softmax(-1)
        self.assertTrue(torch.allclose(output['risk_logit'].sigmoid(), p[:, 2], atol=1e-6))
        self.assertTrue(torch.equal(output['utility'], p[:, 1] - p[:, 2]))

    def test_unknown_connections_do_not_become_negatives_and_ranking_uses_real_examples(self):
        model, points, valid, weights = self.model_batch()
        target = dict(action=torch.tensor([1, 2, 0]), risk=torch.tensor([0., 1., 0.]),
            utility=torch.tensor([1., -1., 0.]), quality=torch.zeros(3, 3),
            point_labels=torch.full(valid.shape, -1, dtype=torch.long), is_real=torch.ones(3, dtype=torch.bool))
        settings = self.config(Path('unused'))['objective']
        output = model(points, valid, weights)
        total, components = loss(output, target, settings)
        self.assertEqual(components['connection_loss'], 0.)
        self.assertGreater(components['ranking_loss'], 0.)
        total.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        target['is_real'].fill_(False)
        _, components = loss(model(points, valid, weights), target, settings)
        self.assertEqual(components['ranking_loss'], 0.)

    def test_policy_uses_one_outcome_distribution_and_auxiliary_fields_cannot_veto(self):
        pairs = np.asarray([[0, 1], [1, 2], [2, 3]])
        scores = dict(action=np.asarray([[.05, .94, .01], [.1, .89, .01], [.7, .29, .01]]))
        settings = dict(destruction_cost=3.)
        selected = policy.select_pairs(pairs, scores, settings, .2)
        np.testing.assert_array_equal(selected, [[0, 1]])
        scores.update(risk=np.ones(3), utility=-np.ones(3), quality=np.zeros((3, 3)))
        np.testing.assert_array_equal(policy.select_pairs(pairs, scores, settings, .2), selected)
        self.assertEqual(len(policy.select_pairs(pairs, scores, settings, 2.)), 0)

    def test_leave_one_forest_out_catches_a_policy_that_fails_on_the_excluded_forest(self):
        metrics = dict(f1=.8, completeness=.8, commission=.2, coverage=.7, point_precision=.8, point_recall=.8)
        trials = []
        for index, threshold in enumerate((2., .8, .3)):
            forests = []
            for plot in ('a', 'b', 'c'):
                lost = int(index == 2 and plot == 'a')
                recovered = int(index == 1 and plot == 'a') + 2 * int(index == 2 and plot != 'a')
                forests.append(dict(plot=plot, baseline=metrics, refined=dict(metrics, f1=.7 if lost else .8),
                    lost=lost, recovered=recovered, accepted_pairs=int(index != 0)))
            trials.append(dict(threshold=threshold, forests=forests, macro=metrics,
                recovered=sum(r['recovered'] for r in forests), accepted_pairs=sum(r['accepted_pairs'] for r in forests),
                eligible=index != 2))
        chosen, stress = calibration.choose(trials)
        self.assertFalse(stress['all_source_folds_safe'])
        self.assertEqual(chosen['threshold'], 2.)

    def test_nonempty_local_write_preserves_unselected_point_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.asarray([0, 10, 20, 30, 20, 40, 0], dtype=np.int64)
            np.save(root / 'baseline.npy', labels)
            result = self.quiet(policy.write_prediction, root / 'baseline.npy', root / 'prediction.npy',
                                np.asarray([10, 20, 30, 40]), np.asarray([[0, 1]]), 3)
            np.testing.assert_array_equal(np.load(root / 'prediction.npy'), [0, 10, 10, 30, 10, 40, 0])
            self.assertEqual(result['changed_foreground_points'], 0)
            self.assertEqual(result['changed_unselected_points'], 0)

    def test_complete_prepare_check_train_mine_resume_evaluate_and_gt_free_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.setup_files(Path(directory))
            self.quiet(workflow.prepare, config)
            self.quiet(workflow.check, config)
            self.quiet(workflow.train, config)
            self.quiet(workflow.train, config, resume=True)
            self.quiet(workflow.evaluate, config)
            (Path(config['external_cache_root']) / 'outside/labels.npy').unlink()
            self.quiet(workflow.predict, config)
            marker = json.loads((Path(config['work_dir']) / 'external_evaluation/outside/prediction.json').read_text())
            self.assertFalse(marker['gt_opened'])
            self.assertEqual(marker['changed_foreground_points'], 0)
            self.assertEqual(marker['changed_unselected_points'], 0)
            report = json.loads((Path(config['work_dir']) / 'evaluation/brief.json').read_text())
            self.assertEqual(report['lost_gt'], 0)
            state = torch.load(Path(config['work_dir']) / 'best.pth', map_location='cpu', weights_only=False)
            self.assertEqual(state['schema'], 3)
            self.assertIn('structure_encoder.0.weight', state['model'])


if __name__ == '__main__':
    unittest.main()
