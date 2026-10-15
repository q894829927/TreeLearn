"""Server-side CPU tests. No forest files, CUDA or spconv are required."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tree_learn.mask_refinement.data import (
    export_cache, load_cache, load_candidates, point_inputs,
    prepare_regions, stratified_sample,
)
from tree_learn.mask_refinement.evaluation import audit_forest, evaluate_labels
from tree_learn.mask_refinement.inference import predict_forest, update_winners
from tree_learn.mask_refinement.model import CandidateMaskMLP, mask_loss

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mask_workflow', ROOT / 'tools/mask_refinement/run.py')
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)

REGIONS = dict(margin_xy_m=2.0, minimum_scale_m=1.0, max_region_points=100)
SUPERVISION = dict(min_labeled_fraction=0.5, min_match_iou=0.1, min_match_purity=0.4)


def synthetic_cache(directory, labels=None, shift=None):
    coords = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                       [1, 0, 0], [1, 0, 1], [1, 0, 2],
                       [0.5, 0, 0], [0.5, 0, 1]], dtype=np.float32)
    if shift is not None:
        coords += np.asarray(shift, dtype=np.float32)
    features = np.arange(8 * 32, dtype=np.float32).reshape(8, 32) / 100
    baseline = np.array([10, 10, 20, 20, 20, 10, 0, 0])
    if labels is None:
        labels = np.array([1, 1, 1, 2, 2, 2, 0, -1])
    export_cache(directory, coords, features, np.zeros_like(coords),
                 np.tile([2., -2.], (8, 1)), baseline, labels,
                 dict(signature='synthetic', xyz_mean=[0, 0, 0]))
    prepare_regions(directory, REGIONS)


class CandidateMaskDataTests(unittest.TestCase):
    def test_regions_and_features_do_not_depend_on_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'a', Path(tmp) / 'b'
            synthetic_cache(first)
            synthetic_cache(second, labels=np.full(8, -1))
            a, b = load_candidates(first), load_candidates(second)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            for candidate in a['ids']:
                np.testing.assert_array_equal(
                    np.load(first / 'regions' / f'{candidate}.npy'),
                    np.load(second / 'regions' / f'{candidate}.npy'))
            arrays, _ = load_cache(first)
            self.assertNotIn('labels', arrays)

    def test_point_features_translate_with_candidate_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'a', Path(tmp) / 'b'
            synthetic_cache(first)
            synthetic_cache(second, shift=[10, -20, 5])
            a, _ = load_cache(first)
            b, _ = load_cache(second)
            indices = np.arange(8)
            x = point_inputs(a, load_candidates(first), 0, indices)
            y = point_inputs(b, load_candidates(second), 0, indices)
            self.assertEqual(x.shape, (8, 40))
            np.testing.assert_allclose(x, y, atol=1e-6)

    def test_region_limit_fails_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            Path(tmp, 'regions.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'do not truncate'):
                prepare_regions(tmp, dict(REGIONS, max_region_points=1))

    def test_geometry_reconfiguration_rejects_stale_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                prepare_regions(tmp, dict(REGIONS, margin_xy_m=3.0))

    def test_sampling_ignores_unknowns_and_preserves_population_mass(self):
        labels = np.array([1] * 2 + [2] * 6 + [0] * 12 + [-1] * 10)
        indices, weights = stratified_sample(np.arange(30), labels, 1, 40,
                                             np.random.default_rng(4))
        self.assertFalse(np.any(labels[indices] < 0))
        for label, expected in ((1, .1), (2, .3), (0, .6)):
            self.assertAlmostEqual(float(weights[labels[indices] == label].sum() / weights.sum()), expected, places=6)

    def test_audit_is_separate_from_prediction_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            report = audit_forest(tmp, SUPERVISION)
            self.assertEqual(report['num_trainable'], 2)
            self.assertAlmostEqual(report['oracle_coverage'], 1.0)
            self.assertGreater(report['oracle_coverage_gain'], 0.0)
            with np.load(Path(tmp) / 'supervision.npz') as targets:
                np.testing.assert_array_equal(targets['target_ids'], [1, 2])

    def test_metrics_handle_ignore_abstention_and_strict_iou(self):
        labels = np.array([1, 1, 2, 2, 0, -1])
        perfect = evaluate_labels(labels, np.array([7, 7, 9, 9, 0, 100]))
        self.assertEqual(perfect['f1'], 1.0)
        self.assertEqual(perfect['num_predictions'], 2)
        empty = evaluate_labels(labels, np.zeros(6, dtype=int))
        self.assertEqual(empty['f1'], 0.0)
        self.assertEqual(empty['coverage'], 0.0)
        half = evaluate_labels(np.array([1, 1]), np.array([8, 0]))
        self.assertEqual(half['tp'], 0)  # IoU == .5 is not a match.

    def test_identity_masks_exactly_reproduce_baseline(self):
        baseline = np.array([2, 2, 5, 5, 0])
        predictions = np.zeros(5, dtype=int)
        best = np.full(5, -np.inf, dtype=np.float32)
        indices = np.arange(5)
        for candidate in (5, 2):
            update_winners(baseline, predictions, best, indices,
                           (baseline == candidate).astype(np.float32), candidate, .5)
        np.testing.assert_array_equal(predictions, baseline)

    def test_false_positive_count_uses_labeled_tree_fraction(self):
        # One correct tree, one fragment on a missed tree, one ground-only FP.
        labels = np.array([1, 1, 2, 2, 2, 0, 0])
        prediction = np.array([10, 10, 20, 0, 0, 30, 30])
        metrics = evaluate_labels(labels, prediction)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 1, 1))
        self.assertAlmostEqual(metrics['coverage'], (1 + 1 / 3) / 2)

    def test_equal_scores_keep_original_owner_independent_of_order(self):
        baseline = np.array([2, 5, 0])
        for order in ((2, 5), (5, 2)):
            predictions = np.zeros(3, dtype=int)
            best = np.full(3, -np.inf)
            for candidate in order:
                update_winners(baseline, predictions, best, np.arange(3),
                               np.full(3, .8), candidate, .5)
            np.testing.assert_array_equal(predictions, baseline)


class CandidateMaskModelTests(unittest.TestCase):
    def test_loss_is_finite_with_empty_targets_and_updates_model(self):
        torch.manual_seed(12)
        model = CandidateMaskMLP(16)
        points, descriptors = torch.randn(2, 32, 40), torch.randn(2, 64)
        targets = torch.zeros(2, 32)
        targets[0, :10] = 1
        loss, _, _ = mask_loss(model(points, descriptors), targets, torch.ones_like(targets))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)

    def test_inference_is_chunk_invariant_and_never_opens_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            Path(tmp, 'labels.npy').unlink()
            torch.manual_seed(13)
            model = CandidateMaskMLP(16).eval()
            a, mask_a, _ = predict_forest(model, tmp, torch.device('cpu'), chunk_size=1)
            b, mask_b, _ = predict_forest(model, tmp, torch.device('cpu'), chunk_size=100)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(mask_a, mask_b)

    def test_checkpoint_round_trip_preserves_normalization_and_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = CandidateMaskMLP(16).eval()
            model.feature_mean.fill_(3)
            model.feature_scale.fill_(2)
            points, descriptor = torch.randn(17, 40), torch.randn(64)
            torch.save(model.state_dict(), Path(tmp) / 'model.pth')
            restored = CandidateMaskMLP(16).eval()
            restored.load_state_dict(torch.load(Path(tmp) / 'model.pth', weights_only=True))
            torch.testing.assert_close(model(points, descriptor), restored(points, descriptor))


class CandidateMaskWorkflowTests(unittest.TestCase):
    def settings(self):
        import yaml
        return yaml.safe_load((ROOT / 'configs/experiments/candidate_mask_mlp/step1.yaml').read_text())

    def test_paired_forest_leak_is_rejected(self):
        settings = self.settings()
        settings['splits']['validation'].append('A1N')
        with self.assertRaisesRegex(ValueError, 'leak'):
            WORKFLOW.validate_settings(settings)

    def test_locked_experiment_changes_with_inference_policy(self):
        first = self.settings()
        second = copy.deepcopy(first)
        second['inference']['mask_threshold'] = .8
        self.assertNotEqual(WORKFLOW.experiment_signature(first, {}),
                            WORKFLOW.experiment_signature(second, {}))

    def test_synthetic_train_resume_and_evaluate(self):
        # Tiny CPU integration test exercises checkpoint, manifests and CLI logic.
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings()
            settings.update(output_root=str(Path(tmp) / 'data'),
                            work_dir=str(Path(tmp) / 'work'),
                            splits={'train': ['A1N'], 'validation': ['G4N']},
                            pilot_plots=['A1N', 'G4N'], regions=REGIONS,
                            supervision=SUPERVISION)
            settings['training'].update(device='cpu', hidden_dim=8, epochs=1,
                                         points_per_candidate=12, batch_size=2,
                                         statistics_points_per_forest=4)
            entries = []
            for split, plot in WORKFLOW.selected_plots(settings, True):
                path = Path(settings['output_root']) / split / plot
                synthetic_cache(path)
                entries.append(dict(split=split, plot=plot, cache=f'{split}/{plot}',
                                    signature='synthetic'))
            manifest = dict(schema=1, pilot=True, baseline_signature='synthetic',
                            baseline_request={key: settings[key] for key in ('checkpoint', 'pipeline_template')},
                            regions=REGIONS, forests=entries)
            WORKFLOW.write_json(WORKFLOW.manifest_path(settings, True), manifest)
            self.assertTrue(WORKFLOW.audit(settings, True))
            WORKFLOW.train(settings, pilot=True, smoke=True, resume=False)
            WORKFLOW.train(settings, pilot=True, smoke=True, resume=True)
            self.assertTrue(WORKFLOW.evaluate(settings, pilot=True, smoke=True, export=True))
            result = json.loads((Path(settings['work_dir']) / 'smoke/evaluation/summary.json').read_text())
            self.assertIsNone(result['passed'])
            self.assertTrue(result['smoke'])


if __name__ == '__main__':
    unittest.main()
