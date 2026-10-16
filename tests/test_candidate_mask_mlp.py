"""Server-side CPU tests. No forest files, CUDA or spconv are required."""

import copy
import ast
import importlib.util
import json
import logging
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from tree_learn.mask_refinement.data import (
    export_cache, load_cache, load_candidates, point_inputs,
    prepare_regions, stratified_sample,
)
from tree_learn.mask_refinement.evaluation import audit_forest, evaluate_labels
from tree_learn.mask_refinement.diagnostics import (
    assignment_control, diagnose_assignments, validate_saved_predictions,
)
from tree_learn.mask_refinement.inference import predict_forest, update_winners
from tree_learn.mask_refinement.model import CandidateMaskMLP, mask_loss
from tree_learn.mask_refinement.tiling import tile_generate_and_save_streaming

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mask_workflow', ROOT / 'tools/mask_refinement/run.py')
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)
EXTERNAL_SPEC = importlib.util.spec_from_file_location('mask_external', ROOT / 'tools/mask_refinement/external.py')
EXTERNAL = importlib.util.module_from_spec(EXTERNAL_SPEC)
EXTERNAL_SPEC.loader.exec_module(EXTERNAL)

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
    def test_low_scores_are_distinguished_from_unscored_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            Path(tmp, 'labels.npy').unlink()
            model = CandidateMaskMLP(8).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.mask_head[-1].bias.fill_(-2)
            previous = None
            for chunk_size in (1, 100):
                predictions, abstained, stats = predict_forest(
                    model, tmp, torch.device('cpu'), chunk_size=chunk_size)
                self.assertFalse(predictions.any())
                self.assertEqual(int(abstained.sum()), 6)
                diagnostics = stats['score_diagnostics']
                self.assertEqual(diagnostics['empty_prediction_reason'], 'all_scores_below_threshold')
                self.assertEqual(diagnostics['num_foreground_points_scored'], 6)
                self.assertEqual(diagnostics['num_foreground_points_not_scored'], 0)
                self.assertEqual(diagnostics['num_scored_pairs'], 12)
                self.assertEqual(diagnostics['num_candidates_scored'], 2)
                self.assertEqual(diagnostics['num_candidates_at_or_above_threshold'], 0)
                self.assertEqual(diagnostics['fraction_pairs_at_or_above_threshold'], 0.0)
                self.assertAlmostEqual(diagnostics['max_probability'], 0.11920292, places=6)
                if previous is not None:
                    for key, value in diagnostics.items():
                        if isinstance(value, float):
                            self.assertAlmostEqual(value, previous[key], places=6)
                        else:
                            self.assertEqual(value, previous[key])
                previous = diagnostics
            # Simulate missing/corrupt region coverage. It must not be reported
            # as merely low model confidence, and its JSON must remain finite.
            for region in Path(tmp, 'regions').glob('*.npy'):
                np.save(region, np.array([], dtype=np.int64))
            _, _, stats = predict_forest(model, tmp, torch.device('cpu'))
            diagnostics = stats['score_diagnostics']
            self.assertEqual(diagnostics['empty_prediction_reason'], 'no_foreground_scores')
            self.assertEqual(diagnostics['num_foreground_points_not_scored'], 6)
            self.assertEqual(diagnostics['num_scored_pairs'], 0)
            self.assertIsNone(diagnostics['max_probability'])
            json.dumps(diagnostics, allow_nan=False)

    def test_score_diagnostics_preserve_threshold_ties_and_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            model = CandidateMaskMLP(8).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            # All sigmoid probabilities are exactly .5; original owners win.
            predictions, abstained, stats = predict_forest(model, tmp, torch.device('cpu'))
            np.testing.assert_array_equal(predictions, np.load(Path(tmp, 'baseline.npy')))
            self.assertFalse(abstained.any())
            diagnostics = stats['score_diagnostics']
            self.assertIsNone(diagnostics['empty_prediction_reason'])
            self.assertEqual(diagnostics['num_pairs_at_or_above_threshold'], 12)
            self.assertEqual(diagnostics['fraction_pairs_at_or_above_threshold'], 1.0)
            self.assertEqual(diagnostics['num_candidates_at_or_above_threshold'], 2)
            self.assertEqual(diagnostics['mean_probability'], .5)

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


class CandidateMaskDiagnosticTests(unittest.TestCase):
    def test_controls_separate_abstention_and_reassignment_without_mutation(self):
        baseline = np.array([1, 1, 2, 2, 0, 3])
        refined = np.array([1, 0, 1, 2, 0, 0])
        before, after = baseline.copy(), refined.copy()
        np.testing.assert_array_equal(
            assignment_control(baseline, refined, 'abstention_only'), [1, 0, 2, 2, 0, 0])
        np.testing.assert_array_equal(
            assignment_control(baseline, refined, 'reassignment_only'), [1, 1, 1, 2, 0, 3])
        np.testing.assert_array_equal(baseline, before)
        np.testing.assert_array_equal(refined, after)

    def test_changed_background_and_new_ids_are_rejected(self):
        baseline = np.array([1, 1, 0])
        for refined in (np.array([1, 1, 1]), np.array([2, 1, 0]),
                        np.array([-1, 1, 0]), np.array([1., 1., 0.])):
            with self.assertRaises(ValueError):
                validate_saved_predictions(baseline, refined)

    def test_removing_background_can_create_a_counted_fp_without_a_new_instance(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0, -1])
        baseline = np.array([10, 10, 10, 20, 20, 20, 20, 30])
        refined = np.array([10, 10, 10, 20, 0, 0, 0, 30])
        report = diagnose_assignments(labels, baseline, refined)
        self.assertEqual(report['metrics']['baseline']['fp'], 0)
        self.assertEqual(report['metrics']['refined']['fp'], 1)
        self.assertEqual(report['new_counted_fp_ids'], [20])
        self.assertEqual(report['removed_counted_fp_ids'], [])
        states = {row['candidate_id']: row for row in report['candidates']}
        self.assertEqual(states[20]['baseline']['status'], 'excluded_unmatched')
        self.assertEqual(states[20]['refined']['status'], 'counted_fp')
        self.assertEqual(states[20]['baseline']['tree_fraction'], .25)
        self.assertEqual(states[20]['refined']['tree_fraction'], 1.0)
        self.assertEqual(states[30]['refined']['status'], 'no_labeled_points')
        self.assertEqual(report['point_changes']['abstained']['gt_background_points'], 3)
        self.assertEqual(report['point_changes']['abstained']['gt_tree_points'], 0)
        self.assertFalse(report['eligible_for_advancement'])

    def test_strict_iou_and_disappeared_candidates_are_tracked(self):
        labels = np.array([1, 1, 2, 2, 0])
        baseline = np.array([10, 10, 20, 20, 30])
        refined = np.array([10, 0, 20, 20, 0])
        report = diagnose_assignments(labels, baseline, refined)
        states = {row['candidate_id']: row for row in report['candidates']}
        self.assertEqual(states[10]['baseline']['status'], 'matched')
        self.assertEqual(states[10]['refined']['status'], 'counted_fp')
        self.assertEqual(states[10]['refined']['best_iou'], .5)
        self.assertEqual(states[30]['refined']['status'], 'absent')
        self.assertEqual(report['point_changes']['abstained']['gt_tree_points'], 1)
        self.assertEqual(report['metrics']['reassignment_only'], report['metrics']['baseline'])
        self.assertEqual(report['metrics']['abstention_only'], report['metrics']['refined'])

    def test_stale_saved_evaluation_is_rejected(self):
        labels, baseline = np.array([1, 1]), np.array([10, 10])
        metrics = evaluate_labels(labels, baseline)
        expected = dict(baseline=metrics, refined=dict(metrics, fp=99))
        with self.assertRaisesRegex(ValueError, 'disagree with arrays'):
            diagnose_assignments(labels, baseline, baseline.copy(), expected)


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

    def test_diagnosis_reuses_saved_arrays_without_checkpoint_or_changing_primary_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings()
            settings.update(output_root=str(Path(tmp) / 'data'),
                            work_dir=str(Path(tmp) / 'work'), regions=REGIONS,
                            splits={'train': ['A1N'], 'validation': ['G4N']},
                            pilot_plots=['A1N', 'G4N'])
            entries = []
            for split, plot in WORKFLOW.selected_plots(settings, False):
                synthetic_cache(Path(settings['output_root']) / split / plot)
                entries.append(dict(split=split, plot=plot, cache=f'{split}/{plot}',
                                    signature='synthetic'))
            manifest = dict(schema=1, pilot=False, baseline_signature='synthetic',
                            baseline_request={key: settings[key] for key in ('checkpoint', 'pipeline_template')},
                            regions=REGIONS, forests=entries)
            WORKFLOW.write_json(WORKFLOW.manifest_path(settings, False), manifest)
            arrays, _ = load_cache(Path(settings['output_root']) / 'validation/G4N', with_labels=True)
            metrics = evaluate_labels(arrays['labels'], arrays['baseline'])
            source = Path(settings['work_dir']) / 'evaluation/summary.json'
            WORKFLOW.write_json(source, dict(
                smoke=False, protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5',
                forests=[dict(plot='G4N', baseline=metrics, refined=metrics)], checks={}, passed=False))
            prediction = source.parent / 'G4N/refined_labels.npy'
            prediction.parent.mkdir()
            np.save(prediction, arrays['baseline'])
            saved_source, saved_prediction = source.read_bytes(), prediction.read_bytes()
            report = WORKFLOW.diagnose(settings)
            self.assertTrue(report['posthoc'])
            self.assertFalse(report['eligible_for_advancement'])
            self.assertFalse(report['primary_evaluation_passed'])
            self.assertEqual(source.read_bytes(), saved_source)
            self.assertEqual(prediction.read_bytes(), saved_prediction)
            self.assertTrue((Path(settings['work_dir']) /
                             'diagnostics/assignment/G4N/candidate_transitions.csv').is_file())

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


class CandidateMaskExternalTests(unittest.TestCase):
    def settings(self):
        import yaml
        return yaml.safe_load((ROOT / 'configs/experiments/candidate_mask_mlp/step1.yaml').read_text())

    def test_external_paths_preserve_training_artifacts_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            training = self.settings()
            training.update(output_root=str(Path(tmp) / 'train_cache'),
                            work_dir=str(Path(tmp) / 'train_work'))
            source = Path(tmp) / 'input/wytham_vox0.1.npz'
            source.parent.mkdir()
            np.savez(source, points=np.zeros((1, 3)), labels=np.ones(1))
            config = dict(forest_path=str(source), output_root=str(Path(tmp) / 'external_cache'),
                          work_dir=str(Path(tmp) / 'external_work'))
            original = copy.deepcopy(training)
            derived = EXTERNAL.external_settings(config, training)
            self.assertEqual(training, original)
            self.assertEqual(derived['splits'], {'external': ['wytham_vox0.1']})
            for key in ('checkpoint', 'pipeline_template', 'regions', 'inference', 'training'):
                self.assertEqual(derived[key], training[key])
            for destination in (training['output_root'], str(Path(training['work_dir']) / 'external')):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    EXTERNAL.external_settings(dict(config, output_root=destination), training)

    def test_external_checkpoint_rejects_smoke_and_policy_changes(self):
        settings, manifest = self.settings(), {'test_manifest': True}
        checkpoint = dict(smoke=False, signature=WORKFLOW.experiment_signature(settings, manifest),
                          manifest_digest=WORKFLOW.digest(manifest))
        EXTERNAL.check_checkpoint(checkpoint, settings, manifest)
        with self.assertRaisesRegex(ValueError, 'not smoke'):
            EXTERNAL.check_checkpoint(dict(checkpoint, smoke=True), settings, manifest)
        changed = copy.deepcopy(settings)
        changed['inference']['mask_threshold'] = .4
        with self.assertRaisesRegex(ValueError, 'differs'):
            EXTERNAL.check_checkpoint(checkpoint, changed, manifest)

    def test_external_lock_rejects_checkpoint_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock.json'
            request = dict(checkpoint_sha256='original', source_validation_passed=False)
            with self.assertRaises(FileNotFoundError):
                EXTERNAL.lock_request(path, request)
            EXTERNAL.lock_request(path, request, create=True)
            self.assertEqual(EXTERNAL.lock_request(path, request)['request'], request)
            with self.assertRaisesRegex(ValueError, 'lock changed'):
                EXTERNAL.lock_request(path, dict(request, checkpoint_sha256='different'))

    def test_failed_external_attempt_can_bind_streaming_without_changing_weight_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = dict(output_root=tmp, splits={'external': ['wytham_vox0.1']})
            request = dict(checkpoint_sha256='unchanged', training_signature='unchanged')
            lock = dict(request=copy.deepcopy(request))
            policy = EXTERNAL.preparation_policy(dict(stream_tiles=True))
            EXTERNAL.check_preparation(lock, policy, settings, create=True)
            self.assertEqual(lock['request'], request)
            self.assertEqual(lock['preparation'], policy)
            EXTERNAL.check_preparation(lock, policy, settings)
            changed = dict(policy, source_code={'changed': 'hash'})
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                EXTERNAL.check_preparation(lock, changed, settings, create=True)
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                EXTERNAL.check_preparation(lock, None, settings)

    def test_completed_external_attempt_cannot_adopt_a_different_tile_implementation(self):
        for marker in ('manifest', 'cache', 'lock_digest'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                settings = dict(output_root=tmp, splits={'external': ['wytham_vox0.1']})
                lock = dict(request={})
                if marker == 'lock_digest':
                    lock['external_manifest_digest'] = 'completed'
                else:
                    path = Path(tmp) / ('manifest.json' if marker == 'manifest' else
                                        'external/wytham_vox0.1/cache.json')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    EXTERNAL.check_preparation(lock, EXTERNAL.preparation_policy(dict(stream_tiles=True)),
                                              settings, create=True)

    def test_external_evaluation_allows_failed_source_gate_without_training_or_overwrite(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings()
            settings.update(output_root=str(Path(tmp) / 'train_cache'),
                            work_dir=str(Path(tmp) / 'train_work'), regions=REGIONS,
                            splits={'train': ['A1N'], 'validation': ['G4N']},
                            pilot_plots=['A1N', 'G4N'])
            settings['training'].update(device='cpu', hidden_dim=8)
            entries = []
            for split, plot in WORKFLOW.selected_plots(settings, False):
                synthetic_cache(Path(settings['output_root']) / split / plot)
                entries.append(dict(split=split, plot=plot, cache=f'{split}/{plot}', signature='synthetic'))
            manifest = dict(schema=1, pilot=False, baseline_signature='synthetic',
                            baseline_request={key: settings[key] for key in ('checkpoint', 'pipeline_template')},
                            regions=REGIONS, forests=entries)
            WORKFLOW.write_json(WORKFLOW.manifest_path(settings, False), manifest)
            source_report_path = Path(settings['work_dir']) / 'evaluation/summary.json'
            WORKFLOW.write_json(source_report_path, dict(smoke=False, passed=False, forests=[dict(plot='G4N')]))
            model = CandidateMaskMLP(8)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.feature_mean.fill_(3)
                model.feature_scale.fill_(2)
            checkpoint_path = Path(settings['work_dir']) / 'best.pth'
            torch.save(dict(smoke=False, epoch=26, hidden_dim=8, model=model.state_dict(),
                            signature=WORKFLOW.experiment_signature(settings, manifest),
                            manifest_digest=WORKFLOW.digest(manifest)), checkpoint_path)
            training_config = Path(tmp) / 'training.yaml'
            training_config.write_text(yaml.safe_dump(settings), encoding='utf-8')
            source = Path(tmp) / 'input/wytham_vox0.1.npz'
            source.parent.mkdir()
            np.savez(source, points=np.zeros((1, 3)), labels=np.ones(1))
            config = dict(training_config=str(training_config), checkpoint=str(checkpoint_path),
                          forest_path=str(source), output_root=str(Path(tmp) / 'external_cache'),
                          work_dir=str(Path(tmp) / 'external_work'))
            _, _, derived, _, request = EXTERNAL.load_context(config)
            cache = Path(derived['output_root']) / 'external/wytham_vox0.1'
            synthetic_cache(cache)
            external_manifest = dict(manifest, forests=[dict(
                split='external', plot='wytham_vox0.1', cache='external/wytham_vox0.1', signature='synthetic')])
            WORKFLOW.write_json(WORKFLOW.manifest_path(derived, False), external_manifest)
            WORKFLOW.write_json(Path(derived['output_root']) / 'external_lock.json', dict(
                request=request, external_manifest_digest=WORKFLOW.digest(external_manifest)))
            checkpoint_bytes, source_bytes = checkpoint_path.read_bytes(), source_report_path.read_bytes()
            result = EXTERNAL.evaluate(config)
            self.assertTrue(result['external'])
            self.assertIsNone(result['passed'])
            self.assertFalse(result['locked_training']['source_validation_passed'])
            self.assertEqual(result['locked_training']['checkpoint_epoch'], 26)
            self.assertEqual(result['forests'][0]['assignment']['score_diagnostics']['threshold'], .5)
            self.assertEqual(result['baseline_macro'], result['refined_macro'])
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_bytes)
            self.assertEqual(source_report_path.read_bytes(), source_bytes)


class CandidateMaskStreamingTileTests(unittest.TestCase):
    @staticmethod
    def original_method():
        # Compile the actual reference method without importing open3d/spconv.
        source = ROOT / 'tree_learn/util/data_preparation.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        generator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                         node.name == 'SampleGenerator')
        method = next(node for node in generator.body if isinstance(node, ast.FunctionDef) and
                      node.name == 'tile_generate_and_save')
        namespace = dict(np=np, torch=torch, json=json, os=__import__('os'), tqdm=lambda values: values)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['tile_generate_and_save']

    @staticmethod
    def generator(directory, dtype=np.float32, feature_dim=1):
        directory = Path(directory)
        (directory / 'npz').mkdir(parents=True)
        (directory / 'json').mkdir()
        # Include exact grid edges, their neighbors, and isolated extremes.
        values = np.array([0, .75, 8.25, 15.75, 23.25, 24], dtype=dtype)
        values = np.concatenate([values, np.nextafter(values[1:5], dtype(-np.inf)),
                                 np.nextafter(values[1:5], dtype(np.inf))])
        x, y = np.meshgrid(values, values)
        points = np.column_stack([x.ravel(), y.ravel(), np.linspace(-1, 37.1234, x.size)]).astype(dtype)
        features = np.random.default_rng(42).normal(size=(len(points), feature_dim)).astype(np.float32)
        return SimpleNamespace(points=points, label=(np.arange(len(points)) % 7 - 1).astype(dtype),
                               feats=features, x_range=np.array([0, 24], dtype=dtype),
                               y_range=np.array([0, 24], dtype=dtype), plot_name='synthetic',
                               n_neigh_sor=None, multiplier_sor=None, rad=None, npoints_rad=None,
                               save_dir_data=str(directory / 'npz'), save_dir_meta_data=str(directory / 'json'))

    def compare_outputs(self, first, second):
        names = sorted(path.name for path in (first / 'npz').glob('*.npz'))
        self.assertTrue(names)
        self.assertEqual(names, sorted(path.name for path in (second / 'npz').glob('*.npz')))
        for name in names:
            with np.load(first / 'npz' / name) as expected, np.load(second / 'npz' / name) as actual:
                self.assertEqual(expected.files, actual.files)
                for key in expected.files:
                    self.assertEqual(expected[key].dtype, actual[key].dtype)
                    np.testing.assert_array_equal(expected[key], actual[key])
            meta = Path(name).with_suffix('.json')
            self.assertEqual(json.loads((first / 'json' / meta).read_text()),
                             json.loads((second / 'json' / meta).read_text()))

    def test_streaming_tiles_exactly_match_original_on_cpu(self):
        reference = self.original_method()
        logger = logging.getLogger('streaming_tile_test')
        for dtype in (np.float32, np.float64):
            for width in (1, 3):
                for compressed in (False, True):
                    for stride in (.5, 1.):
                        with self.subTest(dtype=dtype, width=width, compressed=compressed, stride=stride):
                            with tempfile.TemporaryDirectory() as tmp:
                                first, second = Path(tmp) / 'original', Path(tmp) / 'streaming'
                                with mock.patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
                                    reference(self.generator(first, dtype, width), 8., 13.5, stride,
                                              compressed=compressed, logger=logger)
                                    tile_generate_and_save_streaming(self.generator(second, dtype, width),
                                                                    8., 13.5, stride,
                                                                    compressed=compressed, logger=logger)
                                self.compare_outputs(first, second)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; CPU reference comparison still runs')
    def test_streaming_tiles_exactly_match_original_on_cuda(self):
        reference = self.original_method()
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                first, second = Path(tmp) / 'original', Path(tmp) / 'streaming'
                reference(self.generator(first, dtype), 8., 13.5, .5, logger=logging.getLogger('tile_test'))
                tile_generate_and_save_streaming(self.generator(second, dtype), 8., 13.5, .5,
                                                logger=logging.getLogger('tile_test'))
                self.compare_outputs(first, second)

    def test_streaming_releases_saved_chunk_views_before_the_next_save(self):
        references = []
        save = np.savez

        def record_save(path, **values):
            self.assertTrue(all(reference() is None for reference in references))
            references.append(weakref.ref(values['points']))
            save(path, **values)

        with tempfile.TemporaryDirectory() as tmp:
            generator = self.generator(Path(tmp) / 'streaming')
            with mock.patch.object(torch.Tensor, 'cuda', lambda tensor: tensor), \
                    mock.patch('tree_learn.mask_refinement.tiling.np.savez', new=record_save):
                tile_generate_and_save_streaming(generator, 8., 13.5, .5,
                                                logger=logging.getLogger('tile_test'))
        self.assertGreater(len(references), 1)
        self.assertTrue(all(reference() is None for reference in references))

    def test_streaming_rejects_unverified_rotation_or_denoising(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = self.generator(Path(tmp) / 'streaming')
            with self.assertRaisesRegex(ValueError, 'rotation or denoising'):
                tile_generate_and_save_streaming(generator, 8., 13.5, .5,
                                                plot_corners=[[0, 0], [1, 1]])
            generator.n_neigh_sor = 10
            with self.assertRaisesRegex(ValueError, 'rotation or denoising'):
                tile_generate_and_save_streaming(generator, 8., 13.5, .5)
            self.assertEqual(list((Path(tmp) / 'streaming/npz').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
