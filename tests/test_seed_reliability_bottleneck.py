import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / 'tools' / 'diagnostics' /
    'diagnose_seed_reliability_bottleneck.py')
SPEC = importlib.util.spec_from_file_location(
    'seed_reliability_bottleneck_test', SCRIPT_PATH)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class SeedReliabilityBottleneckTests(unittest.TestCase):

    def test_average_ranks_and_spearman_handle_ties(self):
        ranks = DIAGNOSTIC.average_ranks([1.0, 1.0, 3.0, 2.0])
        np.testing.assert_allclose(ranks, [0.5, 0.5, 3.0, 2.0])
        self.assertAlmostEqual(
            DIAGNOSTIC.spearman_correlation(
                [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
            1.0)

    def test_univariate_audit_reports_inverse_direction(self):
        features = np.asarray([
            [0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0],
        ], dtype=np.float32)
        targets = np.asarray([0, 0, 1, 1], dtype=bool)
        top, _ = DIAGNOSTIC.top_univariate_features(
            features, targets, ['backbone_0', 'scalar_a'], 2)
        self.assertEqual({row['direction'] for row in top}, {
            'higher', 'lower'})
        self.assertTrue(all(
            row['separability_auc'] == 1.0 for row in top))

    def test_locked_scores_align_with_validation_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_dir = root / 'validation'
            validation_dir.mkdir(parents=True)
            artifact_path = validation_dir / 'G4N.npz'
            candidate_indices = np.arange(6, dtype=np.int64)
            is_tree = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
            utility = np.asarray(
                [0.9, 0.7, 0.4, 0.2, 0.0, 0.0], dtype=np.float32)
            reliable = is_tree & (utility >= 0.5)
            np.savez_compressed(
                artifact_path,
                candidate_indices=candidate_indices,
                backbone_features=np.arange(
                    12, dtype=np.float32).reshape(6, 2),
                scalar_features=np.asarray([
                    [0.9, 0.1], [0.8, 0.2], [0.4, 0.6],
                    [0.2, 0.8], [0.1, 0.9], [0.0, 1.0],
                ], dtype=np.float32),
                target_is_tree=is_tree,
                target_reliable=reliable,
                target_coverage_critical=np.asarray(
                    [1, 0, 1, 0, 0, 0], dtype=bool),
                target_utility=utility,
                target_vote_error_xy=np.asarray(
                    [0.1, 0.2, 0.4, 0.6, 0.0, 0.0], dtype=np.float32),
                target_vote_cell_purity=np.asarray(
                    [1.0, 0.9, 0.8, 0.7, 0.0, 0.0], dtype=np.float32),
                target_tree_id=np.asarray([1, 1, 2, 2, 0, 0]),
            )
            metadata = {
                'source_plot': 'G4N',
                'scalar_feature_names': ['scalar_a', 'scalar_b'],
            }
            (validation_dir / 'G4N_metadata.json').write_text(
                json.dumps(metadata), encoding='utf-8')
            manifest_path = root / 'manifest.csv'
            with manifest_path.open(
                    'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'source_plot', 'split', 'artifact_path',
                    'num_candidates'])
                writer.writeheader()
                writer.writerow({
                    'source_plot': 'G4N',
                    'split': 'validation',
                    'artifact_path': str(artifact_path),
                    'num_candidates': 6,
                })

            scores = np.asarray(
                [0.95, 0.8, 0.45, 0.3, 0.2, 0.1], dtype=np.float32)
            locked_path = root / 'locked.npz'
            np.savez_compressed(
                locked_path,
                reliability_scores=scores,
                coverage_scores=np.linspace(0.1, 0.6, 6),
                target_reliable=reliable,
                target_coverage_critical=np.asarray(
                    [1, 0, 1, 0, 0, 0], dtype=bool),
                target_utility=utility,
                target_tree_id=np.asarray([1, 1, 2, 2, 0, 0]),
                candidate_index=candidate_indices,
                plot_id=np.zeros(6, dtype=np.int16),
            )
            observed_auc = DIAGNOSTIC.finite_binary_metrics(
                reliable, scores)['roc_auc']
            e1b_path = root / 'summary.json'
            e1b_path.write_text(json.dumps({
                'per_seed': [{
                    'seed': 42,
                    'reliability_roc_auc': observed_auc,
                }],
                'gate': {
                    'reliability_auc_passed': False,
                    'coverage_ap_lift_passed': True,
                    'passed': False,
                },
            }), encoding='utf-8')
            settings = {
                'data_root': str(root),
                'manifest_path': str(manifest_path),
                'e1b_summary_path': str(e1b_path),
                'locked_scores_path': str(locked_path),
                'locked_seed': 42,
                'expected_validation_plots': 1,
                'expected_backbone_dim': 2,
                'expected_scalar_dim': 2,
                'reliability_threshold': 0.5,
                'diagnostic': {
                    'utility_thresholds': [0.4, 0.5, 0.6],
                    'ambiguity_half_widths': [0.05, 0.1],
                    'max_feature_samples_per_plot': 6,
                    'sample_seed': 42,
                    'top_features_to_report': 3,
                },
                'decision': {
                    'target_auc': 1.1,
                    'macro_auc_gap_for_calibration': 0.05,
                    'threshold_auc_gain_for_target_redesign': 0.05,
                    'max_near_threshold_tree_rate': 0.5,
                    'min_tree_only_auc_for_pointwise_signal': 0.7,
                    'min_univariate_auc_for_capacity_signal': 0.8,
                },
            }
            data = DIAGNOSTIC.load_validation_diagnostic_data(settings)
            report = DIAGNOSTIC.build_report(settings, data)
            self.assertTrue(report['integrity_gate']['passed'])
            self.assertEqual(report['alignment']['candidate_mismatches'], 0)
            self.assertEqual(
                report['alignment']['label_definition_mismatches'], 0)
            self.assertEqual(len(report['threshold_sensitivity']), 3)
            self.assertIn('自动建议', DIAGNOSTIC.format_markdown(report))


if __name__ == '__main__':
    unittest.main()
