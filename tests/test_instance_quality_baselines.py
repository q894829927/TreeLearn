import csv
import tempfile
import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'training' / 'train_instance_quality_baselines.py'
)
SPEC = importlib.util.spec_from_file_location(
    'instance_quality_baselines_standalone', MODULE_PATH)
BASELINES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINES)


class InstanceQualityBaselineTests(unittest.TestCase):

    def test_dataset_loader_preserves_forest_split_and_manifest_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_rows = []
            for split, plot, instance_id in (
                    ('train', 'A1N', 1),
                    ('validation', 'G4N', 2)):
                artifact = root / split / f'{plot}.npz'
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    artifact,
                    instance_ids=np.asarray([instance_id]),
                    global_feature_values=np.asarray([[1.0, 2.0]]),
                    global_feature_names=np.asarray(['f1', 'f2']),
                    target_max_iou=np.asarray([0.8]),
                    target_valid=np.asarray([True]),
                    target_classification_valid=np.asarray([True]),
                    target_is_true_tree=np.asarray([True]),
                    source_plot=np.asarray(plot),
                    split=np.asarray(split),
                )
                manifest_rows.append({
                    'instance_id': instance_id,
                    'source_plot': plot,
                    'split': split,
                    'artifact_path': str(artifact),
                })
            manifest = root / 'manifest.csv'
            with manifest.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)
            dataset = BASELINES.load_quality_dataset(root, manifest)

        self.assertEqual(dataset['feature_names'], ['f1', 'f2'])
        self.assertEqual(set(dataset['split']), {'train', 'validation'})
        np.testing.assert_array_equal(dataset['instance_id'], [1, 2])

    def test_rank_correlation_handles_order_and_reverse_order(self):
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(
            BASELINES.rank_correlation(values, values), 1.0)
        self.assertAlmostEqual(
            BASELINES.rank_correlation(values, values[::-1]), -1.0)

    def test_threshold_selection_respects_completeness(self):
        targets = np.asarray([True, True, False, False])
        scores = np.asarray([0.9, 0.8, 0.7, 0.1])
        best, baseline = BASELINES.select_filter_threshold(
            targets, scores, step=0.1, max_completeness_drop=0.0)
        self.assertAlmostEqual(best['threshold'], 0.8)
        self.assertEqual(best['tp'], 2)
        self.assertEqual(best['fp'], 0)
        self.assertAlmostEqual(best['f1'], 1.0)
        self.assertLess(baseline['f1'], best['f1'])

    def test_summary_gate_accepts_stable_aggregate_baseline(self):
        rows = []
        for seed in (42, 43, 44):
            for model, auc, fp_ap in (
                    ('single_feature', 0.82, 0.60),
                    ('logistic_regression', 0.91, 0.72),
                    ('global_mlp', 0.90 + (seed - 43) * 0.005, 0.70)):
                rows.append({
                    'model': model,
                    'seed': seed,
                    'selected_feature': (
                        'confidence_mean' if model == 'single_feature' else ''),
                    'roc_auc': auc,
                    'positive_ap': 0.95,
                    'fp_ap': fp_ap,
                    'iou_mae': 0.15,
                    'spearman': 0.75,
                    'filtered_f1': 0.88,
                    'filtered_completeness': 0.99,
                    'filtered_commission': 0.10,
                    'unfiltered_f1': 0.80,
                })
        settings = {
            'seeds': [42, 43, 44],
            'gate': {
                'min_validation_roc_auc': 0.85,
                'min_validation_fp_ap': 0.65,
                'max_mlp_auc_range': 0.05,
            },
        }
        dataset = {
            'instance_id': np.arange(4),
            'feature_names': ['a', 'b'],
            'source_plot': np.asarray(['A', 'A', 'B', 'B']),
            'split': np.asarray([
                'train', 'train', 'validation', 'validation']),
        }
        summary = BASELINES.summarize(rows, settings, dataset)
        self.assertEqual(summary['best_aggregate_model'], 'logistic_regression')
        self.assertTrue(summary['gate']['passed'])


if __name__ == '__main__':
    unittest.main()
