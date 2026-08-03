import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'training' / 'train_vertical_instance_quality.py')
SPEC = importlib.util.spec_from_file_location(
    'vertical_instance_quality_standalone', MODULE_PATH)
VERTICAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERTICAL)


class VerticalInstanceQualityTests(unittest.TestCase):

    def make_artifacts(self, root):
        manifest_rows = []
        for split, plot, instance_id, offset in (
                ('train', 'A1N', 11, 0.0),
                ('validation', 'G4N', 22, 10.0)):
            artifact = root / split / f'{plot}.npz'
            artifact.parent.mkdir(parents=True, exist_ok=True)
            tokens = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
            tokens += offset
            layer_mask = np.asarray([[True, True, True, False]])
            tokens[:, -1] = 0.0
            np.savez_compressed(
                artifact,
                instance_ids=np.asarray([instance_id]),
                global_feature_values=np.asarray([[1.0, 2.0]]),
                global_feature_names=np.asarray(['global_a', 'global_b']),
                vertical_tokens=tokens,
                layer_valid_mask=layer_mask,
                token_feature_names=np.asarray([
                    'token_a', 'token_b', 'token_c',
                    'token_d', 'token_e', 'token_f']),
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
        return manifest

    def test_vertical_loader_aligns_tokens_with_forest_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = VERTICAL.load_vertical_quality_dataset(
                root, self.make_artifacts(root))
        self.assertEqual(dataset['vertical_tokens'].shape, (2, 4, 6))
        self.assertEqual(dataset['layer_valid_mask'].shape, (2, 4))
        self.assertEqual(dataset['token_feature_names'][0], 'token_a')
        np.testing.assert_array_equal(dataset['instance_id'], [11, 22])
        self.assertEqual(set(dataset['split']), {'train', 'validation'})

    def test_token_standardizer_ignores_and_zeros_invalid_layers(self):
        tokens = np.asarray([
            [[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]],
            [[5.0, 6.0], [7.0, 8.0], [-999.0, -999.0]],
        ], dtype=np.float32)
        mask = np.asarray([
            [True, True, False], [True, True, False]])
        median, scale = VERTICAL.fit_token_standardizer(tokens, mask)
        output = VERTICAL.standardize_tokens(tokens, mask, median, scale)
        np.testing.assert_allclose(median, [4.0, 5.0])
        self.assertTrue(np.all(output[~mask] == 0))
        self.assertTrue(np.isfinite(output).all())

    @staticmethod
    def make_baseline_summary():
        def metric(mean):
            return {
                'mean': mean, 'sample_std': 0.0,
                'min': mean, 'max': mean}
        return {
            'best_aggregate_model': 'global_mlp',
            'models': {'global_mlp': {
                'roc_auc': metric(0.90), 'fp_ap': metric(0.80),
                'iou_mae': metric(0.10), 'spearman': metric(0.75),
                'filtered_f1': metric(0.80)}},
            'gate': {'passed': True},
        }

    @staticmethod
    def make_dataset():
        return {
            'instance_id': np.arange(4),
            'vertical_tokens': np.zeros((4, 8, 6), dtype=np.float32),
            'token_feature_names': [f'f{i}' for i in range(6)],
            'source_plot': np.asarray(['A', 'A', 'B', 'B']),
            'split': np.asarray([
                'train', 'train', 'validation', 'validation']),
        }

    @staticmethod
    def make_settings():
        return {'seeds': [42, 43, 44], 'gate': {
            'min_iou_mae_improvement': 0.002,
            'min_fp_ap_improvement': 0.002,
            'max_f1_sample_std': 0.01,
            'f1_tolerance': 1e-6}}

    @staticmethod
    def make_rows(iou_mae=0.09, fp_ap=0.80):
        return [{
            'model': 'vertical_mlp', 'seed': seed,
            'roc_auc': 0.91, 'positive_ap': 0.95,
            'fp_ap': fp_ap, 'iou_mae': iou_mae,
            'spearman': 0.78, 'filtered_f1': 0.80,
            'filtered_completeness': 0.99,
            'filtered_commission': 0.10,
            'unfiltered_f1': 0.79,
        } for seed in (42, 43, 44)]

    def test_gate_passes_when_vertical_iou_clearly_improves(self):
        summary = VERTICAL.summarize_vertical(
            self.make_rows(), self.make_baseline_summary(),
            self.make_settings(), self.make_dataset())
        self.assertTrue(summary['gate']['iou_mae_improved'])
        self.assertTrue(summary['gate']['passed'])

    def test_gate_stops_without_vertical_information_gain(self):
        summary = VERTICAL.summarize_vertical(
            self.make_rows(iou_mae=0.099, fp_ap=0.801),
            self.make_baseline_summary(),
            self.make_settings(), self.make_dataset())
        self.assertFalse(summary['gate']['vertical_information_improved'])
        self.assertFalse(summary['gate']['passed'])

    @unittest.skipUnless(
        importlib.util.find_spec('torch') is not None,
        'PyTorch is unavailable in this test environment.')
    def test_vertical_mlp_masks_layers_and_isolates_batches(self):
        import torch

        torch.manual_seed(4)
        model = VERTICAL.build_vertical_mlp(6, {
            'token_hidden_dim': 8, 'token_mlp_layers': 2,
            'instance_hidden_dims': [8], 'dropout': 0.0}).eval()
        tokens = torch.randn(2, 4, 6)
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False]])
        first = model(tokens, mask)
        changed = tokens.clone()
        changed[0, 2:] = 1e6
        changed[1] = -1e6
        second = model(changed, mask)
        self.assertEqual(first[0].shape, (2,))
        self.assertEqual(first[1].shape, (2,))
        self.assertTrue(torch.allclose(first[0][0], second[0][0]))
        self.assertTrue(torch.allclose(first[1][0], second[1][0]))
        self.assertTrue(torch.all((first[1] >= 0) & (first[1] <= 1)))


if __name__ == '__main__':
    unittest.main()