import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / 'tree_learn' / 'util' / 'seed_quality.py'
SEED_SPEC = importlib.util.spec_from_file_location(
    'seed_quality_e1b_test', SEED_PATH)
SEED = importlib.util.module_from_spec(SEED_SPEC)
SEED_SPEC.loader.exec_module(SEED)

TRAIN_PATH = ROOT / 'tools' / 'training' / 'train_seed_quality_mlp.py'
TRAIN_SPEC = importlib.util.spec_from_file_location(
    'train_seed_quality_mlp_test', TRAIN_PATH)
TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN)


class SeedMLPModelTests(unittest.TestCase):

    def test_dual_head_shapes_and_gradients(self):
        model = SEED.build_seed_quality_mlp(
            5, {'hidden_dims': [8, 4], 'dropout': 0.0})
        values = torch.randn(7, 5)
        reliability, coverage = model(values)
        self.assertEqual(reliability.shape, (7,))
        self.assertEqual(coverage.shape, (7,))
        loss = reliability.square().mean() + coverage.square().mean()
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()))

    def test_exact_selector_protects_coverage_then_fills_reliability(self):
        reliability = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
        coverage = np.asarray([0.1, 0.2, 0.3, 1.0, 0.0])
        selected, metadata = SEED.select_seed_candidates(
            reliability, coverage,
            keep_ratio=0.6, coverage_protect_ratio=0.2)
        np.testing.assert_array_equal(
            np.flatnonzero(selected), [0, 1, 3])
        self.assertEqual(metadata['num_kept'], 3)
        self.assertEqual(metadata['num_coverage_protected'], 1)

    def test_top_k_ties_are_resolved_by_smaller_index(self):
        selected = SEED.exact_top_k_mask(np.ones(5), 2)
        np.testing.assert_array_equal(np.flatnonzero(selected), [0, 1])


class SeedMLPDataAndMetricTests(unittest.TestCase):

    def test_plot_balanced_sampling_and_train_only_standardizer(self):
        plot_ids = np.asarray([0] * 5 + [1] * 2 + [2] * 4)
        train = plot_ids < 2
        sampled = TRAIN.balanced_plot_sample_indices(
            plot_ids, train, examples_per_plot=4, seed=42)
        self.assertEqual(np.count_nonzero(plot_ids[sampled] == 0), 4)
        self.assertEqual(np.count_nonzero(plot_ids[sampled] == 1), 4)
        features = np.column_stack([
            np.arange(len(plot_ids), dtype=np.float32),
            np.ones(len(plot_ids), dtype=np.float32),
        ])
        mean, scale, _ = TRAIN.fit_sampled_standardizer(
            features, plot_ids, train, max_points=8, seed=42)
        changed = features.copy()
        changed[~train] += 10000
        changed_mean, changed_scale, _ = TRAIN.fit_sampled_standardizer(
            changed, plot_ids, train, max_points=8, seed=42)
        np.testing.assert_array_equal(mean, changed_mean)
        np.testing.assert_array_equal(scale, changed_scale)

    def test_numpy_binary_metrics_handle_perfect_and_tied_scores(self):
        targets = np.asarray([False, True, False, True])
        perfect = TRAIN.binary_metrics(
            targets, np.asarray([0.1, 0.9, 0.2, 0.8]))
        self.assertAlmostEqual(perfect['roc_auc'], 1.0)
        self.assertAlmostEqual(perfect['average_precision'], 1.0)
        tied = TRAIN.binary_metrics(targets, np.ones(4))
        self.assertAlmostEqual(tied['roc_auc'], 0.5)
        self.assertAlmostEqual(tied['average_precision'], 0.5)

    def test_selector_applies_exact_keep_ratio_per_forest(self):
        reliability = np.linspace(0.1, 1.0, 10)
        coverage = reliability[::-1].copy()
        reliable = np.asarray([0, 1] * 5, dtype=bool)
        critical = np.asarray([
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        ], dtype=bool)
        utility = reliable.astype(np.float32)
        tree_ids = np.asarray([1, 1, 2, 2, 3, 1, 1, 2, 2, 3])
        plot_ids = np.asarray([0] * 5 + [1] * 5)
        result = TRAIN.evaluate_seed_scores(
            reliability, coverage, reliable, critical, utility,
            tree_ids, plot_ids, {
                'keep_ratio': 0.6,
                'coverage_protect_ratio': 0.2,
                'random_control_seeds': [11, 12, 13],
            })
        self.assertEqual(result['selection_info']['num_kept'], 6)
        self.assertEqual(result['selection_info']['num_forests'], 2)
        self.assertAlmostEqual(result['per_plot']['0']['keep_rate'], 0.6)
        self.assertAlmostEqual(result['per_plot']['1']['keep_rate'], 0.6)

    def test_aggregate_summary_and_markdown_cover_all_gate_fields(self):
        reliability = np.linspace(0.1, 1.0, 10)
        coverage = reliability[::-1].copy()
        reliable = np.asarray([0, 1] * 5, dtype=bool)
        critical = np.asarray(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
        utility = reliable.astype(np.float32)
        tree_ids = np.asarray([1, 1, 2, 2, 3, 1, 1, 2, 2, 3])
        plot_ids = np.asarray([0] * 5 + [1] * 5)
        selector = {
            'keep_ratio': 0.6,
            'coverage_protect_ratio': 0.2,
            'random_control_seeds': [11, 12, 13],
        }
        metrics = TRAIN.evaluate_seed_scores(
            reliability, coverage, reliable, critical, utility,
            tree_ids, plot_ids, selector)
        results = [{
            'seed': seed, 'best_epoch': 1, 'device': 'cpu',
            'parameter_count': 10, 'standardizer_sample_size': 8,
            'metrics': metrics,
        } for seed in (42, 43, 44)]
        settings = {
            'seeds': [42, 43, 44], 'locked_seed': 42,
            'selector': selector,
            'gate': {
                'min_reliability_roc_auc': 0.0,
                'min_coverage_ap_lift': 0.0,
                'min_critical_recall_gain_over_random': -1.0,
                'min_critical_recall_gain_over_reliability': -1.0,
                'min_selected_utility_relative_gain': -1.0,
                'min_critical_tree_coverage': 0.0,
                'min_plot_critical_tree_coverage': 0.0,
                'max_critical_recall_sample_std': 1.0,
            },
        }
        dataset = {
            'features': np.zeros((10, 4), dtype=np.float32),
            'split_id': np.asarray([0] * 5 + [1] * 5),
            'input_dim': 4,
            'feature_names': ['a', 'b', 'c', 'd'],
            'plot_names': ['A1N', 'G4N'],
            'plot_splits': ['train', 'validation'],
        }
        summary = TRAIN.aggregate_results(results, settings, dataset)
        self.assertTrue(summary['gate']['passed'])
        self.assertIn('PASS', TRAIN.format_markdown(summary))
    @staticmethod
    def write_artifact(root, plot, split, start):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{plot}.npz'
        count = 4
        np.savez_compressed(
            path,
            candidate_indices=np.arange(start, start + count),
            backbone_features=np.arange(
                count * 2, dtype=np.float32).reshape(count, 2),
            scalar_features=np.ones((count, 2), dtype=np.float32),
            target_valid=np.ones(count, dtype=bool),
            target_is_tree=np.asarray([1, 1, 1, 0], dtype=bool),
            target_reliable=np.asarray([1, 0, 1, 0], dtype=bool),
            target_coverage_critical=np.asarray([1, 0, 0, 0], dtype=bool),
            target_utility=np.asarray([1.0, 0.2, 0.8, 0.0], dtype=np.float32),
            target_tree_id=np.asarray([1, 1, 2, 0]),
        )
        metadata = {
            'source_plot': plot,
            'split': split,
            'scalar_feature_names': ['scalar_a', 'scalar_b'],
        }
        (directory / f'{plot}_metadata.json').write_text(
            json.dumps(metadata), encoding='utf-8')
        return path

    def test_dataset_loader_checks_schema_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = self.write_artifact(root, 'A1N', 'train', 0)
            val_path = self.write_artifact(root, 'G4N', 'validation', 10)
            manifest = root / 'manifest.csv'
            with manifest.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'source_plot', 'split', 'artifact_path',
                    'num_candidates'])
                writer.writeheader()
                writer.writerow({
                    'source_plot': 'A1N', 'split': 'train',
                    'artifact_path': str(train_path), 'num_candidates': 4})
                writer.writerow({
                    'source_plot': 'G4N', 'split': 'validation',
                    'artifact_path': str(val_path), 'num_candidates': 4})
            dataset = TRAIN.load_seed_quality_dataset(
                root, manifest, expected_backbone_dim=2,
                expected_scalar_dim=2)
            self.assertEqual(dataset['features'].shape, (8, 4))
            self.assertEqual(dataset['feature_names'], [
                'backbone_0', 'backbone_1', 'scalar_a', 'scalar_b'])
            np.testing.assert_array_equal(
                np.unique(dataset['split_id']), [0, 1])


if __name__ == '__main__':
    unittest.main()