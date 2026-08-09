import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = (
    ROOT / 'tools' / 'training' / 'train_seed_relation_attention.py')
SPEC = importlib.util.spec_from_file_location(
    'seed_relation_training_smoke', TRAIN_PATH)
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class SeedRelationTrainingSmokeTests(unittest.TestCase):

    def test_one_epoch_relation_attention_is_finite(self):
        generator = np.random.default_rng(42)
        num_points = 16
        features = generator.normal(size=(num_points, 4)).astype(np.float32)
        coords = generator.normal(size=(num_points, 3)).astype(np.float32)
        votes = coords[:, :2] * 0.1
        plot_ids = np.asarray([0] * 8 + [1] * 8, dtype=np.int16)
        split_ids = np.asarray([0] * 8 + [1] * 8, dtype=np.int8)
        local_is_tree = np.asarray(
            [1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        local_utility = np.asarray(
            [0.9, 0.8, 0.3, 0.2, 0.7, 0.1, 0.0, 0.0],
            dtype=np.float32)
        local_reliable = local_is_tree & (local_utility >= 0.5)
        local_critical = np.asarray(
            [1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
        local_tree_ids = np.asarray([1, 1, 1, 2, 2, 2, 0, 0])
        neighbours = np.empty((num_points, 2), dtype=np.int32)
        for start in (0, 8):
            for local in range(8):
                index = start + local
                neighbours[index] = [index, start + (local + 1) % 8]
        dataset = {
            'features': features,
            'coords': coords,
            'base_votes_xy': votes,
            'neighbor_index': neighbours,
            'neighbor_valid': np.ones((num_points, 2), dtype=bool),
            'target_is_tree': np.tile(local_is_tree, 2),
            'target_reliable': np.tile(local_reliable, 2),
            'target_coverage_critical': np.tile(local_critical, 2),
            'target_utility': np.tile(local_utility, 2),
            'target_tree_id': np.tile(local_tree_ids, 2),
            'plot_id': plot_ids,
            'split_id': split_ids,
            'input_dim': 4,
        }
        settings = {
            'geometry_dim': 6,
            'model': {
                'control_hidden_dims': [16, 8],
                'attention_hidden_dim': 8,
                'attention_feedforward_dim': 16,
                'dropout': 0.0,
            },
            'neighborhood': {
                'num_neighbors': 2,
                'radius': 1.0,
                'vertical_scale': 2.0,
            },
            'training': {
                'epochs': 1,
                'patience': 1,
                'validation_frequency': 1,
                'examples_per_plot': 8,
                'batch_size': 4,
                'validation_batch_size': 4,
                'max_standardizer_points': 8,
                'learning_rate': 1e-3,
                'min_learning_rate': 1e-4,
                'warmup_epochs': 1,
                'weight_decay': 0.0,
                'utility_loss_weight': 0.5,
                'coverage_loss_weight': 1.0,
                'max_pos_weight': 10.0,
                'grad_norm_clip': 5.0,
                'amp': False,
            },
            'selector': {
                'keep_ratio': 0.75,
                'coverage_protect_ratio': 0.25,
                'random_control_seeds': [11, 12, 13],
            },
        }
        result = TRAIN.train_one_model_seed(
            dataset, settings, 'relation_attention', 42)
        self.assertEqual(result['best_epoch'], 1)
        self.assertTrue(np.isfinite(
            result['metrics']['tree_only_reliability']['roc_auc']))
        self.assertTrue(all(
            torch.isfinite(value).all()
            for value in result['state_dict'].values()))


if __name__ == '__main__':
    unittest.main()
