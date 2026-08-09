import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'tree_learn' / 'util' / 'seed_attention.py'
MODEL_SPEC = importlib.util.spec_from_file_location(
    'seed_attention_model_test', MODEL_PATH)
MODEL = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(MODEL)
TRAIN_PATH = (
    ROOT / 'tools' / 'training' / 'train_seed_relation_attention.py')
TRAIN_SPEC = importlib.util.spec_from_file_location(
    'seed_relation_training_test', TRAIN_PATH)
TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN)


MODEL_CONFIG = {
    'control_hidden_dims': [24, 12],
    'attention_hidden_dim': 8,
    'attention_feedforward_dim': 16,
    'dropout': 0.0,
}


class SeedRelationModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.query = torch.randn(5, 6)
        self.neighbours = torch.randn(5, 4, 6)
        self.geometry = torch.randn(5, 4, 3)
        self.valid = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ], dtype=torch.bool)

    def test_both_models_have_finite_dual_heads_and_gradients(self):
        for model_type in ('neighborhood_mlp', 'relation_attention'):
            model = MODEL.build_seed_relation_model(
                6, 3, model_type, MODEL_CONFIG)
            reliability, coverage = model(
                self.query, self.neighbours, self.geometry, self.valid)
            self.assertEqual(reliability.shape, (5,))
            self.assertEqual(coverage.shape, (5,))
            self.assertTrue(torch.isfinite(reliability).all())
            (reliability.mean() + coverage.mean()).backward()
            self.assertTrue(all(
                parameter.grad is not None and
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()))

    def test_invalid_neighbours_cannot_change_outputs(self):
        for model_type in ('neighborhood_mlp', 'relation_attention'):
            model = MODEL.build_seed_relation_model(
                6, 3, model_type, MODEL_CONFIG).eval()
            first = model(
                self.query, self.neighbours,
                self.geometry, self.valid)
            changed_features = self.neighbours.clone()
            changed_geometry = self.geometry.clone()
            changed_features[~self.valid] += 10000
            changed_geometry[~self.valid] -= 10000
            second = model(
                self.query, changed_features,
                changed_geometry, self.valid)
            torch.testing.assert_close(first[0], second[0])
            torch.testing.assert_close(first[1], second[1])

    def test_neighbour_order_is_invariant(self):
        permutation = torch.tensor([2, 0, 3, 1])
        model = MODEL.build_seed_relation_model(
            6, 3, 'relation_attention', MODEL_CONFIG).eval()
        first = model(
            self.query, self.neighbours, self.geometry, self.valid)
        second = model(
            self.query,
            self.neighbours[:, permutation],
            self.geometry[:, permutation],
            self.valid[:, permutation])
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    def test_preregistered_parameter_counts_are_matched(self):
        config = {
            'model_types': ['neighborhood_mlp', 'relation_attention'],
            'expected_input_dim': 61,
            'geometry_dim': 6,
            'model': {
                'control_hidden_dims': [176, 96],
                'attention_hidden_dim': 64,
                'attention_feedforward_dim': 128,
                'dropout': 0.1,
            },
        }
        counts = TRAIN.model_parameter_counts(config)
        gap = abs(
            counts['neighborhood_mlp'] - counts['relation_attention']) / max(
                counts.values())
        self.assertLessEqual(gap, 0.10)


class SeedRelationBatchTests(unittest.TestCase):

    def test_batch_geometry_and_invalid_fallback_are_finite(self):
        dataset = {
            'features': np.arange(20, dtype=np.float32).reshape(5, 4),
            'coords': np.asarray([
                [0, 0, 0], [1, 0, 1], [0, 1, 2],
                [5, 5, 3], [6, 5, 4],
            ], dtype=np.float32),
            'base_votes_xy': np.asarray([
                [0, 0], [0.2, 0], [0, 0.2], [5, 5], [5.2, 5],
            ], dtype=np.float32),
            'neighbor_index': np.asarray([
                [0, 1, 2], [1, 0, 1], [2, 0, 2],
                [3, 4, 3], [4, 3, 4],
            ], dtype=np.int32),
            'neighbor_valid': np.asarray([
                [1, 1, 1], [1, 1, 0], [1, 1, 0],
                [1, 1, 0], [1, 1, 0],
            ], dtype=bool),
        }
        query, neighbours, geometry, valid = TRAIN.prepare_relation_batch(
            dataset, np.asarray([1, 3]),
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            {'radius': 0.6, 'vertical_scale': 4.0})
        self.assertEqual(query.shape, (2, 4))
        self.assertEqual(neighbours.shape, (2, 3, 4))
        self.assertEqual(geometry.shape, (2, 3, 6))
        self.assertTrue(np.isfinite(geometry).all())
        self.assertTrue(np.all(geometry[~valid] == 0))


if __name__ == '__main__':
    unittest.main()
