import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'training' / 'train_instance_merge_pairs.py')
SPEC = importlib.util.spec_from_file_location(
    'instance_merge_training_standalone', MODULE_PATH)
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


class InstanceMergeTrainingTests(unittest.TestCase):

    def test_best_target_is_per_source_and_ties_use_smaller_target_id(self):
        indices = TRAINING.select_best_target_indices(
            scores=np.asarray([0.8, 0.8, 0.7, 0.9]),
            source_keys=np.asarray(['A:1', 'A:1', 'A:2', 'A:2']),
            target_ids=np.asarray([20, 10, 5, 7]),
        )
        np.testing.assert_array_equal(indices, [1, 3])

    def test_threshold_selection_enforces_precision_and_unsafe_rate(self):
        labels = np.asarray([True, False, False, True, False])
        scores = np.asarray([0.9, 0.1, 0.8, 0.7, 0.2])
        sources = np.asarray(['A:1', 'A:1', 'A:2', 'A:3', 'A:3'])
        targets = np.asarray([10, 20, 30, 40, 50])
        selected, _ = TRAINING.select_merge_threshold(
            labels,
            scores,
            sources,
            targets,
            {
                'min_pair_precision': 0.95,
                'min_positive_source_recall': 0.30,
                'max_unsafe_merge_rate': 0.0,
            },
        )
        self.assertEqual(selected['num_correct_merges'], 1)
        self.assertEqual(selected['num_unsafe_merges'], 0)
        self.assertAlmostEqual(selected['positive_source_recall'], 0.5)
        self.assertTrue(selected['passed'])

    @staticmethod
    def model_rows(mlp_recall):
        rows = []
        recalls = {
            'distance_rule': 0.30,
            'logistic_regression': 0.50,
            'pair_mlp': mlp_recall,
        }
        for model, recall in recalls.items():
            for seed in (42, 43, 44):
                rows.append({
                    'model': model,
                    'seed': seed,
                    'gate_passed': True,
                    'threshold': 0.5,
                    'pair_roc_auc': 0.9,
                    'pair_average_precision': 0.8,
                    'pair_precision': 0.96,
                    'positive_source_recall': recall,
                    'unsafe_merge_rate': 0.004,
                    'num_proposed_merges': 10,
                    'num_correct_merges': 9,
                    'num_unsafe_merges': 1,
                })
        return rows

    @staticmethod
    def selection_settings():
        return {
            'seeds': [42, 43, 44],
            'selection': {
                'locked_seed': 42,
                'min_gate_pass_seeds': 2,
                'min_logistic_recall_gain_over_distance': 0.01,
                'min_mlp_recall_gain_over_simpler': 0.02,
            },
            'gate': {
                'max_mlp_recall_sample_std': 0.05,
            },
        }

    def test_model_selection_keeps_logistic_without_sufficient_mlp_gain(self):
        _, _, selected, _, gate = TRAINING.choose_deployment_model(
            self.model_rows(mlp_recall=0.51),
            self.selection_settings(),
        )
        self.assertEqual(selected, 'logistic_regression')
        self.assertTrue(gate['passed'])

    def test_model_selection_accepts_mlp_only_after_registered_gain(self):
        _, _, selected, _, gate = TRAINING.choose_deployment_model(
            self.model_rows(mlp_recall=0.53),
            self.selection_settings(),
        )
        self.assertEqual(selected, 'pair_mlp')
        self.assertTrue(gate['passed'])

    def test_loader_rejects_plot_leakage(self):
        fieldnames = [
            'source_plot', 'split', 'source_instance_id',
            'target_instance_id', 'source_best_gt_id',
            'target_best_gt_id', 'source_max_iou', 'target_max_iou',
            'source_is_true_tree', 'target_is_true_tree',
            'pair_valid', 'pair_positive', 'base_vote_distance',
        ]
        rows = []
        for split, source_id, positive in (
                ('train', 1, True), ('validation', 2, False)):
            rows.append({
                'source_plot': 'P1',
                'split': split,
                'source_instance_id': source_id,
                'target_instance_id': 10,
                'source_best_gt_id': 1,
                'target_best_gt_id': 1 if positive else 2,
                'source_max_iou': 0.2,
                'target_max_iou': 0.8,
                'source_is_true_tree': False,
                'target_is_true_tree': True,
                'pair_valid': True,
                'pair_positive': positive,
                'base_vote_distance': 0.5,
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.csv'
            with path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'leak'):
                TRAINING.load_pair_dataset(
                    path, ['base_vote_distance'])


if __name__ == '__main__':
    unittest.main()