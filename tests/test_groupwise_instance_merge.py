import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'training' / 'train_groupwise_instance_merge.py')
SPEC = importlib.util.spec_from_file_location(
    'groupwise_instance_merge_standalone', MODULE_PATH)
GROUPWISE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GROUPWISE)


class GroupwiseInstanceMergeTests(unittest.TestCase):

    def test_source_groups_are_sorted_and_keep_candidate_labels(self):
        groups = GROUPWISE.build_source_groups(
            features=np.asarray([
                [20.0], [5.0], [10.0], [7.0]], dtype=np.float32),
            labels=np.asarray([False, True, True, False]),
            source_keys=np.asarray(['P:2', 'P:1', 'P:1', 'P:2']),
            target_ids=np.asarray([20, 5, 10, 7]),
        )
        self.assertEqual(
            [group['source_key'] for group in groups], ['P:1', 'P:2'])
        np.testing.assert_array_equal(groups[0]['target_ids'], [5, 10])
        np.testing.assert_array_equal(
            groups[0]['positive_mask'], [True, True])
        np.testing.assert_array_equal(groups[1]['target_ids'], [7, 20])

    def test_model_shapes_and_padding_do_not_change_valid_outputs(self):
        import torch

        torch.manual_seed(1)
        model = GROUPWISE.build_groupwise_merge_model(
            3, {'hidden_dims': [4], 'dropout': 0.0})
        model.eval()
        values = torch.randn(2, 3, 3)
        mask = torch.tensor([
            [True, True, False],
            [True, True, True],
        ])
        first_candidates, first_null = model(values, mask)
        modified = values.clone()
        modified[0, 2] = 1000.0
        second_candidates, second_null = model(modified, mask)
        self.assertEqual(tuple(first_candidates.shape), (2, 3))
        self.assertEqual(tuple(first_null.shape), (2,))
        torch.testing.assert_close(
            first_candidates[0, :2], second_candidates[0, :2])
        torch.testing.assert_close(first_null[0], second_null[0])
    def test_listwise_loss_handles_positive_set_and_no_merge_target(self):
        import torch

        candidate_logits = torch.zeros(
            (2, 2), dtype=torch.float32, requires_grad=True)
        no_merge_logits = torch.zeros(
            2, dtype=torch.float32, requires_grad=True)
        candidate_mask = torch.ones((2, 2), dtype=torch.bool)
        positive_mask = torch.tensor([
            [True, False],
            [False, False],
        ])
        loss = GROUPWISE.groupwise_list_loss(
            candidate_logits,
            no_merge_logits,
            candidate_mask,
            positive_mask,
        )
        self.assertAlmostEqual(loss.item(), np.log(3.0), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(candidate_logits.grad).all())
        self.assertTrue(torch.isfinite(no_merge_logits.grad).all())

    @staticmethod
    def records():
        return [
            {
                'source_key': 'P:1',
                'target_instance_id': 10,
                'confidence': 0.9,
                'no_merge_probability': 0.05,
                'chosen_correct': True,
                'has_positive_target': True,
            },
            {
                'source_key': 'P:2',
                'target_instance_id': 20,
                'confidence': 0.8,
                'no_merge_probability': 0.1,
                'chosen_correct': False,
                'has_positive_target': True,
            },
            {
                'source_key': 'P:3',
                'target_instance_id': 30,
                'confidence': 0.7,
                'no_merge_probability': 0.2,
                'chosen_correct': False,
                'has_positive_target': False,
            },
        ]

    def test_binary_average_precision_is_deterministic(self):
        value = GROUPWISE.binary_average_precision(
            labels=[True, False, True],
            scores=[0.9, 0.8, 0.7],
        )
        self.assertAlmostEqual(value, (1.0 + 2.0 / 3.0) / 2.0)
    def test_source_recall_denominator_includes_misranked_positive_source(self):
        metrics = GROUPWISE.group_source_metrics(
            self.records(), threshold=0.85)
        self.assertEqual(metrics['num_positive_sources'], 2)
        self.assertEqual(metrics['num_correct_merges'], 1)
        self.assertEqual(metrics['num_unsafe_merges'], 0)
        self.assertAlmostEqual(metrics['positive_source_recall'], 0.5)

    def test_tiny_training_run_produces_finite_checkpoint_metrics(self):
        features = np.asarray([
            [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.5], [1.0, 1.0],
            [0.2, 0.8], [0.8, 0.2],
        ], dtype=np.float32)
        groups = GROUPWISE.build_source_groups(
            features=features,
            labels=np.asarray([True, False, False, False, True, False]),
            source_keys=np.asarray([
                'P:1', 'P:1', 'P:2', 'P:2', 'P:3', 'P:3']),
            target_ids=np.asarray([10, 11, 20, 21, 30, 31]),
        )
        result = GROUPWISE.train_groupwise_model(
            groups,
            groups,
            input_dim=2,
            model_config={'hidden_dims': [4], 'dropout': 0.0},
            training_config={
                'batch_size': 2,
                'epochs': 2,
                'patience': 2,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
            },
            constraints={
                'min_pair_precision': 0.0,
                'min_positive_source_recall': 0.0,
                'max_unsafe_merge_rate': 1.0,
            },
            seed=42,
        )
        self.assertTrue(result['state_dict'])
        self.assertTrue(np.isfinite(
            result['metrics']['source_average_precision']))
        self.assertGreaterEqual(result['metrics']['best_epoch'], 1)
    def test_threshold_selection_preserves_safety_before_recall(self):
        selected = GROUPWISE.select_group_threshold(
            self.records(),
            {
                'min_pair_precision': 0.95,
                'min_positive_source_recall': 0.30,
                'max_unsafe_merge_rate': 0.0,
            },
        )
        self.assertAlmostEqual(selected['threshold'], 0.9)
        self.assertTrue(selected['passed'])
        self.assertAlmostEqual(selected['positive_source_recall'], 0.5)


if __name__ == '__main__':
    unittest.main()