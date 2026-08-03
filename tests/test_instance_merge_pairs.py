import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'data_gen' / 'gen_instance_merge_pairs.py')
SPEC = importlib.util.spec_from_file_location(
    'instance_merge_pairs_standalone', MODULE_PATH)
PAIRS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRS)


class InstanceMergePairTests(unittest.TestCase):

    @staticmethod
    def make_artifact():
        token_names = np.asarray([
            'occupancy_fraction',
            'semantic_probability_mean',
            'verticality_mean',
            'base_vote_radius_rms',
        ])
        tokens = np.ones((4, 2, 4), dtype=np.float32)
        return {
            'instance_ids': np.asarray([1, 2, 3, 4]),
            'global_feature_names': np.asarray(['num_points']),
            'global_feature_values': np.asarray(
                [[100], [120], [20], [50]], dtype=np.float32),
            'vertical_tokens': tokens,
            'layer_valid_mask': np.ones((4, 2), dtype=bool),
            'token_feature_names': token_names,
            'target_max_iou': np.asarray([0.8, 0.8, 0.2, 0.7]),
            'target_best_gt_id': np.asarray([10, 20, 10, 30]),
            'target_valid': np.ones(4, dtype=bool),
            'target_is_true_tree': np.asarray([True, True, False, True]),
            'instance_xy_centroid': np.asarray([
                [0.0, 0.0], [10.0, 0.0], [0.2, 0.0], [30.0, 0.0]],
                dtype=np.float32),
            'instance_base_vote_xy_centroid': np.asarray([
                [0.0, 0.0], [10.0, 0.0], [0.1, 0.0], [30.0, 0.0]],
                dtype=np.float32),
            'instance_z_min': np.zeros(4, dtype=np.float32),
            'instance_z_max': np.asarray([10, 12, 9, 8], dtype=np.float32),
            'source_plot': 'P1',
            'split': 'train',
        }

    def test_pair_generation_labels_same_gt_fragment_positive(self):
        artifact = self.make_artifact()
        scores = np.asarray([0.9, 0.8, 0.1, 0.2])
        rows, summary = PAIRS.build_pairs_for_artifact(
            artifact, scores, keep_ratio=0.5,
            num_neighbors=2, max_base_vote_distance=8.0)
        source_three = [
            row for row in rows if row['source_instance_id'] == 3]
        self.assertEqual(len(source_three), 1)
        self.assertEqual(source_three[0]['target_instance_id'], 1)
        self.assertTrue(source_three[0]['pair_positive'])
        self.assertEqual(summary['num_low_quality'], 2)
        self.assertEqual(summary['num_low_with_positive'], 1)

    def test_pair_features_are_finite(self):
        artifact = self.make_artifact()
        scores = np.asarray([0.9, 0.8, 0.1, 0.2])
        row = PAIRS.pair_features(
            artifact, scores, source=2, target=0, neighbor_rank=1)
        self.assertTrue(all(np.isfinite(value) for value in row.values()))
        self.assertAlmostEqual(row['base_vote_distance'], 0.1, places=5)
        self.assertAlmostEqual(row['z_overlap_min_ratio'], 1.0)

    def test_top_ratio_validation_and_deterministic_tie_break(self):
        scores = np.asarray([0.5, 0.5, 0.1])
        instance_ids = np.asarray([2, 1, 3])
        high, low = PAIRS.top_ratio_masks(
            scores, instance_ids, keep_ratio=1 / 3)
        np.testing.assert_array_equal(high, [False, True, False])
        np.testing.assert_array_equal(low, [True, False, True])
        with self.assertRaises(ValueError):
            PAIRS.top_ratio_masks(scores, instance_ids, keep_ratio=0.0)
        with self.assertRaises(ValueError):
            PAIRS.top_ratio_masks(
                np.asarray([0.5, np.nan, 0.1]),
                instance_ids,
                keep_ratio=0.5)
    def test_pilot_gate_requires_positive_and_negative_pairs(self):
        rows = [
            {
                'source_plot': 'P1', 'split': 'train',
                'pair_valid': True, 'pair_positive': True,
                'source_is_true_tree': False,
                'target_is_true_tree': True,
                'source_instance_id': 1, 'target_instance_id': 2,
                'value': 0.5,
            },
            {
                'source_plot': 'P1', 'split': 'train',
                'pair_valid': True, 'pair_positive': False,
                'source_is_true_tree': False,
                'target_is_true_tree': True,
                'source_instance_id': 3, 'target_instance_id': 2,
                'value': 1.0,
            },
        ]
        plot_summaries = [{
            'source_plot': 'P1',
            'split': 'train',
            'num_low_quality': 2,
            'num_low_with_neighbor': 2,
            'num_low_with_positive': 1,
        }]
        summary = PAIRS.summarize(
            rows, plot_summaries, {'gate': {}}, pilot=True)
        self.assertTrue(summary['gate']['passed'])


if __name__ == '__main__':
    unittest.main()