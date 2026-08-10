import unittest

import numpy as np

from tree_learn.util.instance_split_diagnostics import (
    analyze_instance_split_oracle,
    apply_gt_extraction_ceiling,
    detection_metrics_from_labels,
    identify_undersegmentation_targets,
    split_features,
)


class InstanceSplitDiagnosticsTests(unittest.TestCase):

    @staticmethod
    def merged_case():
        tree_one = np.column_stack([
            np.linspace(0.0, 0.2, 6), np.zeros(6), np.arange(6)])
        tree_two = np.column_stack([
            np.linspace(4.0, 4.2, 4), np.zeros(4), np.arange(4)])
        coords = np.vstack([tree_one, tree_two]).astype(np.float32)
        labels = np.asarray([1] * 6 + [2] * 4, dtype=np.int64)
        predictions = np.full(10, 7, dtype=np.int64)
        offsets = np.zeros_like(coords)
        return coords, offsets, labels, predictions

    def test_strict_match_exposes_one_undersegmented_tree(self):
        _, _, labels, predictions = self.merged_case()
        metrics = detection_metrics_from_labels(labels, predictions)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']),
                         (1, 0, 1))
        targets = identify_undersegmentation_targets(labels, predictions)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]['prediction_id'], 7)
        self.assertEqual(targets[0]['missed_gt_ids'], [2])
        self.assertEqual(targets[0]['oracle_num_children'], 2)

    def test_gt_extraction_recovers_omission_without_losing_match(self):
        _, _, labels, predictions = self.merged_case()
        targets = identify_undersegmentation_targets(labels, predictions)
        split, metadata = apply_gt_extraction_ceiling(
            labels, predictions, targets)
        metrics = detection_metrics_from_labels(labels, split)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']),
                         (2, 0, 0))
        self.assertEqual(metadata['accepted_splits'], 1)

    def test_geometry_oracles_recover_separated_trees(self):
        coords, offsets, labels, predictions = self.merged_case()
        report = analyze_instance_split_oracle(
            coords, offsets, labels, predictions,
            max_fit_points=100, random_state=42)
        for mode in (
                'raw_xy_kmeans_oracle',
                'base_vote_kmeans_oracle',
                'vertical_axis_kmeans_oracle'):
            self.assertEqual(
                report['modes'][mode]['recovered_undersegmented_trees'], 1)
            self.assertEqual(report['modes'][mode]['lost_baseline_trees'], 0)
            self.assertAlmostEqual(report['modes'][mode]['f1'], 1.0)

    def test_vertical_features_are_finite_and_height_conditioned(self):
        coords, offsets, _, _ = self.merged_case()
        offsets[:, 0] = 1.0
        features = split_features(
            coords, offsets, 'vertical_axis_kmeans_oracle')
        self.assertEqual(features.shape, (10, 4))
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(np.allclose(features[:, :2], features[:, 2:]))

    def test_entirely_unmatched_parent_does_not_add_phantom_child(self):
        labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
        predictions = np.ones(4, dtype=np.int64)
        targets = identify_undersegmentation_targets(labels, predictions)
        self.assertEqual(len(targets), 1)
        self.assertEqual(sorted(targets[0]['missed_gt_ids']), [1, 2])
        self.assertEqual(targets[0]['oracle_num_children'], 2)


if __name__ == '__main__':
    unittest.main()
