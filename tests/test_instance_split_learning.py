import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tree_learn.util.instance_split_learning import (
    build_instance_split_learning_artifact,
    evaluate_split_decisions,
    proposal_geometry_features,
    save_instance_split_learning_artifact,
    validate_instance_split_learning_artifact,
)


class InstanceSplitLearningTests(unittest.TestCase):
    def test_proposal_features_are_finite(self):
        coords = np.asarray([
            [0.0, 0.0, 0.0], [0.1, 0.0, 1.0],
            [2.0, 0.0, 0.0], [2.1, 0.0, 1.0],
        ])
        values = proposal_geometry_features(
            coords, np.zeros_like(coords), np.asarray([0, 0, 1, 1]))
        self.assertEqual(values.shape, (16,))
        self.assertTrue(np.isfinite(values).all())

    def test_exact_contingency_evaluation_recovers_two_trees(self):
        data = {
            'instance_ids': np.asarray([7]),
            'base_gt_ids': np.asarray([1, 2]),
            'base_gt_counts': np.asarray([2, 2]),
            'base_pred_counts': np.asarray([4]),
            'base_intersections': np.asarray([[2, 2]]),
            'proposal_instance_index': np.asarray([0]),
            'proposal_k': np.asarray([2]),
            'proposal_child_counts': np.asarray([[2, 2]]),
            'proposal_child_intersection_offsets': np.asarray([0, 1, 2]),
            'proposal_child_gt_indices': np.asarray([0, 1]),
            'proposal_child_intersection_values': np.asarray([2, 2]),
        }
        baseline = evaluate_split_decisions(data, [])
        split = evaluate_split_decisions(data, [0])
        self.assertEqual((baseline['tp'], baseline['fp'], baseline['fn']),
                         (0, 1, 2))
        self.assertEqual((split['tp'], split['fp'], split['fn']), (2, 0, 0))
        self.assertGreater(split['f1'], baseline['f1'])

    def test_duplicate_parent_proposals_are_rejected(self):
        data = {
            'instance_ids': np.asarray([7]),
            'base_gt_ids': np.asarray([1]),
            'base_gt_counts': np.asarray([2]),
            'base_pred_counts': np.asarray([2]),
            'base_intersections': np.asarray([[2]]),
            'proposal_instance_index': np.asarray([0, 0]),
            'proposal_k': np.asarray([2, 3]),
            'proposal_child_counts': np.asarray([[1, 1], [1, 1]]),
            'proposal_child_intersection_offsets': np.asarray(
                [0, 1, 2, 2, 3]),
            'proposal_child_gt_indices': np.asarray([0, 0, 0]),
            'proposal_child_intersection_values': np.asarray([1, 1, 1]),
        }
        with self.assertRaisesRegex(ValueError, 'one split proposal'):
            evaluate_split_decisions(data, [0, 1])

    def test_build_artifact_uses_q3_consistent_parent_target(self):
        coords = np.asarray([
            [0.0, 0.0, 0.0], [0.1, 0.0, 1.0], [0.0, 0.1, 2.0],
            [3.0, 0.0, 0.0], [3.1, 0.0, 1.0], [3.0, 0.1, 2.0],
        ], dtype=np.float32)
        labels = np.asarray([1, 1, 1, 2, 2, 2])
        predictions = np.ones(6, dtype=np.int64)
        logits = np.tile(np.asarray([[6.0, -6.0]], dtype=np.float32), (6, 1))
        offsets = np.zeros((6, 3), dtype=np.float32)
        artifact = build_instance_split_learning_artifact(
            coords=coords,
            semantic_logits=logits,
            offset_predictions=offsets,
            instance_labels=labels,
            instance_predictions=predictions,
            initial_instance_predictions=predictions,
            backbone_features=np.ones((6, 4), dtype=np.float32),
            verticality=np.ones(6, dtype=np.float32),
            axis_confidence=np.zeros(6, dtype=np.float32),
            split='validation',
            min_parent_points=2,
            max_children=2,
            tau_min=1,
            max_fit_points=100,
            expected_undersegmented_gt_ids=[1, 2])
        self.assertEqual(artifact['metadata']['num_candidate_parents'], 1)
        self.assertEqual(artifact['metadata']['num_undersegmented_gt_trees'], 2)
        self.assertEqual(artifact['metadata']['num_known_k_safe_parents'], 1)
        self.assertTrue(artifact['arrays']['proposal_safety_target'][0])
        with tempfile.TemporaryDirectory() as directory:
            npz_path, metadata_path = save_instance_split_learning_artifact(
                artifact, directory, 'V1', 'validation')
            metadata = validate_instance_split_learning_artifact(
                npz_path, metadata_path)
            self.assertEqual(metadata['source_plot'], 'V1')
            self.assertEqual(metadata['artifact_version'], 1)

    def test_expected_q3_ids_are_strict(self):
        coords = np.asarray([
            [0.0, 0.0, 0.0], [0.1, 0.0, 1.0],
            [3.0, 0.0, 0.0], [3.1, 0.0, 1.0],
        ], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'differ from fixed Q3'):
            build_instance_split_learning_artifact(
                coords=coords,
                semantic_logits=np.tile(
                    np.asarray([[6.0, -6.0]], dtype=np.float32), (4, 1)),
                offset_predictions=np.zeros((4, 3), dtype=np.float32),
                instance_labels=np.asarray([1, 1, 2, 2]),
                instance_predictions=np.ones(4, dtype=np.int64),
                initial_instance_predictions=np.ones(4, dtype=np.int64),
                backbone_features=np.ones((4, 2), dtype=np.float32),
                verticality=np.ones(4, dtype=np.float32),
                split='validation', min_parent_points=2, max_children=2,
                expected_undersegmented_gt_ids=[1])


if __name__ == '__main__':
    unittest.main()