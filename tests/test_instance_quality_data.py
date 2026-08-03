import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tree_learn' / 'util' / 'instance_quality.py'
)
SPEC = importlib.util.spec_from_file_location(
    'instance_quality_standalone', MODULE_PATH)
INSTANCE_QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTANCE_QUALITY)


class VerticalInstanceTokenTests(unittest.TestCase):

    def setUp(self):
        self.coords = np.asarray([
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 1.0],
            [0.0, 0.2, 3.0],
            [5.0, 5.0, 0.0],
            [5.2, 5.0, 2.0],
            [5.0, 5.2, 4.0],
        ], dtype=np.float32)
        self.instances = np.asarray([1, 1, 1, 2, 2, 2])
        self.backbone = np.asarray([
            [1.0, 2.0], [2.0, 3.0], [4.0, 6.0],
            [10.0, 20.0], [12.0, 24.0], [14.0, 28.0],
        ], dtype=np.float32)
        self.logits = np.tile(
            np.asarray([[4.0, -2.0]], dtype=np.float32), (6, 1))
        self.offsets = np.zeros((6, 3), dtype=np.float32)
        self.verticality = np.linspace(0.2, 0.9, 6, dtype=np.float32)
        self.confidence = np.linspace(0.4, 0.9, 6, dtype=np.float32)

    def compute(self, backbone=None):
        return INSTANCE_QUALITY.compute_vertical_instance_tokens(
            self.coords,
            self.instances,
            self.backbone if backbone is None else backbone,
            self.logits,
            self.offsets,
            self.verticality,
            self.confidence,
            num_layers=4,
        )

    def test_shapes_masks_and_finite_values(self):
        result = self.compute()
        self.assertEqual(result['vertical_tokens'].shape, (2, 4, 12))
        self.assertEqual(result['layer_valid_mask'].shape, (2, 4))
        np.testing.assert_array_equal(result['instance_ids'], [1, 2])
        self.assertTrue(np.isfinite(result['vertical_tokens']).all())

        occupancy_index = result['token_feature_names'].index(
            'occupancy_fraction')
        occupancy = result['vertical_tokens'][:, :, occupancy_index]
        np.testing.assert_allclose(occupancy.sum(axis=1), np.ones(2))
        self.assertTrue(np.all(
            result['vertical_tokens'][~result['layer_valid_mask']] == 0))

    def test_instance_features_are_isolated(self):
        baseline = self.compute()['vertical_tokens']
        changed = self.backbone.copy()
        changed[self.instances == 1] += 1000.0
        modified = self.compute(changed)['vertical_tokens']
        self.assertFalse(np.array_equal(baseline[0], modified[0]))
        np.testing.assert_array_equal(baseline[1], modified[1])


class InstanceGeometryTests(unittest.TestCase):

    def test_geometry_centroids_votes_and_z_ranges_align_by_instance(self):
        coords = np.asarray([
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 3.0],
            [10.0, 4.0, 2.0],
            [12.0, 6.0, 8.0],
            [99.0, 99.0, 0.0],
        ])
        predictions = np.asarray([1, 1, 3, 3, 0])
        offsets = np.asarray([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        result = INSTANCE_QUALITY.compute_instance_geometry(
            coords, predictions, offsets)
        np.testing.assert_array_equal(result['instance_ids'], [1, 3])
        np.testing.assert_allclose(
            result['xy_centroid'], [[1.0, 0.0], [11.0, 5.0]])
        np.testing.assert_allclose(
            result['base_vote_xy_centroid'],
            [[1.0, 0.0], [11.0, 5.0]])
        np.testing.assert_allclose(result['z_min'], [1.0, 2.0])
        np.testing.assert_allclose(result['z_max'], [3.0, 8.0])

    def test_quality_artifact_persists_optional_geometry(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest('Pandas is required for artifact persistence.')
        instance_ids = np.asarray([1, 3], dtype=np.int64)
        global_features = pd.DataFrame({
            'instance_id': instance_ids,
            'num_points': [2.0, 2.0],
        })
        token_data = {
            'instance_ids': instance_ids,
            'vertical_tokens': np.ones((2, 2, 1), dtype=np.float32),
            'layer_valid_mask': np.ones((2, 2), dtype=bool),
            'token_feature_names': ['occupancy_fraction'],
            'num_layers': 2,
        }
        target_data = {
            'instance_ids': instance_ids,
            'target_max_iou': np.asarray([0.8, 0.2], dtype=np.float32),
            'target_best_gt_id': np.asarray([10, 10], dtype=np.int64),
            'target_labeled_fraction': np.ones(2, dtype=np.float32),
            'target_tree_point_fraction': np.ones(2, dtype=np.float32),
            'target_is_edge': np.zeros(2, dtype=bool),
            'target_valid': np.ones(2, dtype=bool),
            'target_classification_valid': np.ones(2, dtype=bool),
            'target_is_true_tree': np.asarray([True, False]),
        }
        geometry_data = {
            'instance_ids': instance_ids,
            'xy_centroid': np.asarray(
                [[1.0, 0.0], [11.0, 5.0]], dtype=np.float32),
            'base_vote_xy_centroid': np.asarray(
                [[1.0, 0.0], [11.0, 5.0]], dtype=np.float32),
            'z_min': np.asarray([1.0, 2.0], dtype=np.float32),
            'z_max': np.asarray([3.0, 8.0], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            npz_path, _, _ = INSTANCE_QUALITY.save_instance_quality_data(
                global_features,
                token_data,
                target_data,
                directory,
                source_plot='P1',
                split='train',
                geometry_data=geometry_data,
            )
            with np.load(npz_path, allow_pickle=False) as artifact:
                np.testing.assert_allclose(
                    artifact['instance_xy_centroid'],
                    geometry_data['xy_centroid'])
                np.testing.assert_allclose(
                    artifact['instance_base_vote_xy_centroid'],
                    geometry_data['base_vote_xy_centroid'])
                np.testing.assert_allclose(
                    artifact['instance_z_min'], geometry_data['z_min'])
                np.testing.assert_allclose(
                    artifact['instance_z_max'], geometry_data['z_max'])


class InstanceQualityTargetTests(unittest.TestCase):

    def test_positive_ambiguous_and_validity_targets(self):
        coords = np.column_stack([
            np.arange(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        ])
        predictions = np.asarray([
            10, 10, 10, 20, 20, 30, 30, 40, 40, 0])
        labels = np.asarray([1, 1, 1, 2, 2, 2, 0, 0, 0, 0])
        result = INSTANCE_QUALITY.compute_instance_quality_targets(
            coords,
            predictions,
            labels,
            min_labeled_fraction=0.5,
            match_iou_threshold=0.5,
            negative_iou_threshold=0.25,
            edge_margin_m=0.0,
        )

        np.testing.assert_array_equal(
            result['instance_ids'], [10, 20, 30, 40])
        np.testing.assert_allclose(
            result['target_max_iou'], [1.0, 2.0 / 3.0, 0.25, 0.0])
        np.testing.assert_array_equal(
            result['target_is_true_tree'], [True, True, False, False])
        np.testing.assert_array_equal(
            result['target_valid'], [True, True, True, True])
        np.testing.assert_array_equal(
            result['target_classification_valid'],
            [True, True, False, True])

    def test_invalid_iou_threshold_order_is_rejected(self):
        coords = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            INSTANCE_QUALITY.compute_instance_quality_targets(
                coords,
                np.asarray([1, 1]),
                np.asarray([1, 1]),
                match_iou_threshold=0.5,
                negative_iou_threshold=0.5,
                edge_margin_m=0.0,
            )


class InstanceQualityFilterTests(unittest.TestCase):

    def test_zero_threshold_preserves_labels_exactly(self):
        predictions = np.asarray([0, 10, 10, 20, 30, 30], dtype=np.int64)
        result = INSTANCE_QUALITY.apply_instance_quality_filter(
            predictions,
            np.asarray([10, 20, 30]),
            np.asarray([0.2, 0.4, 0.8]),
            threshold=0.0,
        )
        np.testing.assert_array_equal(result['predictions'], predictions)
        self.assertEqual(result['rejected_instance_ids'].size, 0)
        self.assertEqual(result['label_mapping'], {10: 10, 20: 20, 30: 30})

    def test_filter_removes_low_score_and_remaps_both_stages(self):
        final_predictions = np.asarray(
            [0, 10, 10, 20, 20, 30, 30], dtype=np.int64)
        result = INSTANCE_QUALITY.apply_instance_quality_filter(
            final_predictions,
            np.asarray([10, 20, 30]),
            np.asarray([0.9, 0.1, 0.8]),
            threshold=0.5,
        )
        np.testing.assert_array_equal(
            result['predictions'], [0, 1, 1, 0, 0, 2, 2])
        self.assertEqual(result['label_mapping'], {10: 1, 30: 2})
        np.testing.assert_array_equal(
            result['rejected_instance_ids'], [20])

        initial_predictions = np.asarray(
            [-1, 10, 20, 30, 0], dtype=np.int64)
        remapped = INSTANCE_QUALITY.remap_instance_predictions(
            initial_predictions,
            result['label_mapping'],
            result['rejected_instance_ids'],
        )
        np.testing.assert_array_equal(remapped, [-1, 1, 0, 2, 0])

    def test_top_ratio_filter_is_deterministic_and_uses_ceiling(self):
        predictions = np.asarray([10, 10, 20, 20, 30, 30])
        result = INSTANCE_QUALITY.apply_instance_quality_filter(
            predictions,
            np.asarray([30, 10, 20]),
            np.asarray([0.5, 0.5, 0.5]),
            threshold=0.0,
            mode='top_ratio',
            keep_ratio=0.34,
        )
        np.testing.assert_array_equal(
            result['kept_instance_ids'], [10, 20])
        np.testing.assert_array_equal(
            result['rejected_instance_ids'], [30])
        np.testing.assert_array_equal(
            result['predictions'], [1, 1, 2, 2, 0, 0])
        self.assertEqual(result['filter_mode'], 'top_ratio')
        self.assertEqual(result['keep_ratio'], 0.34)
    def test_filter_rejects_missing_instance_scores(self):
        with self.assertRaises(ValueError):
            INSTANCE_QUALITY.apply_instance_quality_filter(
                np.asarray([0, 1, 2]),
                np.asarray([1]),
                np.asarray([0.9]),
                threshold=0.5,
            )

    def test_score_only_metadata_records_identical_label_digest(self):
        score_result = {
            'instance_ids': np.asarray([1, 2]),
            'validity_probability': np.asarray([0.8, 0.4]),
            'predicted_iou': np.asarray([0.9, 0.5]),
            'quality_score': np.asarray([0.72, 0.2]),
            'checkpoint_seed': 42,
            'checkpoint_best_epoch': 15,
            'device': 'cpu',
            'parameter_count': 123,
            'scoring_seconds': 0.25,
            'pre_filter_label_sha256': 'same',
            'post_filter_label_sha256': 'same',
        }
        with tempfile.TemporaryDirectory() as directory:
            _, metadata_path = (
                INSTANCE_QUALITY.save_instance_quality_scores(
                    score_result,
                    directory,
                    threshold=0.5,
                    filter_enabled=False))
            with open(metadata_path, encoding='utf-8') as file:
                metadata = json.load(file)
        self.assertTrue(metadata['score_only_labels_identical'])
        self.assertEqual(metadata['num_kept'], 2)
        self.assertEqual(metadata['num_passes_threshold'], 1)

if __name__ == '__main__':
    unittest.main()
