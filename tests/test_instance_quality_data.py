import importlib.util
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


if __name__ == '__main__':
    unittest.main()
