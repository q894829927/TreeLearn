import importlib.util
import pathlib
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'tree_learn' / 'dataset' / 'dataset.py'
SPEC = importlib.util.spec_from_file_location(
    'tree_dataset_standalone', MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)
TreeDataset = DATASET_MODULE.TreeDataset


def make_dataset(upper_anchor_mode='stem_axis'):
    dataset = TreeDataset.__new__(TreeDataset)
    dataset.base_anchor_mode = 'legacy'
    dataset.base_anchor_height = 0.5
    dataset.base_anchor_floor_quantile = 0.01
    dataset.upper_anchor_mode = upper_anchor_mode
    dataset.upper_anchor_lower_ratio = 0.55
    dataset.upper_anchor_upper_ratio = 0.75
    dataset.upper_anchor_min_points = 3
    dataset.stem_axis_lower_ratio = 0.10
    dataset.stem_axis_upper_ratio = 0.50
    dataset.stem_axis_num_bins = 4
    dataset.stem_axis_verticality_threshold = 0.60
    dataset.stem_axis_min_points_per_bin = 3
    dataset.stem_axis_min_valid_bins = 3
    dataset.upper_anchor_target_ratio = 0.65
    return dataset


def leaning_tree():
    heights = np.linspace(0.0, 10.0, 101, dtype=np.float32)
    points = np.stack(
        [0.10 * heights, 0.05 * heights, heights], axis=1)
    return points


class StemAxisLabelTests(unittest.TestCase):

    def test_stem_axis_recovers_synthetic_lean(self):
        dataset = make_dataset()
        points = leaning_tree()
        upper_anchor, valid_bins = dataset._get_stem_axis_upper_anchor(
            points, np.ones(len(points)), min_z=0.0, tree_height=10.0)
        self.assertEqual(valid_bins, 4)
        np.testing.assert_allclose(
            upper_anchor,
            np.array([0.65, 0.325, 6.5], dtype=np.float32),
            atol=1e-5)

    def test_insufficient_vertical_bins_is_invalid(self):
        dataset = make_dataset()
        points = leaning_tree()
        verticality = np.zeros(len(points), dtype=np.float32)
        upper_anchor, valid_bins = dataset._get_stem_axis_upper_anchor(
            points, verticality, min_z=0.0, tree_height=10.0)
        self.assertIsNone(upper_anchor)
        self.assertEqual(valid_bins, 0)

    def test_get_offset_supports_one_dimensional_features(self):
        dataset = make_dataset()
        points = leaning_tree()
        instances = np.ones(len(points), dtype=np.int64)
        semantic = np.zeros(len(points), dtype=np.int64)
        features = np.ones(len(points), dtype=np.float32)
        base, upper, base_mask, upper_mask = dataset.getOffset(
            points, instances, semantic, input_feat=features)
        self.assertTrue(base_mask.all())
        self.assertTrue(upper_mask.all())
        axis_xy = upper[0, :2] - base[0, :2]
        self.assertTrue(np.isfinite(axis_xy).all())

    def test_crown_median_mode_preserves_existing_definition(self):
        dataset = make_dataset(upper_anchor_mode='crown_median')
        points = leaning_tree()
        instances = np.ones(len(points), dtype=np.int64)
        semantic = np.zeros(len(points), dtype=np.int64)
        base, upper, _, upper_mask = dataset.getOffset(
            points, instances, semantic)

        min_z = np.partition(points[:, 2], 10)[3]
        height = points[:, 2].max() - min_z
        relative_height = (points[:, 2] - min_z) / height
        crown_points = points[
            (relative_height >= 0.55) & (relative_height <= 0.75)]
        expected_upper = np.array([
            np.median(crown_points[:, 0]),
            np.median(crown_points[:, 1]),
            min_z + 0.65 * height,
        ], dtype=np.float32)

        self.assertTrue(upper_mask.all())
        np.testing.assert_allclose(
            points[0] + upper[0], expected_upper, atol=1e-6)
        self.assertTrue(np.isfinite(base).all())


if __name__ == '__main__':
    unittest.main()
