import unittest
import numpy as np

from tree_learn.experiments.ptv3_stage_v1.validation import inside, complete_tree_ids, masked_gt, tile_contributes


class StageRegionTests(unittest.TestCase):
    def test_cut_trees_become_unknown_and_never_non_tree(self):
        xyz = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 2], [25, 0, 2],
                        [2, 2, 0], [3, 3, 0]], dtype=float)
        labels = np.array([90000, 90000, 7, 7, 0, -1])
        coords, target, complete = masked_gt(xyz, labels)
        self.assertEqual(complete.tolist(), [90000])
        self.assertEqual(target.tolist(), [90000, 90000, -1, 0, -1])
        self.assertEqual(len(coords), 5)
        self.assertEqual(labels.tolist(), [90000, 90000, 7, 7, 0, -1])

    def test_empty_positive_and_core_boundaries(self):
        xyz = np.array([[-24, 24, 0], [24, -24, 0], [24.001, 0, 0]])
        self.assertEqual(inside(xyz).tolist(), [True, True, False])
        self.assertEqual(complete_tree_ids(xyz, np.zeros(3)).tolist(), [])

    def test_tile_selection_uses_global_translation_and_full_inner_overlap(self):
        self.assertTrue(tile_contributes([30, 0, 0], [-2, 0, 0]))
        self.assertFalse(tile_contributes([30.03, 0, 0], [-2, 0, 0]))
        self.assertTrue(tile_contributes([-30, -30, 0], [2, 2, 0]))


if __name__ == "__main__": unittest.main()
