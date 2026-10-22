"""Posthoc diagnostic integrity: bins, GT masks, topology and circle evidence."""

import unittest
import numpy as np

from tools.local_split_v1.analyze_failure_mechanisms import (
    accumulate, band_indices, circle_pair, contact_bands, edge_classes, finish_groups, fit_circle)


class FailureMechanismTests(unittest.TestCase):
    def test_half_open_bins_and_outside_values(self):
        np.testing.assert_array_equal(band_indices([0, .1, .199, .2], [0, .1, .2, np.inf]), [0, 1, 1, 2])
        for values in [[-1], [np.nan], [1.]]:
            with self.assertRaises(ValueError):
                band_indices(values, [0, .1, 1.])

    def test_unknowns_and_background_are_not_other_tree_edges(self):
        labels = np.array([-1, 0, 1, 1, 2])
        edges = np.array([[0, 2], [1, 2], [2, 3], [3, 4], [1, 1]])
        masks = edge_classes(labels, edges)
        np.testing.assert_array_equal(masks['same'], [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(masks['other_tree'], [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(masks['tree_background'], [0, 1, 0, 0, 0])

    def test_contact_hops_distinguish_interior_same_tree_edges(self):
        edges = np.column_stack((np.arange(10), np.arange(1, 11)))
        labels = np.r_[np.ones(6, int), np.full(5, 2)]
        bands = contact_bands(11, edges, edge_classes(labels, edges)['other_tree'])
        self.assertEqual(bands[0], 3)
        self.assertEqual(bands[3], 1)
        self.assertEqual(bands[5], 0)

    def test_error_contributions_sum_to_population_counts(self):
        groups = {}
        scores = np.array([.1, .7, .2, .8])
        masks = dict(same=np.array([1, 1, 0, 0], bool), other_tree=np.array([0, 0, 1, 1], bool))
        accumulate(groups, 'distance', np.array([0, 1, 0, 1]), masks, scores, 'example')
        rows = finish_groups(groups)
        self.assertEqual(sum(r['count'] for r in rows), 4)
        self.assertEqual(sum(r['cut'] for r in rows), 2)
        self.assertTrue(all(np.isfinite(r['mean_bce']) for r in rows))

    def test_one_cylinder_halves_have_separated_centroids_but_shared_circle(self):
        angle = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        ring = np.column_stack([.3 * np.cos(angle), .3 * np.sin(angle)])
        first, second = ring[ring[:, 0] >= 0], ring[ring[:, 0] < 0]
        result = circle_pair(first, second)
        self.assertTrue(result['valid'])
        self.assertGreater(result['centroid_separation_m'], .3)
        self.assertLess(result['center_to_centroid_separation_ratio'], 1e-10)
        self.assertLess(result['shared']['relative_rmse'], 1e-10)

    def test_two_cylinders_have_distinct_fitted_centers_and_translation_invariance(self):
        angle = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        ring = np.column_stack([.06 * np.cos(angle), .06 * np.sin(angle)])
        result = circle_pair(ring, ring + [.6, 0])
        shifted = circle_pair(ring + [100, -50], ring + [100.6, -50])
        self.assertAlmostEqual(result['fitted_center_separation_m'], .6)
        self.assertAlmostEqual(result['center_to_centroid_separation_ratio'], 1.)
        self.assertAlmostEqual(result['fitted_center_separation_m'], shifted['fitted_center_separation_m'])

    def test_degenerate_line_is_unidentifiable_circle(self):
        result = fit_circle(np.column_stack([np.arange(20), np.zeros(20)]))
        self.assertFalse(result['valid'])


if __name__ == '__main__':
    unittest.main()
