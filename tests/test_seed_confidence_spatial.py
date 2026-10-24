"""CPU-only regression tests for opt-in spatial seed confidence ranking."""
import importlib.util
import pathlib
import unittest

import numpy as np


AXIS_PATH = pathlib.Path(__file__).resolve().parents[1] / 'tree_learn' / 'util' / 'axis.py'
SPEC = importlib.util.spec_from_file_location('standalone_seed_confidence_axis', AXIS_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
filter_seeds = MODULE.filter_base_seeds_by_confidence


class SpatialConfidenceSeedTests(unittest.TestCase):
    def test_exact_same_budget_and_sparse_cell_coverage(self):
        active = np.ones(12, dtype=bool)
        confidence = np.r_[np.linspace(.99, .80, 10), [.15, .10]]
        vote_xy = np.vstack([np.tile([[.2, .2]], (10, 1)),
                             np.tile([[8.2, .2]], (2, 1))])
        global_mask = filter_seeds(
            active, confidence, True, mode='top_ratio', keep_ratio=.5)
        spatial_mask = filter_seeds(
            active, confidence, True, mode='spatial_top_ratio',
            keep_ratio=.5, seed_vote_xy=vote_xy, spatial_cell_size=2.0)
        self.assertEqual(int(global_mask.sum()), 6)
        self.assertEqual(int(spatial_mask.sum()), 6)
        self.assertEqual(int(global_mask[10:].sum()), 0)
        self.assertEqual(int(spatial_mask[10:].sum()), 1)

    def test_single_cell_equivalent_to_global_top_ratio(self):
        active = np.array([True, False, True, True, True, False, True])
        confidence = np.array([.4, .99, .9, .9, .1, .92, .3])
        vote_xy = np.tile([[0.2, -0.2]], (len(active), 1))
        global_mask = filter_seeds(
            active, confidence, True, mode='top_ratio', keep_ratio=.6)
        spatial_mask = filter_seeds(
            active, confidence, True, mode='spatial_top_ratio',
            keep_ratio=.6, seed_vote_xy=vote_xy, spatial_cell_size=2.0)
        np.testing.assert_array_equal(global_mask, spatial_mask)

    def test_deterministic_ties_and_no_nonseed_selection(self):
        active = np.array([True, True, False, True, True, True])
        confidence = np.array([.5, .5, .99, .5, .5, .5])
        vote_xy = np.array([[.2, .2], [.1, .1], [100, 100],
                            [-.2, .2], [-.1, .1], [-.1, -.1]])
        opts = dict(mode='spatial_top_ratio', keep_ratio=.6,
                    seed_vote_xy=vote_xy, spatial_cell_size=1.0)
        a = filter_seeds(active, confidence, True, **opts)
        b = filter_seeds(active, confidence, True, **opts)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(int(a.sum()), 3)
        self.assertFalse(a[2])
        self.assertTrue(np.all(a <= active))

    def test_empty_and_disabled_unchanged(self):
        active = np.array([False, False, False])
        out = filter_seeds(active, enabled=True,
                           axis_confidence=np.array([.1,.2,.3]),
                           mode='spatial_top_ratio', keep_ratio=.78)
        self.assertFalse(out.any())
        active = np.array([True, False, True])
        out = filter_seeds(active, enabled=False,
                           mode='spatial_top_ratio')
        np.testing.assert_array_equal(out, active)

    def test_invalid_inputs_fail_closed(self):
        active = np.array([True, True, False])
        confidence = np.array([.1, .2, .3])
        for kwargs in (
            dict(seed_vote_xy=None),
            dict(seed_vote_xy=np.zeros((2, 2))),
            dict(seed_vote_xy=np.zeros((3, 3))),
            dict(seed_vote_xy=np.array([[np.nan,0],[0,0],[0,0]])),
            dict(seed_vote_xy=np.zeros((3, 2)), spatial_cell_size=0),
            dict(seed_vote_xy=np.zeros((3, 2)), spatial_cell_size=np.inf),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                filter_seeds(active, confidence, True,
                             mode='spatial_top_ratio', keep_ratio=.78,
                             **kwargs)
        with self.assertRaises(ValueError):
            filter_seeds(active, [np.nan, .2, .3], True,
                         mode='spatial_top_ratio', keep_ratio=.78,
                         seed_vote_xy=np.zeros((3, 2)))

    def test_large_input_ratio_and_input_immutability(self):
        rng = np.random.default_rng(123)
        n = 10001
        active = rng.random(n) > .23
        confidence = rng.random(n)
        votes = rng.normal(0, 22, (n, 2))
        before = (active.copy(), confidence.copy(), votes.copy())
        kept = filter_seeds(active, confidence, True,
                            mode='spatial_top_ratio', keep_ratio=.78,
                            seed_vote_xy=votes, spatial_cell_size=2.0)
        self.assertEqual(int(kept.sum()),
                         int(np.ceil(int(active.sum()) * .78)))
        np.testing.assert_array_equal(active, before[0])
        np.testing.assert_array_equal(confidence, before[1])
        np.testing.assert_array_equal(votes, before[2])


if __name__ == '__main__':
    unittest.main()
