"""Audit matching, supervision isolation and exact deterministic sampler replay."""

import unittest
import numpy as np

from tools.local_split_v1.audit_negative_supervision import (
    canonical_conflicts, match_negative_rows, replay_sampler, row_keys)


class NegativeAuditTests(unittest.TestCase):
    def test_raw_match_finds_saved_permutation_without_tolerance(self):
        raw = np.arange(4 * 72, dtype=np.float32).reshape(4, 72)
        permutation = [3, 0, 2, 1]
        semantics = np.column_stack((np.ones(4), np.arange(4), np.arange(4) + 1))
        mapping, duplicates = match_negative_rows(raw, raw[permutation], semantics)
        np.testing.assert_array_equal(mapping, permutation)
        self.assertEqual(duplicates, 0)
        changed = raw[permutation].copy(); changed[0, 0] += 1e-4
        with self.assertRaises(ValueError):
            match_negative_rows(raw, changed, semantics)

    def test_missing_or_duplicate_bank_edge_refuses_false_coverage(self):
        raw = np.arange(3 * 72, dtype=np.float32).reshape(3, 72)
        semantics = np.zeros((3, 3), int)
        for bank in (raw[:2], raw[[0, 0, 2]]):
            with self.assertRaises(ValueError):
                match_negative_rows(raw, bank, semantics)

    def test_identical_raw_inputs_with_different_gt_pair_refuse_assignment(self):
        raw = np.zeros((2, 72), np.float32)
        with self.assertRaises(ValueError):
            match_negative_rows(raw, raw, np.array([[1, 2, 3], [2, 0, 3]]))
        mapping, duplicates = match_negative_rows(raw, raw, np.array([[1, 2, 3], [1, 2, 3]]))
        self.assertEqual(duplicates, 1)
        np.testing.assert_array_equal(mapping, [0, 1])

    def test_symmetric_inputs_detect_opposite_target_collision(self):
        raw = np.arange(72, dtype=np.float32)[None, :]
        swapped = np.column_stack((raw[:, 32:64], raw[:, :32], raw[:, 64:]))
        np.testing.assert_array_equal(row_keys(raw, True), row_keys(swapped, True))
        result = canonical_conflicts(np.r_[raw, swapped], np.array([1, 0]))
        self.assertEqual(result, dict(conflicting_input_groups=1, positive_rows=1, negative_rows=1))

    def test_match_does_not_cast_imprecise_inputs_to_float32(self):
        with self.assertRaises(ValueError):
            row_keys(np.zeros((2, 72), dtype=np.float64))

    def test_sampler_matches_original_rng_call_order_and_counts(self):
        records = [dict(category='mixed', plot='A', kind=np.array([0, 1, 2])),
                   dict(category='correct', plot='B', kind=np.array([0, 0, 2]))]
        replay, visits = replay_sampler(records, steps=5, edges_per_step=16)
        rng = np.random.default_rng(42)
        expected = [np.zeros(3, int), np.zeros(3, int)]
        for _ in range(5):
            for index in (0, 0, 1, 1):
                rng.integers(1); rng.integers(1)
                at = rng.integers(3, size=4)
                expected[index] += np.bincount(at, minlength=3)
        self.assertEqual(replay['rng_state'], rng.bit_generator.state)
        for actual, wanted in zip(visits, expected):
            np.testing.assert_array_equal(actual, wanted)
        self.assertEqual(sum(replay['sampled_edges'].values()), 80)
        self.assertAlmostEqual(sum(replay['mean_within_negative_batch_share'].values()), 1.)


if __name__ == '__main__':
    unittest.main()
