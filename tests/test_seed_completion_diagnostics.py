import unittest

import numpy as np

from tree_learn.util.seed_completion_diagnostics import (
    _stable_top_k,
    build_seed_completion_masks,
    evaluate_instance_predictions,
)


class SeedCompletionDiagnosticsTests(unittest.TestCase):
    def test_stable_top_k_uses_point_index_for_ties(self):
        selected = _stable_top_k(
            np.asarray([9, 2, 5]), np.asarray([1.0, 1.0, 1.0]), 2)
        np.testing.assert_array_equal(selected, [2, 5])

    def test_topup_is_exact_and_target_local(self):
        logits = np.tile(
            np.asarray([[6.0, -6.0]], dtype=np.float32), (7, 1))
        labels = np.asarray([1, 1, 1, 1, 2, 2, 2])
        verticality = np.asarray([0.9, 0.1, 0.2, 0.3, 0.9, 0.9, 0.9])
        offsets = np.zeros((7, 3), dtype=np.float32)
        targets = np.zeros((7, 3), dtype=np.float32)
        offsets[1, 0] = 4.0
        offsets[2, 0] = 0.1
        offsets[3, 0] = 0.2
        result = build_seed_completion_masks(
            logits, verticality, offsets, targets, labels, [1],
            tau_vert=0.6, tau_off=4.0, tau_min=3)
        baseline = result['baseline_mask']
        self.assertEqual(int(np.count_nonzero(baseline & (labels == 1))), 1)
        for mode, mask in result['masks'].items():
            self.assertEqual(int(np.count_nonzero(mask & (labels == 1))), 3)
            np.testing.assert_array_equal(
                mask[labels == 2], baseline[labels == 2],
                err_msg=mode)
        self.assertTrue(result['masks']['xy_oracle_topup'][2])
        self.assertTrue(result['masks']['xy_oracle_topup'][3])

    def test_topup_rejects_insufficient_semantic_support(self):
        logits = np.asarray([
            [6.0, -6.0], [-6.0, 6.0], [-6.0, 6.0]],
            dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'cannot be topped up'):
            build_seed_completion_masks(
                logits, np.zeros(3), np.zeros((3, 3)),
                np.zeros((3, 3)), np.ones(3), [1], tau_min=2)

    def test_detection_evaluation_recovers_separated_trees(self):
        labels = np.asarray([1, 1, 2, 2])
        merged = np.asarray([1, 1, 1, 1])
        separated = np.asarray([1, 1, 2, 2])
        baseline = evaluate_instance_predictions(labels, merged)
        improved = evaluate_instance_predictions(labels, separated)
        self.assertEqual((baseline['tp'], baseline['fp'], baseline['fn']),
                         (0, 1, 2))
        self.assertEqual((improved['tp'], improved['fp'], improved['fn']),
                         (2, 0, 0))


if __name__ == '__main__':
    unittest.main()
