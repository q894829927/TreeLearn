import unittest

import numpy as np

from tree_learn.util.seed_completion_proposals import (
    build_multiscale_seed_completion_proposals,
    select_oracle_seed_completion_proposals,
)


def tree_logits(count):
    return np.tile(
        np.asarray([[6.0, -6.0]], dtype=np.float32), (count, 1))


class SeedCompletionProposalTests(unittest.TestCase):
    def test_proposals_are_gt_free_and_top_up_exactly(self):
        coords = np.zeros((6, 3), dtype=np.float32)
        coords[:, 0] = np.asarray([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        offsets = np.zeros((6, 3), dtype=np.float32)
        verticality = np.asarray([0.9, 0.9, 0.9, 0.1, 0.2, 0.3])
        first = build_multiscale_seed_completion_proposals(
            coords, tree_logits(6), offsets, verticality,
            tau_min=5, scales=(1.0,), shift_fractions=(0.0,))
        second = build_multiscale_seed_completion_proposals(
            coords, tree_logits(6), offsets, verticality,
            tau_min=5, scales=(1.0,), shift_fractions=(0.0,))
        self.assertEqual(len(first['proposals']), 1)
        self.assertEqual(len(second['proposals']), 1)
        proposal = first['proposals'][0]
        self.assertEqual(proposal['base_seed_count'], 3)
        self.assertEqual(proposal['added_seed_count'], 2)
        self.assertEqual(len(proposal['selected_indices']), 2)
        np.testing.assert_array_equal(
            proposal['selected_indices'],
            second['proposals'][0]['selected_indices'])

    def test_oracle_activation_is_target_local_and_deterministic(self):
        baseline = np.asarray([True, True, False, False, False, True])
        proposals = [{
            'proposal_id': 1,
            'scale': 1.0,
            'shift_x': 0.0,
            'shift_y': 0.0,
            'cell_x': 0,
            'cell_y': 0,
            'semantic_count': 5,
            'base_seed_count': 2,
            'added_seed_count': 2,
            'tree_probability_mean': 1.0,
            'tree_probability_std': 0.0,
            'verticality_mean': 0.5,
            'verticality_std': 0.1,
            'offset_z_abs_mean': 0.0,
            'offset_z_abs_std': 0.0,
            'vote_radius_rms': 0.1,
            'margin_mean': 2.0,
            'margin_std': 0.1,
            'selected_indices': np.asarray([2, 3]),
        }]
        labels = np.asarray([7, 7, 7, 7, 8, 8])
        result = select_oracle_seed_completion_proposals(
            proposals, baseline, labels, [7], tau_min=4)
        self.assertEqual(result['activated_proposal_ids'], [1])
        self.assertEqual(int(np.count_nonzero(
            result['seed_mask'] & (labels == 7))), 4)
        self.assertEqual(int(np.count_nonzero(
            result['seed_mask'] & (labels == 8))), 1)
        self.assertTrue(result['per_target'][0]['proposal_covered'])

    def test_non_target_proposal_is_not_activated(self):
        baseline = np.asarray([False, False, False])
        template = {
            'proposal_id': 1, 'scale': 1.0,
            'shift_x': 0.0, 'shift_y': 0.0,
            'cell_x': 0, 'cell_y': 0,
            'semantic_count': 3, 'base_seed_count': 0,
            'added_seed_count': 2,
            'tree_probability_mean': 1.0,
            'tree_probability_std': 0.0,
            'verticality_mean': 0.0, 'verticality_std': 0.0,
            'offset_z_abs_mean': 0.0, 'offset_z_abs_std': 0.0,
            'vote_radius_rms': 0.0, 'margin_mean': 1.0,
            'margin_std': 0.0,
            'selected_indices': np.asarray([0, 1]),
        }
        result = select_oracle_seed_completion_proposals(
            [template], baseline, np.asarray([8, 8, 8]), [7], tau_min=2)
        self.assertEqual(result['activated_proposal_ids'], [])
        self.assertFalse(result['seed_mask'].any())

    def test_target_need_not_be_proposal_majority(self):
        baseline = np.asarray([True, False, False, False])
        proposal = {
            'proposal_id': 1, 'scale': 1.0,
            'shift_x': 0.0, 'shift_y': 0.0,
            'cell_x': 0, 'cell_y': 0,
            'semantic_count': 4, 'base_seed_count': 1,
            'added_seed_count': 3,
            'tree_probability_mean': 1.0,
            'tree_probability_std': 0.0,
            'verticality_mean': 0.0, 'verticality_std': 0.0,
            'offset_z_abs_mean': 0.0, 'offset_z_abs_std': 0.0,
            'vote_radius_rms': 0.0, 'margin_mean': 1.0,
            'margin_std': 0.0,
            'selected_indices': np.asarray([1, 2, 3]),
        }
        labels = np.asarray([7, 7, 8, 8])
        result = select_oracle_seed_completion_proposals(
            [proposal], baseline, labels, [7], tau_min=2)
        self.assertEqual(result['activated_proposal_ids'], [1])
        self.assertTrue(result['per_target'][0]['proposal_covered'])
        self.assertEqual(
            result['proposal_rows'][0]['oracle_dominant_tree_id'], 8)
        self.assertEqual(
            result['proposal_rows'][0]['oracle_target_tree_ids'], '7')

if __name__ == '__main__':
    unittest.main()
