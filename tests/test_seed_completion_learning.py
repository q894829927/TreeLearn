import tempfile
import unittest

from tree_learn.util.seed_completion_learning import (
    PROPOSAL_FEATURE_NAMES,
    proposal_center,
    save_learning_artifact,
    validate_learning_artifact,
    validate_proposal_rows,
)


def proposal_row(proposal_id, activated=False, covered=False, target_ids=''):
    row = {
        'proposal_id': proposal_id,
        'cell_x': 2,
        'cell_y': 3,
        'oracle_activated': activated,
        'oracle_target_proposal': covered,
        'oracle_target_added_points': 4 if covered else 0,
        'oracle_selected_purity': 0.8 if covered else 0.0,
        'oracle_target_tree_ids': target_ids,
    }
    for index, name in enumerate(PROPOSAL_FEATURE_NAMES):
        row[name] = 1.0 + index
    row['scale'] = 0.5
    row['shift_x'] = 0.5
    row['shift_y'] = 0.0
    return row


class SeedCompletionLearningTests(unittest.TestCase):
    def test_targets_and_centers_are_aligned(self):
        rows = [
            proposal_row(1, activated=True, covered=True, target_ids='7;8'),
            proposal_row(2),
        ]
        arrays = validate_proposal_rows(rows)
        self.assertEqual(arrays['features'].shape, (2, 15))
        self.assertEqual(arrays['activation_target'].tolist(), [True, False])
        self.assertEqual(arrays['coverage_target'].tolist(), [True, False])
        self.assertEqual(arrays['target_tree_ids'][0], (7, 8))
        self.assertEqual(proposal_center(rows[0]).tolist(), [1.5, 1.75])

    def test_activated_proposal_must_cover_target(self):
        with self.assertRaisesRegex(ValueError, 'activated proposal'):
            validate_proposal_rows([
                proposal_row(1, activated=True, covered=False)])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not unique'):
            validate_proposal_rows([proposal_row(1), proposal_row(1)])

    def test_saved_artifact_round_trip(self):
        rows = [
            proposal_row(1, activated=True, covered=True, target_ids='7'),
            proposal_row(2),
        ]
        metadata = {
            'artifact_version': 1,
            'source_plot': 'V1',
            'split': 'validation',
            'feature_names': list(PROPOSAL_FEATURE_NAMES),
            'num_proposals': 2,
            'num_activation_positive': 1,
            'num_coverage_positive': 1,
            'num_target_trees': 1,
            'num_proposal_covered_target_trees': 1,
            'target_tree_ids': [7],
        }
        with tempfile.TemporaryDirectory() as directory:
            csv_path, metadata_path = save_learning_artifact(
                {'rows': rows, 'metadata': metadata}, directory)
            observed, saved_rows, arrays = validate_learning_artifact(
                csv_path, metadata_path)
            self.assertEqual(observed['source_plot'], 'V1')
            self.assertEqual(len(saved_rows), 2)
            self.assertEqual(int(arrays['activation_target'].sum()), 1)


if __name__ == '__main__':
    unittest.main()
