import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'train_seed_completion_activation_test',
    ROOT / 'tools' / 'training' /
    'train_seed_completion_activation.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_settings():
    return {
        'splits': copy.deepcopy(MODULE.EXPECTED_SPLITS),
        'seeds': [42, 43, 44],
        'locked_seed': 42,
        'selection': {
            'model_priority': [
                'multitask_mlp', 'logistic_regression', 'fixed_rule'],
            'keep_ratios': [0.001, 0.01],
            'min_activation_precision': 0.02,
            'min_activation_recall': 0.25,
            'min_target_tree_recall': 0.5,
        },
        'gate': {
            'min_activation_precision': 0.02,
            'min_activation_recall': 0.25,
            'min_target_tree_recall': 0.5,
            'min_mlp_seed_passes': 2,
        },
    }


class SeedCompletionActivationTests(unittest.TestCase):
    def test_binary_metrics_handle_perfect_and_tied_scores(self):
        targets = np.asarray([False, False, True, True])
        perfect = MODULE.binary_metrics(
            targets, np.asarray([0.0, 0.1, 0.8, 0.9]))
        tied = MODULE.binary_metrics(targets, np.ones(4))
        self.assertAlmostEqual(perfect['roc_auc'], 1.0)
        self.assertAlmostEqual(perfect['average_precision'], 1.0)
        self.assertAlmostEqual(tied['roc_auc'], 0.5)

    def test_selection_is_deterministic_and_suppresses_neighbors(self):
        record = {
            'proposal_ids': np.asarray([30, 20, 10]),
            'centers': np.asarray([
                [0.0, 0.0], [0.1, 0.0], [2.0, 0.0]],
                dtype=np.float32),
        }
        scores = np.asarray([0.9, 0.9, 0.8])
        first = MODULE.deterministic_select(
            record, scores, keep_ratio=1.0,
            nms_radius=0.3, max_selected=3)
        second = MODULE.deterministic_select(
            record, scores, keep_ratio=1.0,
            nms_radius=0.3, max_selected=3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.tolist(), [False, True, True])

    def test_target_tree_recall_uses_selected_proposals(self):
        record = {
            'plot': 'V1',
            'split': 'validation',
            'proposal_ids': np.asarray([1, 2]),
            'centers': np.asarray([[0.0, 0.0], [2.0, 0.0]]),
            'activation_target': np.asarray([True, False]),
            'coverage_target': np.asarray([True, False]),
            'target_tree_ids': [(7,), ()],
            'all_target_tree_ids': (7,),
        }
        result = MODULE.selection_metrics(
            [record], {'V1': np.asarray([1.0, 0.0])}, {
                'keep_ratio': 0.5,
                'nms_radius': 0.1,
                'max_selected_per_plot': 1,
            })
        self.assertEqual(result['selected'], 1)
        self.assertAlmostEqual(result['activation_precision'], 1.0)
        self.assertAlmostEqual(result['target_tree_recall'], 1.0)

    def test_preregistered_settings_reject_wytham_and_drift(self):
        settings = valid_settings()
        MODULE.validate_settings(settings)
        invalid = copy.deepcopy(settings)
        invalid['splits']['validation'][-1] = 'Wytham'
        with self.assertRaisesRegex(ValueError, 'Wytham'):
            MODULE.validate_settings(invalid)
        invalid = copy.deepcopy(settings)
        invalid['locked_seed'] = 43
        with self.assertRaisesRegex(ValueError, 'locked seed'):
            MODULE.validate_settings(invalid)


if __name__ == '__main__':
    unittest.main()
