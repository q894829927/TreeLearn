import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPEC = importlib.util.spec_from_file_location(
    'train_instance_split_heads_test',
    ROOT / 'tools' / 'training' / 'train_instance_split_heads.py')
TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN)
GEN_SPEC = importlib.util.spec_from_file_location(
    'gen_instance_split_learning_test',
    ROOT / 'tools' / 'data_gen' / 'gen_instance_split_learning_data.py')
GEN = importlib.util.module_from_spec(GEN_SPEC)
GEN_SPEC.loader.exec_module(GEN)


class InstanceSplitHeadsTests(unittest.TestCase):
    def test_decision_uses_predicted_k_and_both_thresholds(self):
        data = {
            'proposal_instance_index': np.asarray([0, 0]),
            'proposal_k': np.asarray([2, 3]),
        }
        accepted = TRAIN.decisions_from_scores(
            data,
            candidate_scores=np.asarray([0.9, 0.1]),
            child_predictions=np.asarray([3, 2]),
            safety_scores=np.asarray([0.2, 0.8]),
            candidate_threshold=0.5,
            safety_threshold=0.5)
        np.testing.assert_array_equal(accepted, [1])

    def test_threshold_search_selects_safe_improving_split(self):
        data = {
            'instance_ids': np.asarray([7]),
            'base_gt_ids': np.asarray([1, 2]),
            'base_gt_counts': np.asarray([2, 2]),
            'base_pred_counts': np.asarray([4]),
            'base_intersections': np.asarray([[2, 2]]),
            'proposal_instance_index': np.asarray([0]),
            'proposal_k': np.asarray([2]),
            'proposal_child_counts': np.asarray([[2, 2]]),
            'proposal_child_intersection_offsets': np.asarray([0, 1, 2]),
            'proposal_child_gt_indices': np.asarray([0, 1]),
            'proposal_child_intersection_values': np.asarray([2, 2]),
        }
        records = [{'plot_name': 'V1', 'data': data}]
        outputs = [{
            'candidate_score': np.asarray([0.9]),
            'child_prediction': np.asarray([2]),
            'safety_score': np.asarray([0.9]),
        }]
        settings = {'selection': {
            'candidate_thresholds': [0.5],
            'safety_thresholds': [0.5],
            'match_iou_threshold': 0.5,
            'min_precision_for_counted_fp': 0.5,
            'max_completeness_drop_pp': 0.2,
            'max_commission_increase_pp': 0.5,
        }}
        selected, _ = TRAIN.select_thresholds(records, outputs, settings)
        self.assertEqual(selected['accepted_splits'], 1)
        self.assertGreater(selected['f1_gain_pp'], 0.0)

    def test_model_output_shapes(self):
        import torch
        model = TRAIN.build_model(
            instance_dim=5, proposal_dim=3, max_children=4,
            model_config={
                'hidden_dim': 8, 'safety_hidden_dim': 4, 'dropout': 0.0})
        embedding, candidate, children = model.forward_instances(
            torch.randn(3, 5))
        safety = model.forward_safety(
            embedding, torch.randn(2, 3), torch.tensor([0, 2]))
        self.assertEqual(tuple(candidate.shape), (3,))
        self.assertEqual(tuple(children.shape), (3, 3))
        self.assertEqual(tuple(safety.shape), (2,))

    def test_threshold_selection_keeps_safety_constraints(self):
        baseline = {'tp': 10, 'fp': 2, 'fn': 0}
        safe = {'tp': 11, 'fp': 2, 'fn': 0}
        unsafe = {'tp': 9, 'fp': 4, 'fn': 1}
        self.assertGreater(
            TRAIN.detection_metrics(**safe)['f1'],
            TRAIN.detection_metrics(**baseline)['f1'])
        self.assertLess(
            TRAIN.detection_metrics(**unsafe)['f1'],
            TRAIN.detection_metrics(**baseline)['f1'])

    def test_split_validation_rejects_group_leakage(self):
        with self.assertRaisesRegex(ValueError, 'groups leak'):
            GEN.validate_splits({
                'train': ['G4N'],
                'validation': ['G4W'],
            })

    def test_training_config_rejects_wytham(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(
                "data_root: data/wytham\noutput_dir: logs/test\n",
                encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Wytham'):
                TRAIN.load_settings(path)


if __name__ == '__main__':
    unittest.main()