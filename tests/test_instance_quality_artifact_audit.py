import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'diagnostics' / 'audit_instance_quality_artifacts.py'
)
SPEC = importlib.util.spec_from_file_location(
    'audit_instance_quality_artifacts_standalone', MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class InstanceQualityArtifactAuditTests(unittest.TestCase):

    def make_fixture(self, directory, invalid_occupancy=False):
        root = Path(directory)
        artifact_path = root / 'train' / 'P1.npz'
        artifact_path.parent.mkdir(parents=True)
        tokens = np.zeros((2, 8, 2), dtype=np.float32)
        tokens[:, 0, 0] = 1.0
        if invalid_occupancy:
            tokens[1, 0, 0] = 0.5
        masks = np.zeros((2, 8), dtype=bool)
        masks[:, 0] = True
        np.savez_compressed(
            artifact_path,
            instance_ids=np.asarray([1, 2]),
            global_feature_values=np.ones((2, 3), dtype=np.float32),
            vertical_tokens=tokens,
            layer_valid_mask=masks,
            token_feature_names=np.asarray([
                'occupancy_fraction', 'relative_height_mean']),
            target_max_iou=np.asarray([0.8, 0.1], dtype=np.float32),
            target_labeled_fraction=np.asarray([1.0, 1.0], dtype=np.float32),
            target_tree_point_fraction=np.asarray([0.8, 0.0], dtype=np.float32),
            target_is_edge=np.asarray([False, False]),
            target_valid=np.asarray([True, True]),
            target_classification_valid=np.asarray([True, True]),
            target_is_true_tree=np.asarray([True, False]),
            source_plot=np.asarray('P1'),
            split=np.asarray('train'),
        )
        sample_path = root / 'manual_audit_sample.csv'
        fieldnames = [
            'instance_id', 'target_max_iou', 'target_labeled_fraction',
            'target_tree_point_fraction', 'target_is_edge', 'target_valid',
            'target_classification_valid', 'target_is_true_tree',
            'source_plot', 'split', 'artifact_path',
        ]
        rows = [
            {
                'instance_id': 1,
                'target_max_iou': 0.8,
                'target_labeled_fraction': 1.0,
                'target_tree_point_fraction': 0.8,
                'target_is_edge': False,
                'target_valid': True,
                'target_classification_valid': True,
                'target_is_true_tree': True,
                'source_plot': 'P1',
                'split': 'train',
                'artifact_path': str(artifact_path),
            },
            {
                'instance_id': 2,
                'target_max_iou': 0.1,
                'target_labeled_fraction': 1.0,
                'target_tree_point_fraction': 0.0,
                'target_is_edge': False,
                'target_valid': True,
                'target_classification_valid': True,
                'target_is_true_tree': False,
                'source_plot': 'P1',
                'split': 'train',
                'artifact_path': str(artifact_path),
            },
        ]
        with sample_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return {
            'data_root': str(root),
            'sample_path': str(sample_path),
            'thresholds': {
                'positive_iou': 0.5,
                'negative_iou': 0.25,
            },
            'gate': {
                'min_sample_count': 2,
                'min_plot_count': 1,
                'require_train': True,
                'require_validation': False,
                'require_ambiguous': False,
                'require_edge': False,
            },
        }

    def test_consistent_positive_and_negative_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.make_fixture(directory)
            report, rows = AUDIT.audit_artifacts(settings)
        self.assertTrue(report['gate']['passed'])
        self.assertEqual(report['class_counts'], {
            'negative': 1,
            'positive': 1,
        })
        self.assertEqual(len(rows), 2)

    def test_invalid_occupancy_fails_consistency_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.make_fixture(
                directory, invalid_occupancy=True)
            report, _ = AUDIT.audit_artifacts(settings)
        self.assertFalse(report['gate']['all_consistency_checks_passed'])
        self.assertGreater(report['error_count'], 0)


if __name__ == '__main__':
    unittest.main()
