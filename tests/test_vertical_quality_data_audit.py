import importlib.util
import json
import pathlib
import tempfile
import unittest

import numpy as np


repo_root = pathlib.Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location(
    'vertical_quality_audit_standalone',
    repo_root / 'tools' / 'diagnostics' /
    'audit_vertical_quality_data.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)


class VerticalQualityAuditTests(unittest.TestCase):

    def test_json_default_converts_numpy_scalars_and_arrays(self):
        payload = {
            'passed': np.bool_(True),
            'count': np.int64(3),
            'values': np.asarray([0.25, 0.5], dtype=np.float32),
        }

        serialized = json.dumps(payload, default=module.json_default)
        restored = json.loads(serialized)

        self.assertIs(restored['passed'], True)
        self.assertEqual(restored['count'], 3)
        self.assertEqual(len(restored['values']), 2)

    def test_partial_manual_validation_is_role_eligible_not_full(self):
        settings = {
            'min_label_coverage': 0.95,
            'min_validation_label_coverage': 0.50,
            'min_validation_trees': 100,
        }
        row = {
            'role': 'validation',
            'num_trees': 200,
            'non_tree_point_count': 1_000,
            'label_conflict_count': 0,
            'label_coverage_rate': 0.72814,
        }

        full, role_eligible, threshold, minimum_trees = (
            module.forest_eligibility(row, settings))

        self.assertFalse(full)
        self.assertTrue(role_eligible)
        self.assertEqual(threshold, 0.50)
        self.assertEqual(minimum_trees, 100)

    def test_summarizes_labels_and_boundary_trees(self):
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 5.0, 1.0],
            [5.0, 5.0, 2.0],
            [9.9, 5.0, 3.0],
            [10.0, 10.0, 4.0],
            [5.0, 6.0, 5.0],
        ])
        labels = np.array([-1, 1, 1, 2, 2, 0])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'forest.npy'
            np.save(path, np.column_stack([coords, labels]))
            summary, tree_ids = module.summarize_labeled_arrays(
                path, coords, labels, edge_margin_m=0.2)

        self.assertEqual(summary['point_count'], 6)
        self.assertEqual(summary['tree_point_count'], 4)
        self.assertEqual(summary['non_tree_point_count'], 1)
        self.assertEqual(summary['unlabeled_point_count'], 1)
        self.assertEqual(summary['num_trees'], 2)
        self.assertEqual(summary['boundary_tree_count_bbox'], 2)
        self.assertEqual(tree_ids, {1, 2})
        self.assertAlmostEqual(summary['label_coverage_rate'], 5 / 6)

    def test_gate_requires_consistent_diagnostics(self):
        config = {
            'audit': {
                'expected_branch': 'vertical-instance-quality',
                'min_training_forests': 1,
                'min_validation_forests': 1,
            },
        }
        forest_template = {
            'eligible_full_forest': True,
            'sensor': 'sensor',
            'forest_type': 'forest',
        }
        forests = [
            dict(forest_template, role='train', label_quality='automatic'),
            dict(
                forest_template, role='validation',
                label_quality='manually_corrected'),
        ]
        reproducibility = {
            'git': {'branch': 'vertical-instance-quality'},
            'fixed_configs': [{'exists': True}],
            'checkpoints': [{'exists': True}],
        }
        diagnostics = [{'checks': {'passed': False}}]

        gate = module.build_gate(
            config, forests, diagnostics, reproducibility)

        self.assertFalse(gate['diagnostic_artifacts_consistent'])
        self.assertFalse(gate['passed'])


if __name__ == '__main__':
    unittest.main()
