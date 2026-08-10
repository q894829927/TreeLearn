import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_coverage_preserving_quality_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'coverage_preserving_quality_oracle_integration', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def make_artifact(root, plot, split):
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{plot}.npz'
    ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    np.savez_compressed(
        path,
        instance_ids=ids,
        target_max_iou=np.asarray([0.8, 0.7, 0.1, 0.3]),
        target_best_gt_id=np.asarray([10, 11, 0, 12]),
        target_valid=np.asarray([1, 1, 1, 1], dtype=bool),
        target_classification_valid=np.asarray(
            [1, 1, 1, 0], dtype=bool),
        target_is_true_tree=np.asarray([1, 1, 0, 0], dtype=bool),
        source_plot=np.asarray(plot),
        split=np.asarray(split),
    )
    return path, ids


class CoveragePreservingQualityOracleIntegrationTests(unittest.TestCase):

    def test_loader_checks_manifest_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, ids = make_artifact(root, 'P', 'validation')
            manifest = root / 'manifest.csv'
            with manifest.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'source_plot', 'split', 'artifact_path', 'instance_id'])
                writer.writeheader()
                for instance_id in ids:
                    writer.writerow({
                        'source_plot': 'P',
                        'split': 'validation',
                        'artifact_path': str(artifact),
                        'instance_id': int(instance_id),
                    })
            groups = ORACLE.load_artifact_groups(
                root, manifest, {'train': [], 'validation': ['P']})
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]['targets']['safe_reject'].sum(), 1)
            self.assertEqual(
                groups[0]['targets']['coverage_critical'].sum(), 2)

    def test_loader_forbids_wytham_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'manifest.csv'
            with manifest.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'source_plot', 'split', 'artifact_path', 'instance_id'])
                writer.writeheader()
                writer.writerow({
                    'source_plot': 'wytham',
                    'split': 'validation',
                    'artifact_path': str(root / 'missing.npz'),
                    'instance_id': 1,
                })
            with self.assertRaisesRegex(ValueError, 'Wytham'):
                ORACLE.load_artifact_groups(
                    root, manifest,
                    {'train': [], 'validation': ['wytham']})

    def test_reject_ratio_is_preregistered(self):
        raw = {
            'data_root': 'data',
            'manifest_path': 'manifest.csv',
            'output_dir': 'out',
            'expected_splits': {'train': [], 'validation': []},
            'reject_ratio': 0.20,
            'gate': {},
        }
        with self.assertRaisesRegex(ValueError, '0.15'):
            ORACLE.run_settings(raw)


if __name__ == '__main__':
    unittest.main()
