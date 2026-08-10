import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_multiscale_grouping_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'multiscale_grouping_oracle_integration', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class MultiscaleGroupingOracleIntegrationTests(unittest.TestCase):

    def test_cluster_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            votes = np.asarray([[0.0, 0.0], [1.0, 0.0]])
            prediction = np.asarray([0, -1], dtype=np.int64)
            identity = {'path': 'x', 'size': 1, 'mtime_ns': 2}
            with mock.patch.object(
                    ORACLE, 'cluster_votes', return_value=prediction) as call:
                first = ORACLE.load_or_cluster(
                    votes, 'P', 25, cache_dir, identity, 1)
                second = ORACLE.load_or_cluster(
                    votes, 'P', 25, cache_dir, identity, 1)
            np.testing.assert_array_equal(first, prediction)
            np.testing.assert_array_equal(second, prediction)
            self.assertEqual(call.call_count, 1)

    def test_validation_loader_rejects_wytham(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.csv'
            with path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'source_plot', 'split', 'artifact_path',
                    'num_candidates'])
                writer.writeheader()
                writer.writerow({
                    'source_plot': 'wytham',
                    'split': 'validation',
                    'artifact_path': 'wytham.npz',
                    'num_candidates': 1,
                })
            with self.assertRaises(ValueError):
                ORACLE.load_validation_rows(path, ['wytham'])

    def test_config_rejects_unregistered_scale_grid_before_io(self):
        raw = {
            'data_root': 'data',
            'manifest_path': 'manifest.csv',
            'output_dir': 'out',
            'e2a_summary_path': 'e2a.json',
            'e2a_plot_dir': 'plots',
            'expected_validation_plots': ['P'],
            'cluster_sizes': [25, 50, 100],
            'base_cluster_size': 50,
            'min_iou': 0.5,
            'n_jobs': 1,
            'gate': {},
        }
        with self.assertRaisesRegex(ValueError, 'locked'):
            ORACLE.run_settings(raw)

    def test_reference_requires_original_e2a_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot_dir = root / 'plots'
            plot_dir.mkdir()
            summary = root / 'summary.json'
            summary.write_text(json.dumps({
                'settings': {
                    'run_hdbscan': True,
                    'min_cluster_size': 99,
                    'min_iou': 0.5,
                    'n_jobs': 1,
                },
            }), encoding='utf-8')
            settings = {
                'e2a_summary_path': str(summary),
                'e2a_plot_dir': str(plot_dir),
                'base_cluster_size': 50,
                'min_iou': 0.5,
                'n_jobs': 1,
            }
            with self.assertRaisesRegex(ValueError, 'differs'):
                ORACLE.load_e2a_base_references(settings, ['P'])


if __name__ == '__main__':
    unittest.main()
