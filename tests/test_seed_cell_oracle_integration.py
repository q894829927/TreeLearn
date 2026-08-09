import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_seed_cell_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'seed_cell_oracle_integration', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class SeedCellOracleIntegrationTests(unittest.TestCase):

    def test_one_plot_analysis_is_exact_and_json_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / 'validation' / 'PLOT'
            artifact_dir.mkdir(parents=True)
            artifact_path = artifact_dir / 'seed_quality.npz'
            votes = np.column_stack([
                np.repeat(np.arange(4), 5) + np.tile(
                    np.linspace(0.05, 0.45, 5), 4),
                np.zeros(20),
            ]).astype(np.float32)
            labels = np.asarray(
                [1, 1, 1, 0, 0, 2, 2, 2, 0, 0,
                 3, 3, 3, 0, 0, 4, 4, 4, 0, 0],
                dtype=np.int64)
            utility = np.where(labels > 0, 0.9, 0.0).astype(np.float32)
            critical = np.zeros(20, dtype=bool)
            critical[[0, 5, 10, 15]] = True
            np.savez_compressed(
                artifact_path,
                base_votes_xy=votes,
                target_tree_id=labels,
                target_utility=utility,
                target_vote_error_xy=np.where(
                    labels > 0, 0.1, 0.0).astype(np.float32),
                target_coverage_critical=critical,
                target_valid=np.ones(20, dtype=bool),
            )
            report = ORACLE.analyze_plot({
                'source_plot': 'PLOT',
                'artifact_path': str(artifact_path),
                'num_candidates': '20',
            }, {
                'data_root': str(root),
                'cell_size': 1.0,
                'keep_ratio': 0.8,
                'density_exponent': 0.5,
                'random_seeds': [42, 43, 44],
            })
            self.assertEqual(report['keep_count'], 16)
            self.assertEqual(
                report['modes']['cell_oracle']['selected_count'], 16)
            self.assertEqual(
                report['modes']['cell_oracle']['critical_recall'], 1.0)
            self.assertEqual(
                report['modes']['cell_oracle']['occupied_cell_coverage'], 1.0)
            json.dumps(report)


if __name__ == '__main__':
    unittest.main()
