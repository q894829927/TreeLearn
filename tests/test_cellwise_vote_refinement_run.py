import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_cellwise_vote_refinement_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'cellwise_vote_refinement_run', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class CellwiseVoteRefinementRunTests(unittest.TestCase):

    def test_full_run_writes_summary_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'data'
            artifact_dir = data_root / 'validation'
            artifact_dir.mkdir(parents=True)
            artifact_path = artifact_dir / 'PLOT.npz'
            votes = np.asarray([
                [0.1, 0.1], [0.2, 0.1], [2.1, 0.1],
                [2.2, 0.1], [4.1, 0.1], [4.2, 0.1],
            ], dtype=np.float32)
            labels = np.asarray([1, 1, 2, 2, 0, 0])
            tree = labels > 0
            residual = np.zeros((6, 2), dtype=np.float32)
            residual[:2] = [0.4, 0.0]
            residual[2:4] = [-0.4, 0.0]
            targets = np.zeros_like(votes)
            targets[tree] = votes[tree] + residual[tree]
            np.savez_compressed(
                artifact_path,
                candidate_indices=np.arange(6, dtype=np.int64),
                base_votes_xy=votes,
                target_base_vote_xy=targets,
                target_vote_residual_xy=residual,
                target_vote_error_xy=np.linalg.norm(residual, axis=1),
                target_is_tree=tree,
                target_tree_id=labels,
                target_valid=np.ones(6, dtype=bool),
                target_coverage_critical=np.asarray(
                    [1, 0, 1, 0, 0, 0], dtype=bool),
            )
            manifest_path = data_root / 'manifest.csv'
            with manifest_path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'source_plot', 'split', 'artifact_path',
                    'num_candidates'])
                writer.writeheader()
                writer.writerow({
                    'source_plot': 'PLOT', 'split': 'validation',
                    'artifact_path': str(artifact_path),
                    'num_candidates': 6,
                })
            output_dir = root / 'output'
            config = {
                'data_root': str(data_root),
                'manifest_path': str(manifest_path),
                'output_dir': str(output_dir),
                'expected_validation_plots': ['PLOT'],
                'cell_size': 1.0,
                'topology_cell_size': 1.0,
                'run_hdbscan': False,
                'min_cluster_size': 2,
                'min_iou': 0.5,
                'n_jobs': 1,
                'gate': {
                    'expected_validation_plots': 1,
                    'max_pointwise_oracle_mean_error': 1e-5,
                    'min_mean_error_reduction_relative': 0.1,
                    'min_p90_error_reduction_relative': 0.05,
                    'min_pointwise_improvement_retained': 0.7,
                    'max_cell_purity_drop': 0.0,
                    'max_multi_tree_collision_increase': 0.0,
                    'min_plot_wins': 1,
                },
            }

            ORACLE.run_settings(config, config_path='<test>')
            summary = json.loads(
                (output_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['gate']['passed'])
            cache_path = output_dir / 'plots' / 'PLOT.json'
            self.assertTrue(cache_path.is_file())
            first_cache = cache_path.read_bytes()
            ORACLE.run_settings(config, config_path='<test>')
            self.assertEqual(first_cache, cache_path.read_bytes())


if __name__ == '__main__':
    unittest.main()
