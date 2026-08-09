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
    'cellwise_vote_refinement_integration', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class CellwiseVoteRefinementIntegrationTests(unittest.TestCase):

    def test_exact_artifact_runs_without_clustering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / 'validation'
            artifact_dir.mkdir(parents=True)
            path = artifact_dir / 'PLOT.npz'
            votes = np.asarray([
                [0.1, 0.1], [0.2, 0.1], [1.1, 0.1],
                [1.2, 0.1], [2.1, 0.1], [2.2, 0.1],
            ], dtype=np.float32)
            labels = np.asarray([1, 1, 2, 2, 0, 0])
            tree = labels > 0
            residual = np.zeros((6, 2), dtype=np.float32)
            residual[:2] = [1.0, 0.0]
            residual[2:4] = [-1.0, 0.0]
            targets = np.zeros((6, 2), dtype=np.float32)
            targets[tree] = votes[tree] + residual[tree]
            np.savez_compressed(
                path,
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
            report = ORACLE.analyze_plot({
                'source_plot': 'PLOT',
                'artifact_path': str(path),
                'num_candidates': '6',
            }, {
                'data_root': str(root),
                'cell_size': 1.0,
                'topology_cell_size': 1.0,
                'run_hdbscan': False,
                'min_cluster_size': 2,
                'min_iou': 0.5,
                'n_jobs': 1,
            })
            self.assertEqual(report['num_candidates'], 6)
            self.assertEqual(report['num_tree_candidates'], 4)
            self.assertLess(
                report['modes']['pointwise_oracle']['error'][
                    'mean_error_xy'], 1e-6)
            self.assertEqual(report['candidate_set_sha256'].__class__, str)
            json.dumps(report)


if __name__ == '__main__':
    unittest.main()
