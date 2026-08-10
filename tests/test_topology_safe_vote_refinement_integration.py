import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_topology_safe_vote_refinement_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'topology_safe_vote_refinement_integration', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class TopologySafeVoteRefinementIntegrationTests(unittest.TestCase):

    def test_exact_artifact_reuses_e2a_identity_without_clustering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / 'validation'
            artifact_dir.mkdir(parents=True)
            path = artifact_dir / 'PLOT.npz'
            votes = np.asarray([
                [0.1, 0.1], [0.2, 0.1], [2.1, 0.1],
                [2.2, 0.1], [4.1, 0.1], [4.2, 0.1],
            ], dtype=np.float32)
            labels = np.asarray([1, 1, 2, 2, 0, 0])
            tree = labels > 0
            residual = np.zeros((6, 2), dtype=np.float32)
            residual[:2] = [0.3, 0.0]
            residual[2:4] = [-0.3, 0.0]
            targets = np.zeros_like(votes)
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
            row = {
                'source_plot': 'PLOT',
                'artifact_path': str(path),
                'num_candidates': '6',
            }
            arrays, identity = ORACLE.load_vote_artifact(
                row, str(root))
            self.assertEqual(len(arrays['base_votes_xy']), 6)
            reference = {
                'artifact_identity': identity,
                'clusters': {},
                'cache_sha256': 'synthetic',
            }
            report = ORACLE.analyze_plot(row, {
                'data_root': str(root),
                'cell_size': 1.0,
                'topology_cell_size': 1.0,
                'damping_factors': [0.25, 0.5, 0.75, 1.0],
                'run_hdbscan': False,
                'min_cluster_size': 2,
                'min_iou': 0.5,
                'n_jobs': 1,
            }, reference)
            self.assertEqual(report['num_candidates'], 6)
            self.assertGreater(
                report['effects']['mean_error_reduction_relative'], 0)
            self.assertGreaterEqual(
                report['effects']['cell_purity_gain'], 0)
            self.assertLessEqual(
                report['effects']['multi_tree_collision_increase'], 0)
            json.dumps(report)


if __name__ == '__main__':
    unittest.main()
