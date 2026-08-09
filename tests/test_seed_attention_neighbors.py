import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / 'tools' / 'data_gen' / 'gen_seed_attention_neighbors.py')
SPEC = importlib.util.spec_from_file_location(
    'seed_attention_neighbors_test', SCRIPT_PATH)
NEIGHBORS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NEIGHBORS)


class SeedAttentionNeighbourTests(unittest.TestCase):

    def test_query_is_self_first_and_respects_radius(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest('SciPy is required for cKDTree neighbour queries.')
        votes = np.asarray([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 5.0],
        ], dtype=np.float32)
        result = NEIGHBORS.query_fixed_radius_neighbors(
            votes, num_neighbors=3, radius=0.25,
            query_chunk_size=2, workers=1)
        np.testing.assert_array_equal(result[:, 0], np.arange(4))
        self.assertTrue(np.all(result[3, 1:] == -1))
        for query in range(len(votes)):
            valid = result[query, 1:] >= 0
            if valid.any():
                distances = np.linalg.norm(
                    votes[result[query, 1:][valid]] - votes[query], axis=1)
                self.assertTrue(np.all(distances <= 0.25 + 1e-6))

    def test_cache_validation_detects_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / 'source.npz'
            cache_path = root / 'neighbors.npz'
            np.savez_compressed(
                source_path,
                candidate_indices=np.arange(3),
                base_votes_xy=np.asarray([
                    [0.0, 0.0], [0.1, 0.0], [1.0, 1.0],
                ], dtype=np.float32),
                target_coverage_critical=np.asarray(
                    [1, 0, 1], dtype=bool),
            )
            neighbours = np.asarray([
                [0, 1], [1, 0], [2, -1],
            ], dtype=np.int32)
            np.savez_compressed(
                cache_path,
                candidate_indices=np.arange(3),
                neighbor_indices=neighbours)
            result = NEIGHBORS.validate_neighbor_cache(
                cache_path, source_path, num_neighbors=2, radius=0.2)
            self.assertAlmostEqual(result['relation_coverage'], 2 / 3)
            self.assertAlmostEqual(
                result['critical_relation_coverage'], 0.5)
            self.assertTrue(result['all_self_edges_valid'])


if __name__ == '__main__':
    unittest.main()
