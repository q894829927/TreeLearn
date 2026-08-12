import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tree_learn.util.fragment_graph_diagnostics import (
    analyze_fragment_graph_oracle,
    build_fragment_adjacency,
    merge_contingency_groups,
)
from tree_learn.util.omission_diagnostics import _contingency


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'diagnostics' / 'diagnose_fragment_graph_oracle.py'
SPEC = importlib.util.spec_from_file_location('q6a_config_test', SCRIPT)
Q6A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q6A)


class FragmentGraphOracleTests(unittest.TestCase):

    def test_merge_contingency_groups_sums_rows(self):
        labels = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
        predictions = np.asarray([10, 10, 20, 20, 30, 30, 30, 30])
        raw = _contingency(labels, predictions)
        table = {
            'gt_ids': raw['gt_ids'],
            'gt_counts': raw['gt_counts'],
            'pred_ids': raw['pred_ids'],
            'pred_counts': raw['pred_counts'],
            'intersections': raw['intersections'],
        }
        merged = merge_contingency_groups(table, [(0, 1)])
        np.testing.assert_array_equal(merged['pred_ids'], [10, 30])
        np.testing.assert_array_equal(merged['pred_counts'], [4, 4])
        np.testing.assert_array_equal(
            merged['intersections'], [[4, 0], [0, 4]])

    def test_adjacency_rejects_distant_vote_centers(self):
        geometry = {
            'vote_centers': np.asarray([[0., 0.], [0.5, 0.], [10., 0.]]),
            'raw_centers': np.asarray([[0., 0.], [0.5, 0.], [10., 0.]]),
            'xy_min': np.asarray([[0., 0.], [0.4, 0.], [10., 0.]]),
            'xy_max': np.asarray([[0.1, 0.1], [0.6, 0.1], [10.1, 0.1]]),
            'z_min': np.asarray([0., 0., 0.]),
            'z_max': np.asarray([5., 5., 5.]),
        }
        edges = build_fragment_adjacency(geometry, {
            'max_vote_center_distance': 2.0,
            'max_raw_center_distance': 2.0,
            'max_xy_bbox_gap': 1.0,
            'max_vertical_gap': 1.0,
            'max_candidate_edges': 10,
        })
        self.assertEqual([(row['left'], row['right']) for row in edges], [(0, 1)])

    def test_graph_oracle_recovers_fragmented_tree_without_losing_tp(self):
        # GT 1 is split into predictions 10 and 20; GT 2 is already detected.
        labels = np.asarray([1] * 10 + [2] * 10)
        predictions = np.asarray([10] * 5 + [20] * 5 + [30] * 10)
        coords = np.zeros((20, 3), dtype=np.float32)
        coords[:5, 0] = 0.0
        coords[5:10, 0] = 0.4
        coords[10:, 0] = 10.0
        coords[:, 2] = np.tile(np.arange(10), 2)
        offsets = np.zeros_like(coords)
        logits = np.tile([5.0, 0.0], (20, 1)).astype(np.float32)
        verticality = np.ones(20, dtype=np.float32)
        result = analyze_fragment_graph_oracle(
            coords, logits, offsets, verticality, labels, predictions,
            expected_fragmentation_gt_ids=[1],
            adjacency_settings={
                'max_vote_center_distance': 2.0,
                'max_raw_center_distance': 2.0,
                'max_xy_bbox_gap': 1.0,
                'max_vertical_gap': 1.0,
                'max_candidate_edges': 10,
            },
            min_fragment_overlap_fraction=0.1,
            min_fragment_overlap_points=1)
        self.assertEqual(result['baseline']['tp'], 1)
        graph = result['modes']['fragment_graph_oracle']
        self.assertEqual(graph['tp'], 2)
        self.assertEqual(graph['recovered_fragmentation_trees'], 1)
        self.assertEqual(graph['lost_baseline_trees'], 0)

    def test_config_rejects_wytham(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q6a.yaml'
            path.write_text(
                'output_root: out\nq3_reference_root: q3\n'
                'source_root: data\npipeline_template: x\ncheckpoint: x\n'
                'validation_plots: [Wytham]\nfragment_graph_oracle: {}\n'
                'gate: {}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must not mention Wytham'):
                Q6A.load_settings(path)


if __name__ == '__main__':
    unittest.main()
