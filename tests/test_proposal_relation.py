import tempfile
import unittest
from pathlib import Path

import numpy as np

from tree_learn.util.proposal_relation import (
    build_learning_targets, build_proposal_graph,
    build_vertical_micro_proposals, decode_relation_graph,
    load_inference_graph, point_instances_from_nodes,
    save_inference_graph)


class ProposalRelationGraphTests(unittest.TestCase):
    def _points(self):
        coords = np.array([
            [.1, .1, 0.0], [.1, .1, .4], [.1, .1, 2.0],
            [.7, .1, .1], [.7, .1, .5], [5., 5., 0.]], np.float32)
        probability = np.array([.9, .8, .9, .95, .9, .1], np.float32)
        return coords, probability

    def test_micro_proposals_cover_tree_points_once(self):
        coords, probability = self._points()
        proposals = build_vertical_micro_proposals(
            coords, probability, xy_cell_size=.6, z_bin_size=.5,
            vertical_split_gap=1.)
        ids = proposals['point_node_id']
        self.assertTrue(np.all(ids[probability >= .5] >= 0))
        self.assertTrue(np.all(ids[probability < .5] == -1))
        self.assertEqual(len(np.flatnonzero(ids >= 0)), 5)
        self.assertNotEqual(ids[0], ids[2])

    def test_graph_is_bounded_and_unique(self):
        coords, probability = self._points()
        proposals = build_vertical_micro_proposals(coords, probability)
        graph = build_proposal_graph(proposals, max_neighbors=2)
        edges = list(map(tuple, graph['edge_index'].T.tolist()))
        self.assertEqual(len(edges), len(set(edges)))
        degrees = np.bincount(
            graph['edge_index'][0], minlength=len(graph['node_features']))
        self.assertLessEqual(int(degrees.max(initial=0)), 2)
        self.assertTrue(np.all(np.isfinite(graph['node_features'])))
        self.assertTrue(np.all(np.isfinite(graph['edge_features'])))

    def test_learning_labels_are_separate_from_inference_graph(self):
        coords, probability = self._points()
        graph = build_proposal_graph(
            build_vertical_micro_proposals(coords, probability))
        target = build_learning_targets(
            graph, np.array([1, 1, 2, 1, 1, 0]))
        self.assertEqual(len(target['edge_label']), graph['edge_index'].shape[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'graph.npz'
            save_inference_graph(path, graph, {'split': 'inference'})
            loaded, metadata = load_inference_graph(path)
            self.assertNotIn('node_gt', loaded)
            self.assertEqual(metadata['split'], 'inference')
            with self.assertRaises(ValueError):
                save_inference_graph(
                    path, {**graph, 'gt_label': np.ones(1)}, {})

    def test_union_find_is_order_invariant_and_handles_isolates(self):
        graph = {
            'node_features': np.zeros((3, 2), np.float32),
            'node_bounds': np.array([
                [0, 0, 0, 1, 1, 2], [1, 0, 0, 2, 1, 2],
                [8, 8, 0, 9, 9, 2]], np.float32),
            'edge_index': np.array([[0, 1], [1, 0]], np.int64)}
        first = decode_relation_graph(graph, np.array([.9, .8]), .5)
        reversed_graph = {**graph, 'edge_index': graph['edge_index'][:, ::-1]}
        second = decode_relation_graph(
            reversed_graph, np.array([.8, .9]), .5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[0], first[1])
        self.assertNotEqual(first[0], first[2])

    def test_empty_and_single_node(self):
        empty = {
            'node_features': np.empty((0, 2)),
            'node_bounds': np.empty((0, 6)),
            'edge_index': np.empty((2, 0), np.int64)}
        self.assertEqual(len(decode_relation_graph(empty, [], .5)), 0)
        ids = point_instances_from_nodes(
            np.array([0, -1]), np.array([1]))
        np.testing.assert_array_equal(ids, np.array([1, 0]))


if __name__ == '__main__':
    unittest.main()
