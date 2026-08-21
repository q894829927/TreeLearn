import unittest

import numpy as np

from tree_learn.util.proposal_relation import (
    EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES)
from tree_learn.util.proposal_relation_coarsening import (
    adaptive_component_assignments, aggregate_proposals,
    build_hierarchical_graph, vertical_profile_assignments,
    xy_block_assignments)


def tiny_graph():
    node_count = 4
    centers = np.asarray([
        [0.0, 0.0, 1.0], [0.5, 0.0, 1.2],
        [2.0, 0.0, 1.1], [6.0, 0.0, 1.0]], np.float32)
    bounds = np.concatenate(
        [centers-np.asarray([.1, .1, .5]),
         centers+np.asarray([.1, .1, .5])], axis=1).astype(np.float32)
    features = np.zeros((node_count, len(NODE_FEATURE_NAMES)), np.float32)
    features[:, NODE_FEATURE_NAMES.index('tree_probability_mean')] = .9
    features[:, NODE_FEATURE_NAMES.index('verticality_mean')] = .8
    features[:, NODE_FEATURE_NAMES.index('x_span')] = .2
    features[:, NODE_FEATURE_NAMES.index('y_span')] = .2
    features[:, NODE_FEATURE_NAMES.index('height')] = 1.
    histograms = np.zeros((node_count, 8), np.float32)
    histograms[:, :2] = .5
    edge_index = np.asarray([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]], np.int64)
    edge_features = np.zeros(
        (edge_index.shape[1], len(EDGE_FEATURE_NAMES)), np.float32)
    for name, value in {
            'xy_distance': .5, 'xy_bbox_gap': .1, 'vertical_gap': 0.,
            'vertical_overlap': .9, 'vertical_histogram_cosine': 1.,
            'tree_probability_difference': 0., 'shared_xy_boundary': 1.,
            'x_span_ratio': 1., 'y_span_ratio': 1., 'height_ratio': 1.,
    }.items():
        edge_features[:, EDGE_FEATURE_NAMES.index(name)] = value
    return {
        'point_node_id': np.asarray([0, 0, 1, 2, 3, -1], np.int32),
        'node_features': features,
        'node_bounds': bounds,
        'node_centers': centers,
        'node_point_counts': np.asarray([2, 1, 1, 1], np.int64),
        'vertical_histograms': histograms,
        'edge_index': edge_index,
        'edge_features': edge_features,
    }


class ProposalRelationCoarseningTests(unittest.TestCase):
    def test_xy_block_is_deterministic_and_compresses(self):
        graph = tiny_graph()
        first = xy_block_assignments(graph, block_size=1.0)
        second = xy_block_assignments(graph, block_size=1.0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[0], first[1])
        self.assertNotEqual(first[1], first[2])

    def test_aggregation_preserves_point_ownership(self):
        graph = tiny_graph()
        merged = aggregate_proposals(
            graph, np.asarray([0, 0, 1, 2], np.int64))
        np.testing.assert_array_equal(
            merged['point_node_id'],
            np.asarray([0, 0, 0, 1, 2, -1], np.int32))
        self.assertEqual(int(merged['node_point_counts'].sum()), 5)
        self.assertTrue(np.all(np.isfinite(merged['node_features'])))

    def test_vertical_profile_uses_only_graph_features(self):
        labels = vertical_profile_assignments(tiny_graph())
        self.assertEqual(len(np.unique(labels)), 1)

    def test_adaptive_component_respects_diameter(self):
        labels = adaptive_component_assignments(
            tiny_graph(), maximum_component_xy_diameter=1.0)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[1], labels[2])

    def test_hierarchical_graph_is_finite_and_bounded(self):
        settings = {
            'xy_block_size': 1.0,
            'graph_maximum_xy_distance': 2.5,
            'graph_maximum_xy_bbox_gap': .6,
            'graph_maximum_vertical_gap': 1.5,
            'graph_max_neighbors': 2,
        }
        graph, assignments = build_hierarchical_graph(
            tiny_graph(), 'xy_block', settings)
        self.assertEqual(len(assignments), 4)
        self.assertLessEqual(
            graph['edge_index'].shape[1],
            2*len(graph['node_features']))
        self.assertTrue(np.all(np.isfinite(graph['edge_features'])))


if __name__ == '__main__':
    unittest.main()
