"""Label-free hierarchical coarsening for vertical micro-proposal graphs."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .proposal_relation import (
    EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, build_proposal_graph)


def _consecutive(values):
    _, inverse = np.unique(np.asarray(values, np.int64), return_inverse=True)
    return inverse.astype(np.int64)


def _edge_column(graph, name):
    return np.asarray(graph['edge_features'])[:, EDGE_FEATURE_NAMES.index(name)]


def xy_block_assignments(graph, block_size=3.6):
    """Aggregate columns into a fixed horizontal lattice without labels."""
    centers = np.asarray(graph['node_centers'], np.float64)
    if len(centers) == 0:
        return np.empty(0, np.int64)
    if float(block_size) <= 0:
        raise ValueError('block_size must be positive.')
    origin = centers[:, :2].min(0)
    keys = np.floor(
        (centers[:, :2]-origin)/float(block_size)).astype(np.int64)
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    return labels.astype(np.int64)


def vertical_profile_assignments(
        graph, maximum_xy_distance=1.5, minimum_histogram_cosine=.85,
        minimum_vertical_overlap=.5, maximum_tree_probability_difference=.15):
    """Connected components of a strict, label-free canopy-continuity graph."""
    node_count = len(graph['node_features'])
    edge_index = np.asarray(graph['edge_index'], np.int64)
    if node_count == 0 or edge_index.shape[1] == 0:
        return np.arange(node_count, dtype=np.int64)
    mask = (
        (_edge_column(graph, 'xy_distance') <= float(maximum_xy_distance)) &
        (_edge_column(graph, 'vertical_histogram_cosine') >=
         float(minimum_histogram_cosine)) &
        (_edge_column(graph, 'vertical_overlap') >=
         float(minimum_vertical_overlap)) &
        (np.abs(_edge_column(graph, 'tree_probability_difference')) <=
         float(maximum_tree_probability_difference)))
    source, target = edge_index[:, mask]
    if len(source) == 0:
        return np.arange(node_count, dtype=np.int64)
    adjacency = coo_matrix(
        (np.ones(len(source), np.uint8), (source, target)),
        shape=(node_count, node_count)).tocsr()
    _, labels = connected_components(
        adjacency, directed=False, return_labels=True)
    return labels.astype(np.int64)


class _BoundedUnion:
    def __init__(self, bounds):
        self.parent = np.arange(len(bounds), dtype=np.int64)
        self.rank = np.zeros(len(bounds), np.int8)
        self.minimum = np.asarray(bounds[:, :3], np.float64).copy()
        self.maximum = np.asarray(bounds[:, 3:], np.float64).copy()

    def find(self, value):
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def try_union(self, left, right, maximum_xy_diameter,
                  maximum_height):
        left, right = self.find(left), self.find(right)
        if left == right:
            return True
        minimum = np.minimum(self.minimum[left], self.minimum[right])
        maximum = np.maximum(self.maximum[left], self.maximum[right])
        if np.linalg.norm(maximum[:2]-minimum[:2]) > maximum_xy_diameter:
            return False
        if maximum[2]-minimum[2] > maximum_height:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        self.minimum[left], self.maximum[left] = minimum, maximum
        return True


def adaptive_component_assignments(
        graph, maximum_bbox_gap=.3, maximum_vertical_gap=.5,
        minimum_histogram_cosine=.70,
        maximum_tree_probability_difference=.20,
        maximum_component_xy_diameter=4.8,
        maximum_component_height=60.):
    """Greedily consolidate structurally compatible nodes with fixed bounds."""
    node_count = len(graph['node_features'])
    edge_index = np.asarray(graph['edge_index'], np.int64)
    if node_count == 0 or edge_index.shape[1] == 0:
        return np.arange(node_count, dtype=np.int64)
    source, target = edge_index
    mask = (
        (source < target) &
        (_edge_column(graph, 'xy_bbox_gap') <= float(maximum_bbox_gap)) &
        (_edge_column(graph, 'vertical_gap') <= float(maximum_vertical_gap)) &
        (_edge_column(graph, 'vertical_histogram_cosine') >=
         float(minimum_histogram_cosine)) &
        (np.abs(_edge_column(graph, 'tree_probability_difference')) <=
         float(maximum_tree_probability_difference)))
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return np.arange(node_count, dtype=np.int64)
    score = (
        2*_edge_column(graph, 'vertical_histogram_cosine')[candidates] +
        _edge_column(graph, 'vertical_overlap')[candidates] +
        _edge_column(graph, 'shared_xy_boundary')[candidates] -
        _edge_column(graph, 'xy_bbox_gap')[candidates] -
        .5*_edge_column(graph, 'vertical_gap')[candidates] -
        np.abs(_edge_column(
            graph, 'tree_probability_difference')[candidates]))
    order = candidates[np.argsort(-score, kind='stable')]
    union = _BoundedUnion(np.asarray(graph['node_bounds']))
    for edge in order:
        union.try_union(
            source[edge], target[edge],
            float(maximum_component_xy_diameter),
            float(maximum_component_height))
    roots = np.asarray([union.find(node) for node in range(node_count)])
    return _consecutive(roots)


def coarsening_assignments(graph, method, settings):
    if method == 'xy_block':
        return xy_block_assignments(
            graph, block_size=float(settings['xy_block_size']))
    if method == 'vertical_profile':
        return vertical_profile_assignments(
            graph,
            maximum_xy_distance=float(settings['profile_maximum_xy_distance']),
            minimum_histogram_cosine=float(
                settings['profile_minimum_histogram_cosine']),
            minimum_vertical_overlap=float(
                settings['profile_minimum_vertical_overlap']),
            maximum_tree_probability_difference=float(
                settings['profile_maximum_probability_difference']))
    if method == 'adaptive_component':
        return adaptive_component_assignments(
            graph,
            maximum_bbox_gap=float(settings['adaptive_maximum_bbox_gap']),
            maximum_vertical_gap=float(
                settings['adaptive_maximum_vertical_gap']),
            minimum_histogram_cosine=float(
                settings['adaptive_minimum_histogram_cosine']),
            maximum_tree_probability_difference=float(
                settings['adaptive_maximum_probability_difference']),
            maximum_component_xy_diameter=float(
                settings['adaptive_maximum_component_xy_diameter']),
            maximum_component_height=float(
                settings['adaptive_maximum_component_height']))
    raise ValueError(f'Unknown coarsening method: {method}')


def aggregate_proposals(graph, node_to_super):
    """Aggregate a micro graph into super-proposals without target fields."""
    node_to_super = _consecutive(node_to_super)
    node_count = len(graph['node_features'])
    if len(node_to_super) != node_count:
        raise ValueError('node_to_super is not aligned with graph nodes.')
    super_count = int(node_to_super.max())+1 if len(node_to_super) else 0
    old_counts = np.asarray(graph['node_point_counts'], np.float64)
    counts = np.bincount(
        node_to_super, weights=old_counts,
        minlength=super_count).astype(np.int64)
    weighted_features = (
        np.asarray(graph['node_features'], np.float64)*old_counts[:, None])
    features = np.zeros(
        (super_count, len(NODE_FEATURE_NAMES)), np.float64)
    np.add.at(features, node_to_super, weighted_features)
    features /= np.maximum(counts[:, None], 1)
    centers = np.zeros((super_count, 3), np.float64)
    np.add.at(
        centers, node_to_super,
        np.asarray(graph['node_centers'], np.float64)*old_counts[:, None])
    centers /= np.maximum(counts[:, None], 1)
    histogram = np.zeros((super_count, 8), np.float64)
    np.add.at(
        histogram, node_to_super,
        np.asarray(graph['vertical_histograms'], np.float64)*
        old_counts[:, None])
    histogram /= np.maximum(histogram.sum(1, keepdims=True), 1e-9)
    bounds = np.empty((super_count, 6), np.float64)
    bounds[:, :3], bounds[:, 3:] = np.inf, -np.inf
    np.minimum.at(
        bounds[:, :3], node_to_super,
        np.asarray(graph['node_bounds'], np.float64)[:, :3])
    np.maximum.at(
        bounds[:, 3:], node_to_super,
        np.asarray(graph['node_bounds'], np.float64)[:, 3:])
    point_node = np.asarray(graph['point_node_id'], np.int64)
    point_super = np.full(len(point_node), -1, np.int32)
    valid = point_node >= 0
    point_super[valid] = node_to_super[point_node[valid]].astype(np.int32)
    result = {
        'point_node_id': point_super,
        'node_features': features.astype(np.float32),
        'node_bounds': bounds.astype(np.float32),
        'node_centers': centers.astype(np.float32),
        'node_point_counts': counts,
        'vertical_histograms': histogram.astype(np.float32),
    }
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError('Coarsened proposal artifact contains non-finite values.')
    return result


def build_hierarchical_graph(graph, method, settings):
    labels = coarsening_assignments(graph, method, settings)
    proposals = aggregate_proposals(graph, labels)
    result = build_proposal_graph(
        proposals,
        max_xy_distance=float(settings['graph_maximum_xy_distance']),
        max_xy_bbox_gap=float(settings['graph_maximum_xy_bbox_gap']),
        max_vertical_gap=float(settings['graph_maximum_vertical_gap']),
        max_neighbors=int(settings['graph_max_neighbors']))
    return result, labels
