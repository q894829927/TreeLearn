"""Offset-free vertical micro-proposals and sparse relation attention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    F = None

from .omission_diagnostics import _contingency, detection_metrics
from .seed_quality import tree_probabilities


GRAPH_VERSION = 1
NODE_FEATURE_NAMES = (
    'log_num_points', 'center_x', 'center_y', 'center_z',
    'x_span', 'y_span', 'height', 'log_density',
    'tree_probability_mean', 'tree_probability_std',
    'tree_probability_p10', 'tree_probability_p90',
    'verticality_mean', 'verticality_std',
    'linearity', 'planarity', 'scattering',
    'height_p10', 'height_p25', 'height_p50', 'height_p75', 'height_p90',
) + tuple(f'vertical_occupancy_{index}' for index in range(8)) + (
    'radius_lower', 'radius_middle', 'radius_upper',
)
EDGE_FEATURE_NAMES = (
    'delta_x', 'delta_y', 'delta_z', 'xy_distance', 'xyz_distance',
    'xy_bbox_gap', 'vertical_gap', 'vertical_overlap',
    'log_height_ratio', 'log_count_ratio', 'tree_probability_difference',
    'verticality_difference', 'vertical_histogram_cosine',
    'shared_xy_boundary', 'x_span_ratio', 'y_span_ratio', 'height_ratio',
)
VERTICAL_NODE_FEATURE_INDICES = tuple(
    index for index, name in enumerate(NODE_FEATURE_NAMES)
    if name.startswith('height_') or name.startswith('vertical_') or
    name in {'center_z', 'height', 'verticality_mean', 'verticality_std',
             'radius_lower', 'radius_middle', 'radius_upper'})


def _as_probability(values, tree_class_index=0):
    values = np.asarray(values)
    if values.ndim == 1:
        probabilities = values.astype(np.float32, copy=False)
    elif values.ndim == 2:
        probabilities = tree_probabilities(
            values.astype(np.float32, copy=False), tree_class_index)
    else:
        raise ValueError('Tree probability input must have shape [N] or [N, C].')
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('Tree probabilities contain non-finite values.')
    return probabilities


def coordinate_sha256(coords):
    coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    return hashlib.sha256(coords.view(np.uint8)).hexdigest()


def _bbox_gap(minimum_a, maximum_a, minimum_b, maximum_b):
    delta = np.maximum(
        np.maximum(minimum_a - maximum_b, minimum_b - maximum_a), 0.0)
    return float(np.linalg.norm(delta))


def _interval_gap(minimum_a, maximum_a, minimum_b, maximum_b):
    return float(max(minimum_a - maximum_b, minimum_b - maximum_a, 0.0))


def _principal_shape(points):
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    largest = max(float(eigenvalues[0]), 1e-9)
    return (
        float((eigenvalues[0] - eigenvalues[1]) / largest),
        float((eigenvalues[1] - eigenvalues[2]) / largest),
        float(eigenvalues[2] / largest),
    )


def _vertical_profile(points, minimum_z, maximum_z):
    height = max(float(maximum_z - minimum_z), 1e-6)
    normalized = np.clip((points[:, 2] - minimum_z) / height, 0.0, 1.0)
    bins = np.minimum((normalized * 8).astype(np.int64), 7)
    histogram = np.bincount(bins, minlength=8).astype(np.float64)
    histogram /= max(histogram.sum(), 1.0)
    center_xy = points[:, :2].mean(axis=0)
    radial = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    radii = []
    for lower, upper in ((0.0, 1/3), (1/3, 2/3), (2/3, 1.01)):
        mask = (normalized >= lower) & (normalized < upper)
        radii.append(float(np.sqrt(np.mean(radial[mask] ** 2)))
                     if np.any(mask) else 0.0)
    return histogram, radii


def build_vertical_micro_proposals(
        coords, semantic_values, verticality=None, tree_probability_threshold=0.5,
        xy_cell_size=0.6, z_bin_size=0.5, vertical_split_gap=1.0,
        tree_class_index=0):
    """Partition semantic-tree points into deterministic vertical columns."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coords must have shape [N, 3].')
    if not np.all(np.isfinite(coords)):
        raise ValueError('coords contain non-finite values.')
    probabilities = _as_probability(semantic_values, tree_class_index)
    if len(probabilities) != len(coords):
        raise ValueError('Semantic values and coordinates are not aligned.')
    if verticality is None:
        verticality = np.zeros(len(coords), dtype=np.float32)
    verticality = np.asarray(verticality, dtype=np.float32).reshape(-1)
    if len(verticality) != len(coords) or not np.all(np.isfinite(verticality)):
        raise ValueError('verticality must be finite and aligned with coords.')
    if min(xy_cell_size, z_bin_size, vertical_split_gap) <= 0:
        raise ValueError('Micro-proposal spatial parameters must be positive.')

    selected = np.flatnonzero(probabilities >= float(tree_probability_threshold))
    point_node_id = np.full(len(coords), -1, dtype=np.int32)
    empty = {
        'point_node_id': point_node_id,
        'node_features': np.empty((0, len(NODE_FEATURE_NAMES)), np.float32),
        'node_bounds': np.empty((0, 6), np.float32),
        'node_centers': np.empty((0, 3), np.float32),
        'node_point_counts': np.empty(0, np.int64),
        'vertical_histograms': np.empty((0, 8), np.float32),
    }
    if len(selected) == 0:
        return empty

    origin = coords[selected].min(axis=0)
    xy_bins = np.floor(
        (coords[selected, :2] - origin[:2]) / float(xy_cell_size)).astype(np.int64)
    z_bins = np.floor(
        (coords[selected, 2] - origin[2]) / float(z_bin_size)).astype(np.int64)
    order = np.lexsort((z_bins, xy_bins[:, 1], xy_bins[:, 0]))
    sorted_points = selected[order]
    sorted_xy = xy_bins[order]
    sorted_z_bins = z_bins[order]
    new_node = np.ones(len(sorted_points), dtype=bool)
    if len(sorted_points) > 1:
        same_xy = np.all(sorted_xy[1:] == sorted_xy[:-1], axis=1)
        maximum_bin_gap = max(int(np.ceil(
            float(vertical_split_gap) / float(z_bin_size))), 1)
        bin_gap = sorted_z_bins[1:] - sorted_z_bins[:-1]
        new_node[1:] = (~same_xy) | (bin_gap > maximum_bin_gap)
    sorted_node_ids = np.cumsum(new_node, dtype=np.int64) - 1
    point_node_id[sorted_points] = sorted_node_ids.astype(np.int32)
    starts = np.flatnonzero(new_node)
    ends = np.r_[starts[1:], len(sorted_points)]
    node_count = len(starts)

    node_features = np.zeros((node_count, len(NODE_FEATURE_NAMES)), np.float32)
    node_bounds = np.zeros((node_count, 6), np.float32)
    node_centers = np.zeros((node_count, 3), np.float32)
    node_counts = (ends - starts).astype(np.int64)
    histograms = np.zeros((node_count, 8), np.float32)
    for node_id, (start, end) in enumerate(zip(starts, ends)):
        indices = sorted_points[start:end]
        points = coords[indices]
        probs = probabilities[indices]
        vert = verticality[indices]
        minimum, maximum = points.min(0), points.max(0)
        center, spans = points.mean(0), maximum - minimum
        volume = max(float(np.prod(np.maximum(spans, .1))), 1e-6)
        shape = _principal_shape(points)
        quantiles = np.quantile(
            points[:, 2] - minimum[2], (.1, .25, .5, .75, .9))
        histogram, radii = _vertical_profile(points, minimum[2], maximum[2])
        values = (
            np.log1p(len(indices)), center[0]-origin[0],
            center[1]-origin[1], center[2]-origin[2],
            spans[0], spans[1], spans[2], np.log1p(len(indices)/volume),
            probs.mean(), probs.std(), np.quantile(probs, .1),
            np.quantile(probs, .9), vert.mean(), vert.std(), *shape,
            *quantiles, *histogram, *radii,
        )
        node_features[node_id] = np.asarray(values, np.float32)
        node_bounds[node_id] = np.r_[minimum, maximum]
        node_centers[node_id] = center
        histograms[node_id] = histogram
    if not np.all(np.isfinite(node_features)):
        raise ValueError('Generated node features contain non-finite values.')
    return {
        'point_node_id': point_node_id,
        'node_features': node_features,
        'node_bounds': node_bounds,
        'node_centers': node_centers,
        'node_point_counts': node_counts,
        'vertical_histograms': histograms,
    }


def build_proposal_graph(
        proposals, max_xy_distance=2.5, max_xy_bbox_gap=0.6,
        max_vertical_gap=1.5, max_neighbors=16):
    """Build a bounded directed graph without labels or predicted offsets."""
    centers = np.asarray(proposals['node_centers'], np.float64)
    bounds = np.asarray(proposals['node_bounds'], np.float64)
    features = np.asarray(proposals['node_features'], np.float64)
    counts = np.asarray(proposals['node_point_counts'], np.float64)
    histograms = np.asarray(proposals['vertical_histograms'], np.float64)
    if len(centers) < 2:
        return {
            **proposals,
            'edge_index': np.empty((2, 0), np.int64),
            'edge_features': np.empty((0, len(EDGE_FEATURE_NAMES)), np.float32),
        }
    if int(max_neighbors) <= 0:
        raise ValueError('max_neighbors must be positive.')
    k = min(len(centers), max(int(max_neighbors)*4+1, 2))
    distances, neighbors = cKDTree(centers[:, :2]).query(
        centers[:, :2], k=k, distance_upper_bound=float(max_xy_distance))
    if k == 1:
        distances, neighbors = distances[:, None], neighbors[:, None]
    edges, edge_values = [], []
    pi = NODE_FEATURE_NAMES.index('tree_probability_mean')
    vi = NODE_FEATURE_NAMES.index('verticality_mean')
    si = [NODE_FEATURE_NAMES.index(name)
          for name in ('x_span', 'y_span', 'height')]
    for source in range(len(centers)):
        accepted = 0
        for distance, target in zip(distances[source], neighbors[source]):
            target = int(target)
            if target == source or target >= len(centers) or not np.isfinite(distance):
                continue
            xy_gap = _bbox_gap(
                bounds[source, :2], bounds[source, 3:5],
                bounds[target, :2], bounds[target, 3:5])
            if distance > float(max_xy_distance) and xy_gap > float(max_xy_bbox_gap):
                continue
            vertical_gap = _interval_gap(
                bounds[source, 2], bounds[source, 5],
                bounds[target, 2], bounds[target, 5])
            if vertical_gap > float(max_vertical_gap):
                continue
            delta = centers[target] - centers[source]
            ha = max(bounds[source, 5]-bounds[source, 2], 1e-6)
            hb = max(bounds[target, 5]-bounds[target, 2], 1e-6)
            overlap = max(min(bounds[source, 5], bounds[target, 5]) -
                          max(bounds[source, 2], bounds[target, 2]), 0.0)
            norm = max(np.linalg.norm(histograms[source]) *
                       np.linalg.norm(histograms[target]), 1e-9)
            cosine = float(histograms[source] @ histograms[target] / norm)
            sa = np.maximum(features[source, si], 1e-6)
            sb = np.maximum(features[target, si], 1e-6)
            span_ratio = np.minimum(sa, sb) / np.maximum(sa, sb)
            edge_values.append((
                delta[0], delta[1], delta[2], distance, np.linalg.norm(delta),
                xy_gap, vertical_gap, overlap/max(min(ha, hb), 1e-6),
                np.log(ha/hb), np.log(max(counts[source], 1) /
                                     max(counts[target], 1)),
                features[target, pi]-features[source, pi],
                features[target, vi]-features[source, vi],
                cosine, float(xy_gap <= 1e-6), *span_ratio,
            ))
            edges.append((source, target))
            accepted += 1
            if accepted >= int(max_neighbors):
                break
    edge_index = (np.asarray(edges, np.int64).T if edges else
                  np.empty((2, 0), np.int64))
    edge_features = (np.asarray(edge_values, np.float32) if edges else
                     np.empty((0, len(EDGE_FEATURE_NAMES)), np.float32))
    if not np.all(np.isfinite(edge_features)):
        raise ValueError('Generated edge features contain non-finite values.')
    return {**proposals, 'edge_index': edge_index,
            'edge_features': edge_features}


def build_learning_targets(graph, instance_labels, minimum_node_purity=0.5):
    """Create GT targets separately from a label-free graph artifact."""
    point_node_id = np.asarray(graph['point_node_id'], np.int64)
    labels = np.asarray(instance_labels, np.int64).reshape(-1)
    if len(labels) != len(point_node_id):
        raise ValueError('GT labels and graph points are not aligned.')
    node_count = len(graph['node_features'])
    node_gt = np.zeros(node_count, np.int64)
    node_purity = np.zeros(node_count, np.float32)
    for node in range(node_count):
        values = labels[point_node_id == node]
        positive = values[values > 0]
        if len(positive):
            ids, counts = np.unique(positive, return_counts=True)
            best = int(np.argmax(counts))
            node_gt[node] = int(ids[best])
            node_purity[node] = float(counts[best]/max(len(values), 1))
    source, target = np.asarray(graph['edge_index'], np.int64)
    source_clean = node_purity[source] >= float(minimum_node_purity)
    target_clean = node_purity[target] >= float(minimum_node_purity)
    valid = source_clean & target_clean
    positive = (valid & (node_gt[source] > 0) &
                (node_gt[source] == node_gt[target]))
    return {
        'node_gt': node_gt,
        'node_purity': node_purity,
        'edge_label': positive.astype(np.int8),
        'edge_valid': valid.astype(bool),
        'covered_gt_ids': np.unique(node_gt[node_gt > 0]),
    }


class _UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value):
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return left
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return left


def decode_relation_graph(
        graph, edge_scores, threshold, reciprocal_top_k=2,
        max_component_xy_diameter=15.0, max_component_height=60.0):
    edge_index = np.asarray(graph['edge_index'], np.int64)
    scores = np.asarray(edge_scores, np.float64).reshape(-1)
    node_count = len(graph['node_features'])
    if edge_index.shape != (2, len(scores)):
        raise ValueError('Edge scores and edge_index are not aligned.')
    if node_count == 0:
        return np.empty(0, np.int64)
    outgoing = [[] for _ in range(node_count)]
    for edge, (source, target) in enumerate(edge_index.T):
        outgoing[int(source)].append((float(scores[edge]), int(target)))
    top_pairs = set()
    for source, rows in enumerate(outgoing):
        rows.sort(key=lambda row: (-row[0], row[1]))
        top_pairs.update((source, target) for _, target in
                         rows[:int(reciprocal_top_k)])
    pair_scores = {}
    for edge, (source, target) in enumerate(edge_index.T):
        source, target = int(source), int(target)
        if ((source, target) not in top_pairs or
                (target, source) not in top_pairs):
            continue
        pair = (min(source, target), max(source, target))
        pair_scores.setdefault(pair, []).append(float(scores[edge]))
    ranked = sorted(
        ((float(np.mean(values)), pair)
         for pair, values in pair_scores.items()
         if float(np.mean(values)) >= float(threshold)),
        key=lambda row: (-row[0], row[1]))
    bounds = np.asarray(graph['node_bounds'], np.float64)
    component_min, component_max = bounds[:, :3].copy(), bounds[:, 3:].copy()
    union = _UnionFind(node_count)
    for _, (left, right) in ranked:
        rl, rr = union.find(left), union.find(right)
        if rl == rr:
            continue
        minimum = np.minimum(component_min[rl], component_min[rr])
        maximum = np.maximum(component_max[rl], component_max[rr])
        if np.linalg.norm(maximum[:2]-minimum[:2]) > float(
                max_component_xy_diameter):
            continue
        if maximum[2]-minimum[2] > float(max_component_height):
            continue
        root = union.union(rl, rr)
        component_min[root], component_max[root] = minimum, maximum
    roots = np.asarray([union.find(node) for node in range(node_count)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)+1


def point_instances_from_nodes(point_node_id, node_instances):
    point_node_id = np.asarray(point_node_id, np.int64)
    node_instances = np.asarray(node_instances, np.int64)
    output = np.zeros(len(point_node_id), np.int64)
    selected = point_node_id >= 0
    if np.any(selected):
        if point_node_id[selected].max() >= len(node_instances):
            raise ValueError('point_node_id references a missing node.')
        output[selected] = node_instances[point_node_id[selected]]
    return output


def evaluate_instances(labels, predictions, match_iou_threshold=0.5,
                       min_precision_for_counted_fp=0.5):
    table = _contingency(labels, predictions)
    iou = table['iou']
    if iou.size:
        pred_indices, gt_indices = linear_sum_assignment(iou, maximize=True)
        keep = iou[pred_indices, gt_indices] > float(match_iou_threshold)
        matched_pred = set(map(int, pred_indices[keep]))
        matched_gt = set(map(int, gt_indices[keep]))
    else:
        matched_pred, matched_gt = set(), set()
    fraction = np.divide(
        table['intersections'].sum(1), table['pred_counts'],
        out=np.zeros(len(table['pred_counts']), np.float64),
        where=table['pred_counts'] > 0)
    fp = sum(index not in matched_pred and
             fraction[index] >= float(min_precision_for_counted_fp)
             for index in range(len(table['pred_ids'])))
    return (detection_metrics(
        len(matched_gt), fp, len(table['gt_ids'])-len(matched_gt)),
        matched_gt, table)


def evaluate_graph_oracle(graph, instance_labels, baseline_predictions,
                          match_iou_threshold=0.5,
                          min_precision_for_counted_fp=0.5):
    targets = build_learning_targets(graph, instance_labels)
    # P0 is a connectivity ceiling: every candidate same-tree edge is kept
    # and every candidate different-tree edge is rejected. Reciprocal and
    # component-safety constraints belong to deployable P2 decoding, not P0.
    union = _UnionFind(len(graph['node_features']))
    for edge, (source, target) in enumerate(graph['edge_index'].T):
        if targets['edge_label'][edge]:
            union.union(int(source), int(target))
    roots = np.asarray(
        [union.find(node) for node in range(len(graph['node_features']))])
    _, node_instances = np.unique(roots, return_inverse=True)
    node_instances = node_instances.astype(np.int64) + 1
    predictions = point_instances_from_nodes(
        graph['point_node_id'], node_instances)
    baseline, baseline_matches, baseline_table = evaluate_instances(
        instance_labels, baseline_predictions, match_iou_threshold,
        min_precision_for_counted_fp)
    oracle, oracle_matches, _ = evaluate_instances(
        instance_labels, predictions, match_iou_threshold,
        min_precision_for_counted_fp)
    missed = set(range(len(baseline_table['gt_ids'])))-set(baseline_matches)
    recovered = missed & set(oracle_matches)
    missed_ids = {int(baseline_table['gt_ids'][index]) for index in missed}
    covered = missed_ids & set(map(int, targets['covered_gt_ids']))
    return {
        'baseline': baseline, 'oracle': oracle,
        'recovered_trees': int(len(recovered)),
        'missed_trees': int(len(missed)),
        'graph_covered_missed_trees': int(len(covered)),
        'graph_coverage': float(len(covered)/max(len(missed), 1)),
        'num_nodes': int(len(graph['node_features'])),
        'num_edges': int(graph['edge_index'].shape[1]),
        'num_baseline_instances': int(len(baseline_table['pred_ids'])),
        'average_out_degree': float(
            graph['edge_index'].shape[1]/max(len(graph['node_features']), 1)),
        'predictions': predictions, 'targets': targets,
    }


def save_inference_graph(path, graph, metadata):
    forbidden = ('gt', 'label', 'target', 'truth')
    for key in graph:
        if any(token in key.lower() for token in forbidden):
            raise ValueError(f'Inference graph contains forbidden key: {key}')
    if any(any(token in key.lower() for token in forbidden) for key in metadata):
        raise ValueError('Inference metadata contains a GT-like key.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = dict(metadata)
    clean.update({
        'graph_version': GRAPH_VERSION,
        'node_feature_names': list(NODE_FEATURE_NAMES),
        'edge_feature_names': list(EDGE_FEATURE_NAMES),
    })
    arrays = {key: value for key, value in graph.items()
              if isinstance(value, np.ndarray)}
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(clean)))


def load_inference_graph(path):
    with np.load(path, allow_pickle=False) as payload:
        graph = {key: payload[key] for key in payload.files
                 if key != 'metadata_json'}
        metadata = json.loads(str(payload['metadata_json']))
    if int(metadata.get('graph_version', -1)) != GRAPH_VERSION:
        raise ValueError('Unsupported proposal graph version.')
    return graph, metadata


if torch is not None:
    class RelationAttentionLayer(nn.Module):
        def __init__(self, node_dim, edge_dim, num_heads=4, ffn_dim=128,
                     dropout=.1):
            super().__init__()
            if node_dim % num_heads:
                raise ValueError('node_dim must be divisible by num_heads.')
            self.num_heads = int(num_heads)
            self.head_dim = node_dim//num_heads
            self.norm1 = nn.LayerNorm(node_dim)
            self.query = nn.Linear(node_dim, node_dim)
            self.key = nn.Linear(node_dim, node_dim)
            self.value = nn.Linear(node_dim, node_dim)
            self.edge_score = nn.Linear(edge_dim, num_heads)
            self.edge_value = nn.Linear(edge_dim, node_dim)
            self.output = nn.Linear(node_dim, node_dim)
            self.dropout = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(node_dim)
            self.ffn = nn.Sequential(
                nn.Linear(node_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ffn_dim, node_dim))

        @staticmethod
        def _segment_softmax(values, index, size):
            maximum = torch.full(
                (size, values.shape[1]), -torch.inf,
                dtype=values.dtype, device=values.device)
            expanded = index[:, None].expand_as(values)
            maximum.scatter_reduce_(
                0, expanded, values, reduce='amax', include_self=True)
            stabilized = torch.exp(values-maximum[index])
            denominator = torch.zeros_like(maximum)
            denominator.index_add_(0, index, stabilized)
            return stabilized/denominator[index].clamp_min(1e-9)

        def forward(self, nodes, edge_index, edges, return_attention=False):
            if edge_index.numel() == 0:
                output = nodes+self.ffn(self.norm2(nodes))
                empty = nodes.new_empty((0, self.num_heads))
                return (output, empty) if return_attention else output
            normalized = self.norm1(nodes)
            source, target = edge_index
            shape = (-1, self.num_heads, self.head_dim)
            query = self.query(normalized)[target].view(shape)
            key = self.key(normalized)[source].view(shape)
            value = self.value(normalized)[source].view(shape)
            edge_value = self.edge_value(edges).view(shape)
            logits = ((query*key).sum(-1)/self.head_dim**.5 +
                      self.edge_score(edges))
            attention = self._segment_softmax(logits, target, len(nodes))
            messages = attention[..., None]*(value+edge_value)
            aggregated = nodes.new_zeros(
                (len(nodes), self.num_heads, self.head_dim))
            aggregated.index_add_(0, target, messages)
            nodes = nodes+self.dropout(self.output(aggregated.flatten(1)))
            nodes = nodes+self.dropout(self.ffn(self.norm2(nodes)))
            return (nodes, attention) if return_attention else nodes


    class VerticalRelationAttention(nn.Module):
        def __init__(self, node_input_dim=len(NODE_FEATURE_NAMES),
                     edge_input_dim=len(EDGE_FEATURE_NAMES), node_dim=64,
                     edge_dim=32, num_layers=2, num_heads=4, ffn_dim=128,
                     dropout=.1, use_vertical=True):
            super().__init__()
            self.use_vertical = bool(use_vertical)
            self.node_input = nn.Sequential(
                nn.LayerNorm(node_input_dim), nn.Linear(node_input_dim, node_dim),
                nn.GELU())
            self.edge_input = nn.Sequential(
                nn.LayerNorm(edge_input_dim), nn.Linear(edge_input_dim, edge_dim),
                nn.GELU())
            self.layers = nn.ModuleList([
                RelationAttentionLayer(
                    node_dim, edge_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)])
            self.edge_head = nn.Sequential(
                nn.Linear(node_dim*2+edge_dim, node_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(node_dim, 1))
            self.reliability_head = nn.Linear(node_dim, 1)

        def forward(self, node_features, edge_index, edge_features,
                    return_attention=False):
            if not self.use_vertical:
                node_features = node_features.clone()
                node_features[:, VERTICAL_NODE_FEATURE_INDICES] = 0
            nodes = self.node_input(node_features)
            edges = self.edge_input(edge_features)
            attentions = []
            for layer in self.layers:
                if return_attention:
                    nodes, attention = layer(
                        nodes, edge_index, edges, return_attention=True)
                    attentions.append(attention)
                else:
                    nodes = layer(nodes, edge_index, edges)
            if edge_index.numel():
                source, target = edge_index
                logits = self.edge_head(torch.cat(
                    (nodes[source], nodes[target], edges), 1)).squeeze(1)
            else:
                logits = nodes.new_empty(0)
            result = {
                'edge_logits': logits, 'node_embeddings': nodes,
                'node_reliability_logits':
                    self.reliability_head(nodes).squeeze(1)}
            if return_attention:
                result['attention'] = attentions
            return result


    class EdgeMLP(nn.Module):
        def __init__(self, node_input_dim=len(NODE_FEATURE_NAMES),
                     edge_input_dim=len(EDGE_FEATURE_NAMES), hidden_dim=96,
                     depth=3, dropout=.1):
            super().__init__()
            current = node_input_dim*2+edge_input_dim
            blocks = []
            for _ in range(int(depth)):
                blocks.extend((nn.Linear(current, hidden_dim), nn.GELU(),
                               nn.Dropout(dropout)))
                current = hidden_dim
            self.encoder = nn.Sequential(*blocks)
            self.edge_head = nn.Linear(current, 1)

        def forward(self, node_features, edge_index, edge_features,
                    return_attention=False):
            if edge_index.numel():
                source, target = edge_index
                hidden = self.encoder(torch.cat(
                    (node_features[source], node_features[target],
                     edge_features), 1))
                logits = self.edge_head(hidden).squeeze(1)
            else:
                logits = node_features.new_empty(0)
            return {
                'edge_logits': logits, 'node_embeddings': node_features,
                'node_reliability_logits':
                    node_features.new_zeros(len(node_features))}


    def focal_edge_loss(logits, targets, gamma=2., positive_weight=1.):
        targets = targets.to(logits.dtype)
        base = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        probability = torch.sigmoid(logits)
        pt = torch.where(targets > .5, probability, 1-probability)
        weight = torch.where(
            targets > .5,
            torch.as_tensor(positive_weight, dtype=logits.dtype,
                            device=logits.device), 1.)
        return (weight*(1-pt).pow(gamma)*base).mean()


    def supervised_contrastive_loss(embeddings, labels, temperature=.1):
        valid = labels > 0
        embeddings, labels = embeddings[valid], labels[valid]
        if len(embeddings) < 2:
            return embeddings.sum()*0
        embeddings = F.normalize(embeddings, dim=1)
        similarity = embeddings@embeddings.T/float(temperature)
        identity = torch.eye(
            len(labels), dtype=torch.bool, device=labels.device)
        positive = (labels[:, None] == labels[None, :]) & ~identity
        if not torch.any(positive):
            return embeddings.sum()*0
        logits = similarity.masked_fill(identity, -torch.inf)
        log_probability = logits-torch.logsumexp(logits, 1, keepdim=True)
        per_row = -(log_probability.masked_fill(~positive, 0).sum(1) /
                    positive.sum(1).clamp_min(1))
        return per_row[positive.any(1)].mean()


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_parameter_matched_edge_mlp(target_parameters, tolerance=.05, **kwargs):
    if torch is None:
        raise RuntimeError('PyTorch is required for EdgeMLP.')
    candidates = []
    for depth in (2, 3, 4):
        for width in range(16, 513, 4):
            model = EdgeMLP(hidden_dim=width, depth=depth, **kwargs)
            count = parameter_count(model)
            candidates.append((abs(count-target_parameters), count, model))
    _, count, model = min(candidates, key=lambda row: row[0])
    difference = abs(count-target_parameters)/max(target_parameters, 1)
    if difference > float(tolerance):
        raise ValueError(
            f'Cannot parameter-match EdgeMLP: target={target_parameters}, '
            f'control={count}.')
    return model
