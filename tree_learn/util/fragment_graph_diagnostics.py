"""Restricted fragment-graph Oracle diagnostics.

The graph is built without ground-truth labels. Ground truth is used only to
identify the fixed Q3 fragmentation targets and to accept safe merge groups,
so the result is a proposal upper bound rather than a deployable method.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .omission_diagnostics import _contingency, detection_metrics
from .seed_quality import tree_probabilities


MERGE_MODES = (
    'fragment_union_ceiling',
    'fragment_graph_oracle',
)


def _evaluate_table(
        intersections, pred_counts, gt_counts, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5):
    intersections = np.asarray(intersections, dtype=np.int64)
    pred_counts = np.asarray(pred_counts, dtype=np.int64).reshape(-1)
    gt_counts = np.asarray(gt_counts, dtype=np.int64).reshape(-1)
    if intersections.shape != (len(pred_counts), len(gt_counts)):
        raise ValueError('Fragment contingency arrays are not aligned.')
    unions = pred_counts[:, None] + gt_counts[None, :] - intersections
    iou = np.divide(
        intersections, unions,
        out=np.zeros_like(intersections, dtype=np.float64), where=unions > 0)
    if iou.size:
        pred_index, gt_index = linear_sum_assignment(iou, maximize=True)
        valid = iou[pred_index, gt_index] > float(match_iou_threshold)
        matched_pred = set(map(int, pred_index[valid]))
        matched_gt = set(map(int, gt_index[valid]))
    else:
        matched_pred, matched_gt = set(), set()
    labeled_fraction = np.divide(
        intersections.sum(axis=1), pred_counts,
        out=np.zeros(len(pred_counts), dtype=np.float64),
        where=pred_counts > 0)
    false_positives = sum(
        index not in matched_pred and
        labeled_fraction[index] >= float(min_precision_for_counted_fp)
        for index in range(len(pred_counts)))
    metrics = detection_metrics(
        len(matched_gt), false_positives, len(gt_counts) - len(matched_gt))
    return metrics, matched_gt, matched_pred


def merge_contingency_groups(table, groups):
    """Merge disjoint groups of prediction rows and return a new table."""
    num_predictions = len(table['pred_ids'])
    parent = np.arange(num_predictions, dtype=np.int64)

    def find(index):
        index = int(index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    used = set()
    for group in groups:
        members = sorted(set(map(int, group)))
        if len(members) < 2:
            raise ValueError('Every fragment merge group needs two members.')
        if members[0] < 0 or members[-1] >= num_predictions:
            raise ValueError('Fragment merge group index is invalid.')
        overlap = used & set(members)
        if overlap:
            raise ValueError(
                f'Fragment merge groups must be disjoint: {sorted(overlap)}.')
        used.update(members)
        root = find(members[0])
        for member in members[1:]:
            parent[find(member)] = root

    roots = np.asarray([find(index) for index in range(num_predictions)])
    unique_roots = np.unique(roots)
    positions = np.searchsorted(unique_roots, roots)
    intersections = np.zeros(
        (len(unique_roots), len(table['gt_ids'])), dtype=np.int64)
    pred_counts = np.zeros(len(unique_roots), dtype=np.int64)
    np.add.at(intersections, positions, table['intersections'])
    np.add.at(pred_counts, positions, table['pred_counts'])
    return {
        'gt_ids': np.asarray(table['gt_ids'], dtype=np.int64),
        'gt_counts': np.asarray(table['gt_counts'], dtype=np.int64),
        'pred_ids': np.asarray(table['pred_ids'], dtype=np.int64)[unique_roots],
        'pred_counts': pred_counts,
        'intersections': intersections,
    }


def _instance_geometry(
        coords, semantic_logits, offset_predictions, verticality,
        instance_predictions, pred_ids, tree_class_index=0):
    coords = np.asarray(coords, dtype=np.float64)
    offsets = np.asarray(offset_predictions, dtype=np.float64)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    verticality = np.asarray(verticality, dtype=np.float64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('Fragment coordinates must have shape [N, 3].')
    if offsets.shape != coords.shape or len(predictions) != len(coords) or \
            len(verticality) != len(coords):
        raise ValueError('Fragment pointwise arrays are not aligned.')
    logits = np.asarray(semantic_logits, dtype=np.float32)
    if logits.ndim != 2 or len(logits) != len(coords):
        raise ValueError('Fragment semantic logits must have shape [N, C].')

    positive = predictions > 0
    point_pred_ids = predictions[positive]
    positions = np.searchsorted(pred_ids, point_pred_ids)
    if np.any(positions >= len(pred_ids)) or np.any(
            pred_ids[positions] != point_pred_ids):
        raise ValueError('Prediction IDs differ between geometry and table.')
    count = len(pred_ids)
    point_counts = np.bincount(positions, minlength=count).astype(np.int64)
    xy = coords[positive, :2]
    z = coords[positive, 2]
    votes = xy + offsets[positive, :2]
    tree_prob = tree_probabilities(logits, tree_class_index)[positive]
    vert = verticality[positive]

    xy_min = np.full((count, 2), np.inf, dtype=np.float64)
    xy_max = np.full((count, 2), -np.inf, dtype=np.float64)
    z_min = np.full(count, np.inf, dtype=np.float64)
    z_max = np.full(count, -np.inf, dtype=np.float64)
    np.minimum.at(xy_min, positions, xy)
    np.maximum.at(xy_max, positions, xy)
    np.minimum.at(z_min, positions, z)
    np.maximum.at(z_max, positions, z)

    sums_xy = np.zeros((count, 2), dtype=np.float64)
    sums_vote = np.zeros((count, 2), dtype=np.float64)
    sums_prob = np.zeros(count, dtype=np.float64)
    sums_vert = np.zeros(count, dtype=np.float64)
    np.add.at(sums_xy, positions, xy)
    np.add.at(sums_vote, positions, votes)
    np.add.at(sums_prob, positions, tree_prob)
    np.add.at(sums_vert, positions, vert)
    denominator = np.maximum(point_counts, 1)
    raw_centers = sums_xy / denominator[:, None]
    vote_centers = sums_vote / denominator[:, None]
    tree_probability_mean = sums_prob / denominator
    verticality_mean = sums_vert / denominator
    vote_residual = votes - vote_centers[positions]
    vote_squared = np.zeros(count, dtype=np.float64)
    np.add.at(vote_squared, positions, np.sum(vote_residual ** 2, axis=1))
    vote_radius_rms = np.sqrt(vote_squared / denominator)
    return {
        'point_counts': point_counts,
        'raw_centers': raw_centers,
        'vote_centers': vote_centers,
        'xy_min': xy_min,
        'xy_max': xy_max,
        'z_min': z_min,
        'z_max': z_max,
        'tree_probability_mean': tree_probability_mean,
        'verticality_mean': verticality_mean,
        'vote_radius_rms': vote_radius_rms,
    }


def _bbox_gap(minimum_a, maximum_a, minimum_b, maximum_b):
    separation = np.maximum(
        np.maximum(minimum_a - maximum_b, minimum_b - maximum_a), 0.0)
    return float(np.linalg.norm(separation))


def _interval_gap(minimum_a, maximum_a, minimum_b, maximum_b):
    return float(max(minimum_a - maximum_b, minimum_b - maximum_a, 0.0))


def build_fragment_adjacency(geometry, settings):
    """Build deterministic GT-free edges in base-vote/geometry space."""
    vote_centers = np.asarray(geometry['vote_centers'], dtype=np.float64)
    if len(vote_centers) < 2:
        return []
    maximum_vote_distance = float(settings['max_vote_center_distance'])
    raw_pairs = cKDTree(vote_centers).query_pairs(
        r=maximum_vote_distance, output_type='ndarray')
    raw_pairs = np.asarray(raw_pairs, dtype=np.int64).reshape(-1, 2)
    edges = []
    for left, right in raw_pairs:
        raw_distance = float(np.linalg.norm(
            geometry['raw_centers'][left] - geometry['raw_centers'][right]))
        if raw_distance > float(settings['max_raw_center_distance']):
            continue
        xy_gap = _bbox_gap(
            geometry['xy_min'][left], geometry['xy_max'][left],
            geometry['xy_min'][right], geometry['xy_max'][right])
        if xy_gap > float(settings['max_xy_bbox_gap']):
            continue
        vertical_gap = _interval_gap(
            geometry['z_min'][left], geometry['z_max'][left],
            geometry['z_min'][right], geometry['z_max'][right])
        if vertical_gap > float(settings['max_vertical_gap']):
            continue
        vote_distance = float(np.linalg.norm(
            vote_centers[left] - vote_centers[right]))
        edges.append({
            'left': int(left),
            'right': int(right),
            'vote_center_distance': vote_distance,
            'raw_center_distance': raw_distance,
            'xy_bbox_gap': xy_gap,
            'vertical_gap': vertical_gap,
        })
    edges.sort(key=lambda row: (row['left'], row['right']))
    maximum_edges = int(settings.get('max_candidate_edges', 100000))
    if len(edges) > maximum_edges:
        raise RuntimeError(
            f'Fragment graph produced {len(edges):,} edges, exceeding '
            f'max_candidate_edges={maximum_edges:,}.')
    return edges


def _connected_fragment_groups(fragment_indices, edges):
    members = sorted(set(map(int, fragment_indices)))
    member_set = set(members)
    neighbors = defaultdict(set)
    for edge in edges:
        left, right = int(edge['left']), int(edge['right'])
        if left in member_set and right in member_set:
            neighbors[left].add(right)
            neighbors[right].add(left)
    groups = []
    visited = set()
    for start in members:
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current], reverse=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= 2:
            groups.append(tuple(sorted(component)))
    return groups


def _oracle_select_groups(
        table, proposals, baseline_matched_gt, match_iou_threshold,
        min_precision_for_counted_fp):
    base_metrics, _, _ = _evaluate_table(
        table['intersections'], table['pred_counts'], table['gt_counts'],
        match_iou_threshold, min_precision_for_counted_fp)
    ranked = []
    for proposal in proposals:
        merged = merge_contingency_groups(table, [proposal['members']])
        metrics, matched_gt, _ = _evaluate_table(
            merged['intersections'], merged['pred_counts'],
            merged['gt_counts'], match_iou_threshold,
            min_precision_for_counted_fp)
        target_index = int(proposal['target_gt_index'])
        if target_index not in matched_gt:
            continue
        if not set(baseline_matched_gt).issubset(matched_gt):
            continue
        ranked.append((
            metrics['f1'] - base_metrics['f1'],
            metrics['completeness'] - base_metrics['completeness'],
            -len(proposal['members']), -int(proposal['target_gt_id']),
            proposal))
    ranked.sort(key=lambda row: row[:-1], reverse=True)

    accepted = []
    used_predictions = set()
    accepted_targets = set()
    current_metrics = base_metrics
    current_matched_gt = set(baseline_matched_gt)
    for *_, proposal in ranked:
        target_id = int(proposal['target_gt_id'])
        members = set(map(int, proposal['members']))
        if target_id in accepted_targets or members & used_predictions:
            continue
        candidate_groups = [row['members'] for row in accepted] + [
            proposal['members']]
        merged = merge_contingency_groups(table, candidate_groups)
        metrics, matched_gt, _ = _evaluate_table(
            merged['intersections'], merged['pred_counts'],
            merged['gt_counts'], match_iou_threshold,
            min_precision_for_counted_fp)
        target_index = int(proposal['target_gt_index'])
        safe = (
            target_index in matched_gt and
            set(baseline_matched_gt).issubset(matched_gt) and
            metrics['f1'] > current_metrics['f1'] + 1e-12)
        if not safe:
            continue
        accepted.append(proposal)
        used_predictions.update(members)
        accepted_targets.add(target_id)
        current_metrics = metrics
        current_matched_gt = matched_gt
    return accepted, current_metrics, current_matched_gt


def analyze_fragment_graph_oracle(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, instance_predictions,
        expected_fragmentation_gt_ids, adjacency_settings,
        tree_class_index=0, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5,
        min_fragment_overlap_fraction=0.05,
        min_fragment_overlap_points=20):
    """Compare an unrestricted union ceiling with a GT-free graph Oracle."""
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    if len(labels) != len(predictions):
        raise ValueError('Fragment GT and prediction labels are not aligned.')
    table_raw = _contingency(labels, predictions)
    table = {
        'gt_ids': table_raw['gt_ids'],
        'gt_counts': table_raw['gt_counts'],
        'pred_ids': table_raw['pred_ids'],
        'pred_counts': table_raw['pred_counts'],
        'intersections': table_raw['intersections'],
    }
    expected_ids = sorted(set(map(int, expected_fragmentation_gt_ids)))
    if len(expected_ids) != len(list(expected_fragmentation_gt_ids)):
        raise ValueError('Expected fragmentation GT IDs contain duplicates.')
    missing = sorted(set(expected_ids) - set(map(int, table['gt_ids'])))
    if missing:
        raise ValueError(f'Fragmentation GT IDs are absent: {missing}.')

    baseline, baseline_matched_gt, _ = _evaluate_table(
        table['intersections'], table['pred_counts'], table['gt_counts'],
        match_iou_threshold, min_precision_for_counted_fp)
    geometry = _instance_geometry(
        coords, semantic_logits, offset_predictions, verticality,
        predictions, table['pred_ids'], tree_class_index)
    edges = build_fragment_adjacency(geometry, adjacency_settings)

    unrestricted = []
    graph_proposals = []
    per_target = []
    edge_pairs = {(row['left'], row['right']) for row in edges}
    for gt_id in expected_ids:
        gt_index = int(np.searchsorted(table['gt_ids'], gt_id))
        gt_count = int(table['gt_counts'][gt_index])
        threshold = max(
            int(min_fragment_overlap_points),
            int(np.ceil(float(min_fragment_overlap_fraction) * gt_count)))
        fragment_indices = np.flatnonzero(
            table['intersections'][:, gt_index] >= threshold)
        base = {
            'target_gt_id': int(gt_id),
            'target_gt_index': gt_index,
        }
        if len(fragment_indices) >= 2:
            unrestricted.append({
                **base, 'members': tuple(map(int, fragment_indices))})
        components = (
            _connected_fragment_groups(fragment_indices, edges)
            if len(fragment_indices) >= 2 else [])
        for component in components:
            graph_proposals.append({**base, 'members': component})
        direct_edges = sum(
            (min(int(left), int(right)), max(int(left), int(right))) in
            edge_pairs
            for offset, left in enumerate(fragment_indices)
            for right in fragment_indices[offset + 1:])
        per_target.append({
            'gt_tree_id': int(gt_id),
            'significant_fragments': int(len(fragment_indices)),
            'candidate_edges': int(direct_edges),
            'candidate_components': int(len(components)),
            'graph_covered': bool(components),
        })

    ceiling_accepted, ceiling_metrics, ceiling_matched = (
        _oracle_select_groups(
            table, unrestricted, baseline_matched_gt,
            match_iou_threshold, min_precision_for_counted_fp))
    graph_accepted, graph_metrics, graph_matched = _oracle_select_groups(
        table, graph_proposals, baseline_matched_gt,
        match_iou_threshold, min_precision_for_counted_fp)
    expected_indices = {
        int(np.searchsorted(table['gt_ids'], gt_id)) for gt_id in expected_ids}

    def mode_result(accepted, metrics, matched):
        recovered_ids = sorted(
            int(table['gt_ids'][index]) for index in
            (set(matched) - set(baseline_matched_gt)) & expected_indices)
        lost_ids = sorted(
            int(table['gt_ids'][index]) for index in
            set(baseline_matched_gt) - set(matched))
        return {
            **metrics,
            'accepted_groups': int(len(accepted)),
            'merged_predictions': int(sum(
                len(row['members']) for row in accepted)),
            'recovered_fragmentation_trees': int(len(recovered_ids)),
            'recovered_gt_ids': recovered_ids,
            'lost_baseline_trees': int(len(lost_ids)),
            'lost_baseline_gt_ids': lost_ids,
            'accepted': [{
                'target_gt_id': int(row['target_gt_id']),
                'prediction_ids': [
                    int(table['pred_ids'][index]) for index in row['members']],
            } for row in accepted],
        }

    return {
        'baseline': baseline,
        'num_fragmentation_gt_trees': int(len(expected_ids)),
        'expected_fragmentation_gt_ids': expected_ids,
        'num_predicted_instances': int(len(table['pred_ids'])),
        'num_candidate_edges': int(len(edges)),
        'num_graph_proposals': int(len(graph_proposals)),
        'graph_covered_target_trees': int(sum(
            row['graph_covered'] for row in per_target)),
        'per_target': per_target,
        'modes': {
            'fragment_union_ceiling': mode_result(
                ceiling_accepted, ceiling_metrics, ceiling_matched),
            'fragment_graph_oracle': mode_result(
                graph_accepted, graph_metrics, graph_matched),
        },
        'adjacency_settings': {
            key: (int(value) if key == 'max_candidate_edges' else float(value))
            for key, value in adjacency_settings.items()
        },
        'parameters': {
            'tree_class_index': int(tree_class_index),
            'match_iou_threshold': float(match_iou_threshold),
            'min_precision_for_counted_fp': float(
                min_precision_for_counted_fp),
            'min_fragment_overlap_fraction': float(
                min_fragment_overlap_fraction),
            'min_fragment_overlap_points': int(
                min_fragment_overlap_points),
        },
    }
