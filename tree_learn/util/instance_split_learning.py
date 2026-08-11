"""Learning artifacts and exact evaluation for deployable instance splits.

The Q4b1 artifact is self-contained because instance identifiers are only
meaningful inside one TreeLearn run. Features, targets, Raw-XY proposals and
the base contingency table are therefore produced together.
"""

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .instance_diagnostics import compute_instance_features
from .instance_quality import compute_vertical_instance_tokens
from .instance_split_diagnostics import (
    _fit_kmeans,
    _indices_by_prediction,
    _partition_quality,
    identify_undersegmentation_targets,
    split_features,
)
from .omission_diagnostics import (
    _contingency,
    compute_gt_omission_diagnostics,
    detection_metrics,
)


SPLIT_LEARNING_ARTIFACT = 'instance_split_learning.npz'
SPLIT_LEARNING_METADATA = 'metadata.json'
SPLIT_LEARNING_VERSION = 1

PROPOSAL_FEATURE_NAMES = (
    'log1p_num_points',
    'num_children',
    'min_child_fraction',
    'max_child_fraction',
    'child_fraction_std',
    'xy_centroid_separation_min_norm',
    'xy_centroid_separation_mean_norm',
    'xy_centroid_separation_max_norm',
    'within_xy_sse_ratio',
    'within_vote_sse_ratio',
    'height_overlap_mean',
    'height_overlap_max',
    'child_height_cv',
    'child_point_count_cv',
    'parent_xy_radius_rms',
    'parent_vote_radius_rms',
)


def _rms_radius(xy):
    values = np.asarray(xy, dtype=np.float64)
    center = values.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum((values - center) ** 2, axis=1))))


def _pairwise_distances(values):
    points = np.asarray(values, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(1, dtype=np.float64)
    distances = np.sqrt(np.sum(
        (points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    return distances[np.triu_indices(len(points), k=1)]


def _height_overlaps(z_min, z_max):
    overlaps = []
    for first in range(len(z_min)):
        for second in range(first + 1, len(z_min)):
            intersection = max(
                0.0, min(z_max[first], z_max[second]) -
                max(z_min[first], z_min[second]))
            union = max(z_max[first], z_max[second]) - min(
                z_min[first], z_min[second])
            overlaps.append(intersection / max(union, 1e-6))
    return np.asarray(overlaps or [0.0], dtype=np.float64)


def proposal_geometry_features(coords, offsets, partition):
    """Describe one Raw-XY proposal without using ground-truth labels."""
    xyz = np.asarray(coords, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    labels = np.asarray(partition, dtype=np.int64).reshape(-1)
    if xyz.shape != offsets.shape or xyz.shape != (len(labels), 3):
        raise ValueError('Proposal geometry arrays are not aligned.')
    child_ids, child_counts = np.unique(labels, return_counts=True)
    if len(child_ids) < 2:
        raise ValueError('A split proposal needs at least two children.')

    raw_xy = xyz[:, :2]
    votes_xy = raw_xy + offsets[:, :2]
    parent_xy_radius = max(_rms_radius(raw_xy), 1e-6)
    parent_vote_radius = max(_rms_radius(votes_xy), 1e-6)
    fractions = child_counts.astype(np.float64) / len(labels)
    xy_centroids = []
    z_min = []
    z_max = []
    heights = []
    within_xy = 0.0
    within_vote = 0.0
    for child in child_ids:
        mask = labels == child
        child_xy = raw_xy[mask]
        child_votes = votes_xy[mask]
        xy_center = child_xy.mean(axis=0)
        vote_center = child_votes.mean(axis=0)
        xy_centroids.append(xy_center)
        within_xy += float(np.sum((child_xy - xy_center) ** 2))
        within_vote += float(np.sum((child_votes - vote_center) ** 2))
        low = float(xyz[mask, 2].min())
        high = float(xyz[mask, 2].max())
        z_min.append(low)
        z_max.append(high)
        heights.append(high - low)

    xy_separation = _pairwise_distances(xy_centroids) / parent_xy_radius
    parent_xy_sse = float(np.sum(
        (raw_xy - raw_xy.mean(axis=0, keepdims=True)) ** 2))
    parent_vote_sse = float(np.sum(
        (votes_xy - votes_xy.mean(axis=0, keepdims=True)) ** 2))
    overlaps = _height_overlaps(z_min, z_max)
    heights = np.asarray(heights, dtype=np.float64)
    values = np.asarray([
        np.log1p(len(labels)),
        len(child_ids),
        fractions.min(),
        fractions.max(),
        fractions.std(),
        xy_separation.min(),
        xy_separation.mean(),
        xy_separation.max(),
        within_xy / max(parent_xy_sse, 1e-6),
        within_vote / max(parent_vote_sse, 1e-6),
        overlaps.mean(),
        overlaps.max(),
        heights.std() / max(heights.mean(), 1e-6),
        child_counts.std() / max(child_counts.mean(), 1e-6),
        parent_xy_radius,
        parent_vote_radius,
    ], dtype=np.float32)
    if len(values) != len(PROPOSAL_FEATURE_NAMES) or not np.isfinite(
            values).all():
        raise ValueError('Proposal features are invalid.')
    return values


def _child_contingency(instance_labels, point_indices, partition, gt_ids):
    local_labels = np.asarray(instance_labels, dtype=np.int64)[point_indices]
    child_ids, child_counts = np.unique(partition, return_counts=True)
    intersections = np.zeros((len(child_ids), len(gt_ids)), dtype=np.int32)
    valid = local_labels > 0
    if np.any(valid):
        child_indices = np.searchsorted(child_ids, partition[valid])
        gt_indices = np.searchsorted(gt_ids, local_labels[valid])
        np.add.at(intersections, (child_indices, gt_indices), 1)
    return child_counts.astype(np.int64), intersections


def _align_instance_features(
        coords, predictions, initial_predictions, semantic_logits, offsets,
        verticality, axis_confidence, backbone_features, num_layers):
    confidence = (
        np.zeros(len(predictions), dtype=np.float32)
        if axis_confidence is None else np.asarray(axis_confidence).reshape(-1))
    frame = compute_instance_features(
        coords=coords,
        instance_predictions=predictions,
        initial_instance_predictions=initial_predictions,
        semantic_prediction_logits=semantic_logits,
        offset_predictions=offsets,
        verticality=verticality,
        axis_confidence=confidence,
        tree_class_index=0)
    tokens = compute_vertical_instance_tokens(
        coords=coords,
        instance_predictions=predictions,
        backbone_features=backbone_features,
        semantic_prediction_logits=semantic_logits,
        offset_predictions=offsets,
        verticality=verticality,
        axis_confidence=confidence,
        num_layers=num_layers,
        tree_class_index=0)
    instance_ids = tokens['instance_ids'].astype(np.int64)
    indexed = frame.set_index('instance_id')
    if set(instance_ids.tolist()) != set(indexed.index.astype(int)):
        raise ValueError('Global features and vertical tokens do not align.')
    indexed = indexed.loc[instance_ids]
    return {
        'instance_ids': instance_ids,
        'global_feature_values': indexed.to_numpy(dtype=np.float32),
        'global_feature_names': np.asarray(list(indexed.columns)),
        'vertical_tokens': tokens['vertical_tokens'].astype(np.float32),
        'layer_valid_mask': tokens['layer_valid_mask'].astype(bool),
        'token_feature_names': np.asarray(tokens['token_feature_names']),
    }


def build_instance_split_learning_artifact(
        coords, semantic_logits, offset_predictions, instance_labels,
        instance_predictions, initial_instance_predictions, backbone_features,
        verticality, axis_confidence=None, split='train', num_layers=8,
        min_parent_points=100, max_children=4, max_fit_points=50000,
        random_state=20260811, max_train_negative_parents_per_plot=500,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0, tau_min=50,
        match_iou_threshold=0.5, min_precision_for_counted_fp=0.5,
        min_recall_for_undersegmentation=0.5,
        min_fragment_overlap_fraction=0.05,
        min_fragment_overlap_points=20,
        expected_undersegmented_gt_ids=None):
    """Build one compact forest artifact for the Q4b1 three-head control."""
    xyz = np.asarray(coords)
    logits = np.asarray(semantic_logits)
    offsets = np.asarray(offset_predictions)
    gt = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    initial = np.asarray(
        initial_instance_predictions, dtype=np.int64).reshape(-1)
    verticality = np.asarray(verticality).reshape(-1)
    if xyz.shape != offsets.shape or xyz.shape != (len(gt), 3):
        raise ValueError('Split-learning point arrays are not aligned.')
    if any(len(values) != len(gt) for values in (
            logits, predictions, initial, verticality, backbone_features)):
        raise ValueError('Split-learning prediction arrays are not aligned.')
    if split not in {'train', 'validation'}:
        raise ValueError('split must be train or validation.')
    if int(max_children) < 2:
        raise ValueError('max_children must be at least two.')

    features = _align_instance_features(
        xyz, predictions, initial, logits, offsets, verticality,
        axis_confidence, backbone_features, int(num_layers))
    instance_ids = features['instance_ids']
    id_to_index = {
        int(instance_id): index
        for index, instance_id in enumerate(instance_ids)}

    omission = compute_gt_omission_diagnostics(
        coords=xyz,
        semantic_logits=logits,
        offset_predictions=offsets,
        verticality=verticality,
        instance_labels=gt,
        instance_predictions=predictions,
        initial_instance_predictions=initial,
        tree_conf_thresh=tree_conf_thresh,
        tau_vert=tau_vert,
        tau_off=tau_off,
        tau_min=tau_min,
        tree_class_index=0,
        match_iou_threshold=match_iou_threshold,
        min_precision_for_counted_fp=min_precision_for_counted_fp,
        min_recall_for_undersegmentation=min_recall_for_undersegmentation,
        min_fragment_overlap_fraction=min_fragment_overlap_fraction,
        min_fragment_overlap_points=min_fragment_overlap_points)
    undersegmented_gt_ids = sorted(
        int(row['gt_tree_id']) for row in omission['rows']
        if row['category'] == 'undersegmentation')
    if expected_undersegmented_gt_ids is not None:
        expected = sorted(set(map(int, expected_undersegmented_gt_ids)))
        if undersegmented_gt_ids != expected:
            raise ValueError(
                'Q4b1 undersegmentation IDs differ from fixed Q3: '
                f'observed={undersegmented_gt_ids}, expected={expected}.')
    targets = identify_undersegmentation_targets(
        gt, predictions,
        match_iou_threshold=match_iou_threshold,
        min_recall_for_undersegmentation=min_recall_for_undersegmentation,
        allowed_undersegmented_gt_ids=undersegmented_gt_ids)
    target_by_prediction = {
        int(target['prediction_id']): target for target in targets}

    candidate_target = np.zeros(len(instance_ids), dtype=bool)
    child_count_target = np.zeros(len(instance_ids), dtype=np.int64)
    child_count_valid = np.zeros(len(instance_ids), dtype=bool)
    overflow_parent_ids = []
    for prediction_id, target in target_by_prediction.items():
        if prediction_id not in id_to_index:
            raise ValueError(
                f'Undersegmented parent {prediction_id} misses features.')
        index = id_to_index[prediction_id]
        candidate_target[index] = True
        children = int(target['oracle_num_children'])
        child_count_target[index] = children
        child_count_valid[index] = children <= int(max_children)
        if children > int(max_children):
            overflow_parent_ids.append(prediction_id)

    table = _contingency(gt, predictions)
    if not np.array_equal(table['pred_ids'], instance_ids):
        raise ValueError('Contingency rows and instance features do not align.')
    point_indices = _indices_by_prediction(predictions, instance_ids)
    point_counts = np.asarray(table['pred_counts'], dtype=np.int64)
    eligible = point_counts >= int(min_parent_points)
    positive_indices = np.flatnonzero(candidate_target & eligible)
    negative_indices = np.flatnonzero(~candidate_target & eligible)
    if split == 'train' and int(max_train_negative_parents_per_plot) >= 0:
        limit = min(
            len(negative_indices),
            int(max_train_negative_parents_per_plot))
        if limit < len(negative_indices):
            positions = np.linspace(
                0, len(negative_indices) - 1, limit, dtype=np.int64)
            negative_indices = negative_indices[positions]
    proposal_parent_indices = np.sort(np.concatenate([
        positive_indices, negative_indices])).astype(np.int64)
    proposal_instance_index = []
    proposal_k = []
    proposal_values = []
    proposal_safety = []
    proposal_child_counts = []
    sparse_gt_indices = []
    sparse_values = []
    sparse_offsets = [0]
    gt_ids = np.asarray(table['gt_ids'], dtype=np.int64)
    gt_counts = np.asarray(table['gt_counts'], dtype=np.int64)

    for instance_index in proposal_parent_indices:
        prediction_id = int(instance_ids[instance_index])
        indices = point_indices[prediction_id]
        parent_partition = np.zeros(len(indices), dtype=np.int64)
        parent_quality = _partition_quality(
            gt, indices, parent_partition, gt_ids, gt_counts,
            match_iou_threshold, min_precision_for_counted_fp)
        raw_features = split_features(
            xyz[indices], offsets[indices], 'raw_xy_kmeans_oracle')
        for children in range(2, int(max_children) + 1):
            partition = _fit_kmeans(
                raw_features, children, max_fit_points, random_state)
            if partition is None:
                continue
            child_quality = _partition_quality(
                gt, indices, partition, gt_ids, gt_counts,
                match_iou_threshold, min_precision_for_counted_fp)
            child_counts, child_intersections = _child_contingency(
                gt, indices, partition, gt_ids)
            padded_counts = np.zeros(int(max_children), dtype=np.int64)
            padded_counts[:len(child_counts)] = child_counts
            proposal_instance_index.append(int(instance_index))
            proposal_k.append(children)
            proposal_values.append(proposal_geometry_features(
                xyz[indices], offsets[indices], partition))
            proposal_safety.append(child_quality > parent_quality)
            proposal_child_counts.append(padded_counts)
            for child in range(int(max_children)):
                if child < len(child_counts):
                    nonzero = np.flatnonzero(child_intersections[child])
                    sparse_gt_indices.extend(nonzero.tolist())
                    sparse_values.extend(
                        child_intersections[child, nonzero].tolist())
                sparse_offsets.append(len(sparse_values))

    arrays = dict(features)
    arrays.update({
        'candidate_target': candidate_target,
        'child_count_target': child_count_target,
        'child_count_valid': child_count_valid,
        'proposal_instance_index': np.asarray(
            proposal_instance_index, dtype=np.int64),
        'proposal_k': np.asarray(proposal_k, dtype=np.int64),
        'proposal_feature_values': np.asarray(
            proposal_values, dtype=np.float32).reshape(
                -1, len(PROPOSAL_FEATURE_NAMES)),
        'proposal_feature_names': np.asarray(PROPOSAL_FEATURE_NAMES),
        'proposal_safety_target': np.asarray(proposal_safety, dtype=bool),
        'proposal_child_counts': np.asarray(
            proposal_child_counts, dtype=np.int64).reshape(
                -1, int(max_children)),
        'proposal_child_intersection_offsets': np.asarray(
            sparse_offsets, dtype=np.int64),
        'proposal_child_gt_indices': np.asarray(
            sparse_gt_indices, dtype=np.int64),
        'proposal_child_intersection_values': np.asarray(
            sparse_values, dtype=np.int32),
        'base_gt_ids': gt_ids,
        'base_gt_counts': gt_counts,
        'base_pred_counts': np.asarray(table['pred_counts'], dtype=np.int64),
        'base_intersections': np.asarray(
            table['intersections'], dtype=np.int32),
    })
    metadata = {
        'artifact_version': SPLIT_LEARNING_VERSION,
        'split': split,
        'num_instances': int(len(instance_ids)),
        'num_candidate_parents': int(candidate_target.sum()),
        'num_undersegmented_gt_trees': int(len(undersegmented_gt_ids)),
        'undersegmented_gt_ids': undersegmented_gt_ids,
        'num_child_count_overflow': int(len(overflow_parent_ids)),
        'overflow_parent_ids': list(map(int, overflow_parent_ids)),
        'num_proposals': int(len(proposal_k)),
        'num_safe_proposals': int(np.count_nonzero(proposal_safety)),
        'num_known_k_safe_parents': int(sum(
            bool(proposal_safety[row])
            for row, instance_index in enumerate(proposal_instance_index)
            if candidate_target[instance_index] and
            int(proposal_k[row]) == int(
                child_count_target[instance_index]))),
        'baseline': omission['baseline'],
        'max_children': int(max_children),
        'min_parent_points': int(min_parent_points),
        'max_fit_points': int(max_fit_points),
        'random_state': int(random_state),
        'global_feature_names': arrays['global_feature_names'].tolist(),
        'token_feature_names': arrays['token_feature_names'].tolist(),
        'proposal_feature_names': list(PROPOSAL_FEATURE_NAMES),
    }
    return {'arrays': arrays, 'metadata': metadata}


def save_instance_split_learning_artifact(
        artifact, output_dir, source_plot, split, metadata=None,
        savez_function=None):
    directory = os.fspath(output_dir)
    os.makedirs(directory, exist_ok=True)
    arrays = artifact['arrays']
    payload = dict(artifact['metadata'])
    payload.update(metadata or {})
    payload.update({'source_plot': str(source_plot), 'split': str(split)})
    npz_path = os.path.join(directory, SPLIT_LEARNING_ARTIFACT)
    metadata_path = os.path.join(directory, SPLIT_LEARNING_METADATA)
    writer = np.savez_compressed if savez_function is None else savez_function
    writer(npz_path, **arrays)
    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return npz_path, metadata_path


def validate_instance_split_learning_artifact(npz_path, metadata_path):
    required = {
        'instance_ids', 'global_feature_values', 'global_feature_names',
        'vertical_tokens', 'layer_valid_mask', 'token_feature_names',
        'candidate_target', 'child_count_target', 'child_count_valid',
        'proposal_instance_index', 'proposal_k', 'proposal_feature_values',
        'proposal_feature_names', 'proposal_safety_target',
        'proposal_child_counts', 'proposal_child_intersection_offsets',
        'proposal_child_gt_indices', 'proposal_child_intersection_values',
        'base_gt_ids', 'base_gt_counts', 'base_pred_counts',
        'base_intersections',
    }
    with np.load(npz_path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f'Split artifact misses {sorted(missing)}.')
        instances = len(data['instance_ids'])
        proposals = len(data['proposal_instance_index'])
        if instances == 0 or proposals == 0:
            raise ValueError('Split artifact cannot be empty.')
        for name in (
                'global_feature_values', 'vertical_tokens',
                'layer_valid_mask', 'candidate_target',
                'child_count_target', 'child_count_valid',
                'base_pred_counts', 'base_intersections'):
            if len(data[name]) != instances:
                raise ValueError(f'{name} does not align with instances.')
        for name in (
                'proposal_k', 'proposal_feature_values',
                'proposal_safety_target', 'proposal_child_counts'):
            if len(data[name]) != proposals:
                raise ValueError(f'{name} does not align with proposals.')
        if data['base_intersections'].shape != (
                instances, len(data['base_gt_ids'])):
            raise ValueError('Base contingency shape is invalid.')
        child_slots = proposals * data['proposal_child_counts'].shape[1]
        offsets = data['proposal_child_intersection_offsets']
        if len(offsets) != child_slots + 1 or offsets[0] != 0:
            raise ValueError('Sparse proposal offsets are invalid.')
        if offsets[-1] != len(data['proposal_child_gt_indices']) or \
                offsets[-1] != len(
                    data['proposal_child_intersection_values']):
            raise ValueError('Sparse proposal values are not aligned.')
        if np.any(data['proposal_instance_index'] < 0) or np.any(
                data['proposal_instance_index'] >= instances):
            raise ValueError('Proposal instance indices are invalid.')
        for name in ('global_feature_values', 'vertical_tokens',
                     'proposal_feature_values'):
            if not np.isfinite(data[name]).all():
                raise ValueError(f'{name} contains non-finite values.')
    with open(metadata_path, encoding='utf-8') as file:
        metadata = json.load(file)
    if int(metadata['num_instances']) != instances or int(
            metadata['num_proposals']) != proposals:
        raise ValueError('Split metadata counts do not match the NPZ.')
    return metadata


def detection_metrics_from_contingency(
        intersections, pred_counts, gt_counts, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5):
    intersections = np.asarray(intersections, dtype=np.int64)
    pred_counts = np.asarray(pred_counts, dtype=np.int64).reshape(-1)
    gt_counts = np.asarray(gt_counts, dtype=np.int64).reshape(-1)
    if intersections.shape != (len(pred_counts), len(gt_counts)):
        raise ValueError('Contingency arrays are not aligned.')
    unions = pred_counts[:, None] + gt_counts[None, :] - intersections
    iou = np.divide(
        intersections, unions,
        out=np.zeros_like(intersections, dtype=np.float64), where=unions > 0)
    if iou.size:
        pred_index, gt_index = linear_sum_assignment(iou, maximize=True)
        accepted = iou[pred_index, gt_index] > float(match_iou_threshold)
        matched_pred = set(pred_index[accepted].tolist())
        tp = int(accepted.sum())
    else:
        matched_pred = set()
        tp = 0
    tree_fraction = np.divide(
        intersections.sum(axis=1), pred_counts,
        out=np.zeros(len(pred_counts), dtype=np.float64),
        where=pred_counts > 0)
    fp = int(sum(
        index not in matched_pred and
        tree_fraction[index] >= float(min_precision_for_counted_fp)
        for index in range(len(pred_counts))))
    return detection_metrics(tp, fp, len(gt_counts) - tp)


def _decode_proposal_children(data, proposal_index):
    max_children = int(data['proposal_child_counts'].shape[1])
    child_counts = data['proposal_child_counts'][proposal_index]
    offsets = data['proposal_child_intersection_offsets']
    gt_indices = data['proposal_child_gt_indices']
    values = data['proposal_child_intersection_values']
    rows = []
    counts = []
    for child in range(max_children):
        count = int(child_counts[child])
        if count <= 0:
            continue
        slot = proposal_index * max_children + child
        start, end = int(offsets[slot]), int(offsets[slot + 1])
        row = np.zeros(len(data['base_gt_ids']), dtype=np.int64)
        row[gt_indices[start:end]] = values[start:end]
        rows.append(row)
        counts.append(count)
    return rows, counts


def evaluate_split_decisions(
        data, accepted_proposal_indices, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5):
    """Exactly evaluate a set containing at most one proposal per parent."""
    accepted = np.asarray(
        accepted_proposal_indices, dtype=np.int64).reshape(-1)
    if len(accepted) and (accepted.min() < 0 or accepted.max() >= len(
            data['proposal_instance_index'])):
        raise ValueError('Accepted proposal indices are invalid.')
    parent_indices = data['proposal_instance_index'][accepted]
    if len(np.unique(parent_indices)) != len(parent_indices):
        raise ValueError('Only one split proposal per parent may be accepted.')
    removed = set(map(int, parent_indices))
    keep = np.asarray([
        index not in removed
        for index in range(len(data['instance_ids']))], dtype=bool)
    rows = [row for row in data['base_intersections'][keep]]
    counts = data['base_pred_counts'][keep].astype(np.int64).tolist()
    for proposal_index in accepted:
        child_rows, child_counts = _decode_proposal_children(
            data, int(proposal_index))
        rows.extend(child_rows)
        counts.extend(child_counts)
    intersections = np.asarray(rows, dtype=np.int64).reshape(
        -1, len(data['base_gt_ids']))
    metrics = detection_metrics_from_contingency(
        intersections, np.asarray(counts), data['base_gt_counts'],
        match_iou_threshold=match_iou_threshold,
        min_precision_for_counted_fp=min_precision_for_counted_fp)
    metrics['accepted_splits'] = int(len(accepted))
    return metrics