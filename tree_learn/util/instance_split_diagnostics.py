"""GT-only upper bounds for splitting undersegmented TreeLearn instances."""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .omission_diagnostics import (
    _contingency,
    _matched_indices,
    detection_metrics,
)


SPLIT_MODES = (
    'gt_extraction_ceiling',
    'raw_xy_kmeans_oracle',
    'base_vote_kmeans_oracle',
    'vertical_axis_kmeans_oracle',
)

SPLIT_DEPLOYABILITY_MODES = (
    'known_k_oracle_accept',
    'known_k_accept_all',
    'fixed_k2_accept_all',
)


def detection_metrics_from_labels(
        instance_labels, instance_predictions, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5):
    """Reproduce TreeLearn's detection counts for aligned point labels."""
    table = _contingency(instance_labels, instance_predictions)
    matched_pred, matched_gt = _matched_indices(
        table['iou'], match_iou_threshold)
    matched_pred_set = set(matched_pred.tolist())
    tree_fraction = np.divide(
        table['intersections'].sum(axis=1), table['pred_counts'],
        out=np.zeros(len(table['pred_ids']), dtype=np.float64),
        where=table['pred_counts'] > 0)
    false_positives = sum(
        index not in matched_pred_set and
        tree_fraction[index] >= float(min_precision_for_counted_fp)
        for index in range(len(table['pred_ids'])))
    metrics = detection_metrics(
        len(matched_gt), false_positives,
        len(table['gt_ids']) - len(matched_gt))
    metrics.update({
        'matched_gt_ids': table['gt_ids'][matched_gt].astype(int).tolist(),
        'matched_prediction_ids': table['pred_ids'][matched_pred].astype(
            int).tolist(),
    })
    return metrics


def identify_undersegmentation_targets(
        instance_labels, instance_predictions, match_iou_threshold=0.5,
        min_recall_for_undersegmentation=0.5,
        allowed_undersegmented_gt_ids=None):
    """Group missed, high-recall GT trees by their best prediction."""
    allowed_ids = (
        None if allowed_undersegmented_gt_ids is None else
        {int(value) for value in allowed_undersegmented_gt_ids})
    table = _contingency(instance_labels, instance_predictions)
    matched_pred, matched_gt = _matched_indices(
        table['iou'], match_iou_threshold)
    detected = np.zeros(len(table['gt_ids']), dtype=bool)
    detected[matched_gt] = True
    groups = defaultdict(list)
    for gt_index, gt_id in enumerate(table['gt_ids']):
        if allowed_ids is not None and int(gt_id) not in allowed_ids:
            continue
        if detected[gt_index] or len(table['pred_ids']) == 0:
            continue
        pred_index = int(np.argmax(table['recall'][:, gt_index]))
        best_recall = float(table['recall'][pred_index, gt_index])
        if best_recall >= float(min_recall_for_undersegmentation):
            groups[int(table['pred_ids'][pred_index])].append(int(gt_id))

    matched_by_prediction = {
        int(table['pred_ids'][pred_index]): int(table['gt_ids'][gt_index])
        for pred_index, gt_index in zip(matched_pred, matched_gt)}
    targets = []
    for prediction_id in sorted(groups):
        missed = sorted(groups[prediction_id])
        target = {
            'prediction_id': int(prediction_id),
            'missed_gt_ids': missed,
            # The split count is an Oracle input.  One slot preserves the
            # original/dominant component and one slot is added per omission.
            'oracle_num_children': int(max(
                2, len(missed) + int(
                    prediction_id in matched_by_prediction))),
            'matched_gt_id': matched_by_prediction.get(prediction_id),
        }
        targets.append(target)
    if allowed_ids is not None:
        observed_ids = {
            int(gt_id) for target in targets
            for gt_id in target['missed_gt_ids']}
        if observed_ids != allowed_ids:
            missing = sorted(allowed_ids - observed_ids)
            unexpected = sorted(observed_ids - allowed_ids)
            raise ValueError(
                'Q3 undersegmentation IDs cannot be reproduced from the '
                f'aligned Q4a payload; missing={missing}, '
                f'unexpected={unexpected}.')
    return targets


def _fit_kmeans(features, num_clusters, max_fit_points, random_state):
    values = np.asarray(features, dtype=np.float64)
    clusters = int(num_clusters)
    if values.ndim != 2 or len(values) < clusters or clusters < 2:
        return None
    if not np.isfinite(values).all():
        raise ValueError('Split features contain non-finite values.')
    fit_values = values
    if len(values) > int(max_fit_points):
        indices = np.linspace(
            0, len(values) - 1, int(max_fit_points), dtype=np.int64)
        fit_values = values[indices]
    if len(np.unique(fit_values, axis=0)) < clusters:
        return None
    from sklearn.cluster import KMeans
    model = KMeans(
        n_clusters=clusters, n_init=10, random_state=int(random_state),
        algorithm='lloyd')
    model.fit(fit_values)
    return model.predict(values).astype(np.int64)


def split_features(coords, offset_predictions, mode, vertical_power=2.0,
                   vertical_feature_weight=1.0):
    """Construct parameter-free geometry controls for one instance."""
    xyz = np.asarray(coords, dtype=np.float64)
    offsets = np.asarray(offset_predictions, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or offsets.shape != xyz.shape:
        raise ValueError('Split coordinates and offsets must have shape [N, 3].')
    raw_xy = xyz[:, :2]
    votes_xy = raw_xy + offsets[:, :2]
    if mode == 'raw_xy_kmeans_oracle':
        return raw_xy
    if mode == 'base_vote_kmeans_oracle':
        return votes_xy
    if mode != 'vertical_axis_kmeans_oracle':
        raise ValueError(f'Unsupported geometry split mode: {mode}')

    z_low, z_high = np.quantile(xyz[:, 2], [0.02, 0.98])
    normalized_height = np.clip(
        (xyz[:, 2] - z_low) / max(z_high - z_low, 1e-6), 0.0, 1.0)
    lower_structure_weight = np.power(
        1.0 - normalized_height, float(vertical_power))[:, None]
    # Near the stem, raw XY is trusted; toward the crown, frozen base votes
    # provide the stable reference.  Concatenating both traces lets K-means
    # test whether vertical trajectories contain separable child axes.
    height_conditioned_xy = (
        votes_xy + lower_structure_weight * (raw_xy - votes_xy))
    origin = np.median(votes_xy, axis=0, keepdims=True)
    return np.concatenate([
        votes_xy - origin,
        float(vertical_feature_weight) * (height_conditioned_xy - origin),
    ], axis=1)


def _partition_quality(
        instance_labels, point_indices, partition_labels, gt_ids, gt_counts,
        match_iou_threshold, min_precision_for_counted_fp):
    """Score a parent or child partition without using labels to fit it."""
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    partition = np.asarray(partition_labels, dtype=np.int64).reshape(-1)
    gt_ids = np.asarray(gt_ids, dtype=np.int64).reshape(-1)
    gt_counts = np.asarray(gt_counts, dtype=np.int64).reshape(-1)
    if len(partition) != len(indices) or len(gt_ids) != len(gt_counts):
        raise ValueError('Local split partition is not aligned.')
    child_ids, child_counts = np.unique(partition, return_counts=True)
    intersections = np.zeros((len(child_ids), len(gt_ids)), dtype=np.int64)
    local_gt = labels[indices]
    valid = local_gt > 0
    if np.any(valid):
        child_index = np.searchsorted(child_ids, partition[valid])
        gt_index = np.searchsorted(gt_ids, local_gt[valid])
        np.add.at(intersections, (child_index, gt_index), 1)
    unions = child_counts[:, None] + gt_counts[None, :] - intersections
    iou = np.divide(
        intersections, unions,
        out=np.zeros_like(intersections, dtype=np.float64), where=unions > 0)
    rows, columns = linear_sum_assignment(iou, maximize=True)
    accepted = iou[rows, columns] > float(match_iou_threshold)
    matched_rows = set(rows[accepted].tolist())
    tree_fraction = np.divide(
        intersections.sum(axis=1), child_counts,
        out=np.zeros(len(child_ids), dtype=np.float64),
        where=child_counts > 0)
    false_positives = sum(
        index not in matched_rows and
        tree_fraction[index] >= float(min_precision_for_counted_fp)
        for index in range(len(child_ids)))
    accepted_iou = float(iou[rows[accepted], columns[accepted]].sum())
    # Lexicographic safety: recover more trees first, then avoid counted
    # false positives, then prefer the cleaner accepted partition.
    return (int(accepted.sum()), -int(false_positives), accepted_iou)


def _indices_by_prediction(instance_predictions, prediction_ids):
    """Index all requested parents with one full-array pass."""
    predictions = np.asarray(
        instance_predictions, dtype=np.int64).reshape(-1)
    requested = np.asarray(
        sorted(set(map(int, prediction_ids))), dtype=np.int64)
    if len(requested) == 0:
        return {}
    selected = np.flatnonzero(np.isin(predictions, requested))
    if len(selected) == 0:
        return {}
    order = np.argsort(predictions[selected], kind='stable')
    selected = selected[order]
    ordered_ids = predictions[selected]
    starts = np.r_[
        0, np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(selected)]
    return {
        int(ordered_ids[start]): selected[start:end]
        for start, end in zip(starts, ends)
    }


def apply_gt_extraction_ceiling(
        instance_labels, instance_predictions, targets):
    """Extract only each missed GT's points from its merged parent."""
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    output = np.asarray(instance_predictions, dtype=np.int64).reshape(-1).copy()
    next_id = int(output.max(initial=0)) + 1
    extracted = 0
    parent_indices = _indices_by_prediction(
        instance_predictions,
        [target['prediction_id'] for target in targets])
    for target in targets:
        indices = parent_indices.get(int(target['prediction_id']))
        if indices is None:
            continue
        for gt_id in target['missed_gt_ids']:
            move = labels[indices] == int(gt_id)
            if np.any(move):
                output[indices[move]] = next_id
                next_id += 1
                extracted += 1
    return output, {'accepted_splits': int(extracted)}


def apply_geometry_split_oracle(
        coords, offset_predictions, instance_labels, instance_predictions,
        targets, mode, max_fit_points=50000, random_state=0,
        vertical_power=2.0, vertical_feature_weight=1.0,
        match_iou_threshold=0.5, min_precision_for_counted_fp=0.5,
        num_children_mode='oracle', acceptance_mode='oracle'):
    """Apply a geometry split with explicit K and acceptance controls."""
    if num_children_mode not in {'oracle', 'fixed_k2'}:
        raise ValueError(f'Unsupported child-count mode: {num_children_mode}')
    if acceptance_mode not in {'oracle', 'accept_all'}:
        raise ValueError(f'Unsupported split acceptance: {acceptance_mode}')
    # Preserve the pipeline dtype for the full forest.  Only the points of
    # one selected parent are promoted to float64 by split_features().
    xyz = np.asarray(coords)
    offsets = np.asarray(offset_predictions)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    original = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    output = original.copy()
    if xyz.shape != offsets.shape or xyz.shape != (len(labels), 3):
        raise ValueError('Geometry split arrays are not aligned.')
    if len(original) != len(labels):
        raise ValueError('Geometry split predictions are not aligned.')
    next_id = int(original.max(initial=0)) + 1
    proposed = accepted = failed = 0
    accepted_prediction_ids = []
    gt_ids, gt_counts = np.unique(labels[labels > 0], return_counts=True)
    parent_indices = _indices_by_prediction(
        original, [target['prediction_id'] for target in targets])
    for target in targets:
        prediction_id = int(target['prediction_id'])
        indices = parent_indices.get(prediction_id)
        count = 0 if indices is None else len(indices)
        children = (
            int(target['oracle_num_children'])
            if num_children_mode == 'oracle' else 2)
        if count < children or children < 2:
            failed += 1
            continue
        features = split_features(
            xyz[indices], offsets[indices], mode,
            vertical_power=vertical_power,
            vertical_feature_weight=vertical_feature_weight)
        partition = _fit_kmeans(
            features, children, max_fit_points, random_state)
        if partition is None:
            failed += 1
            continue
        proposed += 1
        if acceptance_mode == 'oracle':
            parent_partition = np.zeros(count, dtype=np.int64)
            parent_quality = _partition_quality(
                labels, indices, parent_partition, gt_ids, gt_counts,
                match_iou_threshold, min_precision_for_counted_fp)
            child_quality = _partition_quality(
                labels, indices, partition, gt_ids, gt_counts,
                match_iou_threshold, min_precision_for_counted_fp)
            if child_quality <= parent_quality:
                continue
        for child in np.unique(partition):
            output[indices[partition == child]] = next_id
            next_id += 1
        accepted += 1
        accepted_prediction_ids.append(prediction_id)
    return output, {
        'proposed_splits': int(proposed),
        'accepted_splits': int(accepted),
        'failed_splits': int(failed),
        'accepted_prediction_ids': accepted_prediction_ids,
        'num_children_mode': str(num_children_mode),
        'acceptance_mode': str(acceptance_mode),
    }


def analyze_instance_split_deployability(
        coords, offset_predictions, instance_labels, instance_predictions,
        match_iou_threshold=0.5, min_precision_for_counted_fp=0.5,
        min_recall_for_undersegmentation=0.5, max_fit_points=50000,
        random_state=0, allowed_undersegmented_gt_ids=None):
    """Decompose the GT K and GT acceptance assumptions of raw-XY splits."""
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    baseline = detection_metrics_from_labels(
        labels, predictions, match_iou_threshold,
        min_precision_for_counted_fp)
    targets = identify_undersegmentation_targets(
        labels, predictions, match_iou_threshold,
        min_recall_for_undersegmentation,
        allowed_undersegmented_gt_ids=allowed_undersegmented_gt_ids)
    undersegmented_gt_ids = sorted({
        gt_id for target in targets for gt_id in target['missed_gt_ids']})
    baseline_detected = set(baseline['matched_gt_ids'])
    specifications = {
        'known_k_oracle_accept': ('oracle', 'oracle'),
        'known_k_accept_all': ('oracle', 'accept_all'),
        'fixed_k2_accept_all': ('fixed_k2', 'accept_all'),
    }
    modes = {}
    for name in SPLIT_DEPLOYABILITY_MODES:
        num_children_mode, acceptance_mode = specifications[name]
        split_predictions, metadata = apply_geometry_split_oracle(
            coords, offset_predictions, labels, predictions, targets,
            'raw_xy_kmeans_oracle', max_fit_points=max_fit_points,
            random_state=random_state,
            match_iou_threshold=match_iou_threshold,
            min_precision_for_counted_fp=min_precision_for_counted_fp,
            num_children_mode=num_children_mode,
            acceptance_mode=acceptance_mode)
        metrics = detection_metrics_from_labels(
            labels, split_predictions, match_iou_threshold,
            min_precision_for_counted_fp)
        modes[name] = _mode_report(
            baseline, metrics, metadata, baseline_detected,
            undersegmented_gt_ids)
    return {
        'baseline': _without_match_lists(baseline),
        'num_split_target_predictions': int(len(targets)),
        'num_undersegmented_gt_trees': int(len(undersegmented_gt_ids)),
        'expected_undersegmented_gt_ids': (
            undersegmented_gt_ids if allowed_undersegmented_gt_ids is None
            else sorted({
                int(value) for value in allowed_undersegmented_gt_ids})),
        'split_targets': targets,
        'modes': modes,
        'parameters': {
            'geometry': 'raw_xy_kmeans',
            'match_iou_threshold': float(match_iou_threshold),
            'min_precision_for_counted_fp': float(
                min_precision_for_counted_fp),
            'min_recall_for_undersegmentation': float(
                min_recall_for_undersegmentation),
            'max_fit_points': int(max_fit_points),
            'random_state': int(random_state),
        },
    }


def analyze_instance_split_oracle(
        coords, offset_predictions, instance_labels, instance_predictions,
        match_iou_threshold=0.5, min_precision_for_counted_fp=0.5,
        min_recall_for_undersegmentation=0.5, max_fit_points=50000,
        random_state=0, vertical_power=2.0,
        vertical_feature_weight=1.0,
        allowed_undersegmented_gt_ids=None):
    """Run the exact ceiling and three known-K geometry proposal Oracles."""
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    baseline = detection_metrics_from_labels(
        labels, predictions, match_iou_threshold,
        min_precision_for_counted_fp)
    targets = identify_undersegmentation_targets(
        labels, predictions, match_iou_threshold,
        min_recall_for_undersegmentation,
        allowed_undersegmented_gt_ids=allowed_undersegmented_gt_ids)
    undersegmented_gt_ids = sorted({
        gt_id for target in targets for gt_id in target['missed_gt_ids']})
    baseline_detected = set(baseline['matched_gt_ids'])
    modes = {}

    ceiling_predictions, metadata = apply_gt_extraction_ceiling(
        labels, predictions, targets)
    ceiling_metrics = detection_metrics_from_labels(
        labels, ceiling_predictions, match_iou_threshold,
        min_precision_for_counted_fp)
    modes['gt_extraction_ceiling'] = _mode_report(
        baseline, ceiling_metrics, metadata, baseline_detected,
        undersegmented_gt_ids)

    for mode in SPLIT_MODES[1:]:
        split_predictions, metadata = apply_geometry_split_oracle(
            coords, offset_predictions, labels, predictions, targets, mode,
            max_fit_points=max_fit_points, random_state=random_state,
            vertical_power=vertical_power,
            vertical_feature_weight=vertical_feature_weight,
            match_iou_threshold=match_iou_threshold,
            min_precision_for_counted_fp=min_precision_for_counted_fp)
        metrics = detection_metrics_from_labels(
            labels, split_predictions, match_iou_threshold,
            min_precision_for_counted_fp)
        modes[mode] = _mode_report(
            baseline, metrics, metadata, baseline_detected,
            undersegmented_gt_ids)

    return {
        'baseline': _without_match_lists(baseline),
        'num_split_target_predictions': int(len(targets)),
        'num_undersegmented_gt_trees': int(len(undersegmented_gt_ids)),
        'expected_undersegmented_gt_ids': (
            undersegmented_gt_ids if allowed_undersegmented_gt_ids is None
            else sorted({
                int(value) for value in allowed_undersegmented_gt_ids})),
        'split_targets': targets,
        'modes': modes,
        'parameters': {
            'match_iou_threshold': float(match_iou_threshold),
            'min_precision_for_counted_fp': float(
                min_precision_for_counted_fp),
            'min_recall_for_undersegmentation': float(
                min_recall_for_undersegmentation),
            'max_fit_points': int(max_fit_points),
            'random_state': int(random_state),
            'vertical_power': float(vertical_power),
            'vertical_feature_weight': float(vertical_feature_weight),
        },
    }


def _without_match_lists(metrics):
    return {
        key: value for key, value in metrics.items()
        if key not in {'matched_gt_ids', 'matched_prediction_ids'}
    }


def _mode_report(
        baseline, metrics, metadata, baseline_detected,
        undersegmented_gt_ids):
    detected = set(metrics['matched_gt_ids'])
    recovered = detected & set(undersegmented_gt_ids)
    lost = baseline_detected - detected
    report = _without_match_lists(metrics)
    report.update(metadata)
    report.update({
        'recovered_undersegmented_trees': int(len(recovered)),
        'recovered_undersegmented_gt_ids': sorted(map(int, recovered)),
        'lost_baseline_trees': int(len(lost)),
        'lost_baseline_gt_ids': sorted(map(int, lost)),
        'f1_gain_pp': float(100 * (metrics['f1'] - baseline['f1'])),
        'completeness_gain_pp': float(
            100 * (metrics['completeness'] - baseline['completeness'])),
        'commission_reduction_pp': float(
            100 * (baseline['commission'] - metrics['commission'])),
    })
    return report
