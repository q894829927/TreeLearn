"""Fixed-validation diagnostics for recovering missing base-seed support."""

import numpy as np

from .omission_diagnostics import (
    _contingency,
    _matched_indices,
    compute_gt_omission_diagnostics,
    detection_metrics,
)
from .pipeline import (
    assign_remaining_points_nearest_neighbor,
    group_hdbscan,
)
from .seed_quality import candidate_seed_mask, tree_probabilities


SEED_COMPLETION_MODES = (
    'margin_topup',
    'xy_oracle_topup',
)


def _stable_top_k(indices, scores, count):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(indices) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Top-k indices and scores are invalid.')
    count = int(count)
    if not 0 <= count <= len(indices):
        raise ValueError('Top-k count is outside the candidate set.')
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((indices, -scores))
    return indices[order[:count]]


def build_seed_completion_masks(
        semantic_logits, verticality, offset_predictions, offset_labels,
        instance_labels, target_tree_ids, tree_conf_thresh=0.5,
        tau_vert=0.6, tau_off=4.0, tau_min=50,
        tree_class_index=0):
    """Top up each GT-selected support-failure tree to ``tau_min`` seeds.

    GT selects only the target tree. ``margin_topup`` ranks candidates using
    deployable predictions, while ``xy_oracle_topup`` supplies an upper bound
    by ranking the same candidates with target XY vote error. Neither mode
    changes a vote or adds a non-semantic point.
    """
    logits = np.asarray(semantic_logits)
    verticality = np.asarray(verticality, dtype=np.float64).reshape(-1)
    offsets = np.asarray(offset_predictions, dtype=np.float64)
    targets = np.asarray(offset_labels, dtype=np.float64)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    if offsets.shape != targets.shape or offsets.shape != (len(labels), 3):
        raise ValueError('Seed-completion point arrays are not aligned.')
    if len(logits) != len(labels) or len(verticality) != len(labels):
        raise ValueError('Seed-completion logits are not aligned.')
    if int(tau_min) <= 0:
        raise ValueError('tau_min must be positive.')

    probabilities = tree_probabilities(logits, tree_class_index)
    semantic = probabilities >= float(tree_conf_thresh)
    baseline = candidate_seed_mask(
        logits, verticality, offsets,
        tree_conf_thresh=tree_conf_thresh,
        tau_vert=tau_vert,
        tau_off=tau_off,
        tree_class_index=tree_class_index)
    masks = {mode: baseline.copy() for mode in SEED_COMPLETION_MODES}
    per_tree = []
    for tree_id in sorted(set(map(int, target_tree_ids))):
        tree = labels == tree_id
        current = int(np.count_nonzero(baseline & tree))
        semantic_count = int(np.count_nonzero(semantic & tree))
        need = max(int(tau_min) - current, 0)
        candidates = np.flatnonzero(tree & semantic & ~baseline)
        if len(candidates) < need:
            raise ValueError(
                f'Target tree {tree_id} has only {semantic_count} semantic '
                f'points and cannot be topped up to tau_min={tau_min}.')

        margin_score = (
            probabilities[candidates].astype(np.float64) +
            np.clip(verticality[candidates], 0.0, 1.0) +
            np.exp(-np.abs(offsets[candidates, 2]) /
                   max(float(tau_off), 1e-6)))
        xy_error = np.linalg.norm(
            offsets[candidates, :2] - targets[candidates, :2], axis=1)
        selected_margin = _stable_top_k(candidates, margin_score, need)
        selected_xy = _stable_top_k(candidates, -xy_error, need)
        masks['margin_topup'][selected_margin] = True
        masks['xy_oracle_topup'][selected_xy] = True
        per_tree.append({
            'gt_tree_id': tree_id,
            'baseline_seed_count': current,
            'semantic_point_count': semantic_count,
            'added_seed_count': need,
            'margin_selected_mean_xy_error': float(
                xy_error[np.isin(candidates, selected_margin)].mean()
                if need else 0.0),
            'oracle_selected_mean_xy_error': float(
                xy_error[np.isin(candidates, selected_xy)].mean()
                if need else 0.0),
        })
    return {
        'baseline_mask': baseline,
        'semantic_mask': semantic,
        'masks': masks,
        'per_tree': per_tree,
    }


def cluster_from_base_seed_mask(
        coords, offset_predictions, semantic_logits, seed_mask,
        tree_conf_thresh=0.5, tau_min=50, tree_class_index=0,
        knn_chunk_size=200000, max_cluster_seed_points=None,
        logger=None):
    """Run the original base-only 2D grouping with an explicit seed mask."""
    xyz = np.asarray(coords, dtype=np.float32)
    offsets = np.asarray(offset_predictions, dtype=np.float32)
    seeds = np.asarray(seed_mask, dtype=bool).reshape(-1)
    if xyz.shape != offsets.shape or xyz.shape != (len(seeds), 3):
        raise ValueError('Explicit seed mask is not aligned with points.')
    tree = tree_probabilities(
        semantic_logits, tree_class_index) >= float(tree_conf_thresh)
    if np.any(seeds & ~tree):
        raise ValueError('Explicit seed mask contains non-tree points.')
    seed_indices = np.flatnonzero(seeds)
    if max_cluster_seed_points is not None and int(
            max_cluster_seed_points) > 0 and len(seed_indices) > int(
                max_cluster_seed_points):
        raise RuntimeError('Seed-completion mask exceeds the seed limit.')
    predictions = np.zeros(len(xyz), dtype=np.int64)
    predictions[tree] = -1
    if len(seed_indices) < int(tau_min):
        return predictions, predictions.copy()
    seed_votes = (xyz[seed_indices, :2] + offsets[seed_indices, :2])
    if logger is not None:
        logger.info(
            f'Q5a clustering {len(seed_indices):,} explicit base seeds in 2D')
    clustered = group_hdbscan(
        seed_votes, int(tau_min), -1, 1)
    predictions[seed_indices] = clustered
    initial = predictions.copy()
    tree_votes = xyz[tree, :2] + offsets[tree, :2]
    predictions[tree] = assign_remaining_points_nearest_neighbor(
        tree_votes, predictions[tree], -1,
        chunk_size=int(knn_chunk_size), logger=logger)
    if np.any(predictions[tree] < 0):
        raise RuntimeError('Q5a left semantic tree points unassigned.')
    return predictions, initial


def evaluate_instance_predictions(
        instance_labels, predictions, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5):
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    table = _contingency(labels, predictions)
    pred_indices, gt_indices = _matched_indices(
        table['iou'], float(match_iou_threshold))
    matched_predictions = set(pred_indices.tolist())
    tree_fraction = np.divide(
        table['intersections'].sum(axis=1), table['pred_counts'],
        out=np.zeros(len(table['pred_ids']), dtype=np.float64),
        where=table['pred_counts'] > 0)
    fp = int(sum(
        index not in matched_predictions and
        tree_fraction[index] >= float(min_precision_for_counted_fp)
        for index in range(len(table['pred_ids']))))
    metrics = detection_metrics(
        len(gt_indices), fp, len(table['gt_ids']) - len(gt_indices))
    metrics['detected_gt_ids'] = sorted(map(
        int, table['gt_ids'][gt_indices].tolist()))
    return metrics


def analyze_seed_completion_oracle(
        coords, semantic_logits, offset_predictions, offset_labels,
        verticality, instance_labels, baseline_predictions,
        baseline_initial_predictions, expected_target_tree_ids,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0, tau_min=50,
        tree_class_index=0, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5,
        min_recall_for_undersegmentation=0.5,
        min_fragment_overlap_fraction=0.05,
        min_fragment_overlap_points=20, knn_chunk_size=200000,
        max_cluster_seed_points=None, logger=None):
    """Evaluate GT-targeted but otherwise prediction-only seed completion."""
    q3 = compute_gt_omission_diagnostics(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, baseline_predictions, baseline_initial_predictions,
        tree_conf_thresh=tree_conf_thresh, tau_vert=tau_vert,
        tau_off=tau_off, tau_min=tau_min,
        tree_class_index=tree_class_index,
        match_iou_threshold=match_iou_threshold,
        min_precision_for_counted_fp=min_precision_for_counted_fp,
        min_recall_for_undersegmentation=min_recall_for_undersegmentation,
        min_fragment_overlap_fraction=min_fragment_overlap_fraction,
        min_fragment_overlap_points=min_fragment_overlap_points)
    observed_targets = sorted(
        int(row['gt_tree_id']) for row in q3['rows']
        if row['category'] == 'base_seed_support_failure')
    expected_targets = sorted(set(map(int, expected_target_tree_ids)))
    if observed_targets != expected_targets:
        raise ValueError(
            'Q5a base-seed-support targets differ from fixed Q3: '
            f'observed={observed_targets}, expected={expected_targets}.')
    baseline = evaluate_instance_predictions(
        instance_labels, baseline_predictions,
        match_iou_threshold, min_precision_for_counted_fp)
    if any(int(baseline[key]) != int(q3['baseline'][key])
           for key in ('tp', 'fp', 'fn')):
        raise ValueError('Q5a baseline does not reproduce Q3.')
    baseline_detected = set(baseline.pop('detected_gt_ids'))

    completion = build_seed_completion_masks(
        semantic_logits, verticality, offset_predictions, offset_labels,
        instance_labels, expected_targets,
        tree_conf_thresh=tree_conf_thresh, tau_vert=tau_vert,
        tau_off=tau_off, tau_min=tau_min,
        tree_class_index=tree_class_index)
    modes = {}
    for mode in SEED_COMPLETION_MODES:
        if logger is None:
            print(
                f'Q5a: clustering mode={mode} with '
                f'{int(completion["masks"][mode].sum()):,} seeds...',
                flush=True)
        else:
            logger.info(f'Q5a: clustering mode={mode}')
        predictions, _ = cluster_from_base_seed_mask(
            coords, offset_predictions, semantic_logits,
            completion['masks'][mode],
            tree_conf_thresh=tree_conf_thresh, tau_min=tau_min,
            tree_class_index=tree_class_index,
            knn_chunk_size=knn_chunk_size,
            max_cluster_seed_points=max_cluster_seed_points,
            logger=logger)
        metrics = evaluate_instance_predictions(
            instance_labels, predictions,
            match_iou_threshold, min_precision_for_counted_fp)
        detected = set(metrics.pop('detected_gt_ids'))
        recovered = sorted(
            (detected - baseline_detected) & set(expected_targets))
        lost = sorted(baseline_detected - detected)
        metrics.update({
            'num_seed_points': int(completion['masks'][mode].sum()),
            'num_added_seed_points': int(
                completion['masks'][mode].sum() -
                completion['baseline_mask'].sum()),
            'recovered_target_tree_ids': recovered,
            'recovered_target_trees': int(len(recovered)),
            'lost_baseline_tree_ids': lost,
            'lost_baseline_trees': int(len(lost)),
        })
        modes[mode] = metrics
        if logger is None:
            print(
                f'Q5a: finished mode={mode}; '
                f'recovered={metrics["recovered_target_trees"]}, '
                f'lost={metrics["lost_baseline_trees"]}, '
                f'F1={100.0 * metrics["f1"]:.3f}%.',
                flush=True)
    return {
        'baseline': baseline,
        'num_target_trees': int(len(expected_targets)),
        'expected_target_tree_ids': expected_targets,
        'num_baseline_seeds': int(completion['baseline_mask'].sum()),
        'per_tree_topup': completion['per_tree'],
        'modes': modes,
    }
