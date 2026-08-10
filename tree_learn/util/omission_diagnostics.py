"""Ground-truth tree omission diagnostics for frozen TreeLearn pipelines."""

import csv
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .seed_quality import candidate_seed_mask, tree_probabilities


OMISSION_CATEGORIES = (
    'semantic_support_failure',
    'base_seed_support_failure',
    'density_clustering_failure',
    'undersegmentation',
    'fragmentation',
    'partial_or_localization',
)


def _validate_pointwise_arrays(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, instance_predictions, initial_instance_predictions):
    coords = np.asarray(coords, dtype=np.float64)
    logits = np.asarray(semantic_logits, dtype=np.float32)
    offsets = np.asarray(offset_predictions, dtype=np.float32)
    verticality = np.asarray(verticality, dtype=np.float32).reshape(-1)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    initial = np.asarray(
        initial_instance_predictions, dtype=np.int64).reshape(-1)
    count = len(coords)
    if coords.shape != (count, 3):
        raise ValueError('coords must have shape [N, 3].')
    if logits.ndim != 2 or len(logits) != count:
        raise ValueError('semantic_logits must have shape [N, C].')
    if offsets.shape != (count, 3):
        raise ValueError('offset_predictions must have shape [N, 3].')
    if any(len(values) != count for values in (
            verticality, labels, predictions, initial)):
        raise ValueError('Omission diagnostic arrays are not aligned.')
    return coords, logits, offsets, verticality, labels, predictions, initial


def _contingency(instance_labels, instance_predictions):
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    gt_ids, gt_counts = np.unique(labels[labels > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(
        predictions[predictions > 0], return_counts=True)
    if len(gt_ids) == 0:
        raise ValueError('No positive GT trees are available.')
    intersections = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    both = (labels > 0) & (predictions > 0)
    if np.any(both) and len(pred_ids):
        pred_index = np.searchsorted(pred_ids, predictions[both])
        gt_index = np.searchsorted(gt_ids, labels[both])
        np.add.at(intersections, (pred_index, gt_index), 1)
    unions = (
        pred_counts[:, None] + gt_counts[None, :] - intersections
        if len(pred_ids) else np.zeros((0, len(gt_ids)), dtype=np.int64))
    iou = np.divide(
        intersections, unions,
        out=np.zeros_like(intersections, dtype=np.float64), where=unions > 0)
    precision = np.divide(
        intersections, pred_counts[:, None],
        out=np.zeros_like(intersections, dtype=np.float64),
        where=pred_counts[:, None] > 0)
    recall = intersections / gt_counts[None, :] if len(pred_ids) else np.zeros(
        (0, len(gt_ids)), dtype=np.float64)
    return {
        'gt_ids': gt_ids,
        'gt_counts': gt_counts,
        'pred_ids': pred_ids,
        'pred_counts': pred_counts,
        'intersections': intersections,
        'iou': iou,
        'precision': precision,
        'recall': recall,
    }


def _matched_indices(iou, threshold):
    if iou.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pred_indices, gt_indices = linear_sum_assignment(iou, maximize=True)
    keep = iou[pred_indices, gt_indices] > float(threshold)
    return pred_indices[keep], gt_indices[keep]


def detection_metrics(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'completeness': float(tp / max(tp + fn, 1)),
        'commission': float(fp / max(tp + fp, 1)),
        'f1': float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def compute_gt_omission_diagnostics(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, instance_predictions, initial_instance_predictions,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0, tau_min=50,
        tree_class_index=0, match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5,
        min_recall_for_undersegmentation=0.5,
        min_fragment_overlap_fraction=0.05,
        min_fragment_overlap_points=20):
    """Classify every GT tree by the first failed TreeLearn stage."""
    (coords, logits, offsets, verticality, labels, predictions,
     initial) = _validate_pointwise_arrays(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, instance_predictions, initial_instance_predictions)
    if int(tau_min) <= 0:
        raise ValueError('tau_min must be positive.')
    if not 0 <= float(match_iou_threshold) <= 1:
        raise ValueError('match_iou_threshold must be in [0, 1].')

    table = _contingency(labels, predictions)
    gt_ids = table['gt_ids']
    gt_counts = table['gt_counts']
    pred_ids = table['pred_ids']
    intersections = table['intersections']
    iou = table['iou']
    precision = table['precision']
    recall = table['recall']
    matched_pred_indices, matched_gt_indices = _matched_indices(
        iou, match_iou_threshold)
    detected = np.zeros(len(gt_ids), dtype=bool)
    detected[matched_gt_indices] = True

    tree_probs = tree_probabilities(logits, tree_class_index)
    semantic_mask = tree_probs >= float(tree_conf_thresh)
    seeds = candidate_seed_mask(
        logits, verticality, offsets,
        tree_conf_thresh=tree_conf_thresh,
        tau_vert=tau_vert,
        tau_off=tau_off,
        tree_class_index=tree_class_index)
    clustered_seed_mask = seeds & (initial > 0)

    positive_pred_mask = predictions > 0
    pred_tree_fraction = np.zeros(len(pred_ids), dtype=np.float64)
    if len(pred_ids):
        pred_tree_fraction = intersections.sum(axis=1) / table['pred_counts']
    matched_pred_set = set(matched_pred_indices.tolist())
    counted_fp = sum(
        index not in matched_pred_set and
        pred_tree_fraction[index] >= float(min_precision_for_counted_fp)
        for index in range(len(pred_ids)))

    rows = []
    for gt_index, (gt_id, gt_count) in enumerate(zip(gt_ids, gt_counts)):
        gt_mask = labels == gt_id
        semantic_count = int(np.count_nonzero(gt_mask & semantic_mask))
        seed_count = int(np.count_nonzero(gt_mask & seeds))
        clustered_seed_count = int(np.count_nonzero(
            gt_mask & clustered_seed_mask))
        assigned_count = int(np.count_nonzero(
            gt_mask & positive_pred_mask))
        best_pred_index = (
            int(np.argmax(iou[:, gt_index])) if len(pred_ids) else -1)
        best_pred_id = (
            int(pred_ids[best_pred_index]) if best_pred_index >= 0 else 0)
        best_iou = (
            float(iou[best_pred_index, gt_index])
            if best_pred_index >= 0 else 0.0)
        best_precision = (
            float(precision[best_pred_index, gt_index])
            if best_pred_index >= 0 else 0.0)
        best_recall = (
            float(recall[best_pred_index, gt_index])
            if best_pred_index >= 0 else 0.0)
        minimum_fragment_points = max(
            int(min_fragment_overlap_points),
            int(np.ceil(float(min_fragment_overlap_fraction) * gt_count)))
        significant_predictions = int(np.count_nonzero(
            intersections[:, gt_index] >= minimum_fragment_points))
        total_coverage = float(assigned_count / gt_count)
        if detected[gt_index]:
            category = 'detected'
        elif semantic_count < int(tau_min):
            category = 'semantic_support_failure'
        elif seed_count < int(tau_min):
            category = 'base_seed_support_failure'
        elif clustered_seed_count == 0:
            category = 'density_clustering_failure'
        elif best_recall >= float(min_recall_for_undersegmentation):
            category = 'undersegmentation'
        elif total_coverage >= float(min_recall_for_undersegmentation) and \
                significant_predictions >= 2:
            category = 'fragmentation'
        else:
            category = 'partial_or_localization'
        rows.append({
            'gt_tree_id': int(gt_id),
            'num_points': int(gt_count),
            'height': float(np.ptp(coords[gt_mask, 2])),
            'detected': bool(detected[gt_index]),
            'category': category,
            'semantic_tree_points': semantic_count,
            'semantic_tree_fraction': float(semantic_count / gt_count),
            'base_seed_points': seed_count,
            'base_seed_fraction': float(seed_count / gt_count),
            'clustered_seed_points': clustered_seed_count,
            'clustered_seed_fraction': float(
                clustered_seed_count / max(seed_count, 1)),
            'assigned_tree_points': assigned_count,
            'assigned_tree_fraction': total_coverage,
            'significant_prediction_count': significant_predictions,
            'best_prediction_id': best_pred_id,
            'best_iou': best_iou,
            'best_precision': best_precision,
            'best_recall': best_recall,
        })

    baseline = detection_metrics(
        len(matched_gt_indices), counted_fp,
        len(gt_ids) - len(matched_gt_indices))
    category_counts = {
        category: sum(row['category'] == category for row in rows)
        for category in ('detected',) + OMISSION_CATEGORIES
    }
    if sum(category_counts.values()) != len(gt_ids):
        raise AssertionError('GT omission categories do not form a partition.')
    return {
        'rows': rows,
        'baseline': baseline,
        'category_counts': category_counts,
        'num_gt_trees': int(len(gt_ids)),
        'num_predicted_instances': int(len(pred_ids)),
        'num_counted_false_positives': int(counted_fp),
        'parameters': {
            'tree_conf_thresh': float(tree_conf_thresh),
            'tau_vert': float(tau_vert),
            'tau_off': float(tau_off),
            'tau_min': int(tau_min),
            'tree_class_index': int(tree_class_index),
            'match_iou_threshold': float(match_iou_threshold),
            'min_precision_for_counted_fp': float(
                min_precision_for_counted_fp),
            'min_recall_for_undersegmentation': float(
                min_recall_for_undersegmentation),
            'min_fragment_overlap_fraction': float(
                min_fragment_overlap_fraction),
            'min_fragment_overlap_points': int(
                min_fragment_overlap_points),
        },
    }


def save_gt_omission_diagnostics(result, output_dir, source_plot, split):
    """Save per-tree rows plus a compact JSON summary."""
    os.makedirs(output_dir, exist_ok=True)
    rows = result['rows']
    if not rows:
        raise ValueError('No GT omission rows are available to save.')
    csv_path = os.path.join(output_dir, 'gt_trees.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: value for key, value in result.items() if key != 'rows'
    }
    summary.update({'source_plot': str(source_plot), 'split': str(split)})
    json_path = os.path.join(output_dir, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return csv_path, json_path
