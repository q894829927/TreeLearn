"""Aligned-voxel diagnostics. Negative GT labels are ignored in all metrics."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import digest, load_cache, load_candidates, write_json


def contingency(labels, predictions, max_pairs=20_000_000):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError('Metric inputs must be aligned 1D labels.')
    valid = labels >= 0
    labels, predictions = labels[valid], predictions[valid]
    gt_ids, gt_counts = np.unique(labels[labels > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(predictions[predictions > 0], return_counts=True)
    if len(gt_ids) * len(pred_ids) > max_pairs:
        raise RuntimeError('Metric contingency exceeds max_pairs; review memory budget.')
    intersections = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    both = (labels > 0) & (predictions > 0)
    np.add.at(intersections, (np.searchsorted(pred_ids, predictions[both]),
                              np.searchsorted(gt_ids, labels[both])), 1)
    union = pred_counts[:, None] + gt_counts[None, :] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(union, dtype=float),
                    where=union > 0)
    return gt_ids, gt_counts, pred_ids, pred_counts, intersections, iou


def evaluate_labels(labels, predictions):
    gt, gc, pred, pc, intersection, iou = contingency(labels, predictions)
    if not len(gt):
        raise ValueError('Evaluation requires at least one labeled GT tree.')
    matched_pred, matched_gt = linear_sum_assignment(-iou)
    matched = iou[matched_pred, matched_gt] > 0.5
    matched_pred, matched_gt = matched_pred[matched], matched_gt[matched]
    counted = intersection.sum(axis=1) / np.maximum(pc, 1) >= 0.5
    counted[matched_pred] = False
    tp, fp, fn = len(matched_gt), int(counted.sum()), len(gt) - len(matched_gt)
    coverage, point_precision, point_recall = np.zeros((3, len(gt)))
    if len(pred):
        best = iou.argmax(axis=0)
        coverage = iou[best, np.arange(len(gt))]
        point_precision = intersection[best, np.arange(len(gt))] / pc[best]
        point_recall = intersection[best, np.arange(len(gt))] / gc
    return dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / max(2 * tp + fp + fn, 1),
                completeness=tp / len(gt), commission=fp / max(tp + fp, 1),
                coverage=float(coverage.mean()),
                point_precision=float(point_precision.mean()),
                point_recall=float(point_recall.mean()),
                matched_gt_ids=gt[matched_gt].astype(int).tolist(),
                num_predictions=len(pred), num_gt=len(gt),
                ignored_points=int(np.count_nonzero(np.asarray(labels) < 0)))


def audit_forest(directory, settings):
    """Construct supervision separately from deployable regions/descriptors."""
    directory = Path(directory)
    arrays, _ = load_cache(directory, with_labels=True)
    candidates = load_candidates(directory)
    labels, baseline = arrays['labels'], arrays['baseline']
    gt, gc, pred, pc, intersection, iou = contingency(labels, baseline)
    if not len(gt):
        raise ValueError('No GT trees in audit forest.')
    ids = candidates['ids']
    if len(ids) * len(gt) > 20_000_000:
        raise RuntimeError('Oracle matrix exceeds memory guard.')
    ceiling = np.zeros((len(ids), len(gt)), dtype=np.float32)
    target_ids = np.zeros(len(ids), dtype=np.int64)
    valid = np.zeros(len(ids), dtype=bool)
    baseline_iou = np.zeros(len(ids), dtype=np.float32)
    labeled_fraction = np.zeros(len(ids), dtype=np.float32)
    # Counts on all points are only used to assess annotation completeness.
    positive = baseline[baseline > 0]
    all_ids, all_counts = np.unique(positive, return_counts=True)
    for row, candidate_id in enumerate(ids):
        location = int(np.searchsorted(pred, candidate_id))
        labeled_count = (int(pc[location]) if location < len(pred) and
                         pred[location] == candidate_id else 0)
        total_count = int(all_counts[np.searchsorted(all_ids, candidate_id)])
        labeled_fraction[row] = labeled_count / total_count
        if labeled_count:
            best = int(np.argmax(iou[location]))
            baseline_iou[row] = iou[location, best]
            purity = intersection[location, best] / labeled_count
            if intersection[location].sum() == 0:
                valid[row] = labeled_fraction[row] >= settings['min_labeled_fraction']
            else:
                target_ids[row] = gt[best]
                valid[row] = (
                    labeled_fraction[row] >= settings['min_labeled_fraction'] and
                    baseline_iou[row] >= settings['min_match_iou'] and
                    purity >= settings['min_match_purity'])
        indices = np.load(directory / 'regions' / f'{int(candidate_id)}.npy',
                          mmap_mode='r', allow_pickle=False)
        # Stage A1 preserves original foreground. Semantic false negatives are
        # intentionally outside this experiment's reachable set.
        local_gt = labels[indices]
        reachable = (local_gt > 0) & (baseline[indices] > 0)
        present, counts = np.unique(local_gt[reachable], return_counts=True)
        positions = np.searchsorted(gt, present)
        ceiling[row, positions] = counts / gc[positions]
    rows, cols = linear_sum_assignment(-ceiling)
    assigned_ceiling = np.zeros(len(gt))
    assigned_ceiling[cols] = ceiling[rows, cols]
    baseline_metrics = evaluate_labels(labels, baseline)
    result = dict(
        supervision_settings_digest=digest(settings),
        baseline=baseline_metrics, num_candidates=len(ids),
        num_trainable=int(valid.sum()),
        num_positive=int(np.count_nonzero(valid & (target_ids > 0))),
        num_negative=int(np.count_nonzero(valid & (target_ids == 0))),
        num_ambiguous_or_unlabeled=int(np.count_nonzero(~valid)),
        # GT-only masks, at most one GT per candidate. Not a deployable result.
        oracle_coverage=float(assigned_ceiling.mean()),
        oracle_coverage_gain=float(assigned_ceiling.mean() - baseline_metrics['coverage']),
        oracle_completeness=float(np.mean(assigned_ceiling > 0.5)),
        region_reachable_gt_fraction=float(np.mean(ceiling.max(axis=0) > 0.5)))
    np.savez(directory / 'supervision.npz', ids=ids, target_ids=target_ids,
             valid=valid, baseline_iou=baseline_iou, labeled_fraction=labeled_fraction)
    write_json(directory / 'audit.json', result)
    return result


def macro_metrics(rows):
    keys = ('f1', 'completeness', 'commission', 'coverage',
            'point_precision', 'point_recall')
    if not rows:
        raise ValueError('No forests to summarize.')
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def compare_metrics(baseline, refined):
    return {key + '_delta_pp': 100 * (refined[key] - baseline[key]) for key in (
        'f1', 'completeness', 'commission', 'coverage',
        'point_precision', 'point_recall')}
