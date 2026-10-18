"""CPU analysis of saved instance masks; never used by prediction or training."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from tree_learn.mask_refinement.evaluation import compare_metrics


def subset_metrics(table, keep):
    """Reuse one contingency table for whole-instance removal experiments."""
    gt, gc, pred, pc, intersection, iou = table
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != pred.shape or not len(gt):
        raise ValueError('Expected one keep flag per prediction and at least one GT tree.')
    rows = np.flatnonzero(keep)
    values = iou[rows]
    matched_pred, matched_gt = linear_sum_assignment(-values)
    valid = values[matched_pred, matched_gt] > .5
    matched_pred, matched_gt = matched_pred[valid], matched_gt[valid]
    tp_mask = np.zeros(len(pred), dtype=bool)
    tp_mask[rows[matched_pred]] = True
    fp_mask = keep & (intersection.sum(axis=1) / np.maximum(pc, 1) >= .5) & ~tp_mask
    tp, fp = int(tp_mask.sum()), int(fp_mask.sum())
    coverage, precision, recall = np.zeros((3, len(gt)))
    if len(rows):
        best = rows[values.argmax(axis=0)]
        columns = np.arange(len(gt))
        coverage = iou[best, columns]
        precision = intersection[best, columns] / pc[best]
        recall = intersection[best, columns] / gc
    metrics = dict(tp=tp, fp=fp, fn=len(gt) - tp,
                   f1=2 * tp / max(2 * tp + fp + len(gt) - tp, 1),
                   completeness=tp / len(gt), commission=fp / max(tp + fp, 1),
                   coverage=float(coverage.mean()), point_precision=float(precision.mean()),
                   point_recall=float(recall.mean()), num_predictions=len(rows), num_gt=len(gt),
                   matched_gt_ids=gt[matched_gt].astype(int).tolist())
    return metrics, tp_mask, fp_mask


def analyze_instances(table, records, quality_threshold=.4):
    """GT scores outcomes; filter decisions use saved score/retention only."""
    gt, gc, pred, pc, intersection, iou = table
    by_id = {int(record['instance_id']): record for record in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate saved instance IDs.')
    try:
        selected = [by_id[int(identity)] for identity in pred]
    except KeyError as error:
        raise ValueError('Prediction labels and instance metadata disagree.') from error
    raw = np.array([r['count'] for r in selected], dtype=np.float64)
    final = np.array([r['final_count'] for r in selected], dtype=np.float64)
    score = np.array([r['score'] for r in selected], dtype=np.float64)
    if (not np.isfinite(raw).all() or not np.isfinite(final).all() or
            np.any(raw <= 0) or np.any(final <= 0) or np.any(final > raw) or
            np.any(pc > final) or not np.isfinite(score).all() or
            np.any((score < 0) | (score > 1))):
        raise ValueError('Invalid raw/final mask counts or quality scores.')
    retention = final / raw
    original, tp, fp = subset_metrics(table, np.ones(len(pred), dtype=bool))
    best_iou = iou.max(axis=1) if len(pred) else np.empty(0)
    best_gt = gt[iou.argmax(axis=1)] if len(pred) else np.empty(0, dtype=np.int64)
    related = (best_iou > 0) & np.isin(best_gt, original['matched_gt_ids'])

    def group(mask):
        return dict(instances=int(mask.sum()), tp=int((mask & tp).sum()), fp=int((mask & fp).sum()),
                    non_counted_unmatched=int((mask & ~(tp | fp)).sum()),
                    mean_score=float(score[mask].mean()) if mask.any() else None,
                    mean_best_final_iou=float(best_iou[mask].mean()) if mask.any() else None,
                    mean_retained_fraction=float(retention[mask].mean()) if mask.any() else None)

    masks = dict(unchanged=np.ones(len(pred), dtype=bool), retained_at_least_half=retention >= .5,
                 quality_at_least_0_6=score >= .6,
                 quality_times_retention=score * retention >= quality_threshold)
    policies = {}
    for name, keep in masks.items():
        metrics, _, _ = subset_metrics(table, keep)
        policies[name] = dict(metrics=metrics, removed_tp=int((~keep & tp).sum()),
                              removed_fp=int((~keep & fp).sum()), removed_instances=int((~keep).sum()),
                              delta_from_v1_pp=compare_metrics(original, metrics))
    bins = []
    for low, high in ((0., .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.0000001)):
        bins.append(dict(lower_inclusive=low, upper_exclusive=min(high, 1.),
                         includes_one=high > 1., **group((retention >= low) & (retention < high))))
    report = dict(original=original, groups=dict(tp=group(tp), fp=group(fp),
                  non_counted_unmatched=group(~(tp | fp))), retention_bins=bins,
                  fp_retained_below_half=int((fp & (retention < .5)).sum()),
                  fp_quality_at_least_0_7=int((fp & (score >= .7)).sum()),
                  fp_best_gt_already_matched=int((fp & related).sum()),
                  policies=policies, policy_parameters=dict(retained_fraction=.5, quality=.6,
                  quality_times_retention=quality_threshold))
    details = [dict(instance_id=int(identity), score=float(score[row]), raw_count=int(raw[row]),
                    final_count=int(final[row]), retained_fraction=float(retention[row]),
                    best_final_iou=float(best_iou[row]), best_gt_id=int(best_gt[row]),
                    tp=bool(tp[row]), counted_fp=bool(fp[row]),
                    best_gt_already_matched=bool(related[row])) for row, identity in enumerate(pred)]
    return report, details
