"""Post-hoc CPU diagnostics of saved A1 assignments, never a new primary result."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluation import compare_metrics, contingency, evaluate_labels


def validate_saved_predictions(baseline, refined):
    if (baseline.ndim != 1 or baseline.shape != refined.shape or
            not np.issubdtype(baseline.dtype, np.integer) or
            not np.issubdtype(refined.dtype, np.integer)):
        raise ValueError('Saved assignments must be aligned integer vectors.')
    if np.any(baseline < 0) or np.any(refined < 0):
        raise ValueError('Completed assignments cannot contain negative IDs.')
    if np.any((baseline == 0) & (refined != 0)):
        raise ValueError('A1 predictions changed the fixed semantic background.')
    if not np.isin(np.unique(refined[refined > 0]), np.unique(baseline[baseline > 0])).all():
        raise ValueError('A1 predictions introduced new candidate IDs.')


def assignment_control(baseline, refined, mode):
    """Remove one of two observed operations, without scores, GT or retuning.

    These are descriptive counterfactuals on validation predictions. They do not
    retrain a model or estimate independent/additive causal effects.
    """
    if mode == 'abstention_only':
        return np.where(refined == 0, 0, baseline)
    if mode == 'reassignment_only':
        return np.where(refined > 0, refined, baseline)
    raise ValueError(f'Unknown assignment control: {mode}')


def instance_states(labels, predictions):
    """Use exactly the primary metric's matching and counted-FP definition."""
    gt, _, pred, pc, intersection, iou = contingency(labels, predictions)
    rows, cols = linear_sum_assignment(-iou)
    good = iou[rows, cols] > 0.5
    matched = {int(row): int(gt[col]) for row, col in zip(rows[good], cols[good])}
    all_ids, all_counts = np.unique(predictions[predictions > 0], return_counts=True)
    states = {}
    for candidate_id, count in zip(all_ids, all_counts):
        row = int(np.searchsorted(pred, candidate_id))
        present = row < len(pred) and pred[row] == candidate_id
        tree_points = int(intersection[row].sum()) if present else 0
        labeled_points = int(pc[row]) if present else 0
        tree_fraction = tree_points / labeled_points if labeled_points else None
        best = int(iou[row].argmax()) if present and len(gt) else None
        best_iou = float(iou[row, best]) if best is not None else 0.0
        if not present:
            status = 'no_labeled_points'
        elif row in matched:
            status = 'matched'
        elif tree_fraction >= 0.5:
            status = 'counted_fp'
        else:
            status = 'excluded_unmatched'
        states[int(candidate_id)] = dict(
            status=status, num_points=int(count), num_labeled_points=labeled_points,
            num_tree_points=tree_points, tree_fraction=tree_fraction,
            best_iou=best_iou, best_gt_id=int(gt[best]) if best_iou > 0 else None,
            matched_gt_id=matched.get(row) if present else None)
    return states


def verify_saved_metrics(actual, expected, name):
    for key, value in actual.items():
        saved = expected.get(key)
        if isinstance(value, float):
            equal = isinstance(saved, (int, float)) and np.isclose(value, saved, rtol=0, atol=1e-12)
        elif isinstance(value, list):
            equal = isinstance(saved, list) and sorted(value) == sorted(saved)
        else:
            equal = value == saved
        if not equal:
            raise ValueError(f'Saved {name} metrics disagree with arrays: {key}. '
                             'Check the evaluation directory and cache provenance.')


def diagnose_assignments(labels, baseline, refined, expected=None):
    validate_saved_predictions(baseline, refined)
    metrics = dict(baseline=evaluate_labels(labels, baseline),
                   refined=evaluate_labels(labels, refined))
    if expected is not None:
        for name in ('baseline', 'refined'):
            verify_saved_metrics(metrics[name], expected[name], name)
    for mode in ('abstention_only', 'reassignment_only'):
        control = assignment_control(baseline, refined, mode)
        metrics[mode] = evaluate_labels(labels, control)
        del control

    abstained = (baseline > 0) & (refined == 0)
    reassigned = (baseline > 0) & (refined > 0) & (baseline != refined)
    point_changes = {}
    for name, mask in (('abstained', abstained), ('reassigned', reassigned)):
        point_changes[name] = dict(
            all_points=int(mask.sum()),
            gt_tree_points=int(np.count_nonzero(mask & (labels > 0))),
            gt_background_points=int(np.count_nonzero(mask & (labels == 0))),
            ignored_points=int(np.count_nonzero(mask & (labels < 0))))

    before, after = instance_states(labels, baseline), instance_states(labels, refined)
    absent = dict(status='absent', num_points=0, num_labeled_points=0,
                  num_tree_points=0, tree_fraction=None, best_iou=0.0,
                  best_gt_id=None, matched_gt_id=None)
    candidates, transitions = [], Counter()
    new_fp, removed_fp = [], []
    for candidate_id in sorted(before):
        old, new = before[candidate_id], after.get(candidate_id, absent)
        transitions[(old['status'], new['status'])] += 1
        candidates.append(dict(candidate_id=candidate_id, baseline=old, refined=new))
        if new['status'] == 'counted_fp' and old['status'] != 'counted_fp':
            new_fp.append(candidate_id)
        if old['status'] == 'counted_fp' and new['status'] != 'counted_fp':
            removed_fp.append(candidate_id)
    # These checks protect the diagnostic from silently redefining the metric.
    for name, states in (('baseline', before), ('refined', after)):
        for state, metric in (('matched', 'tp'), ('counted_fp', 'fp')):
            if sum(row['status'] == state for row in states.values()) != metrics[name][metric]:
                raise RuntimeError('Candidate-state audit disagrees with primary metrics.')
    if len(new_fp) - len(removed_fp) != metrics['refined']['fp'] - metrics['baseline']['fp']:
        raise RuntimeError('FP transitions do not reconcile.')
    return dict(
        posthoc=True, eligible_for_advancement=False, metrics=metrics,
        delta_vs_baseline={name: compare_metrics(metrics['baseline'], value)
                           for name, value in metrics.items() if name != 'baseline'},
        point_changes=point_changes,
        candidate_state_transitions=[dict(baseline=old, refined=new, count=count)
                                     for (old, new), count in sorted(transitions.items())],
        new_counted_fp_ids=new_fp, removed_counted_fp_ids=removed_fp,
        candidates=candidates)
