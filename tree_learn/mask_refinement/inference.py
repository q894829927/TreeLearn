"""Full-region streaming inference with explicit abstentions and stable ties."""

from pathlib import Path

import numpy as np
import torch

from .data import load_cache, load_candidates, point_inputs


def update_winners(baseline, predictions, best, indices, scores,
                   candidate_id, threshold):
    """Original owner wins exact ties, then smaller candidate ID wins."""
    old_scores, old_predictions = best[indices], predictions[indices]
    ties = scores == old_scores
    original_owner = baseline[indices] == candidate_id
    old_is_owner = old_predictions == baseline[indices]
    tie_wins = original_owner | (~old_is_owner & (candidate_id < old_predictions))
    accept = ((baseline[indices] > 0) & (scores >= threshold) &
              ((scores > old_scores) | (ties & tie_wins)))
    selected = indices[accept]
    predictions[selected] = candidate_id
    best[selected] = scores[accept]


@torch.no_grad()
def predict_forest(model, directory, device, chunk_size=16384, threshold=0.5):
    if chunk_size <= 0 or not 0 < threshold < 1:
        raise ValueError('Invalid chunk size or mask threshold.')
    # GT and supervision arrays are deliberately never opened here.
    arrays, _ = load_cache(directory, with_labels=False)
    candidates = load_candidates(directory)
    baseline = arrays['baseline']
    predictions = np.zeros(len(baseline), dtype=np.int64)
    best = np.full(len(baseline), -np.inf, dtype=np.float32)
    # Diagnostics use predictions/features only, and never affect assignment.
    # A point may be scored by several candidates; distinguish pairs from points.
    scored_points = np.zeros(len(baseline), dtype=bool)
    score_count, passing_count = 0, 0
    score_sum, score_min, score_max = 0.0, float('inf'), -float('inf')
    candidates_scored, candidates_passing = 0, 0
    model.eval()
    for row, candidate_id in enumerate(candidates['ids']):
        indices = np.load(Path(directory) / 'regions' / f'{int(candidate_id)}.npy',
                          mmap_mode='r', allow_pickle=False)
        descriptor = torch.as_tensor(candidates['descriptors'][row], device=device)
        candidate_scored, candidate_passing = False, False
        for start in range(0, len(indices), chunk_size):
            selected = indices[start:start + chunk_size]
            # Keep the semantic foreground fixed in this first experiment.
            selected = selected[baseline[selected] > 0]
            if not len(selected):
                continue
            inputs = torch.from_numpy(point_inputs(arrays, candidates, row, selected)).to(device)
            scores = model(inputs, descriptor).sigmoid().cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError('Nonfinite mask scores.')
            scored_points[selected] = True
            passing = int(np.count_nonzero(scores >= threshold))
            score_count += len(scores)
            passing_count += passing
            score_sum += float(scores.sum(dtype=np.float64))
            score_min = min(score_min, float(scores.min()))
            score_max = max(score_max, float(scores.max()))
            candidate_scored = True
            candidate_passing |= passing > 0
            update_winners(baseline, predictions, best, selected, scores,
                           int(candidate_id), threshold)
        candidates_scored += int(candidate_scored)
        candidates_passing += int(candidate_passing)
    old_ids = np.unique(baseline[baseline > 0])
    new_ids = np.unique(predictions[predictions > 0])
    abstained = (baseline > 0) & (predictions == 0)
    foreground_count = int(np.count_nonzero(baseline > 0))
    scored_point_count = int(scored_points.sum())
    empty_reason = None
    if not len(new_ids):
        if not foreground_count:
            empty_reason = 'no_baseline_foreground'
        elif not score_count:
            empty_reason = 'no_foreground_scores'
        elif not passing_count:
            empty_reason = 'all_scores_below_threshold'
        else:
            empty_reason = 'assignment_inconsistency'
    score_diagnostics = dict(
        threshold=float(threshold),
        num_scored_pairs=score_count,
        num_pairs_at_or_above_threshold=passing_count,
        fraction_pairs_at_or_above_threshold=passing_count / score_count if score_count else None,
        min_probability=score_min if score_count else None,
        max_probability=score_max if score_count else None,
        mean_probability=score_sum / score_count if score_count else None,
        num_candidates_scored=candidates_scored,
        num_candidates_at_or_above_threshold=candidates_passing,
        num_foreground_points=foreground_count,
        num_foreground_points_scored=scored_point_count,
        num_foreground_points_not_scored=foreground_count - scored_point_count,
        empty_prediction_reason=empty_reason)
    stats = dict(
        num_baseline_instances=len(old_ids), num_refined_instances=len(new_ids),
        disappeared_instances=np.setdiff1d(old_ids, new_ids).astype(int).tolist(),
        num_unassigned_tree_points=int(abstained.sum()),
        unassigned_tree_fraction=float(abstained.sum() / max(foreground_count, 1)),
        num_changed_points=int(np.count_nonzero(predictions != baseline)),
        score_diagnostics=score_diagnostics)
    return predictions, abstained, stats
