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
    model.eval()
    for row, candidate_id in enumerate(candidates['ids']):
        indices = np.load(Path(directory) / 'regions' / f'{int(candidate_id)}.npy',
                          mmap_mode='r', allow_pickle=False)
        descriptor = torch.as_tensor(candidates['descriptors'][row], device=device)
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
            update_winners(baseline, predictions, best, selected, scores,
                           int(candidate_id), threshold)
    old_ids = np.unique(baseline[baseline > 0])
    new_ids = np.unique(predictions[predictions > 0])
    abstained = (baseline > 0) & (predictions == 0)
    stats = dict(
        num_baseline_instances=len(old_ids), num_refined_instances=len(new_ids),
        disappeared_instances=np.setdiff1d(old_ids, new_ids).astype(int).tolist(),
        num_unassigned_tree_points=int(abstained.sum()),
        unassigned_tree_fraction=float(abstained.sum() / max(np.count_nonzero(baseline > 0), 1)),
        num_changed_points=int(np.count_nonzero(predictions != baseline)))
    return predictions, abstained, stats
