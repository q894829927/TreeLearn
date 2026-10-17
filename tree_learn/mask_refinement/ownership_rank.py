"""Source-GT supervision for comparing a challenger with a point's original owner.

Features remain GT-free. GT and frozen candidate targets determine the loss only;
no ranking labels, owner features or new model parameters are needed at inference.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .data import point_inputs


def owner_pairs(arrays, candidates, target_ids, row, selected):
    """Return owner features and +1/-1/0 preference labels on the same samples.

    +1: challenger targets this GT, owner does not. -1: owner targets this GT,
    challenger does not. Skip same-owner, background/unknown points, both-correct
    duplicate queries and both-incorrect queries. Targets follow the original
    frozen audit for all owners, including owners excluded from primary training.
    """
    ids = candidates['ids']
    target_ids = np.asarray(target_ids)
    if target_ids.shape != ids.shape or not len(ids):
        raise ValueError('Frozen candidate targets must align with candidate IDs.')
    labels = arrays['labels'][selected]
    owners = arrays['baseline'][selected]
    foreground = owners > 0
    owner_rows = np.searchsorted(ids, np.where(foreground, owners, ids[0]))
    if np.any(owner_rows >= len(ids)) or np.any(ids[owner_rows[foreground]] != owners[foreground]):
        raise ValueError('Original owner missing from candidate IDs.')
    challenger_correct = (target_ids[row] > 0) & (labels == target_ids[row])
    owner_correct = (target_ids[owner_rows] > 0) & (labels == target_ids[owner_rows])
    usable = foreground & (labels > 0) & (owners != ids[row]) & (challenger_correct != owner_correct)
    direction = np.zeros(len(selected), dtype=np.float32)
    direction[usable & challenger_correct] = 1.
    direction[usable & owner_correct] = -1.
    inputs = np.zeros((len(selected), 40), dtype=np.float32)
    descriptors = np.zeros((len(selected), 64), dtype=np.float32)
    for owner_row in np.unique(owner_rows[usable]):
        positions = np.flatnonzero(usable & (owner_rows == owner_row))
        inputs[positions] = point_inputs(arrays, candidates, int(owner_row), selected[positions])
        descriptors[positions] = candidates['descriptors'][owner_row]
    return inputs, descriptors, direction


def ownership_rank_loss(model, challenger_logits, owner_inputs, owner_descriptors, direction, weights):
    """Weighted logistic pair loss per candidate, then equal candidate averaging."""
    if challenger_logits.ndim != 2 or direction.shape != challenger_logits.shape or weights.shape != direction.shape:
        raise ValueError('Ownership rank loss requires aligned [B,N] logits, directions and weights.')
    active = direction != 0
    if not torch.any(active):
        return challenger_logits.sum() * 0.
    positions = torch.nonzero(active.flatten(), as_tuple=False).flatten()
    # The existing model supports [B,N,40]/[B,64]. Treat each owner pair as a
    # one-point batch item, allowing a different owner descriptor for every point.
    owner_logits = model(owner_inputs.reshape(-1, 40)[positions, None, :],
                         owner_descriptors.reshape(-1, 64)[positions]).squeeze(-1)
    margin = direction.flatten()[positions] * (challenger_logits.flatten()[positions] - owner_logits)
    values = torch.zeros_like(challenger_logits).flatten().index_copy(
        0, positions, F.softplus(-margin)).reshape_as(challenger_logits)
    pair_weights = weights * active.to(weights.dtype)
    per_candidate = (values * pair_weights).sum(dim=-1) / pair_weights.sum(dim=-1).clamp_min(1e-8)
    return per_candidate.mean()
