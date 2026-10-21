"""Instance-balanced structural targets, all derived from training labels only."""

import numpy as np
import torch
from torch.nn import functional as F


def sample_queries(batch, settings, seed):
    rng = np.random.default_rng(seed)
    xyz = batch['coords'].detach().cpu().numpy()
    labels = batch['instance_labels'].detach().cpu().numpy()
    batches = batch['batch_ids'].detach().cpu().numpy()
    valid = batch['masks_off'].detach().cpu().numpy().astype(bool) & (labels > 0)
    rows = []
    for batch_id in np.unique(batches[valid]):
        tree_ids = np.unique(labels[valid & (batches == batch_id)])
        if len(tree_ids) > settings['max_trees']:
            tree_ids = np.sort(rng.choice(tree_ids, settings['max_trees'], replace=False))
        for tree_id in tree_ids:
            pool = np.flatnonzero(valid & (batches == batch_id) & (labels == tree_id))
            pool = pool[np.argsort(xyz[pool, 2], kind='stable')]
            # Equal-count height strata preserve low, middle and high samples.
            for part in np.array_split(pool, min(len(pool), settings['points_per_tree'])):
                rows.append(int(rng.choice(part)))
    if not rows:
        raise ValueError('No supervised tree queries in this crop.')
    return torch.tensor(rows, dtype=torch.long)


def group_average(values, inverse, groups):
    """Average points within each tree, then give every tree equal weight."""
    sums = values.new_zeros(groups).index_add(0, inverse, values)
    counts = torch.bincount(inverse, minlength=groups).clamp_min(1)
    return (sums / counts).mean()


def structure_loss(result, batch, base_offsets, settings):
    device = result['delta_xy'].device
    rows, delta = result['rows'], result['delta_xy'].float()
    labels = batch['instance_labels'].to(device)
    batches = batch['batch_ids'].to(device)
    xy = batch['coords'].to(device).float()[:, :2]
    target = batch['offset_labels'].to(device).float()[rows, :2]
    corrected = base_offsets.detach().float()[rows, :2] + delta
    keys = torch.stack((batches[rows], labels[rows]), dim=1)
    groups, inverse = keys.unique(dim=0, sorted=True, return_inverse=True)
    count = len(groups)
    vote_error = F.smooth_l1_loss(corrected, target, reduction='none').mean(-1)
    vote = group_average(vote_error, inverse, count)
    predicted_votes = xy[rows] + corrected
    means = predicted_votes.new_zeros((count, 2)).index_add(0, inverse, predicted_votes)
    means = means / torch.bincount(inverse, minlength=count)[:, None]
    deviations = F.smooth_l1_loss(predicted_votes, means[inverse], reduction='none').mean(-1)
    consistency = group_average(deviations, inverse, count)
    regularization = group_average(delta.square().mean(-1), inverse, count)
    neighbor = result['neighbors']
    known = (labels[neighbor] >= 0) & (labels[rows, None] > 0)
    valid = result['edge_valid'] & known & (batches[neighbor] == batches[rows, None])
    same = labels[neighbor] == labels[rows, None]
    logits = result['edge_logits'].float()
    # Balance same-tree and different-tree edges per query, then per instance.
    bce = F.binary_cross_entropy_with_logits(logits, same.float(), reduction='none')
    positive, negative = valid & same, valid & ~same
    positive_count, negative_count = positive.sum(1), negative.sum(1)
    edge_per_query = ((bce * positive).sum(1) / positive_count.clamp_min(1)
                      + (bce * negative).sum(1) / negative_count.clamp_min(1))
    edge_per_query = edge_per_query / ((positive_count > 0).float() + (negative_count > 0).float()).clamp_min(1)
    # A tree with no graph edges does not dilute the connection objective.
    usable = valid.any(1)
    if bool(usable.any()):
        _, edge_inverse = keys[usable].unique(dim=0, return_inverse=True)
        connection = group_average(edge_per_query[usable], edge_inverse, int(edge_inverse.max()) + 1)
    else:
        connection = logits.sum() * 0
    total = (settings['vote_weight'] * vote + settings['connection_weight'] * connection
             + settings['consistency_weight'] * consistency + settings['regularization_weight'] * regularization)
    with torch.no_grad():
        base_error = group_average(F.smooth_l1_loss(base_offsets.detach().float()[rows, :2],
            target, reduction='none').mean(-1), inverse, count)
    return total, dict(loss=float(total.detach()), vote_loss=float(vote.detach()),
        base_vote_loss=float(base_error), connection_loss=float(connection.detach()),
        consistency_loss=float(consistency.detach()), regularization=float(regularization.detach()),
        delta_mean_m=float(torch.linalg.vector_norm(delta.detach(), dim=-1).mean()),
        sampled_trees=count, sampled_points=len(rows), positive_edges=int(positive.sum()),
        negative_edges=int(negative.sum()),
        connection_accuracy=float(((logits.detach() >= 0) == same)[valid].float().mean())
                            if bool(valid.any()) else None)
