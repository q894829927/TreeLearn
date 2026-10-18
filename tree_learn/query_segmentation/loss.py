"""Anchor identity, population-weighted masks, IoU quality and ISA losses."""

import torch
import torch.nn.functional as F


def query_weights(targets, negative_weight=.25):
    weights = torch.zeros(len(targets), dtype=torch.float32, device=targets.device)
    identities = torch.unique(targets[targets > 0])
    for identity in identities:
        same = targets == identity
        weights[same] = 1. / (len(identities) * same.sum())
    background = targets == 0
    if background.any():
        weights[background] = negative_weight / background.sum()
    return weights


def discriminative_loss(embedding, labels):
    positive = labels > 0
    if not positive.any():
        return embedding.sum() * 0.
    labels, embedding = labels[positive], embedding[positive]
    identities, inverse = torch.unique(labels, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(identities)).to(embedding.dtype)
    centers = embedding.new_zeros((len(identities), embedding.shape[1]))
    centers.index_add_(0, inverse, embedding)
    centers = centers / counts[:, None]
    errors = (embedding - centers[inverse]).norm(dim=-1).sub(.5).clamp_min(0).square()
    sums = embedding.new_zeros(len(identities)).index_add_(0, inverse, errors)
    pull = (sums / counts).mean()
    push = embedding.sum() * 0.
    if len(identities) > 1:
        distances = torch.pdist(centers)
        push = (1.5 - distances).clamp_min(0).square().mean()
    return pull + push + .001 * centers.norm(dim=-1).mean()


def instance_loss(output, query_targets, point_labels, population_weights, full_gt_counts):
    weights = population_weights * (point_labels >= 0).to(population_weights.dtype)
    normalized = weights / weights.sum().clamp_min(1e-8)
    qweights = query_weights(query_targets)
    targets = ((query_targets[:, None] == point_labels[None, :]) &
               (query_targets[:, None] > 0)).to(normalized.dtype)
    mask_losses = []
    for logits in output['masks']:
        bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * normalized).sum(-1)
        probability = logits.sigmoid()
        intersection = (probability * targets * normalized).sum(-1)
        mass = ((probability + targets) * normalized).sum(-1)
        dice = 1 - (2 * intersection + 1e-6) / (mass + 1e-6)
        dice = torch.where(query_targets > 0, dice, torch.zeros_like(dice))
        mask_losses.append(((bce + dice) * qweights).sum())
    mask = mask_losses[-1] + .5 * sum(mask_losses[:-1])
    with torch.no_grad():
        # Inverse population weights estimate full fine-point counts, not the
        # artificial stratified sample balance. Full forest GT size accounts
        # for truncated trees. This is a sampled estimate, not an exact IoU.
        binary = (output['masks'][-1].sigmoid() >= .5).to(weights.dtype)
        predicted_size = (binary * weights).sum(-1)
        intersection = (binary * targets * weights).sum(-1)
        target_quality = intersection / (predicted_size + full_gt_counts - intersection).clamp_min(1.)
        target_quality = torch.where(query_targets > 0, target_quality.clamp(0, 1), 0.)
    quality = ((output['quality_logits'].sigmoid() - target_quality).square() * qweights).sum()
    quality = quality / qweights.sum().clamp_min(1e-8)
    foreground = (F.binary_cross_entropy_with_logits(output['foreground'],
                   (point_labels > 0).to(weights.dtype), reduction='none') * normalized).sum()
    embedding = discriminative_loss(output['embedding'], point_labels)
    total = mask + .5 * quality + .2 * foreground + .1 * embedding
    logs = dict(loss=float(total.detach()), mask_loss=float(mask.detach()),
                quality_loss=float(quality.detach()), foreground_loss=float(foreground.detach()),
                embedding_loss=float(embedding.detach()),
                mean_quality=float(output['quality_logits'].detach().sigmoid().mean()),
                mean_target_quality=float(target_quality.mean()),
                positive_queries=int((query_targets > 0).sum()),
                background_queries=int((query_targets == 0).sum()))
    return total, logs
