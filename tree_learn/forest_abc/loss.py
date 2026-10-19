"""Binary forest adaptation of anchor-based one-to-many mask supervision."""

import torch
from torch.nn import functional as F


def embedding_loss(embedding, labels):
    valid = labels > 0
    if not valid.any():
        return embedding.sum() * 0
    values = embedding[valid]
    ids, inverse, counts = labels[valid].unique(return_inverse=True, return_counts=True)
    means = values.new_zeros((len(ids), values.shape[1]))
    means.index_add_(0, inverse, values)
    means = means / counts[:, None]
    variance = F.relu((values - means[inverse]).abs().sum(-1) - .5).square()
    variance_per_tree = values.new_zeros(len(ids)).index_add_(0, inverse, variance) / counts
    distance = torch.cdist(means, means, p=1)
    off_diagonal = ~torch.eye(len(ids), dtype=torch.bool, device=values.device)
    separation = (F.relu(3 - distance[off_diagonal]).square().mean()
                  if len(ids) > 1 else values.sum() * 0)
    return variance_per_tree.mean() + separation + .001 * means.abs().sum(-1).mean()


def instance_loss(prediction, block, config, return_components=False):
    labels, weights = block['labels'], block['weights']
    known = labels >= 0
    if not known.any():
        raise ValueError('Block has no labeled voxels.')
    semantic = (labels[known] > 0).long()
    mass = weights[known]
    mass = mass / mass.sum().clamp_min(1)
    point_semantic = (F.cross_entropy(prediction['semantic'][known], semantic,
                                      reduction='none') * mass).sum()
    anchors = prediction['anchors']
    query_labels = labels[anchors]
    supervised = query_labels >= 0
    positive = query_labels > 0
    target = ((query_labels[:, None] == labels[None, known]) & positive[:, None]).float()
    mask_losses, quality_losses, semantic_losses = [], [], []
    for layer in prediction['layers']:
        logits = layer['masks'][:-2, known]
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * mass).sum(-1)
        probability = logits.sigmoid()
        intersection = (probability * target * mass).sum(-1)
        dice = 1 - (2 * intersection + 1e-5) / ((probability + target).mul(mass).sum(-1) + 1e-5)
        if supervised.any():
            mask_losses.append((bce + config['dice_weight'] * dice)[supervised].mean())
        else:
            mask_losses.append(logits.sum() * 0)
        with torch.no_grad():
            hard = (logits >= 0).float()
            # The full-tree denominator downweights proposals cut by the crop,
            # using source counts, not inference GT or source teacher masks.
            intersection_mass = (hard * target * weights[known]).sum(-1)
            predicted_mass = (hard * weights[known]).sum(-1)
            denominator = block['whole_sizes'][anchors] + predicted_mass - intersection_mass
            target_quality = torch.where(positive, intersection_mass /
                                         denominator.clamp_min(1), 0.).clamp(0, 1)
        score = layer['scores'][:-2].sigmoid()
        quality_losses.append(F.mse_loss(score[supervised], target_quality[supervised])
                              if supervised.any() else score.sum() * 0)
        semantic_logits = layer['masks'][-2:, known].T
        semantic_losses.append((F.cross_entropy(semantic_logits, semantic, reduction='none') * mass).sum())
    mask = torch.stack(mask_losses).mean()
    quality = torch.stack(quality_losses).mean()
    decoder_semantic = torch.stack(semantic_losses).mean()
    embedding = embedding_loss(prediction['embedding'], labels)
    total = (mask + config['quality_weight'] * quality + point_semantic
             + config['semantic_query_weight'] * decoder_semantic + config['embedding_weight'] * embedding)
    statistics = dict(loss=float(total.detach()), mask_loss=float(mask.detach()),
                      quality_loss=float(quality.detach()), point_semantic_loss=float(point_semantic.detach()),
                      decoder_semantic_loss=float(decoder_semantic.detach()), embedding_loss=float(embedding.detach()),
                      positive_queries=int(positive.sum()), background_queries=int((query_labels == 0).sum()))
    if return_components:
        return total, statistics, dict(mask=mask, quality=quality, embedding=embedding)
    return total, statistics
