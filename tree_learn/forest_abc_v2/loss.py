"""One-to-many auxiliary masks, one-to-one final ownership, tree-balanced centers."""

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from tree_learn.forest_abc.loss import embedding_loss


def tree_mean(values, labels):
    if not len(values):
        return values.sum() * 0
    _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
    sums = values.new_zeros(len(counts)).index_add(0, inverse, values)
    return (sums / counts).mean()


def balanced_mass(labels, weights):
    """Each visible tree gets equal positive mass; background gets 25% if present."""
    result = torch.zeros_like(weights)
    positive, background = labels > 0, labels == 0
    if positive.any():
        _, inverse = labels[positive].unique(return_inverse=True)
        counts = weights.new_zeros(int(inverse.max()) + 1).index_add(0, inverse, weights[positive])
        result[positive] = weights[positive] / counts[inverse].clamp_min(1e-8) / len(counts)
        if background.any():
            result[positive] *= .75
    if background.any():
        result[background] = weights[background] / weights[background].sum().clamp_min(1e-8)
        if positive.any():
            result[background] *= .25
    return result


def mask_terms(logits, target, mass, dice_weight):
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * mass).sum(-1)
    probability = logits.sigmoid()
    intersection = (probability * target * mass).sum(-1)
    dice = 1 - (2 * intersection + 1e-5) / ((probability + target).mul(mass).sum(-1) + 1e-5)
    return bce + dice_weight * dice


@torch.no_grad()
def match_instances(logits, targets, mass, dice_weight):
    if len(targets) > len(logits):
        raise ValueError('More GT trees than queries; increase num_queries, do not truncate GT.')
    probability = logits.sigmoid()
    weighted_targets = targets * mass
    bce = (F.softplus(logits) * mass).sum(-1)[:, None] - logits @ weighted_targets.T
    intersection = probability @ weighted_targets.T
    dice = 1 - (2 * intersection + 1e-5) / (
        (probability * mass).sum(-1)[:, None] + weighted_targets.sum(-1)[None] + 1e-5)
    cost = bce + dice_weight * dice
    if not torch.isfinite(cost).all():
        raise FloatingPointError('Non-finite instance matching costs.')
    queries, trees = linear_sum_assignment(cost.detach().cpu().numpy())
    order = trees.argsort()
    return torch.as_tensor(queries[order], device=logits.device, dtype=torch.long)


def instance_loss(prediction, block, config, return_components=False):
    labels = block['labels']
    known = labels >= 0
    if not known.any() or not (labels > 0).any():
        raise ValueError('V2 training crop must have known tree voxels.')
    weights = block['weights'].to(prediction['semantic'])
    known_labels, raw_mass = labels[known], weights[known]
    mass = balanced_mass(known_labels, raw_mass)
    ids = known_labels[known_labels > 0].unique(sorted=True)
    targets = (ids[:, None] == known_labels[None]).to(mass)
    query_labels = labels[prediction['anchors']]
    supervised, positive = query_labels >= 0, query_labels > 0
    anchor_targets = ((query_labels[:, None] == known_labels[None]) & positive[:, None]).to(mass)
    whole_sizes = block['whole_sizes'].to(mass)
    auxiliary, auxiliary_quality, semantic_losses = [], [], []
    for layer in prediction['layers']:
        semantic_losses.append((F.cross_entropy(layer['masks'][-2:, known].T,
                                (known_labels > 0).long(), reduction='none') * mass).sum())
    for layer in prediction['layers'][:-1]:
        logits = layer['masks'][:-2, known]
        terms = mask_terms(logits, anchor_targets, mass, config['dice_weight'])
        auxiliary.append(tree_mean(terms[supervised], query_labels[supervised]))
        with torch.no_grad():
            hard = (logits >= 0).to(mass)
            intersection = (hard * anchor_targets * raw_mass).sum(-1)
            union = whole_sizes[prediction['anchors']] + (hard * raw_mass).sum(-1) - intersection
            target_quality = torch.where(positive, intersection / union.clamp_min(1), 0.).clamp(0, 1)
        score_loss = (layer['scores'][:-2].sigmoid() - target_quality).square()
        auxiliary_quality.append(tree_mean(score_loss[supervised], query_labels[supervised]))
    final = prediction['layers'][-1]
    logits = final['masks'][:-2, known]
    selected = match_instances(logits, targets, mass, config['dice_weight'])
    final_mask = mask_terms(logits[selected], targets, mass, config['dice_weight']).mean()
    # All other instance queries compete as negatives, rather than being silently
    # omitted from the point ownership softmax. Class 0 is the background query.
    background = final['masks'][-2, known] - final['masks'][-1, known]
    classes = torch.cat((background[None], logits), dim=0)
    target_class = torch.zeros(len(known_labels), dtype=torch.long, device=labels.device)
    for column, query in enumerate(selected):
        target_class[known_labels == ids[column]] = query + 1
    ownership = (F.cross_entropy(classes.T, target_class, reduction='none') * mass).sum()
    with torch.no_grad():
        winner = classes.argmax(0)
        predicted_mass = raw_mass.new_zeros(len(logits) + 1).index_add(0, winner, raw_mass)[1:]
        quality_target = logits.new_zeros(len(logits))
        for column, query in enumerate(selected):
            intersection = raw_mass[(winner == query + 1) & (known_labels == ids[column])].sum()
            full_size = whole_sizes[known][known_labels == ids[column]].max()
            quality_target[query] = intersection / (full_size + predicted_mass[query] - intersection).clamp_min(1)
    quality_errors = (final['scores'][:-2].sigmoid() - quality_target).square()
    matched = torch.zeros(len(logits), dtype=torch.bool, device=labels.device)
    matched[selected] = True
    quality = quality_errors[matched].mean()
    negatives = ~matched & supervised
    if negatives.any():
        quality = .5 * (quality + quality_errors[negatives].mean())
    aux_mask = torch.stack(auxiliary).mean() if auxiliary else logits.sum() * 0
    aux_quality = torch.stack(auxiliary_quality).mean() if auxiliary_quality else logits.sum() * 0
    mask = final_mask + config['auxiliary_weight'] * aux_mask
    quality = quality + config['auxiliary_weight'] * aux_quality
    point_semantic = (F.cross_entropy(prediction['semantic'][known],
                      (known_labels > 0).long(), reduction='none') * mass).sum()
    decoder_semantic = torch.stack(semantic_losses).mean()
    point_offsets = prediction['offsets'][labels > 0]
    offset_error = F.smooth_l1_loss(point_offsets / config['center_scale'],
                    block['center_offsets'][labels > 0].to(point_offsets) / config['center_scale'],
                    reduction='none').sum(-1)
    positive_labels = labels[labels > 0]
    center_mass = balanced_mass(positive_labels, weights[labels > 0])
    center = (offset_error * center_mass).sum()
    embedding = embedding_loss(prediction['embedding'], labels)
    mask_objective = mask + config['ownership_weight'] * ownership
    total = (mask_objective + config['quality_weight'] * quality + point_semantic
             + config['semantic_query_weight'] * decoder_semantic
             + config['center_weight'] * center + config['embedding_weight'] * embedding)
    components = dict(mask=mask_objective, quality=quality, center=center, ownership=ownership,
                      embedding=embedding)
    statistics = dict(loss=float(total.detach()), mask_loss=float(mask.detach()),
                      ownership_loss=float(ownership.detach()), center_loss=float(center.detach()),
                      quality_loss=float(quality.detach()), point_semantic_loss=float(point_semantic.detach()),
                      decoder_semantic_loss=float(decoder_semantic.detach()), embedding_loss=float(embedding.detach()),
                      matched_trees=len(ids), positive_queries=int(positive.sum()),
                      background_queries=int((query_labels == 0).sum()))
    return (total, statistics, components) if return_components else (total, statistics)
