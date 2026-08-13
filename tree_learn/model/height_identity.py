"""Identity-preserving objectives for frozen TreeLearn adapters."""

import torch
import torch.nn.functional as F


def get_base_seed_mask(base_semantic_logits, base_offsets, input_features,
                       tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0):
    """Return the exact base-only TreeLearn seed mask from teacher outputs."""
    base_tree_probability = F.softmax(
        base_semantic_logits.detach().float(), dim=-1)[:, 0]
    verticality = input_features[:, -1].to(
        base_tree_probability.device).float()
    base_offsets = base_offsets.detach().float()
    return (
        (base_tree_probability >= float(tree_conf_thresh)) &
        (verticality > float(tau_vert)) &
        (torch.abs(base_offsets[:, 2]) < float(tau_off)))


def seed_semantic_identity_loss(
        base_semantic_logits, adapted_semantic_logits, base_offsets,
        input_features, tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0,
        probability_margin=0.02, threshold_epsilon=1e-3):
    """One-sided teacher loss that prevents removal of original base seeds.

    The adapter may increase tree probability freely. For every teacher base
    seed it may reduce probability by at most ``probability_margin``, but not
    below the original TreeLearn decision threshold plus a small epsilon.
    """
    if probability_margin < 0:
        raise ValueError('probability_margin must be non-negative.')
    if threshold_epsilon <= 0:
        raise ValueError('threshold_epsilon must be positive.')

    base_probability = F.softmax(
        base_semantic_logits.detach().float(), dim=-1)[:, 0]
    adapted_probability = F.softmax(
        adapted_semantic_logits.float(), dim=-1)[:, 0]
    base_seed_mask = get_base_seed_mask(
        base_semantic_logits,
        base_offsets,
        input_features,
        tree_conf_thresh=tree_conf_thresh,
        tau_vert=tau_vert,
        tau_off=tau_off)
    if not torch.any(base_seed_mask):
        return 0 * adapted_probability.sum(), base_seed_mask

    threshold_floor = adapted_probability.new_tensor(
        float(tree_conf_thresh) + float(threshold_epsilon))
    minimum_probability = torch.maximum(
        base_probability[base_seed_mask] - float(probability_margin),
        threshold_floor)
    violation = torch.relu(
        minimum_probability - adapted_probability[base_seed_mask])
    return violation.square().mean(), base_seed_mask


def seed_retention_metrics(
        base_semantic_logits, adapted_semantic_logits, base_offsets,
        adapted_offsets, input_features, tree_conf_thresh=0.5,
        tau_vert=0.6, tau_off=4.0):
    """Return teacher seed count and semantic/full adapted retention counts."""
    base_seed_mask = get_base_seed_mask(
        base_semantic_logits,
        base_offsets,
        input_features,
        tree_conf_thresh=tree_conf_thresh,
        tau_vert=tau_vert,
        tau_off=tau_off)
    adapted_probability = F.softmax(
        adapted_semantic_logits.float(), dim=-1)[:, 0]
    adapted_tree = adapted_probability >= float(tree_conf_thresh)
    adapted_geometry = (
        torch.abs(adapted_offsets.float()[:, 2]) < float(tau_off))
    return {
        'base_seed_count': int(base_seed_mask.sum().item()),
        'semantic_retained_count': int(
            (base_seed_mask & adapted_tree).sum().item()),
        'full_retained_count': int(
            (base_seed_mask & adapted_tree & adapted_geometry).sum().item()),
    }


def height_checkpoint_is_eligible(semantic_retention, minimum_retention=0.0):
    """Return whether a validation result satisfies the fixed seed gate."""
    semantic_retention = float(semantic_retention)
    minimum_retention = float(minimum_retention)
    if not 0.0 <= semantic_retention <= 1.0:
        raise ValueError('semantic_retention must be in [0, 1].')
    if not 0.0 <= minimum_retention <= 1.0:
        raise ValueError('minimum_retention must be in [0, 1].')
    return semantic_retention >= minimum_retention
