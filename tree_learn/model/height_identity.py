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


def project_seed_membership_logits(
        base_semantic_logits, adapted_semantic_logits, base_offsets,
        input_features, tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0,
        logit_margin=1e-3, adapted_offsets=None):
    """Project semantic flips inside the base/adapted seed-geometry union.

    The adapted logits keep their per-point mean and may move freely while
    remaining on the teacher side of the tree decision boundary. Points
    outside the union seed geometry are unchanged.
    """
    if base_semantic_logits.shape[-1] != 2:
        raise ValueError('Seed membership projection requires two classes.')
    if not 0.0 < float(tree_conf_thresh) < 1.0:
        raise ValueError('tree_conf_thresh must be in (0, 1).')
    if float(logit_margin) <= 0:
        raise ValueError('logit_margin must be positive.')

    base_logits = base_semantic_logits.detach().float()
    adapted_logits = adapted_semantic_logits.float()
    base_probability = F.softmax(base_logits, dim=-1)[:, 0]
    base_tree = base_probability >= float(tree_conf_thresh)
    verticality = input_features[:, -1].to(base_logits.device).float()
    base_geometry = (
        torch.abs(base_offsets.detach().float()[:, 2]) < float(tau_off))
    if adapted_offsets is None:
        adapted_geometry = base_geometry
    else:
        adapted_geometry = (
            torch.abs(adapted_offsets.detach().float()[:, 2]) <
            float(tau_off))
    # The union blocks semantic flips for points that can be seeds either
    # before or after the adapter offset residual is applied.
    geometry_mask = (
        (verticality > float(tau_vert)) &
        (base_geometry | adapted_geometry))

    probability = adapted_logits.new_tensor(float(tree_conf_thresh))
    boundary = torch.log(probability / (1.0 - probability))
    adapted_margin = adapted_logits[:, 0] - adapted_logits[:, 1]
    tree_floor = boundary + float(logit_margin)
    non_tree_ceiling = boundary - float(logit_margin)
    projected_margin = torch.where(
        base_tree,
        torch.maximum(adapted_margin, tree_floor),
        torch.minimum(adapted_margin, non_tree_ceiling))
    projected_margin = torch.where(
        geometry_mask, projected_margin, adapted_margin)

    center = adapted_logits.mean(dim=-1)
    projected = torch.stack([
        center + 0.5 * projected_margin,
        center - 0.5 * projected_margin,
    ], dim=-1)
    changed_mask = geometry_mask & (projected_margin != adapted_margin)
    return projected.to(adapted_semantic_logits.dtype), geometry_mask, changed_mask


def project_seed_offset_membership(
        base_semantic_logits, base_offsets, adapted_offsets, input_features,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0,
        offset_margin=1e-3):
    """Project adapted offset-z so the frozen teacher seed set is preserved."""
    if not 0.0 < float(tree_conf_thresh) < 1.0:
        raise ValueError('tree_conf_thresh must be in (0, 1).')
    if not 0.0 < float(offset_margin) < float(tau_off):
        raise ValueError('offset_margin must be in (0, tau_off).')

    base_logits = base_semantic_logits.detach().float()
    base_offsets = base_offsets.detach().float()
    adapted = adapted_offsets.float()
    base_tree = (
        F.softmax(base_logits, dim=-1)[:, 0] >= float(tree_conf_thresh))
    vertical = (
        input_features[:, -1].to(base_logits.device).float() >
        float(tau_vert))
    protected = base_tree & vertical
    base_inside = torch.abs(base_offsets[:, 2]) < float(tau_off)

    adapted_z = adapted[:, 2]
    adapted_abs = torch.abs(adapted_z)
    inside_ceiling = adapted_abs.new_tensor(
        float(tau_off) - float(offset_margin))
    outside_floor = adapted_abs.new_tensor(
        float(tau_off) + float(offset_margin))
    projected_abs = torch.where(
        base_inside,
        torch.minimum(adapted_abs, inside_ceiling),
        torch.maximum(adapted_abs, outside_floor))
    projected_abs = torch.where(protected, projected_abs, adapted_abs)

    adapted_sign = torch.sign(adapted_z)
    base_sign = torch.sign(base_offsets[:, 2])
    sign = torch.where(adapted_sign != 0, adapted_sign, base_sign)
    sign = torch.where(sign != 0, sign, torch.ones_like(sign))
    projected_z = sign * projected_abs

    projected = adapted.clone()
    projected[:, 2] = projected_z
    changed = protected & (projected_z != adapted_z)
    return projected.to(adapted_offsets.dtype), protected, changed
