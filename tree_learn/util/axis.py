import numpy as np


def get_axis_fused_features(
        coords, offset, axis_xy=None, axis_confidence=None,
        axis_fusion_weight=0.0, use_axis_confidence=True):
    """Build confidence-gated 2D votes while preserving base-only exactly."""
    base_votes = coords[:, :2] + offset[:, :2]
    if (
        axis_fusion_weight == 0 or
        axis_xy is None or
        axis_confidence is None
    ):
        return base_votes
    if use_axis_confidence:
        confidence = np.asarray(axis_confidence).reshape(-1, 1)
    else:
        confidence = np.ones((len(base_votes), 1), dtype=base_votes.dtype)
    return (
        base_votes +
        axis_fusion_weight * confidence * np.asarray(axis_xy))


def filter_base_seeds_by_confidence(
        base_seed_mask, axis_confidence=None, enabled=False,
        threshold=0.0):
    """Optionally retain only base seeds whose learned confidence is high."""
    base_seed_mask = np.asarray(base_seed_mask, dtype=bool)
    if not enabled:
        return base_seed_mask
    if axis_confidence is None:
        raise ValueError(
            'axis_confidence is required when seed confidence filtering '
            'is enabled.')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            'seed_confidence_threshold must be between 0 and 1.')
    confidence = np.asarray(axis_confidence).reshape(-1)
    if len(confidence) != len(base_seed_mask):
        raise ValueError(
            'axis_confidence and base_seed_mask must have the same length.')
    return base_seed_mask & (confidence >= threshold)
