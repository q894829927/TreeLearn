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
