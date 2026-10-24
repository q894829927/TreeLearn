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
        threshold=0.0, mode='threshold', keep_ratio=1.0,
        random_seed=42, seed_vote_xy=None, spatial_cell_size=2.0):
    """Filter base seeds. Spatial mode keeps an equal global budget per vote cell."""
    base_seed_mask = np.asarray(base_seed_mask, dtype=bool)
    if not enabled:
        return base_seed_mask
    valid_modes = ('threshold', 'top_ratio', 'random_ratio',
                   'spatial_top_ratio')
    if mode not in valid_modes:
        raise ValueError(
            'seed_confidence_filter_mode must be threshold, top_ratio, '
            'random_ratio, or spatial_top_ratio.')
    if mode in ('top_ratio', 'random_ratio', 'spatial_top_ratio'):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(
                'seed_confidence_keep_ratio must be in (0, 1].')
        seed_indices = np.flatnonzero(base_seed_mask)
        if len(seed_indices) == 0 or keep_ratio == 1.0:
            return base_seed_mask
        keep_count = max(1, int(np.ceil(
            len(seed_indices) * keep_ratio)))
        if mode == 'random_ratio':
            random_generator = np.random.default_rng(int(random_seed))
            retained_indices = random_generator.choice(
                seed_indices, size=keep_count, replace=False)
            retained_mask = np.zeros_like(base_seed_mask)
            retained_mask[retained_indices] = True
            return retained_mask

    if axis_confidence is None:
        raise ValueError(
            'axis_confidence is required when seed confidence filtering '
            'is enabled.')
    confidence = np.asarray(axis_confidence).reshape(-1)
    if len(confidence) != len(base_seed_mask):
        raise ValueError(
            'axis_confidence and base_seed_mask must have the same length.')
    if mode == 'threshold':
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                'seed_confidence_threshold must be between 0 and 1.')
        return base_seed_mask & (confidence >= threshold)
    if mode == 'spatial_top_ratio':
        return _spatial_top_ratio_seed_mask(
            base_seed_mask, confidence, seed_vote_xy,
            keep_ratio, spatial_cell_size)
    seed_indices = np.flatnonzero(base_seed_mask)
    confidence_order = np.argsort(
        -confidence[seed_indices], kind='stable')
    retained_mask = np.zeros_like(base_seed_mask)
    retained_mask[seed_indices[confidence_order[:keep_count]]] = True
    return retained_mask


def _spatial_top_ratio_seed_mask(
        base_seed_mask, confidence, seed_vote_xy, keep_ratio,
        spatial_cell_size):
    """Keep exactly ceil(ratio*N) seeds using confidence ranks in XY vote cells.

    The quota per cell uses largest-remainder proportional allocation. Inputs
    are predictions only: no GT, instance IDs, or Wytham-specific parameters.
    This is a pre-registered ablation, not a proven improvement over top_ratio.
    """
    if not np.isfinite(spatial_cell_size) or spatial_cell_size <= 0:
        raise ValueError('seed_confidence_spatial_cell_size must be positive and finite.')
    indices = np.flatnonzero(base_seed_mask)
    selected = np.zeros_like(base_seed_mask)
    if len(indices) == 0:
        return selected
    if seed_vote_xy is None:
        raise ValueError('seed_vote_xy is required by spatial_top_ratio.')
    xy = np.asarray(seed_vote_xy)
    if xy.ndim != 2 or xy.shape != (len(base_seed_mask), 2):
        raise ValueError('seed_vote_xy must have shape (num_points, 2).')
    candidate_xy = np.asarray(xy[indices], dtype=np.float64)
    candidate_scores = np.asarray(confidence[indices], dtype=np.float64)
    if not np.isfinite(candidate_xy).all():
        raise ValueError('Seed vote XY coordinates must be finite.')
    if not np.isfinite(candidate_scores).all():
        raise ValueError('Seed confidence scores must be finite.')
    if keep_ratio == 1.0:
        selected[indices] = True
        return selected

    cells = np.floor(candidate_xy / spatial_cell_size).astype(np.int64)
    _, cell_ids, cell_counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True)
    target = int(np.ceil(len(indices) * keep_ratio))
    ideal = cell_counts.astype(np.float64) * float(keep_ratio)
    quotas = np.floor(ideal).astype(np.int64)
    spare = target - int(quotas.sum())
    remainders = ideal - quotas
    # Deterministic ties: cells are lexicographically sorted by np.unique.
    quota_order = np.lexsort((np.arange(len(quotas)), -remainders))
    quotas[quota_order[:spare]] += 1
    assert quotas.sum() == target
    assert np.all(quotas <= cell_counts)

    # Group primarily by cell, then rank by confidence, breaking ties by
    # original point index. No Python loops over millions of individual seeds.
    order = np.lexsort((indices, -candidate_scores, cell_ids))
    ordered_cells = cell_ids[order]
    group_starts = np.empty(len(order), dtype=bool)
    group_starts[0] = True
    group_starts[1:] = ordered_cells[1:] != ordered_cells[:-1]
    starts = np.maximum.accumulate(
        np.where(group_starts, np.arange(len(order)), 0))
    rank_in_cell = np.arange(len(order)) - starts
    selected[indices[order[rank_in_cell < quotas[ordered_cells]]]] = True
    return selected
