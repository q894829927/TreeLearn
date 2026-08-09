"""Frozen-backbone seed features and supervised reliability/coverage targets."""

import json
import os

import numpy as np


SEED_ARTIFACT_NAME = 'seed_candidates.npz'
SEED_METADATA_NAME = 'metadata.json'


def tree_probabilities(semantic_logits, tree_class_index=0):
    logits = np.asarray(semantic_logits, dtype=np.float32)
    if logits.ndim != 2 or not 0 <= int(tree_class_index) < logits.shape[1]:
        raise ValueError('semantic_logits or tree_class_index is invalid.')
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (
        exponentials[:, int(tree_class_index)] /
        exponentials.sum(axis=1)
    ).astype(np.float32)


def candidate_seed_mask(
        semantic_logits, verticality, offset_predictions,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0,
        tree_class_index=0):
    """Match TreeLearn's initial base-seed mask exactly."""
    offsets = np.asarray(offset_predictions)
    verticality = np.asarray(verticality).reshape(-1)
    if offsets.shape != (len(verticality), 3):
        raise ValueError('offset_predictions and verticality are not aligned.')
    probabilities = tree_probabilities(
        semantic_logits, tree_class_index=tree_class_index)
    if len(probabilities) != len(verticality):
        raise ValueError('semantic logits and verticality are not aligned.')
    return (
        (probabilities >= float(tree_conf_thresh)) &
        (verticality > float(tau_vert)) &
        (np.abs(offsets[:, 2]) < float(tau_off)))


def _cell_inverse(values_xy, cell_size):
    if float(cell_size) <= 0:
        raise ValueError('Cell sizes must be positive.')
    cells = np.floor(
        np.asarray(values_xy, dtype=np.float64) / float(cell_size)
    ).astype(np.int64)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError('Cell coordinates must have shape [N, 2].')
    cell_dtype = np.dtype([('x', '<i8'), ('y', '<i8')])
    keys = np.ascontiguousarray(cells).view(cell_dtype).reshape(-1)
    _, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    return cells, inverse, counts.astype(np.int64)


def _group_mean(values, inverse, counts):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sums = np.bincount(
        inverse, weights=values, minlength=len(counts))
    return sums / np.maximum(counts, 1)


def _group_std(values, inverse, counts):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    means = _group_mean(values, inverse, counts)
    squared_means = _group_mean(values ** 2, inverse, counts)
    return np.sqrt(np.maximum(squared_means - means ** 2, 0.0))


def compute_local_complexity_features(
        base_votes_xy, tree_probability, verticality, normalized_height,
        cell_sizes=(0.3, 0.6, 1.2)):
    """Compute GT-free multiscale statistics in predicted vote space."""
    votes = np.asarray(base_votes_xy, dtype=np.float64)
    probability = np.asarray(tree_probability, dtype=np.float64).reshape(-1)
    verticality = np.asarray(verticality, dtype=np.float64).reshape(-1)
    height = np.asarray(normalized_height, dtype=np.float64).reshape(-1)
    if votes.shape != (len(probability), 2):
        raise ValueError('base_votes_xy and scalar inputs are not aligned.')
    if len(verticality) != len(probability) or len(height) != len(probability):
        raise ValueError('Local-complexity scalar inputs are not aligned.')
    sizes = [float(size) for size in cell_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError('At least one positive complexity scale is required.')

    columns = []
    names = []
    for cell_size in sizes:
        _, inverse, counts = _cell_inverse(votes, cell_size)
        mean_x = _group_mean(votes[:, 0], inverse, counts)
        mean_y = _group_mean(votes[:, 1], inverse, counts)
        squared_radius = (
            (votes[:, 0] - mean_x[inverse]) ** 2 +
            (votes[:, 1] - mean_y[inverse]) ** 2)
        vote_rms_by_cell = np.sqrt(_group_mean(
            squared_radius, inverse, counts))

        scale_tag = f'{cell_size:g}m'.replace('.', 'p')
        scale_columns = [
            np.log1p(counts[inverse]),
            vote_rms_by_cell[inverse],
            _group_mean(verticality, inverse, counts)[inverse],
            _group_std(verticality, inverse, counts)[inverse],
            _group_mean(height, inverse, counts)[inverse],
            _group_std(height, inverse, counts)[inverse],
            _group_mean(probability, inverse, counts)[inverse],
            _group_std(probability, inverse, counts)[inverse],
        ]
        scale_names = [
            'log_count', 'vote_radius_rms',
            'verticality_mean', 'verticality_std',
            'height_mean', 'height_std',
            'tree_probability_mean', 'tree_probability_std',
        ]
        columns.extend(scale_columns)
        names.extend(
            f'{name}_{scale_tag}' for name in scale_names)
    values = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Local complexity features contain non-finite values.')
    return values, names


def vote_cell_purity(base_votes_xy, labels, cell_size):
    votes = np.asarray(base_votes_xy, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if votes.shape != (len(labels), 2):
        raise ValueError('base_votes_xy and labels are not aligned.')
    cells, cell_inverse, cell_counts = _cell_inverse(votes, cell_size)
    pairs = np.column_stack([cells, labels])
    pair_dtype = np.dtype([
        ('x', '<i8'), ('y', '<i8'), ('label', '<i8')])
    pair_keys = np.ascontiguousarray(pairs).view(pair_dtype).reshape(-1)
    _, pair_inverse, pair_counts = np.unique(
        pair_keys, return_inverse=True, return_counts=True)
    return (
        pair_counts[pair_inverse] / cell_counts[cell_inverse]
    ).astype(np.float32)


def compute_coverage_targets(
        base_votes_xy, labels, utility, cell_size=0.6,
        min_seeds_per_tree=50):
    """Create GT-only cell-representative and per-tree quota targets."""
    votes = np.asarray(base_votes_xy, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    if votes.shape != (len(labels), 2) or len(utility) != len(labels):
        raise ValueError('Coverage target arrays are not aligned.')
    if float(cell_size) <= 0:
        raise ValueError('critical_cell_size must be positive.')
    if int(min_seeds_per_tree) < 1:
        raise ValueError('min_seeds_per_tree must be at least one.')
    tree_indices = np.flatnonzero(labels > 0)
    representative = np.zeros(len(labels), dtype=bool)
    quota = np.zeros(len(labels), dtype=bool)
    if len(tree_indices) == 0:
        return {
            'cell_representative': representative,
            'tree_quota': quota,
            'coverage_critical': representative | quota,
        }

    cells = np.floor(
        votes[tree_indices] / float(cell_size)).astype(np.int64)
    tree_labels = labels[tree_indices]
    order = np.lexsort((
        tree_indices,
        -utility[tree_indices],
        cells[:, 1],
        cells[:, 0],
        tree_labels,
    ))
    ordered_indices = tree_indices[order]
    ordered_labels = tree_labels[order]
    ordered_cells = cells[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (
        (ordered_labels[1:] != ordered_labels[:-1]) |
        (ordered_cells[1:, 0] != ordered_cells[:-1, 0]) |
        (ordered_cells[1:, 1] != ordered_cells[:-1, 1]))
    representative[ordered_indices[first]] = True

    for tree_id in np.unique(tree_labels):
        indices = tree_indices[tree_labels == tree_id]
        keep_count = min(int(min_seeds_per_tree), len(indices))
        tree_order = np.lexsort((indices, -utility[indices]))
        quota[indices[tree_order[:keep_count]]] = True
    return {
        'cell_representative': representative,
        'tree_quota': quota,
        'coverage_critical': representative | quota,
    }


def build_seed_quality_artifact(
        coords, semantic_logits, offset_predictions, offset_labels,
        instance_labels, backbone_features, verticality,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0,
        tree_class_index=0, complexity_scales=(0.3, 0.6, 1.2),
        utility_sigma_m=0.3, purity_cell_size=0.6,
        purity_power=1.0, critical_cell_size=0.6,
        min_seeds_per_tree=50, reliability_threshold=0.5):
    """Build one forest's candidate-seed artifact without input leakage."""
    coords = np.asarray(coords, dtype=np.float32)
    logits = np.asarray(semantic_logits, dtype=np.float32)
    predictions = np.asarray(offset_predictions, dtype=np.float32)
    targets = np.asarray(offset_labels, dtype=np.float32)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    if backbone_features is None:
        raise ValueError('backbone_features are required.')
    backbone = np.asarray(backbone_features, dtype=np.float32)
    verticality = np.asarray(verticality, dtype=np.float32).reshape(-1)
    num_points = len(coords)
    if coords.shape != (num_points, 3):
        raise ValueError('coords must have shape [N, 3].')
    expected_lengths = [
        len(logits), len(predictions), len(targets), len(labels),
        len(backbone), len(verticality),
    ]
    if any(length != num_points for length in expected_lengths):
        raise ValueError('Pointwise seed arrays are not aligned.')
    if predictions.shape != (num_points, 3) or targets.shape != (num_points, 3):
        raise ValueError('Offset arrays must have shape [N, 3].')
    if backbone.ndim != 2 or backbone.shape[1] < 1:
        raise ValueError('backbone_features must have shape [N, D].')
    if float(utility_sigma_m) <= 0 or float(purity_power) < 0:
        raise ValueError('Utility parameters are invalid.')
    if not 0 <= float(reliability_threshold) <= 1:
        raise ValueError('reliability_threshold must be in [0, 1].')

    probabilities = tree_probabilities(logits, tree_class_index)
    candidate_mask = candidate_seed_mask(
        logits, verticality, predictions,
        tree_conf_thresh, tau_vert, tau_off, tree_class_index)
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        raise ValueError('No candidate base seeds were found.')

    candidate_coords = coords[candidate_indices]
    candidate_offsets = predictions[candidate_indices]
    candidate_offset_targets = targets[candidate_indices]
    candidate_labels = labels[candidate_indices]
    candidate_probability = probabilities[candidate_indices]
    candidate_verticality = verticality[candidate_indices]
    base_votes = candidate_coords[:, :2] + candidate_offsets[:, :2]
    predicted_height = np.maximum(-candidate_offsets[:, 2], 0.0)
    height_scale = max(float(np.quantile(predicted_height, 0.95)), 1.0)
    normalized_height = np.clip(predicted_height / height_scale, 0.0, 1.0)

    local_values, local_names = compute_local_complexity_features(
        base_votes, candidate_probability, candidate_verticality,
        normalized_height, complexity_scales)
    base_values = np.column_stack([
        candidate_probability,
        candidate_verticality,
        normalized_height,
        np.linalg.norm(candidate_offsets[:, :2], axis=1),
        np.abs(candidate_offsets[:, 2]),
    ]).astype(np.float32)
    base_names = [
        'tree_probability', 'verticality', 'normalized_height',
        'offset_xy_magnitude', 'offset_z_abs',
    ]
    scalar_features = np.column_stack(
        [base_values, local_values]).astype(np.float32)
    scalar_feature_names = base_names + local_names

    valid = candidate_labels >= 0
    tree = candidate_labels > 0
    vote_error = np.zeros(len(candidate_indices), dtype=np.float32)
    vote_error[tree] = np.linalg.norm(
        candidate_offsets[tree, :2] -
        candidate_offset_targets[tree, :2], axis=1)
    purity = vote_cell_purity(
        base_votes, candidate_labels, purity_cell_size)
    accuracy = np.exp(
        -0.5 * (vote_error / float(utility_sigma_m)) ** 2)
    utility = (
        accuracy * np.power(purity, float(purity_power))
    ).astype(np.float32)
    utility[~tree] = 0.0
    coverage = compute_coverage_targets(
        base_votes, candidate_labels, utility,
        critical_cell_size, min_seeds_per_tree)
    reliable = tree & (utility >= float(reliability_threshold))
    critical_tree_ids = np.unique(candidate_labels[
        tree & coverage['coverage_critical']])

    arrays = {
        'candidate_indices': candidate_indices.astype(np.int64),
        'coords': candidate_coords.astype(np.float32),
        'base_votes_xy': base_votes.astype(np.float32),
        'backbone_features': backbone[candidate_indices].astype(np.float16),
        'scalar_features': scalar_features,
        'target_valid': valid.astype(bool),
        'target_is_tree': tree.astype(bool),
        'target_tree_id': candidate_labels.astype(np.int64),
        'target_vote_error_xy': vote_error,
        'target_vote_cell_purity': purity,
        'target_utility': utility,
        'target_reliable': reliable.astype(bool),
        'target_cell_representative': coverage[
            'cell_representative'].astype(bool),
        'target_tree_quota': coverage['tree_quota'].astype(bool),
        'target_coverage_critical': coverage[
            'coverage_critical'].astype(bool),
    }
    if not np.isfinite(scalar_features).all():
        raise ValueError('Seed scalar features contain non-finite values.')
    if not np.isfinite(backbone[candidate_indices]).all():
        raise ValueError('Seed backbone features contain non-finite values.')
    metadata = {
        'num_source_points': int(num_points),
        'num_candidates': int(len(candidate_indices)),
        'num_valid_candidates': int(valid.sum()),
        'num_tree_candidates': int(tree.sum()),
        'num_non_tree_candidates': int(np.count_nonzero(valid & ~tree)),
        'num_unknown_candidates': int(np.count_nonzero(~valid)),
        'num_reliable_candidates': int(reliable.sum()),
        'num_critical_candidates': int(
            coverage['coverage_critical'].sum()),
        'num_supervised_trees': int(len(np.unique(candidate_labels[tree]))),
        'num_critical_trees': int(len(critical_tree_ids)),
        'backbone_dim': int(backbone.shape[1]),
        'scalar_dim': int(scalar_features.shape[1]),
        'scalar_feature_names': scalar_feature_names,
        'complexity_scales': [float(value) for value in complexity_scales],
        'predicted_height_p95_scale': height_scale,
        'target_parameters': {
            'utility_sigma_m': float(utility_sigma_m),
            'purity_cell_size': float(purity_cell_size),
            'purity_power': float(purity_power),
            'critical_cell_size': float(critical_cell_size),
            'min_seeds_per_tree': int(min_seeds_per_tree),
            'reliability_threshold': float(reliability_threshold),
        },
        'candidate_parameters': {
            'tree_conf_thresh': float(tree_conf_thresh),
            'tau_vert': float(tau_vert),
            'tau_off': float(tau_off),
            'tree_class_index': int(tree_class_index),
        },
    }
    return {'arrays': arrays, 'metadata': metadata}


def combine_seed_input_features(backbone_features, scalar_features):
    """Join frozen backbone and GT-free scalar features in FP32."""
    backbone = np.asarray(backbone_features, dtype=np.float32)
    scalar = np.asarray(scalar_features, dtype=np.float32)
    if backbone.ndim != 2 or scalar.ndim != 2:
        raise ValueError('Seed input features must be two-dimensional.')
    if len(backbone) != len(scalar):
        raise ValueError('Backbone and scalar seed features are not aligned.')
    values = np.column_stack([backbone, scalar]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Combined seed input features are non-finite.')
    return values


def build_seed_quality_mlp(input_dim, model_config):
    """Build the E1b shared MLP with reliability and coverage heads."""
    import torch
    from torch import nn

    hidden_dims = [int(value) for value in model_config['hidden_dims']]
    dropout = float(model_config.get('dropout', 0.0))
    if int(input_dim) <= 0 or not hidden_dims or any(
            value <= 0 for value in hidden_dims):
        raise ValueError('Seed-MLP dimensions must be positive.')
    if not 0 <= dropout < 1:
        raise ValueError('Seed-MLP dropout must be in [0, 1).')

    class SeedQualityMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            current_dim = int(input_dim)
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                current_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)
            self.reliability_head = nn.Linear(current_dim, 1)
            self.coverage_head = nn.Linear(current_dim, 1)

        def forward(self, values):
            encoded = self.encoder(values)
            return (
                self.reliability_head(encoded).squeeze(1),
                self.coverage_head(encoded).squeeze(1),
            )

    return SeedQualityMLP()


def exact_top_k_mask(scores, count):
    """Select an exact number of high scores with stable index tie-breaking."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Seed selection scores must be finite.')
    count = int(count)
    if not 0 <= count <= len(values):
        raise ValueError('Top-k count is outside the score array.')
    selected = np.zeros(len(values), dtype=bool)
    if count:
        order = np.lexsort((np.arange(len(values)), -values))
        selected[order[:count]] = True
    return selected


def select_seed_candidates(
        reliability_scores, coverage_scores, keep_ratio=0.78,
        coverage_protect_ratio=0.10):
    """Protect high-coverage candidates, then fill by reliability ranking."""
    reliability = np.asarray(
        reliability_scores, dtype=np.float64).reshape(-1)
    coverage = np.asarray(coverage_scores, dtype=np.float64).reshape(-1)
    if len(reliability) != len(coverage):
        raise ValueError('Reliability and coverage scores are not aligned.')
    if not 0 < float(keep_ratio) <= 1:
        raise ValueError('keep_ratio must be in (0, 1].')
    if not 0 <= float(coverage_protect_ratio) <= float(keep_ratio):
        raise ValueError(
            'coverage_protect_ratio must be in [0, keep_ratio].')
    keep_count = min(
        len(reliability), int(np.ceil(len(reliability) * keep_ratio)))
    protect_count = min(
        keep_count,
        int(np.ceil(len(reliability) * coverage_protect_ratio)))
    protected = exact_top_k_mask(coverage, protect_count)
    selected = protected.copy()
    remaining_count = keep_count - int(selected.sum())
    if remaining_count:
        eligible = np.flatnonzero(~selected)
        fill = exact_top_k_mask(reliability[eligible], remaining_count)
        selected[eligible[fill]] = True
    return selected, {
        'num_candidates': int(len(reliability)),
        'num_kept': int(selected.sum()),
        'num_coverage_protected': int(protected.sum()),
        'keep_ratio': float(keep_ratio),
        'coverage_protect_ratio': float(coverage_protect_ratio),
    }


def save_seed_quality_artifact(
        artifact, output_dir, source_plot, split, metadata=None):
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    arrays = artifact['arrays']
    payload = dict(artifact['metadata'])
    payload.update(metadata or {})
    payload['source_plot'] = str(source_plot)
    payload['split'] = str(split)
    npz_path = os.path.join(output_dir, SEED_ARTIFACT_NAME)
    metadata_path = os.path.join(output_dir, SEED_METADATA_NAME)
    np.savez_compressed(npz_path, **arrays)
    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return npz_path, metadata_path
