import json
import os

import numpy as np
import pandas as pd


def _safe_probabilities(logits, class_index):
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials[:, class_index] / exponentials.sum(axis=1)


def _group_counts(labels, max_label):
    return np.bincount(labels, minlength=max_label + 1).astype(np.int64)


def _group_mean_std(labels, values, counts, max_label):
    values = np.asarray(values, dtype=np.float64)
    sums = np.bincount(labels, weights=values, minlength=max_label + 1)
    squared_sums = np.bincount(
        labels, weights=values * values, minlength=max_label + 1)
    means = np.divide(
        sums, counts, out=np.full(max_label + 1, np.nan), where=counts > 0)
    variances = np.divide(
        squared_sums, counts, out=np.zeros(max_label + 1), where=counts > 0)
    variances = np.maximum(variances - np.nan_to_num(means) ** 2, 0.0)
    standard_deviations = np.sqrt(variances)
    standard_deviations[counts == 0] = np.nan
    return means, standard_deviations


def _bounded_group_quantiles(
        labels, values, counts, max_label, quantiles=(0.1, 0.5, 0.9),
        num_bins=100):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    bin_indices = np.minimum((values * num_bins).astype(np.int64), num_bins - 1)
    flattened = labels * num_bins + bin_indices
    histogram = np.bincount(
        flattened, minlength=(max_label + 1) * num_bins)
    histogram = histogram.reshape(max_label + 1, num_bins)
    cumulative = np.cumsum(histogram, axis=1)
    output = {}
    for quantile in quantiles:
        targets = np.maximum(np.ceil(counts * quantile).astype(np.int64), 1)
        bins = np.argmax(cumulative >= targets[:, None], axis=1)
        estimates = (bins.astype(np.float64) + 0.5) / num_bins
        estimates[counts == 0] = np.nan
        output[quantile] = estimates
    return output


def _group_min_max(labels, values, counts, max_label):
    minimum = np.full(max_label + 1, np.inf, dtype=np.float64)
    maximum = np.full(max_label + 1, -np.inf, dtype=np.float64)
    np.minimum.at(minimum, labels, values)
    np.maximum.at(maximum, labels, values)
    minimum[counts == 0] = np.nan
    maximum[counts == 0] = np.nan
    return minimum, maximum


def _add_bounded_stats(columns, prefix, labels, values, counts, max_label):
    mean, standard_deviation = _group_mean_std(
        labels, values, counts, max_label)
    quantiles = _bounded_group_quantiles(
        labels, values, counts, max_label)
    columns[f'{prefix}_mean'] = mean
    columns[f'{prefix}_std'] = standard_deviation
    columns[f'{prefix}_p10'] = quantiles[0.1]
    columns[f'{prefix}_p50'] = quantiles[0.5]
    columns[f'{prefix}_p90'] = quantiles[0.9]


def _radial_rms(labels, x, y, counts, max_label):
    mean_x, _ = _group_mean_std(labels, x, counts, max_label)
    mean_y, _ = _group_mean_std(labels, y, counts, max_label)
    squared_radius = (
        (x - mean_x[labels]) ** 2 +
        (y - mean_y[labels]) ** 2)
    mean_squared_radius, _ = _group_mean_std(
        labels, squared_radius, counts, max_label)
    return np.sqrt(mean_squared_radius)


def compute_instance_features(
        coords, instance_predictions, initial_instance_predictions,
        semantic_prediction_logits, offset_predictions, verticality,
        axis_confidence, tree_class_index=0):
    arrays = [
        coords, instance_predictions, initial_instance_predictions,
        semantic_prediction_logits, offset_predictions, verticality,
        axis_confidence]
    num_points = len(instance_predictions)
    if any(len(array) != num_points for array in arrays):
        raise ValueError('All pointwise arrays must have the same length.')
    if axis_confidence is None:
        raise ValueError('axis_confidence is required for instance diagnostics.')

    instance_predictions = np.asarray(instance_predictions, dtype=np.int64)
    instance_mask = instance_predictions > 0
    if not np.any(instance_mask):
        raise ValueError('No positive predicted instances are available.')

    labels = instance_predictions[instance_mask]
    max_label = int(labels.max())
    counts = _group_counts(labels, max_label)
    instance_ids = np.flatnonzero(counts > 0)
    instance_ids = instance_ids[instance_ids > 0]

    coords = np.asarray(coords, dtype=np.float64)[instance_mask]
    offsets = np.asarray(offset_predictions, dtype=np.float64)[instance_mask]
    verticality = np.asarray(verticality).reshape(-1)[instance_mask]
    confidence = np.asarray(axis_confidence).reshape(-1)[instance_mask]
    semantic_probability = _safe_probabilities(
        semantic_prediction_logits, tree_class_index)[instance_mask]

    columns = {
        'instance_id': np.arange(max_label + 1, dtype=np.int64),
        'num_points': counts,
    }
    _add_bounded_stats(
        columns, 'confidence', labels, confidence, counts, max_label)
    _add_bounded_stats(
        columns, 'semantic_probability', labels, semantic_probability,
        counts, max_label)
    _add_bounded_stats(
        columns, 'verticality', labels, verticality, counts, max_label)

    initial_predictions = np.asarray(
        initial_instance_predictions, dtype=np.int64)[instance_mask]
    seed_mask = initial_predictions > 0
    seed_labels = labels[seed_mask]
    seed_counts = _group_counts(seed_labels, max_label)
    columns['num_clustered_seeds'] = seed_counts
    columns['seed_fraction'] = np.divide(
        seed_counts, counts,
        out=np.zeros(max_label + 1, dtype=np.float64), where=counts > 0)
    _add_bounded_stats(
        columns, 'seed_confidence', seed_labels, confidence[seed_mask],
        seed_counts, max_label)

    min_x, max_x = _group_min_max(
        labels, coords[:, 0], counts, max_label)
    min_y, max_y = _group_min_max(
        labels, coords[:, 1], counts, max_label)
    min_z, max_z = _group_min_max(
        labels, coords[:, 2], counts, max_label)
    columns['x_span'] = max_x - min_x
    columns['y_span'] = max_y - min_y
    columns['height'] = max_z - min_z
    columns['bbox_area'] = columns['x_span'] * columns['y_span']
    columns['point_density_bbox'] = np.divide(
        counts, np.maximum(columns['bbox_area'], 0.01))
    columns['xy_radius_rms'] = _radial_rms(
        labels, coords[:, 0], coords[:, 1], counts, max_label)

    base_votes = coords[:, :2] + offsets[:, :2]
    columns['base_vote_radius_rms'] = _radial_rms(
        labels, base_votes[:, 0], base_votes[:, 1], counts, max_label)
    columns['seed_base_vote_radius_rms'] = _radial_rms(
        seed_labels, base_votes[seed_mask, 0], base_votes[seed_mask, 1],
        seed_counts, max_label)

    offset_xy_magnitude = np.linalg.norm(offsets[:, :2], axis=1)
    offset_xy_mean, offset_xy_std = _group_mean_std(
        labels, offset_xy_magnitude, counts, max_label)
    offset_z_abs_mean, offset_z_abs_std = _group_mean_std(
        labels, np.abs(offsets[:, 2]), counts, max_label)
    columns['offset_xy_magnitude_mean'] = offset_xy_mean
    columns['offset_xy_magnitude_std'] = offset_xy_std
    columns['offset_z_abs_mean'] = offset_z_abs_mean
    columns['offset_z_abs_std'] = offset_z_abs_std

    frame = pd.DataFrame(columns).iloc[instance_ids].reset_index(drop=True)
    integer_columns = ['instance_id', 'num_points', 'num_clustered_seeds']
    frame[integer_columns] = frame[integer_columns].astype(np.int64)
    return frame


def save_instance_features(frame, output_dir, metadata=None):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'instance_features.csv')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    frame.to_csv(csv_path, index=False)
    payload = dict(metadata or {})
    payload.update({
        'num_instances': int(len(frame)),
        'feature_columns': list(frame.columns),
    })
    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return csv_path, metadata_path
