import csv
import json
import os

import numpy as np


def _group_sum(values, groups, size):
    return np.bincount(groups, weights=values, minlength=size)


def _group_mean(values, groups, counts, size):
    sums = _group_sum(values, groups, size)
    return np.divide(
        sums, counts,
        out=np.zeros(size, dtype=np.float64), where=counts > 0)


def _group_max(values, groups, counts, size):
    maxima = np.full(size, -np.inf, dtype=np.float64)
    np.maximum.at(maxima, groups, values)
    maxima[counts == 0] = 0.0
    return maxima


def _group_std(values, groups, counts, size):
    means = _group_mean(values, groups, counts, size)
    squared_means = _group_mean(
        np.asarray(values, dtype=np.float64) ** 2,
        groups, counts, size)
    return np.sqrt(np.maximum(squared_means - means ** 2, 0.0))


def _tree_probabilities(logits, tree_class_index):
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials[:, tree_class_index] / exponentials.sum(axis=1)


def compute_vertical_instance_tokens(
        coords, instance_predictions, backbone_features,
        semantic_prediction_logits, offset_predictions, verticality,
        axis_confidence, num_layers=8, tree_class_index=0):
    """Aggregate frozen point features into fixed vertical instance tokens."""
    if num_layers <= 0:
        raise ValueError('num_layers must be positive.')
    arrays = [
        coords, instance_predictions, backbone_features,
        semantic_prediction_logits, offset_predictions, verticality]
    num_points = len(instance_predictions)
    if any(len(value) != num_points for value in arrays):
        raise ValueError('All pointwise arrays must have the same length.')
    if backbone_features is None:
        raise ValueError('backbone_features are required for vertical tokens.')
    if axis_confidence is None:
        axis_confidence = np.zeros(num_points, dtype=np.float32)
    if len(axis_confidence) != num_points:
        raise ValueError('axis_confidence must match the point count.')

    predictions = np.asarray(instance_predictions, dtype=np.int64)
    positive_mask = predictions > 0
    if not np.any(positive_mask):
        raise ValueError('No positive predicted instances are available.')
    labels = predictions[positive_mask]
    max_label = int(labels.max())
    instance_counts = np.bincount(
        labels, minlength=max_label + 1).astype(np.int64)
    instance_ids = np.flatnonzero(instance_counts > 0)
    instance_ids = instance_ids[instance_ids > 0]

    coords = np.asarray(coords, dtype=np.float64)[positive_mask]
    backbone = np.asarray(
        backbone_features, dtype=np.float32)[positive_mask]
    offsets = np.asarray(offset_predictions, dtype=np.float64)[positive_mask]
    verticality = np.asarray(verticality).reshape(-1)[positive_mask]
    confidence = np.asarray(axis_confidence).reshape(-1)[positive_mask]
    semantic_probability = _tree_probabilities(
        semantic_prediction_logits, tree_class_index)[positive_mask]

    size = max_label + 1
    counts_float = instance_counts.astype(np.float64)
    min_z = np.full(size, np.inf, dtype=np.float64)
    max_z = np.full(size, -np.inf, dtype=np.float64)
    np.minimum.at(min_z, labels, coords[:, 2])
    np.maximum.at(max_z, labels, coords[:, 2])
    heights = np.maximum(max_z - min_z, 1e-6)
    relative_height = np.clip(
        (coords[:, 2] - min_z[labels]) / heights[labels], 0.0, 1.0)
    layer_indices = np.minimum(
        (relative_height * num_layers).astype(np.int64), num_layers - 1)
    layer_groups = labels * num_layers + layer_indices
    layer_size = (max_label + 1) * num_layers
    layer_counts = np.bincount(
        layer_groups, minlength=layer_size).astype(np.int64)

    mean_x = _group_mean(
        coords[:, 0], labels, counts_float, size)
    mean_y = _group_mean(
        coords[:, 1], labels, counts_float, size)
    xy_radius = np.sqrt(
        (coords[:, 0] - mean_x[labels]) ** 2 +
        (coords[:, 1] - mean_y[labels]) ** 2)
    radius_rms = np.sqrt(_group_mean(
        xy_radius ** 2, labels, counts_float, size))
    normalized_radius = np.divide(
        xy_radius, np.maximum(radius_rms[labels], 0.05))

    base_votes = coords[:, :2] + offsets[:, :2]
    mean_vote_x = _group_mean(
        base_votes[:, 0], labels, counts_float, size)
    mean_vote_y = _group_mean(
        base_votes[:, 1], labels, counts_float, size)
    base_vote_radius = np.sqrt(
        (base_votes[:, 0] - mean_vote_x[labels]) ** 2 +
        (base_votes[:, 1] - mean_vote_y[labels]) ** 2)

    backbone_dim = int(backbone.shape[1])
    scalar_names = [
        'occupancy_fraction',
        'relative_height_mean',
        'normalized_xy_radius_mean',
        'semantic_probability_mean',
        'verticality_mean',
        'axis_confidence_mean',
        'axis_confidence_std',
        'base_vote_radius_rms',
    ]
    feature_names = [
        f'backbone_mean_{index}' for index in range(backbone_dim)
    ] + [
        f'backbone_max_{index}' for index in range(backbone_dim)
    ] + scalar_names
    token_values = np.zeros(
        (max_label + 1, num_layers, len(feature_names)),
        dtype=np.float32)
    layer_counts_float = layer_counts.astype(np.float64)

    for column in range(backbone_dim):
        means = _group_mean(
            backbone[:, column], layer_groups,
            layer_counts_float, layer_size)
        token_values[:, :, column] = means.reshape(
            max_label + 1, num_layers)
        maxima = _group_max(
            backbone[:, column], layer_groups,
            layer_counts, layer_size)
        token_values[:, :, backbone_dim + column] = maxima.reshape(
            max_label + 1, num_layers)

    scalar_values = [
        np.ones(len(labels), dtype=np.float64),
        relative_height,
        normalized_radius,
        semantic_probability,
        verticality,
        confidence,
        confidence,
        base_vote_radius,
    ]
    scalar_start = backbone_dim * 2
    for index, values in enumerate(scalar_values):
        if index == 0:
            means = np.divide(
                layer_counts,
                np.repeat(instance_counts, num_layers),
                out=np.zeros(layer_size, dtype=np.float64),
                where=np.repeat(instance_counts, num_layers) > 0)
        elif index == 6:
            means = _group_std(
                values, layer_groups, layer_counts, layer_size)
        elif index == 7:
            means = np.sqrt(_group_mean(
                values ** 2, layer_groups,
                layer_counts_float, layer_size))
        else:
            means = _group_mean(
                values, layer_groups, layer_counts_float, layer_size)
        token_values[:, :, scalar_start + index] = means.reshape(
            max_label + 1, num_layers)

    layer_valid_mask = layer_counts.reshape(
        max_label + 1, num_layers) > 0
    return {
        'instance_ids': instance_ids.astype(np.int64),
        'vertical_tokens': token_values[instance_ids],
        'layer_valid_mask': layer_valid_mask[instance_ids],
        'token_feature_names': feature_names,
        'num_layers': int(num_layers),
    }


def compute_instance_quality_targets(
        coords, instance_predictions, instance_labels,
        min_labeled_fraction=0.5, match_iou_threshold=0.5,
        negative_iou_threshold=0.25, edge_margin_m=0.5):
    """Compute maximum GT IoU and validity for every predicted instance."""
    coords = np.asarray(coords, dtype=np.float64)
    predictions = np.asarray(instance_predictions, dtype=np.int64).reshape(-1)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    if len(coords) != len(predictions) or len(labels) != len(predictions):
        raise ValueError('coords, predictions and labels must be aligned.')
    pred_mask = predictions > 0
    if not np.any(pred_mask):
        raise ValueError('No positive predicted instances are available.')
    instance_ids, pred_counts = np.unique(
        predictions[pred_mask], return_counts=True)
    gt_ids, gt_counts = np.unique(labels[labels > 0], return_counts=True)
    if len(gt_ids) == 0:
        raise ValueError('No positive GT tree instances are available.')
    if instance_ids.max() > np.iinfo(np.uint32).max:
        raise ValueError('Predicted instance IDs exceed uint32 range.')
    if gt_ids.max() > np.iinfo(np.uint32).max:
        raise ValueError('GT instance IDs exceed uint32 range.')

    both = (predictions > 0) & (labels > 0)
    encoded = (
        (predictions[both].astype(np.uint64) << np.uint64(32)) |
        labels[both].astype(np.uint64))
    encoded_pairs, intersections = np.unique(encoded, return_counts=True)
    max_iou = np.zeros(len(instance_ids), dtype=np.float64)
    best_gt_id = np.zeros(len(instance_ids), dtype=np.int64)
    tree_intersections = np.zeros(len(instance_ids), dtype=np.int64)
    for encoded_pair, intersection in zip(encoded_pairs, intersections):
        integer = int(encoded_pair)
        pred_id = integer >> 32
        gt_id = integer & 0xffffffff
        pred_index = int(np.searchsorted(instance_ids, pred_id))
        gt_index = int(np.searchsorted(gt_ids, gt_id))
        intersection = int(intersection)
        tree_intersections[pred_index] += intersection
        union = pred_counts[pred_index] + gt_counts[gt_index] - intersection
        iou = intersection / union
        if iou > max_iou[pred_index]:
            max_iou[pred_index] = iou
            best_gt_id[pred_index] = gt_id

    tree_point_fraction = tree_intersections / pred_counts
    classified_mask = pred_mask & (labels >= 0)
    classified_by_id = np.bincount(
        predictions[classified_mask],
        minlength=int(instance_ids.max()) + 1)
    labeled_fraction = classified_by_id[instance_ids] / pred_counts
    edge_instance_ids = set()
    if edge_margin_m > 0:
        mins = coords[:, :2].min(axis=0)
        maxs = coords[:, :2].max(axis=0)
        edge_mask = (
            (coords[:, 0] <= mins[0] + edge_margin_m) |
            (coords[:, 0] >= maxs[0] - edge_margin_m) |
            (coords[:, 1] <= mins[1] + edge_margin_m) |
            (coords[:, 1] >= maxs[1] - edge_margin_m))
        edge_instance_ids = set(
            np.unique(predictions[edge_mask & pred_mask]).tolist())
    is_edge = np.asarray([
        int(instance_id) in edge_instance_ids
        for instance_id in instance_ids], dtype=bool)
    target_valid = (
        (labeled_fraction >= min_labeled_fraction) & (~is_edge))
    if not 0 <= negative_iou_threshold < match_iou_threshold <= 1:
        raise ValueError(
            'IoU thresholds must satisfy 0 <= negative < match <= 1.')
    target_is_true_tree = max_iou >= match_iou_threshold
    target_classification_valid = target_valid & (
        target_is_true_tree | (max_iou < negative_iou_threshold))
    return {
        'instance_ids': instance_ids.astype(np.int64),
        'target_max_iou': max_iou.astype(np.float32),
        'target_best_gt_id': best_gt_id,
        'target_labeled_fraction': labeled_fraction.astype(np.float32),
        'target_tree_point_fraction': tree_point_fraction.astype(np.float32),
        'target_is_edge': is_edge,
        'target_valid': target_valid,
        'target_classification_valid': target_classification_valid,
        'target_is_true_tree': target_is_true_tree,
    }


def save_instance_quality_data(
        global_features, token_data, target_data, output_dir,
        source_plot, split, metadata=None):
    os.makedirs(output_dir, exist_ok=True)
    instance_ids = np.asarray(token_data['instance_ids'], dtype=np.int64)
    if not np.array_equal(instance_ids, target_data['instance_ids']):
        raise ValueError('Token and target instance IDs are not aligned.')
    indexed = global_features.set_index('instance_id')
    missing = set(instance_ids.tolist()) - set(indexed.index.astype(int))
    if missing:
        raise ValueError(
            f'Global features miss instance IDs: {sorted(missing)[:10]}')
    indexed = indexed.loc[instance_ids]
    global_feature_names = list(indexed.columns)
    global_feature_values = indexed.to_numpy(dtype=np.float32)

    npz_path = os.path.join(output_dir, 'quality_instances.npz')
    np.savez_compressed(
        npz_path,
        instance_ids=instance_ids,
        global_feature_values=global_feature_values,
        global_feature_names=np.asarray(global_feature_names),
        vertical_tokens=token_data['vertical_tokens'],
        layer_valid_mask=token_data['layer_valid_mask'],
        token_feature_names=np.asarray(token_data['token_feature_names']),
        target_max_iou=target_data['target_max_iou'],
        target_best_gt_id=target_data['target_best_gt_id'],
        target_labeled_fraction=target_data['target_labeled_fraction'],
        target_tree_point_fraction=target_data['target_tree_point_fraction'],
        target_is_edge=target_data['target_is_edge'],
        target_valid=target_data['target_valid'],
        target_classification_valid=(
            target_data['target_classification_valid']),
        target_is_true_tree=target_data['target_is_true_tree'],
        source_plot=np.asarray(source_plot),
        split=np.asarray(split),
    )

    csv_path = os.path.join(output_dir, 'quality_targets.csv')
    fieldnames = [
        'instance_id', 'target_max_iou', 'target_best_gt_id',
        'target_labeled_fraction', 'target_tree_point_fraction',
        'target_is_edge',
        'target_valid', 'target_classification_valid',
        'target_is_true_tree', 'source_plot', 'split']
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, instance_id in enumerate(instance_ids):
            writer.writerow({
                'instance_id': int(instance_id),
                'target_max_iou': float(target_data['target_max_iou'][index]),
                'target_best_gt_id': int(
                    target_data['target_best_gt_id'][index]),
                'target_labeled_fraction': float(
                    target_data['target_labeled_fraction'][index]),
                'target_tree_point_fraction': float(
                    target_data['target_tree_point_fraction'][index]),
                'target_is_edge': bool(target_data['target_is_edge'][index]),
                'target_valid': bool(target_data['target_valid'][index]),
                'target_classification_valid': bool(
                    target_data['target_classification_valid'][index]),
                'target_is_true_tree': bool(
                    target_data['target_is_true_tree'][index]),
                'source_plot': source_plot,
                'split': split,
            })

    payload = dict(metadata or {})
    valid = target_data['target_valid']
    classification_valid = target_data['target_classification_valid']
    positive = target_data['target_is_true_tree'] & classification_valid
    negative = (~target_data['target_is_true_tree']) & classification_valid
    payload.update({
        'source_plot': source_plot,
        'split': split,
        'num_instances': int(len(instance_ids)),
        'num_valid_instances': int(valid.sum()),
        'num_classification_valid_instances': int(
            classification_valid.sum()),
        'num_positive_instances': int(positive.sum()),
        'num_negative_instances': int(negative.sum()),
        'num_layers': int(token_data['num_layers']),
        'global_feature_names': global_feature_names,
        'token_feature_names': token_data['token_feature_names'],
    })
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return npz_path, csv_path, metadata_path
