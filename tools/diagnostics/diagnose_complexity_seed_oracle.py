"""Diagnose the upper bound of GT-guided sparse seed selection.

The diagnostic performs pointwise inference and overlap ensembling once, then
reuses the identical predictions for full-seed, random-ratio, global-oracle,
and tree-balanced-oracle clustering.  It is deliberately an offline Oracle:
ground-truth labels and offset targets must never enter a deployable pipeline.
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np


TREE_CLASS = 0
NON_TREE = 0
NOT_ASSIGNED = -1
START_INSTANCE = 1


def candidate_base_seed_mask(
        semantic_logits, verticality, offset_predictions,
        tree_conf_thresh, tau_vert, tau_off):
    # Match get_instances exactly: its candidate mask uses a FP32 PyTorch
    # softmax. Keeping this identical matters for points close to the semantic
    # threshold because the Oracle score is passed back into get_instances.
    import torch

    logits = np.asarray(semantic_logits)
    probabilities = torch.from_numpy(logits).float().softmax(dim=-1).numpy()
    return (
        (probabilities[:, TREE_CLASS] >= float(tree_conf_thresh)) &
        (np.asarray(verticality).reshape(-1) > float(tau_vert)) &
        (np.abs(np.asarray(offset_predictions)[:, 2]) < float(tau_off)))


def vote_cell_purity(base_votes_xy, labels, voxel_size):
    """Return same-GT-label fraction inside each quantized vote cell."""
    votes = np.asarray(base_votes_xy, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if votes.shape != (len(labels), 2):
        raise ValueError('base_votes_xy and labels are not aligned.')
    if float(voxel_size) <= 0:
        raise ValueError('purity voxel size must be positive.')
    cells = np.floor(votes / float(voxel_size)).astype(np.int64)
    cell_dtype = np.dtype([('x', '<i8'), ('y', '<i8')])
    cell_keys = np.ascontiguousarray(cells).view(cell_dtype).reshape(-1)
    _, cell_inverse, cell_counts = np.unique(
        cell_keys, return_inverse=True, return_counts=True)

    pair_values = np.column_stack([cells, labels])
    pair_dtype = np.dtype([('x', '<i8'), ('y', '<i8'), ('label', '<i8')])
    pair_keys = np.ascontiguousarray(pair_values).view(pair_dtype).reshape(-1)
    _, pair_inverse, pair_counts = np.unique(
        pair_keys, return_inverse=True, return_counts=True)
    return (
        pair_counts[pair_inverse] / cell_counts[cell_inverse]
    ).astype(np.float32)


def compute_seed_utility(
        coords, offset_predictions, offset_labels, instance_labels,
        candidate_mask, sigma_m=0.30, purity_voxel_size=0.60,
        purity_power=1.0):
    """Compute a GT-only seed utility from vote accuracy and local purity."""
    coords = np.asarray(coords, dtype=np.float64)
    predictions = np.asarray(offset_predictions, dtype=np.float64)
    targets = np.asarray(offset_labels, dtype=np.float64)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    num_points = len(coords)
    if (
            predictions.shape != (num_points, 3) or
            targets.shape != (num_points, 3) or
            len(labels) != num_points or len(candidate_mask) != num_points):
        raise ValueError('Oracle point arrays are not aligned.')
    if float(sigma_m) <= 0 or float(purity_power) < 0:
        raise ValueError('sigma must be positive and purity_power non-negative.')

    candidate_indices = np.flatnonzero(candidate_mask)
    utility = np.zeros(num_points, dtype=np.float32)
    vote_error = np.full(num_points, np.nan, dtype=np.float32)
    purity = np.zeros(num_points, dtype=np.float32)
    if len(candidate_indices) == 0:
        return {
            'utility': utility,
            'vote_error_xy': vote_error,
            'purity': purity,
        }

    candidate_error = np.linalg.norm(
        predictions[candidate_indices, :2] -
        targets[candidate_indices, :2], axis=1)
    base_votes = (
        coords[candidate_indices, :2] +
        predictions[candidate_indices, :2])
    candidate_purity = vote_cell_purity(
        base_votes, labels[candidate_indices], purity_voxel_size)
    accuracy = np.exp(
        -0.5 * (candidate_error / float(sigma_m)) ** 2)
    candidate_utility = accuracy * np.power(
        candidate_purity, float(purity_power))
    candidate_utility[labels[candidate_indices] <= 0] = 0.0
    utility[candidate_indices] = candidate_utility.astype(np.float32)
    vote_error[candidate_indices] = candidate_error.astype(np.float32)
    purity[candidate_indices] = candidate_purity
    if not np.isfinite(utility).all():
        raise ValueError('Oracle utility contains non-finite values.')
    return {
        'utility': utility,
        'vote_error_xy': vote_error,
        'purity': purity,
    }


def tree_balanced_oracle_scores(raw_utility, instance_labels, candidate_mask):
    """Convert raw utility to deterministic within-tree percentile scores."""
    utility = np.asarray(raw_utility, dtype=np.float64).reshape(-1)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    if len(utility) != len(labels) or len(labels) != len(candidate_mask):
        raise ValueError('Balanced-oracle arrays are not aligned.')
    scores = np.zeros(len(utility), dtype=np.float32)
    indices = np.arange(len(utility), dtype=np.int64)
    for tree_id in np.unique(labels[candidate_mask & (labels > 0)]):
        tree_indices = indices[candidate_mask & (labels == tree_id)]
        order = np.lexsort((tree_indices, utility[tree_indices]))
        ranks = np.empty(len(tree_indices), dtype=np.float64)
        ranks[order] = np.arange(1, len(tree_indices) + 1)
        scores[tree_indices] = (ranks / len(tree_indices)).astype(np.float32)
    return scores


def seed_set_metrics(
        selected_mask, candidate_mask, labels, vote_error, utility,
        tau_min):
    selected = np.asarray(selected_mask, dtype=bool)
    candidates = np.asarray(candidate_mask, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    valid_selected = selected & candidates
    selected_labels = labels[valid_selected]
    all_tree_ids = np.unique(labels[candidates & (labels > 0)])
    selected_tree_ids, selected_counts = np.unique(
        selected_labels[selected_labels > 0], return_counts=True)
    enough_ids = selected_tree_ids[selected_counts >= int(tau_min)]
    selected_errors = np.asarray(vote_error)[valid_selected]
    selected_errors = selected_errors[np.isfinite(selected_errors)]
    selected_utilities = np.asarray(utility)[valid_selected]
    return {
        'candidate_count': int(candidates.sum()),
        'selected_count': int(valid_selected.sum()),
        'selected_ratio': float(
            valid_selected.sum() / max(candidates.sum(), 1)),
        'tree_seed_coverage': float(
            len(selected_tree_ids) / max(len(all_tree_ids), 1)),
        'tree_tau_min_coverage': float(
            len(enough_ids) / max(len(all_tree_ids), 1)),
        'mean_vote_error_xy': float(
            selected_errors.mean() if len(selected_errors) else np.nan),
        'p90_vote_error_xy': float(
            np.quantile(selected_errors, 0.9)
            if len(selected_errors) else np.nan),
        'mean_raw_utility': float(
            selected_utilities.mean() if len(selected_utilities) else 0.0),
    }


def exact_top_ratio_mask(candidate_mask, scores, keep_ratio):
    candidates = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(candidate_mask):
        raise ValueError('Selection scores must match candidate_mask.')
    if not 0 < float(keep_ratio) <= 1:
        raise ValueError('keep_ratio must be in (0, 1].')
    keep_count = int(np.ceil(float(keep_ratio) * len(candidates)))
    order = np.lexsort((candidates, -scores[candidates]))
    selected = np.zeros(len(candidate_mask), dtype=bool)
    selected[candidates[order[:keep_count]]] = True
    return selected

def streaming_xyz_mean(path, chunk_size=2_000_000):
    """Compute a source-forest mean without loading a large LAS/LAZ at once."""
    if str(path).lower().endswith(('.las', '.laz')):
        import laspy

        coordinate_sum = np.zeros(3, dtype=np.float64)
        count = 0
        with laspy.open(path) as reader:
            for points in reader.chunk_iterator(int(chunk_size)):
                coordinate_sum[0] += np.asarray(points.x).sum(dtype=np.float64)
                coordinate_sum[1] += np.asarray(points.y).sum(dtype=np.float64)
                coordinate_sum[2] += np.asarray(points.z).sum(dtype=np.float64)
                count += len(points)
        if count == 0:
            raise ValueError(f'Cannot compute coordinate mean of empty file: {path}')
        return coordinate_sum / count

    from tree_learn.util import load_data
    coords = load_data(path)[:, :3]
    if len(coords) == 0:
        raise ValueError(f'Cannot compute coordinate mean of empty file: {path}')
    return coords.astype(np.float64).mean(axis=0)

def load_xyz_bounded(path, chunk_size=2_000_000):
    """Load XYZ while bounding temporary LAS/LAZ decompression memory."""
    if str(path).lower().endswith(('.las', '.laz')):
        import laspy

        with laspy.open(path) as reader:
            coords = np.empty(
                (reader.header.point_count, 3), dtype=np.float64)
            start = 0
            for points in reader.chunk_iterator(int(chunk_size)):
                end = start + len(points)
                coords[start:end, 0] = np.asarray(points.x)
                coords[start:end, 1] = np.asarray(points.y)
                coords[start:end, 2] = np.asarray(points.z)
                start = end
        if start != len(coords):
            raise RuntimeError('LAS/LAZ point count changed while loading XYZ.')
        return coords

    from tree_learn.util import load_data
    return load_data(path)[:, :3].astype(np.float64, copy=True)



def load_prediction_xyz_labels_bounded(path, chunk_size=2_000_000):
    """Load saved prediction XYZ and tree IDs with bounded decode memory."""
    if str(path).lower().endswith(('.las', '.laz')):
        import laspy

        with laspy.open(path) as reader:
            num_points = int(reader.header.point_count)
            coords = np.empty((num_points, 3), dtype=np.float64)
            labels = np.empty(num_points, dtype=np.int64)
            start = 0
            for points in reader.chunk_iterator(int(chunk_size)):
                end = start + len(points)
                coords[start:end, 0] = np.asarray(points.x)
                coords[start:end, 1] = np.asarray(points.y)
                coords[start:end, 2] = np.asarray(points.z)
                try:
                    labels[start:end] = np.asarray(
                        points.treeID, dtype=np.int64)
                except AttributeError as error:
                    raise ValueError(
                        f'Prediction file has no treeID field: {path}') from error
                start = end
        if start != len(coords):
            raise RuntimeError('LAS/LAZ point count changed while loading.')
        return coords, labels

    from tree_learn.util import load_data
    data = load_data(path)
    return (
        data[:, :3].astype(np.float64, copy=True),
        data[:, 3].astype(np.int64, copy=True))


def nearest_source_indices(
        source_coords, target_coords, logger=None, chunk_size=1_000_000,
        n_neighbors=1):
    """Map every target point to its K nearest source points."""
    from scipy.spatial import cKDTree

    source = np.asarray(source_coords, dtype=np.float64)
    target = np.asarray(target_coords)
    num_neighbors = int(n_neighbors)
    if len(source) == 0 or len(target) == 0:
        raise ValueError('Nearest-neighbour alignment requires non-empty arrays.')
    if not 1 <= num_neighbors <= len(source):
        raise ValueError('n_neighbors must be between 1 and source size.')
    if len(source) > np.iinfo(np.int32).max:
        raise ValueError('Source array is too large for int32 neighbour indices.')
    tree = cKDTree(source)
    output_shape = (
        (len(target),) if num_neighbors == 1
        else (len(target), num_neighbors))
    indices = np.empty(output_shape, dtype=np.int32)
    distances = np.empty(len(target), dtype=np.float32)
    for start in range(0, len(target), int(chunk_size)):
        end = min(start + int(chunk_size), len(target))
        query_coords = np.asarray(target[start:end], dtype=np.float64)
        try:
            distance, index = tree.query(
                query_coords, k=num_neighbors, workers=-1)
        except TypeError:
            try:
                distance, index = tree.query(
                    query_coords, k=num_neighbors, n_jobs=-1)
            except TypeError:
                distance, index = tree.query(
                    query_coords, k=num_neighbors)
        indices[start:end] = index.astype(np.int32, copy=False)
        nearest_distance = distance if num_neighbors == 1 else distance[:, 0]
        distances[start:end] = nearest_distance.astype(np.float32, copy=False)
        if logger is not None and (
                end == len(target) or end % 5_000_000 == 0):
            logger.info(
                f'Aligned {end:,}/{len(target):,} target points '
                f'with {num_neighbors}-NN ({100 * end / len(target):.1f}%).')
    del tree
    return indices, {
        'num_neighbors': num_neighbors,
        'mean_distance_m': float(distances.mean()),
        'p99_distance_m': float(np.quantile(distances, 0.99)),
        'max_distance_m': float(distances.max()),
    }

def production_neighbour_indices(
        source_coords, target_coords, logger=None, chunk_size=1_000_000,
        n_neighbors=5):
    """Reproduce propagate_preds FP32 sklearn neighbour lookup.

    The production pipeline casts both coordinate arrays to FP32 and uses
    NearestNeighbors with algorithm auto and n_jobs 1. SciPy's FP64 cKDTree is
    close but not equivalent for boundary/equidistant points, which is enough
    to change an official detection count by one instance.
    """
    from sklearn.neighbors import NearestNeighbors

    source = np.asarray(source_coords, dtype=np.float32)
    target = np.asarray(target_coords, dtype=np.float32)
    num_neighbors = int(n_neighbors)
    if len(source) == 0 or len(target) == 0:
        raise ValueError('Nearest-neighbour alignment requires non-empty arrays.')
    if not 1 <= num_neighbors <= len(source):
        raise ValueError('n_neighbors must be between 1 and source size.')
    if len(source) > np.iinfo(np.int32).max:
        raise ValueError('Source array is too large for int32 neighbour indices.')

    model = NearestNeighbors(
        n_neighbors=num_neighbors, algorithm='auto', n_jobs=1)
    model.fit(source)
    output_shape = (
        (len(target),) if num_neighbors == 1
        else (len(target), num_neighbors))
    indices = np.empty(output_shape, dtype=np.int32)
    distances = np.empty(len(target), dtype=np.float32)
    for start in range(0, len(target), int(chunk_size)):
        end = min(start + int(chunk_size), len(target))
        distance, index = model.kneighbors(
            target[start:end], num_neighbors, return_distance=True)
        if num_neighbors == 1:
            indices[start:end] = index[:, 0].astype(np.int32, copy=False)
            distances[start:end] = distance[:, 0].astype(
                np.float32, copy=False)
        else:
            indices[start:end] = index.astype(np.int32, copy=False)
            distances[start:end] = distance[:, 0].astype(
                np.float32, copy=False)
        if logger is not None and (
                end == len(target) or end % 5_000_000 == 0):
            logger.info(
                f'Production-aligned {end:,}/{len(target):,} target points '
                f'with {num_neighbors}-NN ({100 * end / len(target):.1f}%).')
    del model
    return indices, {
        'backend': 'sklearn_fp32_production',
        'num_neighbors': num_neighbors,
        'mean_distance_m': float(distances.mean()),
        'p99_distance_m': float(np.quantile(distances, 0.99)),
        'max_distance_m': float(distances.max()),
    }



def propagate_labels_by_neighbours(
        source_labels, neighbour_indices, chunk_size=1_000_000):
    """Apply TreeLearn's majority vote with smaller-label tie breaking."""
    labels = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    neighbours = np.asarray(neighbour_indices)
    if neighbours.ndim == 1:
        return labels[neighbours]
    if neighbours.ndim != 2 or neighbours.shape[1] < 1:
        raise ValueError('Neighbour indices must have shape [N] or [N, K].')
    if neighbours.min() < 0 or neighbours.max() >= len(labels):
        raise ValueError('Neighbour indices are outside source-label bounds.')
    propagated = np.empty(len(neighbours), dtype=np.int64)
    for start in range(0, len(neighbours), int(chunk_size)):
        end = min(start + int(chunk_size), len(neighbours))
        values = np.sort(labels[neighbours[start:end]], axis=1)
        best_values = values[:, 0].copy()
        best_counts = np.zeros(len(values), dtype=np.int16)
        for column in range(values.shape[1]):
            candidate = values[:, column]
            counts = np.count_nonzero(
                values == candidate[:, None], axis=1)
            improved = counts > best_counts
            best_values[improved] = candidate[improved]
            best_counts[improved] = counts[improved]
        propagated[start:end] = best_values
    return propagated

def build_ensemble_to_original_mapping(
        config, ensemble_coords, original_coords, logger):
    """Precompute the hash propagation used by the production pipeline."""
    import pickle
    from tree_learn.util import propagate_preds_hash_full

    base_dir = str(getattr(
        config, 'pipeline_base_dir',
        os.path.dirname(os.path.dirname(config.forest_path))))
    plot_name = os.path.splitext(os.path.basename(config.forest_path))[0]
    voxelized_dir = os.path.join(
        base_dir,
        f'forest_voxelized{config.sample_generation.voxel_size}')
    hash_mapping_path = os.path.join(
        voxelized_dir, f'{plot_name}_hash_mapping.pkl')
    if not os.path.isfile(hash_mapping_path):
        raise FileNotFoundError(
            f'Production hash mapping does not exist: {hash_mapping_path}')
    logger.info(f'Loading production hash mapping: {hash_mapping_path}')
    with open(hash_mapping_path, 'rb') as file:
        hash_mapping = pickle.load(file)

    source_indices = np.arange(len(ensemble_coords), dtype=np.int64)
    original_mapping, missing_mask = propagate_preds_hash_full(
        ensemble_coords, source_indices, original_coords, hash_mapping)
    del source_indices, hash_mapping
    missing_indices = np.flatnonzero(missing_mask).astype(np.int32)
    original_mapping[missing_mask] = 0
    original_mapping = original_mapping.astype(np.int32, copy=False)
    missing_neighbours = None
    missing_stats = None
    if len(missing_indices):
        logger.info(
            f'Preparing 5-NN fallback for {len(missing_indices):,} '
            'original points not covered by the voxel hash.')
        missing_neighbours, missing_stats = production_neighbour_indices(
            ensemble_coords, original_coords[missing_indices],
            logger=logger, n_neighbors=5)
    return {
        'ensemble_index_for_original': original_mapping,
        'missing_original_indices': missing_indices,
        'ensemble_neighbours_for_missing_original': missing_neighbours,
        'missing_alignment': missing_stats,
        'hash_mapping_path': hash_mapping_path,
    }


def predictions_on_official_gt(predictions, propagation):
    """Apply production ensemble->original->GT label propagation."""
    predictions = np.asarray(predictions, dtype=np.int64)
    original_predictions = predictions[
        propagation['ensemble_index_for_original']]
    missing_indices = propagation['missing_original_indices']
    if len(missing_indices):
        original_predictions[missing_indices] = \
            propagate_labels_by_neighbours(
                predictions,
                propagation['ensemble_neighbours_for_missing_original'])
    return propagate_labels_by_neighbours(
        original_predictions, propagation['original_neighbours_for_gt'])



def legacy_base_anchors(coords, labels, base_anchor_height=0.5):
    """Reproduce TreeDataset legacy base anchors from a labeled forest."""
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(coords) != len(labels):
        raise ValueError('Base-anchor coordinates and labels are not aligned.')
    valid_indices = np.flatnonzero(labels > 0)
    if len(valid_indices) == 0:
        raise ValueError('Ground truth contains no labeled trees.')
    order = np.argsort(labels[valid_indices], kind='stable')
    sorted_indices = valid_indices[order]
    sorted_labels = labels[sorted_indices]
    boundaries = np.r_[
        0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(sorted_labels)]
    tree_ids = []
    anchors = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tree_id = int(sorted_labels[start])
        tree_points = coords[sorted_indices[start:end]]
        if len(tree_points) > 11:
            min_z = np.partition(tree_points[:, 2], 10)[3]
        else:
            min_z = tree_points[:, 2].min()
        base_points = tree_points[
            tree_points[:, 2] <= min_z + float(base_anchor_height)]
        if len(base_points) == 0:
            continue
        tree_ids.append(tree_id)
        anchors.append(base_points.mean(axis=0))
    if not tree_ids:
        raise ValueError('No valid legacy base anchors could be generated.')
    return (
        np.asarray(tree_ids, dtype=np.int64),
        np.asarray(anchors, dtype=np.float32))


def offsets_from_anchors(coords, labels, tree_ids, anchors):
    coords = np.asarray(coords)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    tree_ids = np.asarray(tree_ids, dtype=np.int64).reshape(-1)
    anchors = np.asarray(anchors, dtype=np.float32)
    targets = np.zeros((len(coords), 3), dtype=np.float32)
    positions = np.searchsorted(tree_ids, labels)
    safe_positions = np.minimum(positions, len(tree_ids) - 1)
    valid = (
        (labels > 0) & (positions < len(tree_ids)) &
        (tree_ids[safe_positions] == labels))
    targets[valid] = (
        anchors[safe_positions[valid]] - coords[valid]).astype(np.float32)
    return targets


def prepare_ground_truth(config, settings, arrays, logger):
    from tree_learn.util import load_data

    ground_truth_path = str(settings['ground_truth_path'])
    logger.info(f'Loading Oracle ground truth from {ground_truth_path}...')
    ground_truth = load_data(ground_truth_path)
    ground_truth_labels = ground_truth[:, 3].astype(np.int64)
    ground_truth_coords_absolute = ground_truth[:, :3].astype(
        np.float64, copy=True)
    if not np.any(ground_truth_labels > 0):
        raise ValueError('Oracle ground truth contains no labeled trees.')
    logger.info(
        'Loading original forest coordinates for production propagation...')
    original_coords_absolute = load_xyz_bounded(config.forest_path)
    xyz_mean = original_coords_absolute.mean(axis=0)
    original_coords_centered = original_coords_absolute - xyz_mean
    ground_truth_coords = ground_truth_coords_absolute - xyz_mean
    del ground_truth

    logger.info('Aligning GT labels to ensembled points...')
    gt_index_for_ensemble, gt_to_ensemble_stats = nearest_source_indices(
        ground_truth_coords, arrays['coords'], logger=logger)
    labels_at_ensemble = ground_truth_labels[gt_index_for_ensemble]
    logger.info(
        'Preparing production ensemble-to-original hash propagation...')
    propagation = build_ensemble_to_original_mapping(
        config, arrays['coords'], original_coords_centered, logger)
    del original_coords_centered

    base_dir = str(getattr(
        config, 'pipeline_base_dir',
        os.path.dirname(os.path.dirname(config.forest_path))))
    plot_name = os.path.splitext(os.path.basename(config.forest_path))[0]
    evaluation_coords_path = os.path.join(
        base_dir, str(config.save_cfg.results_dir), 'full_forest',
        f'{plot_name}.laz')
    if not os.path.isfile(evaluation_coords_path):
        raise FileNotFoundError(
            'Baseline prediction coordinate template does not exist: '
            f'{evaluation_coords_path}')
    logger.info(
        'Loading saved baseline coordinates used by official evaluation: '
        f'{evaluation_coords_path}')
    evaluation_original_coords, baseline_original_predictions = \
        load_prediction_xyz_labels_bounded(evaluation_coords_path)
    if len(evaluation_original_coords) != len(original_coords_absolute):
        raise ValueError(
            'Saved baseline and source forest point counts differ; exact '
            'production propagation cannot be reproduced.')
    max_coordinate_difference = 0.0
    for start in range(0, len(original_coords_absolute), 2_000_000):
        end = min(start + 2_000_000, len(original_coords_absolute))
        max_coordinate_difference = max(
            max_coordinate_difference,
            float(np.max(np.abs(
                evaluation_original_coords[start:end] -
                original_coords_absolute[start:end]))))
    if max_coordinate_difference > 0.002:
        raise ValueError(
            'Saved baseline coordinates are not in source-forest order '
            f'(max difference {max_coordinate_difference:.6f} m).')
    del original_coords_absolute

    logger.info(
        'Aligning saved original points to official GT with production 5-NN...')
    original_neighbours_for_gt, original_to_gt_stats = \
        production_neighbour_indices(
            evaluation_original_coords, ground_truth_coords_absolute,
            logger=logger, n_neighbors=5)
    propagation['original_neighbours_for_gt'] = original_neighbours_for_gt
    baseline_evaluation_predictions = propagate_labels_by_neighbours(
        baseline_original_predictions, original_neighbours_for_gt)
    baseline_evaluation_metrics = detection_metrics(
        ground_truth_labels, baseline_evaluation_predictions,
        settings['evaluation_thresholds'])
    if not baseline_matches_reference(
            baseline_evaluation_metrics, settings['baseline_reference']):
        raise RuntimeError(
            'Official saved baseline evaluation reproduction failed before '
            'clustering. This indicates a coordinate/KNN mismatch: '
            f'observed={baseline_evaluation_metrics}, '
            f"expected={settings['baseline_reference']}")
    logger.info(
        'Official saved baseline evaluation reproduced exactly: '
        f"TP={baseline_evaluation_metrics['tp']}, "
        f"FP={baseline_evaluation_metrics['fp']}, "
        f"FN={baseline_evaluation_metrics['fn']}.")
    del baseline_original_predictions, baseline_evaluation_predictions
    del evaluation_original_coords, ground_truth_coords_absolute

    tree_ids, anchors = legacy_base_anchors(
        ground_truth_coords, ground_truth_labels,
        base_anchor_height=float(config.dataset_test.base_anchor_height))
    offset_targets = offsets_from_anchors(
        arrays['coords'], labels_at_ensemble, tree_ids, anchors)
    return {
        'ground_truth_labels': ground_truth_labels,
        'labels_at_ensemble': labels_at_ensemble,
        'offset_targets_at_ensemble': offset_targets,
        'propagation': propagation,
        'alignment': {
            'gt_to_ensemble': gt_to_ensemble_stats,
            'original_to_gt': original_to_gt_stats,
            'saved_baseline_metrics': baseline_evaluation_metrics,
            'missing_original': propagation['missing_alignment'],
            'evaluation_coords_path': evaluation_coords_path,
            'max_coordinate_difference_m': max_coordinate_difference,
            'xyz_mean': xyz_mean.tolist(),
        },
    }


def baseline_matches_reference(metrics, reference):
    return all(
        int(metrics[name]) == int(reference[name])
        for name in ('tp', 'fp', 'fn'))



def detection_metrics(gt_labels, predictions, thresholds):
    from tree_learn.util import (
        get_detections, get_detection_failures, make_labels_consecutive)

    gt = np.asarray(gt_labels, dtype=np.int64).copy()
    pred = np.asarray(predictions, dtype=np.int64).copy()
    if len(gt) != len(pred):
        raise ValueError('GT and predictions must be point-aligned.')
    gt[gt == 0] = -1
    pred[pred == 0] = -1
    gt_mask = gt != -1
    pred_mask = pred != -1
    if not np.any(gt_mask) or not np.any(pred_mask):
        raise ValueError('Detection evaluation needs GT and predicted trees.')
    gt[gt_mask], _ = make_labels_consecutive(gt[gt_mask], start_num=0)
    pred[pred_mask], _ = make_labels_consecutive(pred[pred_mask], start_num=0)
    matched_gt, matched_pred, iou, precision, recall = get_detections(
        gt, pred, float(thresholds['min_iou_for_match']), -1)
    failures = get_detection_failures(
        matched_gt, matched_pred,
        np.arange(np.max(gt) + 1), np.arange(np.max(pred) + 1),
        iou, precision, recall,
        float(thresholds['min_precision_for_pred']),
        float(thresholds['min_recall_for_gt']))
    nonmatched_gt = failures[0]
    nonmatched_pred_corresponding_gt = failures[2]
    counted_fp = int(np.count_nonzero(
        ~np.isnan(nonmatched_pred_corresponding_gt)))
    tp = int(len(matched_gt))
    fn = int(len(nonmatched_gt))
    completeness = tp / max(tp + fn, 1)
    commission = counted_fp / max(tp + counted_fp, 1)
    f1 = 2 * tp / max(2 * tp + counted_fp + fn, 1)
    return {
        'tp': tp,
        'fp': counted_fp,
        'fn': fn,
        'completeness': float(completeness),
        'commission': float(commission),
        'f1': float(f1),
    }


def assess_gate(mode_metrics, random_metrics, settings):
    baseline = mode_metrics['baseline']
    oracle = mode_metrics['oracle_balanced']
    random_f1 = float(np.mean([row['f1'] for row in random_metrics]))
    f1_gain = 100 * (oracle['f1'] - baseline['f1'])
    commission_reduction = 100 * (
        baseline['commission'] - oracle['commission'])
    completeness_drop = 100 * (
        baseline['completeness'] - oracle['completeness'])
    random_gain = 100 * (oracle['f1'] - random_f1)
    gate = {
        'minimum_f1_gain_pp': bool(
            f1_gain >= float(settings['min_f1_gain_pp'])),
        'minimum_commission_reduction_pp': bool(
            commission_reduction >= float(
                settings['min_commission_reduction_pp'])),
        'maximum_completeness_drop_pp': bool(
            completeness_drop <= float(
                settings['max_completeness_drop_pp'])),
        'minimum_gain_over_random_pp': bool(
            random_gain >= float(settings['min_gain_over_random_pp'])),
    }
    gate['passed'] = bool(all(gate.values()))
    return gate, {
        'f1_gain_pp': float(f1_gain),
        'commission_reduction_pp': float(commission_reduction),
        'completeness_drop_pp': float(completeness_drop),
        'f1_gain_over_random_mean_pp': float(random_gain),
        'random_mean_f1': random_f1,
    }


def cluster_mode(
        mode, config, arrays, oracle_scores, keep_ratio, random_seed,
        logger):
    from tree_learn.util import (
        assign_remaining_points_nearest_neighbor,
        get_dual_anchor_features,
        get_instances,
    )

    grouping = copy.deepcopy(config.grouping)
    confidence = None
    if mode == 'baseline':
        grouping.use_seed_confidence_filter = False
    elif mode == 'random':
        grouping.use_seed_confidence_filter = True
        grouping.seed_confidence_filter_mode = 'random_ratio'
        grouping.seed_confidence_keep_ratio = float(keep_ratio)
        grouping.seed_confidence_random_seed = int(random_seed)
    else:
        grouping.use_seed_confidence_filter = True
        grouping.seed_confidence_filter_mode = 'top_ratio'
        grouping.seed_confidence_keep_ratio = float(keep_ratio)
        confidence = oracle_scores

    predictions = get_instances(
        arrays['coords'], arrays['offset_predictions'],
        arrays['upper_offset_predictions'], arrays['semantic_logits'],
        grouping, arrays['verticality'], TREE_CLASS, NON_TREE,
        NOT_ASSIGNED, START_INSTANCE, logger=logger,
        axis_xy=None, axis_confidence=confidence)
    tree_mask = predictions != NON_TREE
    if not np.any(
            predictions[tree_mask] != NOT_ASSIGNED):
        raise RuntimeError(f'{mode} clustering produced no usable instance.')
    features = get_dual_anchor_features(
        arrays['coords'][tree_mask], arrays['offset_predictions'][tree_mask],
        arrays['upper_offset_predictions'][tree_mask],
        grouping.upper_anchor_weight, grouping.axis_height_weight)
    predictions[tree_mask] = assign_remaining_points_nearest_neighbor(
        features, predictions[tree_mask], NOT_ASSIGNED,
        chunk_size=getattr(grouping, 'knn_chunk_size', 200000),
        logger=logger)
    if np.any(predictions == NOT_ASSIGNED):
        raise RuntimeError(f'{mode} left unassigned tree points.')
    return predictions


def pointwise_and_ensemble(config, logger):
    import torch
    from tree_learn.dataset import TreeDataset
    from tree_learn.model import TreeLearn
    from tree_learn.util import (
        build_dataloader, ensemble, get_pointwise_preds, load_checkpoint)

    base_dir = str(getattr(
        config, 'pipeline_base_dir',
        os.path.dirname(os.path.dirname(config.forest_path))))
    config.dataset_test.data_root = os.path.join(base_dir, 'tiles', 'npz')
    if not os.path.isdir(config.dataset_test.data_root):
        raise FileNotFoundError(
            'Pipeline tiles do not exist: '
            f'{config.dataset_test.data_root}. Generate them before E0.')
    if not os.path.isfile(config.pretrain):
        raise FileNotFoundError(
            f'Pipeline checkpoint does not exist: {config.pretrain}')

    model = TreeLearn(**config.model).cuda()
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    loader = build_dataloader(
        dataset, training=False, **config.dataloader)
    load_checkpoint(config.pretrain, logger, model)
    pointwise = get_pointwise_preds(
        model, loader, config.model, logger, return_backbone_feats=False)
    del model
    torch.cuda.empty_cache()
    ensembled = ensemble(
        pointwise[6], pointwise[0], pointwise[1], pointwise[2], pointwise[3],
        pointwise[4], pointwise[5], pointwise[7], pointwise[8],
        pointwise[9], pointwise[10], pointwise[11],
        pointwise[12], pointwise[13], logger=logger)
    return {
        'coords': ensembled[0],
        'semantic_logits': ensembled[1],
        'offset_predictions': ensembled[3],
        'offset_labels': ensembled[4],
        'upper_offset_predictions': ensembled[5],
        'instance_labels': ensembled[7].astype(np.int64),
        'verticality': ensembled[9][:, -1],
    }


def format_markdown(report):
    lines = [
        '# E0 复杂度感知种子注意力：Oracle 诊断',
        '',
        f"- Pipeline config：`{report['pipeline_config']}`",
        f"- 点数：{report['num_points']:,}",
        f"- 基础候选种子：{report['num_candidate_seeds']:,}",
        f"- 固定保留率：{report['keep_ratio']:.3f}",
        '',
        '## 聚类检测结果',
        '',
        '| Mode | TP | FP | FN | Completeness | Commission | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, metrics in report['mode_metrics'].items():
        lines.append(
            f"| {name} | {metrics['tp']} | {metrics['fp']} | "
            f"{metrics['fn']} | {100 * metrics['completeness']:.3f}% | "
            f"{100 * metrics['commission']:.3f}% | "
            f"{100 * metrics['f1']:.3f}% |")
    for row in report['random_metrics']:
        lines.append(
            f"| random_s{row['seed']} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {100 * row['completeness']:.3f}% | "
            f"{100 * row['commission']:.3f}% | {100 * row['f1']:.3f}% |")
    lines.extend([
        '',
        '## Oracle 相对收益',
        '',
        f"- F1 gain：{report['effects']['f1_gain_pp']:+.3f} pp",
        f"- Commission reduction："
        f"{report['effects']['commission_reduction_pp']:+.3f} pp",
        f"- Completeness drop："
        f"{report['effects']['completeness_drop_pp']:+.3f} pp",
        f"- F1 gain over random mean："
        f"{report['effects']['f1_gain_over_random_mean_pp']:+.3f} pp",
        '',
        '## 种子质量',
        '',
        '| Mode | Selected | Tree coverage | ≥tau_min coverage | '
        'Mean vote error | P90 vote error | Mean utility |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for name, metrics in report['seed_metrics'].items():
        lines.append(
            f"| {name} | {metrics['selected_count']:,} | "
            f"{100 * metrics['tree_seed_coverage']:.3f}% | "
            f"{100 * metrics['tree_tau_min_coverage']:.3f}% | "
            f"{metrics['mean_vote_error_xy']:.4f} m | "
            f"{metrics['p90_vote_error_xy']:.4f} m | "
            f"{metrics['mean_raw_utility']:.4f} |")
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    lines.extend([''])
    if report['gate']['passed']:
        lines.append('PASS：Oracle 上限足够，下一步实现 MLP 控制组和种子 Point Transformer。')
    else:
        lines.append('STOP：种子选择上限不足，不训练复杂度感知种子注意力网络。')
    return '\n'.join(lines) + '\n'


def run(config_path):
    import yaml
    from tree_learn.util import get_config, get_root_logger

    with open(config_path, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    required = {
        'pipeline_config', 'ground_truth_path', 'output_dir', 'keep_ratio',
        'random_seeds', 'utility', 'evaluation_thresholds', 'gate',
        'baseline_reference',
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f'Missing E0 config fields: {missing}')
    if not 0 < float(settings['keep_ratio']) <= 1:
        raise ValueError('keep_ratio must be in (0, 1].')
    if not settings['random_seeds']:
        raise ValueError('At least one random control seed is required.')
    if not os.path.isfile(str(settings['pipeline_config'])):
        raise FileNotFoundError(
            'Pipeline config does not exist: '
            f"{settings['pipeline_config']}")
    if not os.path.isfile(str(settings['ground_truth_path'])):
        raise FileNotFoundError(
            'Ground-truth file does not exist: '
            f"{settings['ground_truth_path']}")
    pipeline_config = str(settings['pipeline_config'])
    config = get_config(pipeline_config)
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger(str(output_dir / 'oracle.log'))
    logger.info('Getting the shared pointwise predictions and ensemble...')
    arrays = pointwise_and_ensemble(config, logger)
    oracle_gt = prepare_ground_truth(config, settings, arrays, logger)
    arrays['instance_labels'] = oracle_gt['labels_at_ensemble']
    arrays['offset_labels'] = oracle_gt['offset_targets_at_ensemble']
    logger.info('Oracle GT alignment and legacy base targets are ready.')

    candidate_mask = candidate_base_seed_mask(
        arrays['semantic_logits'], arrays['verticality'],
        arrays['offset_predictions'], config.grouping.tree_conf_thresh,
        config.grouping.tau_vert, config.grouping.tau_off)
    utility_result = compute_seed_utility(
        arrays['coords'], arrays['offset_predictions'],
        arrays['offset_labels'], arrays['instance_labels'], candidate_mask,
        sigma_m=float(settings['utility']['sigma_m']),
        purity_voxel_size=float(
            settings['utility']['purity_voxel_size']),
        purity_power=float(settings['utility']['purity_power']))
    balanced_scores = tree_balanced_oracle_scores(
        utility_result['utility'], arrays['instance_labels'], candidate_mask)
    keep_ratio = float(settings['keep_ratio'])

    selection_masks = {
        'baseline': candidate_mask.copy(),
        'oracle_global': exact_top_ratio_mask(
            candidate_mask, utility_result['utility'], keep_ratio),
        'oracle_balanced': exact_top_ratio_mask(
            candidate_mask, balanced_scores, keep_ratio),
    }
    seed_metrics = {
        name: seed_set_metrics(
            selected, candidate_mask, arrays['instance_labels'],
            utility_result['vote_error_xy'], utility_result['utility'],
            config.grouping.tau_min)
        for name, selected in selection_masks.items()
    }
    thresholds = settings['evaluation_thresholds']
    mode_metrics = {}
    for mode in ('baseline', 'oracle_global', 'oracle_balanced'):
        logger.info(f'===== START {mode} clustering =====')
        score = None
        if mode == 'oracle_global':
            score = utility_result['utility']
        elif mode == 'oracle_balanced':
            score = balanced_scores
        predictions = cluster_mode(
            mode, config, arrays, score, keep_ratio, 0, logger)
        evaluation_predictions = predictions_on_official_gt(
            predictions, oracle_gt['propagation'])
        mode_metrics[mode] = detection_metrics(
            oracle_gt['ground_truth_labels'], evaluation_predictions,
            thresholds)
        if mode == 'baseline' and not baseline_matches_reference(
                mode_metrics[mode], settings['baseline_reference']):
            raise RuntimeError(
                'Baseline reproduction failed before Oracle comparison: '
                f"observed={mode_metrics[mode]}, "
                f"expected={settings['baseline_reference']}")
        logger.info(
            f"DONE {mode}: F1={100 * mode_metrics[mode]['f1']:.3f}%, "
            f"Commission={100 * mode_metrics[mode]['commission']:.3f}%")
        del predictions

    random_metrics = []
    for seed in settings['random_seeds']:
        logger.info(f'===== START random seed {seed} clustering =====')
        predictions = cluster_mode(
            'random', config, arrays, None, keep_ratio, int(seed), logger)
        evaluation_predictions = predictions_on_official_gt(
            predictions, oracle_gt['propagation'])
        metrics = detection_metrics(
            oracle_gt['ground_truth_labels'], evaluation_predictions, thresholds)
        metrics['seed'] = int(seed)
        random_metrics.append(metrics)
        logger.info(
            f"DONE random seed {seed}: F1={100 * metrics['f1']:.3f}%, "
            f"Commission={100 * metrics['commission']:.3f}%")
        del predictions

    gate, effects = assess_gate(
        mode_metrics, random_metrics, settings['gate'])
    report = {
        'config': str(config_path),
        'pipeline_config': pipeline_config,
        'ground_truth_path': str(settings['ground_truth_path']),
        'evaluation_scope': 'official_gt_with_nearest_ensemble_predictions',
        'alignment': oracle_gt['alignment'],
        'num_points': int(len(arrays['coords'])),
        'num_candidate_seeds': int(candidate_mask.sum()),
        'keep_ratio': keep_ratio,
        'random_seeds': [int(value) for value in settings['random_seeds']],
        'utility': settings['utility'],
        'mode_metrics': mode_metrics,
        'random_metrics': random_metrics,
        'seed_metrics': seed_metrics,
        'effects': effects,
        'gate': gate,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    with (output_dir / 'metrics.csv').open(
            'w', newline='', encoding='utf-8') as file:
        rows = [
            {'mode': name, **metrics}
            for name, metrics in mode_metrics.items()
        ] + [
            {'mode': f"random_s{metrics['seed']}", **metrics}
            for metrics in random_metrics
        ]
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'Seed Oracle gate failed; do not train the attention branch.')
    return report


def main():
    parser = argparse.ArgumentParser(
        description='GT Oracle diagnostic for sparse clustering seeds.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
