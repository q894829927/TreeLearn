"""Run one-tile prediction, ensemble, clustering and save for any U-Net candidate."""

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import (
    ensemble,
    get_config,
    get_instances,
    get_pointwise_preds,
    load_checkpoint,
)


def ensemble_pointwise_predictions(pointwise, logger=None):
    """Map get_pointwise_preds output to ensemble's coordinate-first API."""
    (
        semantic_logits,
        semantic_labels,
        offset_predictions,
        offset_labels,
        upper_offset_predictions,
        upper_offset_labels,
        coords,
        instance_labels,
        backbone_features,
        input_features,
        axis_xy_predictions,
        axis_log_variances,
        base_semantic_prediction_logits,
        base_offset_predictions,
    ) = pointwise
    return ensemble(
        coords,
        semantic_logits,
        semantic_labels,
        offset_predictions,
        offset_labels,
        upper_offset_predictions,
        upper_offset_labels,
        instance_labels,
        backbone_features,
        input_features,
        axis_xy_predictions,
        axis_log_variances,
        base_semantic_prediction_logits,
        base_offset_predictions,
        logger=logger)


def resolve_tile_index(dataset_size, requested_index=None):
    if dataset_size <= 0:
        raise ValueError('dataset_size must be positive.')
    tile_index = (
        dataset_size // 2 if requested_index is None else requested_index)
    if not 0 <= tile_index < dataset_size:
        raise ValueError(
            f'tile_index {tile_index} is outside [0, {dataset_size}).')
    return tile_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--tile_index', type=int, default=None,
        help='Dataset tile index; defaults to the middle tile.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('sparse_unet_pipeline_smoke')
    config = get_config(args.config)

    forest_base = os.path.dirname(os.path.dirname(config.forest_path))
    config.dataset_test.data_root = os.path.join(
        forest_base, 'tiles', 'npz')
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    if not len(dataset):
        raise RuntimeError(
            f'No tiles found in {config.dataset_test.data_root}.')
    tile_index = resolve_tile_index(len(dataset), args.tile_index)
    logger.info(
        f'Using representative tile {tile_index}/{len(dataset) - 1}.')
    loader = DataLoader(
        Subset(dataset, [tile_index]), batch_size=1, num_workers=0,
        collate_fn=dataset.collate_fn, shuffle=False)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = TreeLearn(**config.model).cuda().eval()
    load_checkpoint(config.pretrain, logger, model)
    pointwise = get_pointwise_preds(
        model, loader, config.model, logger=logger,
        return_backbone_feats=False)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    parameter_count = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()

    ensembled = ensemble_pointwise_predictions(pointwise, logger=logger)
    (
        coords,
        semantic_logits,
        semantic_labels,
        offset_predictions,
        offset_labels,
        upper_offset_predictions,
        upper_offset_labels,
        instance_labels,
        _,
        input_features,
        axis_xy_predictions,
        axis_log_variances,
        _,
        _,
    ) = ensembled
    axis_confidence = (
        None if axis_log_variances is None
        else 1.0 / (1.0 + np.exp(axis_log_variances)))
    instance_predictions = get_instances(
        coords, offset_predictions, upper_offset_predictions,
        semantic_logits, config.grouping, input_features[:, -1],
        0, 0, -1, 1, logger=logger,
        axis_xy=axis_xy_predictions,
        axis_confidence=axis_confidence)

    predicted_instances = np.unique(
        instance_predictions[instance_predictions > 0])
    if not len(predicted_instances):
        raise RuntimeError(
            f'Tile {tile_index} produced no valid clustered instance; '
            'rerun with --tile_index on a representative forest tile.')

    for name, values in {
        'semantic_logits': semantic_logits,
        'offset_predictions': offset_predictions,
        'coords': coords,
    }.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f'{name} contains NaN or infinity.')
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    elapsed_seconds = time.perf_counter() - start
    np.savez_compressed(
        args.output,
        coords=coords,
        semantic_logits=semantic_logits,
        offset_predictions=offset_predictions,
        instance_predictions=instance_predictions,
        tile_index=np.asarray(tile_index),
        num_instances=np.asarray(len(predicted_instances)),
        peak_memory_gb=np.asarray(peak_memory),
        parameter_count=np.asarray(parameter_count),
        elapsed_seconds=np.asarray(elapsed_seconds))
    logger.info(
        f'PASS: tile={tile_index}, {len(coords):,} points, '
        f'instances={len(predicted_instances)}, peak={peak_memory:.3f} GB, '
        f'params={parameter_count:,}, time={elapsed_seconds:.3f}s; '
        f'saved {args.output}')


if __name__ == '__main__':
    main()
