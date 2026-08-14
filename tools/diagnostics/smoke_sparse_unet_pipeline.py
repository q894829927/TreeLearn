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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
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
    loader = DataLoader(
        Subset(dataset, [0]), batch_size=1, num_workers=0,
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
        peak_memory_gb=np.asarray(peak_memory),
        parameter_count=np.asarray(parameter_count),
        elapsed_seconds=np.asarray(elapsed_seconds))
    logger.info(
        f'PASS: {len(coords):,} points, peak={peak_memory:.3f} GB, '
        f'params={parameter_count:,}, time={elapsed_seconds:.3f}s; '
        f'saved {args.output}')


if __name__ == '__main__':
    main()
