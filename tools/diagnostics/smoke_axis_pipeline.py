"""Run prediction, ensemble, grouping and saving on one existing tile."""

import argparse
import logging
import os

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


TREE_CLASS = 0
NON_TREE_INSTANCE = 0
NOT_ASSIGNED_INSTANCE = -1
FIRST_INSTANCE = 1


def main():
    parser = argparse.ArgumentParser(
        description='Smoke test the Axis-XY pipeline on the first tile.')
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--output', default='logs/axis_pipeline_smoke.npz')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('axis_pipeline_smoke')
    config = get_config(args.config)

    forest_base = os.path.dirname(os.path.dirname(config.forest_path))
    config.dataset_test.data_root = os.path.join(forest_base, 'tiles', 'npz')
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    if len(dataset) == 0:
        raise RuntimeError(
            f'No tiles found in {config.dataset_test.data_root}.')
    dataloader = DataLoader(
        Subset(dataset, [0]),
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        shuffle=False)

    model = TreeLearn(**config.model).cuda().eval()
    load_checkpoint(config.pretrain, logger, model)
    pointwise = get_pointwise_preds(
        model,
        dataloader,
        config.model,
        logger=logger,
        return_backbone_feats=False)
    del model
    torch.cuda.empty_cache()

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
    ensembled = ensemble(
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

    axis_confidence = 1.0 / (1.0 + np.exp(axis_log_variances))
    instance_predictions = get_instances(
        coords,
        offset_predictions,
        upper_offset_predictions,
        semantic_logits,
        config.grouping,
        input_features[:, -1],
        TREE_CLASS,
        NON_TREE_INSTANCE,
        NOT_ASSIGNED_INSTANCE,
        FIRST_INSTANCE,
        logger=logger,
        axis_xy=axis_xy_predictions,
        axis_confidence=axis_confidence)

    for name, values in {
        'axis_xy_predictions': axis_xy_predictions,
        'axis_log_variances': axis_log_variances,
        'axis_confidence': axis_confidence,
    }.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f'{name} contains NaN or infinity.')
    if not ((axis_confidence >= 0) & (axis_confidence <= 1)).all():
        raise RuntimeError('axis_confidence is outside [0, 1].')

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        coords=coords,
        axis_xy_predictions=axis_xy_predictions,
        axis_log_variances=axis_log_variances,
        axis_confidence=axis_confidence,
        instance_predictions=instance_predictions)
    logger.info(
        f'PASS: saved one-tile smoke result with {len(coords):,} points '
        f'to {args.output}')


if __name__ == '__main__':
    main()
