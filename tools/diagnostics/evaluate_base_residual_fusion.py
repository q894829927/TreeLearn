"""Evaluate confidence-gated base-residual votes without rerunning training."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import tqdm

from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import build_dataloader, get_config, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Measure base-only and confidence-gated Base-XY residual errors '
            'on validation tiles.'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--weights', nargs='+', type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    parser.add_argument(
        '--seed_confidence_thresholds', nargs='+', type=float,
        default=[0.0, 0.5, 0.65, 0.75])
    parser.add_argument('--tree_conf_thresh', type=float, default=0.5)
    parser.add_argument('--tau_vert', type=float, default=0.6)
    parser.add_argument('--tau_off', type=float, default=4.0)
    parser.add_argument(
        '--output', default='logs/base_residual_fusion_diagnostic.json')
    return parser.parse_args()


def summarize(errors):
    errors = np.concatenate(errors).astype(np.float64, copy=False)
    return {
        'count': int(len(errors)),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'p90': float(np.percentile(errors, 90)),
    }


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('base_residual_fusion_diagnostic')

    config = get_config(args.config)
    if getattr(config.model, 'axis_target_mode', None) != 'base_residual':
        raise ValueError(
            'The diagnostic requires model.axis_target_mode=base_residual.')

    dataset = TreeDataset(**config.dataset_test, logger=logger)
    loader = build_dataloader(
        dataset,
        training=False,
        seed=int(getattr(config, 'seed', 42)) + 1,
        **config.dataloader.test)

    model = TreeLearn(**config.model).cuda().eval()
    load_checkpoint(args.checkpoint, logger, model)

    weights = list(dict.fromkeys(
        [0.0] + [float(weight) for weight in args.weights]))
    error_chunks = {weight: [] for weight in weights}
    full_residual_errors = []
    seed_base_error_chunks = []
    seed_confidence_chunks = []
    candidate_count = 0
    valid_count = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            output = model(batch, return_loss=False)
            valid_mask = batch['masks_off'].to(
                output['offset_predictions'].device)
            if valid_mask.sum() == 0:
                continue

            target_residual = (
                batch['offset_labels'][valid_mask.cpu(), :2].to(
                    output['offset_predictions'].device).float() -
                output['offset_predictions'][valid_mask, :2].float())
            predicted_residual = output[
                'axis_xy_predictions'][valid_mask].float()
            confidence = output[
                'axis_confidence'][valid_mask].float()
            candidate_mask = output['axis_candidate_mask'][valid_mask]
            tree_probability = output[
                'semantic_prediction_logits'].float().softmax(dim=-1)[
                    valid_mask, 0]
            verticality = batch['input_feats'][
                valid_mask.cpu(), -1].to(confidence.device).float()
            seed_mask = (
                (tree_probability >= args.tree_conf_thresh) &
                (verticality > args.tau_vert) &
                (
                    torch.abs(output['offset_predictions'][
                        valid_mask, 2].float()) < args.tau_off
                ))
            if seed_mask.sum() > 0:
                seed_base_error_chunks.append(
                    torch.linalg.vector_norm(
                        target_residual[seed_mask], dim=1).cpu().numpy())
                seed_confidence_chunks.append(
                    confidence[seed_mask, 0].cpu().numpy())

            valid_count += int(valid_mask.sum().item())
            candidate_count += int(candidate_mask.sum().item())
            full_residual_errors.append(
                torch.linalg.vector_norm(
                    target_residual - predicted_residual,
                    dim=1).cpu().numpy())

            for weight in weights:
                gated_prediction = (
                    weight * confidence * predicted_residual)
                errors = torch.linalg.vector_norm(
                    target_residual - gated_prediction, dim=1)
                error_chunks[weight].append(errors.cpu().numpy())

    if valid_count == 0:
        raise RuntimeError('No valid offset-supervised points were found.')

    results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'valid_points': valid_count,
        'candidate_points': candidate_count,
        'candidate_rate': candidate_count / valid_count,
        'full_residual_without_confidence': summarize(
            full_residual_errors),
        'confidence_gated': {
            str(weight): summarize(error_chunks[weight])
            for weight in weights
        },
    }
    if seed_base_error_chunks:
        seed_base_errors = np.concatenate(
            seed_base_error_chunks).astype(np.float64, copy=False)
        seed_confidences = np.concatenate(
            seed_confidence_chunks).astype(np.float64, copy=False)
        if (
            len(seed_base_errors) > 1 and
            np.std(seed_base_errors) > 0 and
            np.std(seed_confidences) > 0
        ):
            seed_confidence_error_corr = float(np.corrcoef(
                seed_confidences, seed_base_errors)[0, 1])
        else:
            seed_confidence_error_corr = 0.0

        seed_thresholds = list(dict.fromkeys(
            [0.0] + [
                float(threshold)
                for threshold in args.seed_confidence_thresholds
            ]))
        seed_threshold_results = {}
        for threshold in seed_thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    'seed confidence thresholds must be between 0 and 1.')
            retained = seed_confidences >= threshold
            retained_errors = seed_base_errors[retained]
            seed_threshold_results[str(threshold)] = {
                'retained_count': int(retained.sum()),
                'retained_rate': float(np.mean(retained)),
                'mean_base_xy_error': (
                    float(np.mean(retained_errors))
                    if len(retained_errors) else None),
                'median_base_xy_error': (
                    float(np.median(retained_errors))
                    if len(retained_errors) else None),
                'p90_base_xy_error': (
                    float(np.percentile(retained_errors, 90))
                    if len(retained_errors) else None),
            }
        results['actual_base_seeds'] = {
            'definition': {
                'tree_conf_thresh': args.tree_conf_thresh,
                'tau_vert': args.tau_vert,
                'tau_off': args.tau_off,
            },
            'count': int(len(seed_base_errors)),
            'confidence_error_corr': seed_confidence_error_corr,
            'thresholds': seed_threshold_results,
        }
    else:
        results['actual_base_seeds'] = {
            'count': 0,
            'confidence_error_corr': None,
            'thresholds': {},
        }
    ranked = sorted(
        weights,
        key=lambda weight: (
            results['confidence_gated'][str(weight)]['mean'],
            results['confidence_gated'][str(weight)]['p90'],
            weight))
    best_weight = ranked[0]
    base_mean = results['confidence_gated'][str(0.0)]['mean']
    best_mean = results['confidence_gated'][str(best_weight)]['mean']
    results['selection'] = {
        'best_weight': best_weight,
        'base_mean': base_mean,
        'best_mean': best_mean,
        'relative_mean_improvement': (
            (base_mean - best_mean) / max(base_mean, 1e-12)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8')
    logger.info('\n%s', json.dumps(results, indent=2, ensure_ascii=False))
    logger.info('Saved diagnostic to %s', output_path)


if __name__ == '__main__':
    main()
