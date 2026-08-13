"""D1: compare frozen TreeLearn outputs with Height-MLP/HSCA outputs.

This diagnostic never reads ground-truth labels for model selection. It uses
the base predictions embedded in each frozen-adapter forward pass, so point
alignment is exact and no second copy of TreeLearn is needed on the GPU.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import tqdm

from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import build_dataloader, get_config, load_checkpoint
from tree_learn.util.height_context_diagnostics import (
    DriftAccumulator, make_point_metrics)


LOGGER = logging.getLogger('height_context_domain_drift')
MODEL_NAMES = ('a1_height_mlp', 'a2_hsca')
DOMAIN_CONFIGS = {
    'l1w': {
        'a1_height_mlp': (
            'configs/experiments/height_context_attention/'
            'pipeline_l1w_a1_height_mlp.yaml'),
        'a2_hsca': (
            'configs/experiments/height_context_attention/'
            'pipeline_l1w_a2_hsca.yaml'),
    },
    'wytham': {
        'a1_height_mlp': (
            'configs/experiments/height_context_attention/'
            'pipeline_wytham_a1_height_mlp_locked.yaml'),
        'a2_hsca': (
            'configs/experiments/height_context_attention/'
            'pipeline_wytham_a2_hsca_locked.yaml'),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--domains', nargs='+', choices=sorted(DOMAIN_CONFIGS),
        default=['l1w', 'wytham'])
    parser.add_argument(
        '--output_dir',
        default='logs/height_context_attention/d1_domain_drift')
    parser.add_argument(
        '--max_scans', type=int, default=0,
        help='0 uses every tile; positive values are smoke-test only.')
    parser.add_argument('--density_voxel_size', type=float, default=0.6)
    parser.add_argument(
        '--points_per_scan', type=int, default=5000,
        help='Deterministic systematic point sample per tile; 0 uses all points.')
    return parser.parse_args()


def tile_root_from_forest(forest_path):
    base_dir = os.path.dirname(os.path.dirname(forest_path))
    return os.path.join(base_dir, 'tiles', 'npz')


def point_density_category(coords, batch_ids, voxel_size):
    """Return per-point occupancy class of a local 3-D diagnostic voxel."""
    voxel_coords = torch.floor(coords.float() / float(voxel_size)).long()
    keys = torch.cat([batch_ids.long()[:, None], voxel_coords], dim=1)
    _, inverse, counts = torch.unique(
        keys, dim=0, return_inverse=True, return_counts=True)
    occupancy = counts[inverse]
    categories = torch.zeros_like(occupancy)
    categories[occupancy >= 5] = 1
    categories[occupancy >= 17] = 2
    categories[occupancy >= 65] = 3
    return categories


def make_accumulators(num_height_bins):
    return {
        'overall': DriftAccumulator(),
        'height': {
            str(index): DriftAccumulator()
            for index in range(num_height_bins)
        },
        'verticality': {
            name: DriftAccumulator()
            for name in ('0.0-0.2', '0.2-0.4', '0.4-0.6',
                         '0.6-0.8', '0.8-1.0')
        },
        'density': {
            name: DriftAccumulator()
            for name in ('1-4', '5-16', '17-64', '65+')
        },
    }


def update_accumulators(accumulators, metrics, heights, verticality, density):
    accumulators['overall'].update(metrics)
    for index, accumulator in accumulators['height'].items():
        accumulator.update(metrics, heights == int(index))
    verticality_index = np.minimum(
        (np.clip(verticality, 0, 1) * 5).astype(np.int64), 4)
    for index, accumulator in enumerate(accumulators['verticality'].values()):
        accumulator.update(metrics, verticality_index == index)
    for index, accumulator in enumerate(accumulators['density'].values()):
        accumulator.update(metrics, density == index)


def summarize_accumulators(accumulators):
    return {
        group: (
            accumulator.summary()
            if isinstance(accumulator, DriftAccumulator) else
            {name: value.summary() for name, value in accumulator.items()}
        )
        for group, accumulator in accumulators.items()
    }


def run_model(config_path, max_scans, density_voxel_size, points_per_scan):
    config = get_config(config_path)
    if not config.model.use_height_context_adapter:
        raise ValueError(f'Adapter is disabled in {config_path}.')
    config.dataset_test.data_root = tile_root_from_forest(config.forest_path)
    if not Path(config.dataset_test.data_root).is_dir():
        raise FileNotFoundError(
            f'Missing tile directory: {config.dataset_test.data_root}')

    dataset = TreeDataset(**config.dataset_test, logger=LOGGER)
    if max_scans > 0:
        dataset.data_paths = dataset.data_paths[:max_scans]
        LOGGER.warning(
            'Smoke mode: using only %d scans.', len(dataset.data_paths))
    loader = build_dataloader(
        dataset, training=False, seed=20260813, **config.dataloader)

    model = TreeLearn(**config.model).cuda().eval()
    load_checkpoint(config.pretrain, LOGGER, model)
    accumulators = make_accumulators(int(config.model.height_num_bins))

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            batch['voxel_size'] = config.model.voxel_size
            output = model(batch, return_loss=False)
            base_logits = output['base_semantic_prediction_logits'].float()
            adapted_logits = output['semantic_prediction_logits'].float()
            base_probability = base_logits.softmax(dim=-1)[:, 0]
            adapted_probability = adapted_logits.softmax(dim=-1)[:, 0]
            base_offset = output['base_offset_predictions'].float()
            adapted_offset = output['offset_predictions'].float()
            residual = output['height_offset_residual'].float()
            gate = output['height_context_gate'].float()
            height_bins = output['height_bin_indices'].long()
            device = residual.device
            verticality = batch['input_feats'][:, -1].to(device).float()
            coords = batch['coords'].to(device)
            batch_ids = batch['batch_ids'].to(device)
            density = point_density_category(
                coords, batch_ids, density_voxel_size)
            num_points = len(residual)
            if 0 < points_per_scan < num_points:
                sample = torch.linspace(
                    0, num_points - 1, points_per_scan,
                    device=device).long()
            else:
                sample = torch.arange(num_points, device=device)

            metrics = make_point_metrics(
                base_probability[sample].cpu().numpy(),
                adapted_probability[sample].cpu().numpy(),
                residual[sample].cpu().numpy(),
                gate[sample].cpu().numpy(),
                verticality[sample].cpu().numpy(),
                base_offset[sample].cpu().numpy(),
                adapted_offset[sample].cpu().numpy(),
                tree_conf_thresh=float(config.grouping.tree_conf_thresh),
                tau_vert=float(config.grouping.tau_vert),
                tau_off=float(config.grouping.tau_off))
            update_accumulators(
                accumulators,
                metrics,
                height_bins[sample].cpu().numpy(),
                verticality[sample].cpu().numpy(),
                density[sample].cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return {
        'config': config_path,
        'checkpoint': config.pretrain,
        'tile_root': config.dataset_test.data_root,
        'scans': len(dataset.data_paths),
        'density_voxel_size': density_voxel_size,
        'points_per_scan': points_per_scan,
        'statistics': summarize_accumulators(accumulators),
    }


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def build_comparison(results):
    comparison = {}
    metrics = (
        'tree_to_non_tree_rate_among_base_tree',
        'seed_removed_rate_among_base_seed',
        'seed_removed_semantic_rate_among_base_seed',
        'seed_removed_offset_rate_among_base_seed',
        'abs_probability_delta_mean',
        'offset_xy_residual_mean',
        'offset_z_residual_mean',
        'gate_mean',
    )
    for model_name in MODEL_NAMES:
        l1w = results['l1w'][model_name]['statistics']['overall']
        wytham = results['wytham'][model_name]['statistics']['overall']
        comparison[model_name] = {
            metric: {
                'l1w': l1w[metric],
                'wytham': wytham[metric],
                'wytham_to_l1w_ratio': safe_ratio(
                    wytham[metric], l1w[metric]),
            }
            for metric in metrics
        }
    return comparison


def percentage(value):
    return f'{100 * value:.4f}%'


def write_markdown(path, results, comparison, smoke_mode):
    lines = [
        '# D1 HSCA 跨域预测漂移诊断', '',
        '- 本诊断不读取 GT，不选择 checkpoint，不修改 Wytham 参数。',
        f'- 运行模式：{"smoke" if smoke_mode else "full"}', '',
        '| Domain | Model | Points | Tree→non-tree/base-tree | '
        'Seed removed/base-seed | |Δtree prob| | XY residual | Z residual | Gate |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for domain in results:
        for model_name in MODEL_NAMES:
            stats = results[domain][model_name]['statistics']['overall']
            lines.append(
                f'| {domain} | {model_name} | {stats["points"]:,} | '
                f'{percentage(stats["tree_to_non_tree_rate_among_base_tree"])} | '
                f'{percentage(stats["seed_removed_rate_among_base_seed"])} | '
                f'{stats["abs_probability_delta_mean"]:.6f} | '
                f'{stats["offset_xy_residual_mean"]:.4f} m | '
                f'{stats["offset_z_residual_mean"]:.4f} m | '
                f'{stats["gate_mean"]:.4f} |')

    lines.extend(['', '## Wytham / L1W 漂移倍率', ''])
    for model_name, values in comparison.items():
        lines.extend([f'### {model_name}', '', '| Metric | Ratio |', '|---|---:|'])
        for metric, item in values.items():
            ratio = item['wytham_to_l1w_ratio']
            lines.append(
                f'| {metric} | {"N/A" if ratio is None else f"{ratio:.3f}x"} |')
        lines.append('')
    lines.extend([
        '## 解释规则', '',
        '- Tree→non-tree 与 seed removed 明显放大：优先处理 semantic/seed 保守化。',
        '- Offset residual 明显放大而 semantic 稳定：优先处理 offset identity preservation。',
        '- Gate 跨域升高并伴随残差放大：门控或高度归一化发生域偏移。',
        '- A1/A2 都产生相似漂移：问题来自适配目标，而不是注意力本身。',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the full D1 diagnostic.')
    if set(args.domains) != {'l1w', 'wytham'}:
        raise ValueError('D1 comparison requires both l1w and wytham.')
    if args.density_voxel_size <= 0:
        raise ValueError('density_voxel_size must be positive.')
    if args.points_per_scan < 0:
        raise ValueError('points_per_scan must be non-negative.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for domain in args.domains:
        results[domain] = {}
        for model_name in MODEL_NAMES:
            LOGGER.info('===== D1 START %s / %s =====', domain, model_name)
            results[domain][model_name] = run_model(
                DOMAIN_CONFIGS[domain][model_name],
                args.max_scans,
                args.density_voxel_size,
                args.points_per_scan)

    comparison = build_comparison(results)
    report = {
        'uses_ground_truth': False,
        'max_scans': args.max_scans,
        'results': results,
        'domain_comparison': comparison,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    write_markdown(
        output_dir / 'summary.md', results, comparison,
        smoke_mode=args.max_scans > 0)
    print((output_dir / 'summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
