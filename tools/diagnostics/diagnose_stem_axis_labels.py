import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np

from tree_learn.dataset import TreeDataset
from tree_learn.util import get_config, munch_to_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare crown-median and lower-stem axis labels.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_scans', type=int, default=323)
    parser.add_argument(
        '--output_dir', default='logs/stem_axis_diagnostic')
    parser.add_argument('--num_examples', type=int, default=20)
    return parser.parse_args()


def get_tree_min_z(dataset, tree_points):
    if dataset.base_anchor_mode == 'legacy':
        if len(tree_points) > 11:
            return float(np.partition(tree_points[:, 2], 10)[3])
        return float(tree_points[:, 2].min())
    return float(np.quantile(
        tree_points[:, 2], dataset.base_anchor_floor_quantile))


def percentile_summary(values):
    if not values:
        return {'mean': None, 'median': None, 'p90': None}
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
    }


def save_examples(examples, output_path, logger):
    if not examples:
        logger.warning('No examples were available for visualization.')
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            'matplotlib is unavailable; skipping label_examples.png.')
        return

    columns = 4
    rows = int(np.ceil(len(examples) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis, example in zip(axes.flat, examples):
        points = example['points']
        if len(points) > 5000:
            indices = np.linspace(
                0, len(points) - 1, 5000, dtype=np.int64)
            points = points[indices]
        axis.scatter(
            points[:, 0], points[:, 1], s=1, c=points[:, 2],
            cmap='viridis', alpha=0.35)
        axis.scatter(
            *example['base'][:2], marker='x', s=70, c='red',
            label='base')
        axis.scatter(
            *example['crown'][:2], marker='^', s=55, c='orange',
            label='crown')
        axis.scatter(
            *example['stem'][:2], marker='^', s=55, c='blue',
            label='stem')
        axis.plot(
            [example['base'][0], example['stem'][0]],
            [example['base'][1], example['stem'][1]],
            c='blue', linewidth=1)
        axis.set_title(
            f"scan {example['scan']} / tree {example['tree']}")
        axis.set_aspect('equal', adjustable='box')
    for axis in axes.flat[len(examples):]:
        axis.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('stem_axis_diagnostic')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config(args.config)
    stem_kwargs = munch_to_dict(config.dataset_test)
    stem_kwargs['upper_anchor_mode'] = 'stem_axis'
    crown_kwargs = deepcopy(stem_kwargs)
    crown_kwargs['upper_anchor_mode'] = 'crown_median'

    stem_dataset = TreeDataset(**stem_kwargs, logger=logger)
    crown_dataset = TreeDataset(**crown_kwargs, logger=logger)
    scan_count = min(
        len(stem_dataset), len(crown_dataset), max(args.max_scans, 0))

    raw_tree_occurrences = 0
    total_trees = 0
    crown_valid_trees = 0
    stem_valid_trees = 0
    both_valid_trees = 0
    stem_axis_lengths = []
    crown_axis_lengths = []
    valid_bin_histogram = {
        str(index): 0 for index in range(stem_dataset.stem_axis_num_bins + 1)
    }
    examples = []

    for scan_index in range(scan_count):
        stem_item = stem_dataset[scan_index]
        crown_item = crown_dataset[scan_index]
        xyz = stem_item[0].numpy()
        features = stem_item[1].numpy()
        instances = stem_item[2].numpy()
        base_offsets = stem_item[4].numpy()
        stem_offsets = stem_item[5].numpy()
        crown_offsets = crown_item[5].numpy()
        base_masks = stem_item[8].numpy()
        stem_masks = stem_item[9].numpy()
        crown_masks = crown_item[9].numpy()

        for tree_id in np.unique(instances):
            if tree_id <= 0:
                continue
            indices = np.flatnonzero(instances == tree_id)
            if len(indices) == 0:
                continue
            raw_tree_occurrences += 1
            first = indices[0]
            base_valid = bool(np.any(base_masks[indices]))
            if not base_valid:
                continue
            total_trees += 1
            crown_valid = base_valid and bool(np.any(crown_masks[indices]))
            stem_valid = base_valid and bool(np.any(stem_masks[indices]))
            crown_valid_trees += int(crown_valid)
            stem_valid_trees += int(stem_valid)
            both_valid_trees += int(crown_valid and stem_valid)

            tree_points = xyz[indices]
            tree_features = features[indices]
            verticality = (
                tree_features if tree_features.ndim == 1
                else tree_features[:, -1])
            min_z = get_tree_min_z(stem_dataset, tree_points)
            tree_height = float(tree_points[:, 2].max() - min_z)
            _, valid_bins = stem_dataset._get_stem_axis_upper_anchor(
                tree_points, verticality, min_z, tree_height)
            valid_bin_histogram[str(valid_bins)] += 1

            if crown_valid:
                crown_axis = (
                    crown_offsets[first, :2] - base_offsets[first, :2])
                crown_axis_lengths.append(float(np.linalg.norm(crown_axis)))
            if stem_valid:
                stem_axis = (
                    stem_offsets[first, :2] - base_offsets[first, :2])
                stem_axis_lengths.append(float(np.linalg.norm(stem_axis)))

            if (
                crown_valid and stem_valid and
                len(examples) < args.num_examples
            ):
                examples.append({
                    'scan': scan_index,
                    'tree': int(tree_id),
                    'points': tree_points.copy(),
                    'base': xyz[first] + base_offsets[first],
                    'crown': xyz[first] + crown_offsets[first],
                    'stem': xyz[first] + stem_offsets[first],
                })

    crown_rate = crown_valid_trees / max(total_trees, 1)
    stem_rate = stem_valid_trees / max(total_trees, 1)
    coverage_ratio = stem_valid_trees / max(crown_valid_trees, 1)
    stem_lengths = percentile_summary(stem_axis_lengths)
    summary = {
        'config': args.config,
        'scans': scan_count,
        'raw_tree_occurrences': raw_tree_occurrences,
        'supervised_tree_occurrences': total_trees,
        'crown_valid_trees': crown_valid_trees,
        'stem_valid_trees': stem_valid_trees,
        'both_valid_trees': both_valid_trees,
        'crown_valid_rate': crown_rate,
        'stem_valid_rate': stem_rate,
        'stem_to_crown_coverage_ratio': coverage_ratio,
        'crown_axis_xy_length': percentile_summary(crown_axis_lengths),
        'stem_axis_xy_length': stem_lengths,
        'stem_valid_bin_histogram': valid_bin_histogram,
        'gate': {
            'coverage_ratio_at_least_0_70': coverage_ratio >= 0.70,
            'mean_axis_length_at_least_0_30m': (
                stem_lengths['mean'] is not None and
                stem_lengths['mean'] >= 0.30),
        },
    }
    summary['gate']['passed'] = all(summary['gate'].values())

    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    save_examples(
        examples, output_dir / 'label_examples.png', logger)

    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary['gate']['passed']:
        logger.info('PASS: label gate passed; run the stem-axis MLP control.')
    else:
        logger.warning(
            'STOP: label gate failed; do not start MLP/PT training yet.')


if __name__ == '__main__':
    main()
