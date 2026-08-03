import argparse
import csv
import json
from pathlib import Path

import numpy as np


GEOMETRY_FIELDS = (
    'instance_xy_centroid',
    'instance_base_vote_xy_centroid',
    'instance_z_min',
    'instance_z_max',
)


def load_yaml(path):
    import yaml

    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        required_arrays = {
            'instance_ids', 'global_feature_values', 'global_feature_names',
            'vertical_tokens', 'layer_valid_mask', 'token_feature_names',
            'target_max_iou', 'target_best_gt_id', 'target_valid',
            'target_is_true_tree', *GEOMETRY_FIELDS,
        }
        required_scalars = {'source_plot', 'split'}
        required = required_arrays | required_scalars
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f'{path} misses E8b arrays: {sorted(missing)}. '
                'Regenerate this plot with the E8b geometry config.')
        artifact = {
            name: np.asarray(data[name]) for name in required_arrays}
        artifact['source_plot'] = str(np.asarray(data['source_plot']).item())
        artifact['split'] = str(np.asarray(data['split']).item())
    count = len(artifact['instance_ids'])
    for name in required_arrays - {
            'global_feature_names', 'token_feature_names'}:
        if len(artifact[name]) != count:
            raise ValueError(
                f'{path}: {name} has {len(artifact[name])} rows, '
                f'expected {count}.')
    if artifact['instance_xy_centroid'].shape != (count, 2):
        raise ValueError(f'{path}: invalid XY centroid shape.')
    if artifact['instance_base_vote_xy_centroid'].shape != (count, 2):
        raise ValueError(f'{path}: invalid base-vote centroid shape.')
    numeric = [
        'global_feature_values', 'vertical_tokens',
        *GEOMETRY_FIELDS]
    if any(not np.isfinite(artifact[name]).all() for name in numeric):
        raise ValueError(f'{path}: E8b numeric arrays must be finite.')
    return artifact


def score_artifact(artifact, checkpoint, device):
    from tree_learn.util import predict_vertical_instance_quality

    score_result = predict_vertical_instance_quality(
        {
            'instance_ids': artifact['instance_ids'],
            'vertical_tokens': artifact['vertical_tokens'],
            'layer_valid_mask': artifact['layer_valid_mask'],
            'token_feature_names': artifact['token_feature_names'],
        },
        checkpoint,
        device=device,
    )
    if not np.array_equal(
            score_result['instance_ids'], artifact['instance_ids']):
        raise ValueError('Quality scores and artifact instance IDs differ.')
    return np.asarray(score_result['quality_score'], dtype=np.float64)


def top_ratio_masks(scores, instance_ids, keep_ratio):
    scores = np.asarray(scores, dtype=np.float64)
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    if scores.ndim != 1 or instance_ids.ndim != 1:
        raise ValueError('scores and instance_ids must be one-dimensional.')
    if len(scores) == 0 or len(scores) != len(instance_ids):
        raise ValueError('scores and instance_ids must be nonempty and align.')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite.')
    if len(np.unique(instance_ids)) != len(instance_ids):
        raise ValueError('instance_ids must be unique.')
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError('keep_ratio must be in (0, 1].')
    keep_count = max(1, int(np.ceil(float(keep_ratio) * len(scores))))
    order = np.lexsort((instance_ids, -scores))
    high = np.zeros(len(scores), dtype=bool)
    high[order[:keep_count]] = True
    return high, ~high


def cosine_similarity(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def feature_column(artifact, name):
    names = [str(value) for value in artifact['global_feature_names']]
    if name not in names:
        raise ValueError(f'Global instance feature is missing: {name}')
    return artifact['global_feature_values'][:, names.index(name)]


def token_column(artifact, name):
    names = [str(value) for value in artifact['token_feature_names']]
    if name not in names:
        raise ValueError(f'Vertical token feature is missing: {name}')
    return artifact['vertical_tokens'][:, :, names.index(name)]


def pair_features(artifact, scores, source, target, neighbor_rank):
    source_xy = artifact['instance_xy_centroid'][source]
    target_xy = artifact['instance_xy_centroid'][target]
    source_vote = artifact['instance_base_vote_xy_centroid'][source]
    target_vote = artifact['instance_base_vote_xy_centroid'][target]
    source_height = max(
        float(artifact['instance_z_max'][source] -
              artifact['instance_z_min'][source]), 1e-3)
    target_height = max(
        float(artifact['instance_z_max'][target] -
              artifact['instance_z_min'][target]), 1e-3)
    overlap = max(
        0.0,
        min(float(artifact['instance_z_max'][source]),
            float(artifact['instance_z_max'][target])) -
        max(float(artifact['instance_z_min'][source]),
            float(artifact['instance_z_min'][target])))
    valid_layers = (
        artifact['layer_valid_mask'][source] &
        artifact['layer_valid_mask'][target])
    union_layers = (
        artifact['layer_valid_mask'][source] |
        artifact['layer_valid_mask'][target])
    if np.any(valid_layers):
        token_cosine = cosine_similarity(
            artifact['vertical_tokens'][source, valid_layers],
            artifact['vertical_tokens'][target, valid_layers])
    else:
        token_cosine = 0.0
    occupancy = token_column(artifact, 'occupancy_fraction')
    semantic = token_column(artifact, 'semantic_probability_mean')
    verticality = token_column(artifact, 'verticality_mean')
    vote_radius = token_column(artifact, 'base_vote_radius_rms')
    num_points = feature_column(artifact, 'num_points')
    return {
        'neighbor_rank': int(neighbor_rank),
        'source_quality': float(scores[source]),
        'target_quality': float(scores[target]),
        'quality_gap': float(scores[target] - scores[source]),
        'base_vote_distance': float(np.linalg.norm(source_vote - target_vote)),
        'xy_centroid_distance': float(np.linalg.norm(source_xy - target_xy)),
        'source_height': source_height,
        'target_height': target_height,
        'abs_log_height_ratio': float(abs(np.log(source_height / target_height))),
        'z_overlap_min_ratio': float(
            overlap / max(min(source_height, target_height), 1e-3)),
        'layer_overlap_ratio': float(
            valid_layers.sum() / max(union_layers.sum(), 1)),
        'vertical_token_cosine': token_cosine,
        'occupancy_profile_l1': float(np.mean(np.abs(
            occupancy[source] - occupancy[target]))),
        'semantic_profile_l1': float(np.mean(np.abs(
            semantic[source] - semantic[target]))),
        'verticality_profile_l1': float(np.mean(np.abs(
            verticality[source] - verticality[target]))),
        'base_vote_radius_profile_l1': float(np.mean(np.abs(
            vote_radius[source] - vote_radius[target]))),
        'abs_log_num_points_ratio': float(abs(np.log(
            max(float(num_points[source]), 1.0) /
            max(float(num_points[target]), 1.0)))),
    }


def build_pairs_for_artifact(
        artifact, scores, keep_ratio, num_neighbors,
        max_base_vote_distance):
    if int(num_neighbors) <= 0:
        raise ValueError('num_neighbors must be positive.')
    if not np.isfinite(max_base_vote_distance) or max_base_vote_distance <= 0:
        raise ValueError('max_base_vote_distance must be finite and positive.')
    high_mask, low_mask = top_ratio_masks(
        scores, artifact['instance_ids'], keep_ratio)
    high_indices = np.flatnonzero(high_mask)
    low_indices = np.flatnonzero(low_mask)
    rows = []
    low_with_neighbor = set()
    low_with_positive = set()
    for source in low_indices:
        distances = np.linalg.norm(
            artifact['instance_base_vote_xy_centroid'][high_indices] -
            artifact['instance_base_vote_xy_centroid'][source], axis=1)
        order = np.lexsort((
            artifact['instance_ids'][high_indices], distances))
        selected = [
            int(high_indices[position]) for position in order[:num_neighbors]
            if distances[position] <= max_base_vote_distance]
        if selected:
            low_with_neighbor.add(int(source))
        for rank, target in enumerate(selected, start=1):
            source_gt = int(artifact['target_best_gt_id'][source])
            target_gt = int(artifact['target_best_gt_id'][target])
            pair_valid = bool(
                artifact['target_valid'][source] and
                artifact['target_valid'][target])
            pair_positive = bool(
                pair_valid and source_gt > 0 and source_gt == target_gt)
            if pair_positive:
                low_with_positive.add(int(source))
            row = {
                'source_plot': artifact['source_plot'],
                'split': artifact['split'],
                'source_instance_id': int(
                    artifact['instance_ids'][source]),
                'target_instance_id': int(
                    artifact['instance_ids'][target]),
                'source_best_gt_id': source_gt,
                'target_best_gt_id': target_gt,
                'source_max_iou': float(
                    artifact['target_max_iou'][source]),
                'target_max_iou': float(
                    artifact['target_max_iou'][target]),
                'source_is_true_tree': bool(
                    artifact['target_is_true_tree'][source]),
                'target_is_true_tree': bool(
                    artifact['target_is_true_tree'][target]),
                'pair_valid': pair_valid,
                'pair_positive': pair_positive,
                **pair_features(artifact, scores, source, target, rank),
            }
            rows.append(row)
    return rows, {
        'num_instances': int(len(scores)),
        'num_high_quality': int(high_mask.sum()),
        'num_low_quality': int(low_mask.sum()),
        'num_low_with_neighbor': int(len(low_with_neighbor)),
        'num_low_with_positive': int(len(low_with_positive)),
    }


def summarize(rows, plot_summaries, config, pilot):
    valid = [row for row in rows if row['pair_valid']]
    positive = [row for row in valid if row['pair_positive']]
    negative = [row for row in valid if not row['pair_positive']]
    train_positive = [row for row in positive if row['split'] == 'train']
    validation_positive = [
        row for row in positive if row['split'] == 'validation']
    low_count = sum(item['num_low_quality'] for item in plot_summaries)
    low_with_neighbor = sum(
        item['num_low_with_neighbor'] for item in plot_summaries)
    low_with_positive = sum(
        item['num_low_with_positive'] for item in plot_summaries)
    finite_features = all(
        np.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {
            'source_plot', 'split', 'pair_valid', 'pair_positive',
            'source_is_true_tree', 'target_is_true_tree'})
    plot_splits = {}
    for item in plot_summaries:
        plot_splits.setdefault(item['source_plot'], set()).add(item['split'])
    no_plot_leakage = all(
        len(splits) == 1 for splits in plot_splits.values())
    if pilot:
        gate = {
            'pairs_nonempty': len(rows) > 0,
            'positive_pairs_present': len(positive) > 0,
            'negative_pairs_present': len(negative) > 0,
            'all_features_finite': finite_features,
            'no_plot_leakage': no_plot_leakage,
        }
    else:
        gate_cfg = config['gate']
        gate = {
            'enough_train_positive_pairs': (
                len(train_positive) >= int(
                    gate_cfg['min_train_positive_pairs'])),
            'enough_validation_positive_pairs': (
                len(validation_positive) >= int(
                    gate_cfg['min_validation_positive_pairs'])),
            'enough_negative_pairs': (
                len(negative) >= int(gate_cfg['min_negative_pairs'])),
            'neighbor_coverage_passed': (
                low_with_neighbor / max(low_count, 1) >=
                float(gate_cfg['min_neighbor_coverage'])),
            'positive_source_coverage_passed': (
                low_with_positive / max(low_count, 1) >=
                float(gate_cfg['min_positive_source_coverage'])),
            'all_features_finite': finite_features,
            'no_plot_leakage': no_plot_leakage,
        }
    gate['passed'] = bool(all(gate.values()))
    return {
        'mode': 'pilot' if pilot else 'full',
        'num_pairs': int(len(rows)),
        'num_valid_pairs': int(len(valid)),
        'num_positive_pairs': int(len(positive)),
        'num_negative_pairs': int(len(negative)),
        'num_train_positive_pairs': int(len(train_positive)),
        'num_validation_positive_pairs': int(len(validation_positive)),
        'num_low_quality_sources': int(low_count),
        'num_low_with_neighbor': int(low_with_neighbor),
        'num_low_with_positive': int(low_with_positive),
        'neighbor_coverage': float(low_with_neighbor / max(low_count, 1)),
        'positive_source_coverage': float(
            low_with_positive / max(low_count, 1)),
        'plots': plot_summaries,
        'gate': gate,
    }


def write_outputs(rows, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / (
        'pilot_pairs.csv' if summary['mode'] == 'pilot' else 'pairs.csv')
    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    json_path = output_dir / f"{summary['mode']}_summary.json"
    md_path = output_dir / f"{summary['mode']}_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    gate_lines = '\n'.join(
        f'- {name}: **{value}**' for name, value in summary['gate'].items())
    markdown = '\n'.join([
        f"# E8b 邻接对数据审计：{summary['mode']}",
        '',
        f"- 配对总数：{summary['num_pairs']}",
        f"- 有效配对：{summary['num_valid_pairs']}",
        f"- 正配对：{summary['num_positive_pairs']}",
        f"- 负配对：{summary['num_negative_pairs']}",
        f"- 训练集正配对：{summary['num_train_positive_pairs']}",
        f"- 验证集正配对：{summary['num_validation_positive_pairs']}",
        f"- 低质量源实例邻居覆盖率："
        f"{100 * summary['neighbor_coverage']:.3f}%",
        f"- 低质量源实例正目标覆盖率："
        f"{100 * summary['positive_source_coverage']:.3f}%",
        '',
        '## Gate',
        '',
        gate_lines,
        '',
        ('PASS' if summary['gate']['passed'] else 'STOP'),
        '',
    ])
    md_path.write_text(markdown, encoding='utf-8')
    print(markdown)
    return csv_path, json_path, md_path


def run(config_path, pilot=False):
    config = load_yaml(config_path)
    artifact_root = Path(config['artifact_root'])
    checkpoint = Path(config['quality_checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if pilot:
        selected = set(config['pilot_plots'])
    else:
        selected = None
    paths = []
    for split in ('train', 'validation'):
        for path in sorted((artifact_root / split).glob('*.npz')):
            if selected is None or path.stem in selected:
                paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f'No E8b artifacts found below {artifact_root}.')
    rows = []
    plot_summaries = []
    for path in paths:
        print(f'Processing {path}...', flush=True)
        artifact = load_artifact(path)
        scores = score_artifact(
            artifact, checkpoint, config.get('device'))
        plot_rows, plot_summary = build_pairs_for_artifact(
            artifact, scores,
            float(config['keep_ratio']),
            int(config['num_neighbors']),
            float(config['max_base_vote_distance_m']))
        rows.extend(plot_rows)
        plot_summary.update({
            'source_plot': artifact['source_plot'],
            'split': artifact['split'],
            'num_pairs': len(plot_rows),
        })
        plot_summaries.append(plot_summary)
    summary = summarize(rows, plot_summaries, config, pilot)
    write_outputs(rows, summary, config['output_dir'])
    if not summary['gate']['passed']:
        raise RuntimeError('E8b pair-data gate failed.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate quality-guided instance merge pairs.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    run(args.config, pilot=args.pilot)


if __name__ == '__main__':
    main()