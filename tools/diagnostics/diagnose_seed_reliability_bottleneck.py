"""Diagnose why the E1b seed-reliability head missed its fixed AUC gate."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / 'tools' / 'training' / 'train_seed_quality_mlp.py'
TRAIN_SPEC = importlib.util.spec_from_file_location(
    'train_seed_quality_mlp_for_diagnostic', TRAIN_PATH)
TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN)


def finite_binary_metrics(targets, scores):
    targets = np.asarray(targets, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(targets) != len(scores):
        raise ValueError('Targets and scores are not aligned.')
    if len(np.unique(targets)) != 2:
        return None
    if not np.isfinite(scores).all():
        raise ValueError('Scores contain non-finite values.')
    return TRAIN.binary_metrics(targets, scores)


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    _, starts, counts = np.unique(
        sorted_values, return_index=True, return_counts=True)
    sorted_ranks = np.repeat(
        starts + 0.5 * (counts - 1), counts).astype(np.float64)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def spearman_correlation(left, right):
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if left_ranks.std() == 0 or right_ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('The E1a manifest is empty.')
    if len({row['source_plot'] for row in rows}) != len(rows):
        raise ValueError('Manifest plot names must be unique.')
    return rows


def load_validation_diagnostic_data(settings):
    data_root = Path(settings['data_root']).resolve()
    rows = read_manifest(settings['manifest_path'])
    validation_rows = [
        (plot_id, row) for plot_id, row in enumerate(rows)
        if row['split'] == 'validation']
    expected_plots = int(settings['expected_validation_plots'])
    if len(validation_rows) != expected_plots:
        raise ValueError(
            f'Expected {expected_plots} validation plots, '
            f'found {len(validation_rows)}.')

    score_path = Path(settings['locked_scores_path'])
    if not score_path.is_file():
        raise FileNotFoundError(
            f'Missing locked E1b scores: {score_path}')
    with np.load(score_path, allow_pickle=False) as scores:
        required = {
            'reliability_scores', 'coverage_scores', 'target_reliable',
            'target_coverage_critical', 'target_utility',
            'target_tree_id', 'candidate_index', 'plot_id',
        }
        missing = required - set(scores.files)
        if missing:
            raise ValueError(
                f'Locked score artifact misses: {sorted(missing)}')
        locked = {name: scores[name].copy() for name in required}
    lengths = {len(values) for values in locked.values()}
    if len(lengths) != 1:
        raise ValueError('Locked validation score arrays are not aligned.')
    if not np.isfinite(locked['reliability_scores']).all():
        raise ValueError('Locked reliability scores are non-finite.')

    diagnostic = settings['diagnostic']
    sample_count = int(diagnostic['max_feature_samples_per_plot'])
    sample_seed = int(diagnostic['sample_seed'])
    expected_backbone = int(settings['expected_backbone_dim'])
    expected_scalar = int(settings['expected_scalar_dim'])

    full = {
        'target_is_tree': [],
        'target_reliable': [],
        'target_coverage_critical': [],
        'target_utility': [],
        'target_vote_error_xy': [],
        'target_vote_cell_purity': [],
        'plot_id': [],
    }
    sampled_features = []
    sampled_reliable = []
    sampled_is_tree = []
    sampled_utility = []
    sampled_scores = []
    feature_names = None
    plot_names = {}
    candidate_mismatches = 0
    score_target_mismatches = 0
    offset = 0
    for plot_id, row in validation_rows:
        artifact_path = Path(row['artifact_path']).resolve()
        if data_root not in artifact_path.parents or not artifact_path.is_file():
            raise ValueError(f'Invalid E1a artifact path: {artifact_path}')
        metadata_path = artifact_path.with_name(
            f'{artifact_path.stem}_metadata.json')
        with metadata_path.open(encoding='utf-8') as file:
            metadata = json.load(file)
        if metadata['source_plot'] != row['source_plot']:
            raise ValueError('Manifest and artifact plot names differ.')
        scalar_names = [
            str(value) for value in metadata['scalar_feature_names']]
        names = [
            f'backbone_{index}' for index in range(expected_backbone)
        ] + scalar_names
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError('Validation feature schemas are inconsistent.')

        with np.load(artifact_path, allow_pickle=False) as artifact:
            count = len(artifact['candidate_indices'])
            if artifact['backbone_features'].shape != (
                    count, expected_backbone):
                raise ValueError('Unexpected backbone feature dimension.')
            if artifact['scalar_features'].shape != (
                    count, expected_scalar):
                raise ValueError('Unexpected scalar feature dimension.')
            score_mask = locked['plot_id'] == int(plot_id)
            score_indices = np.flatnonzero(score_mask)
            if len(score_indices) != count:
                raise ValueError(
                    f'Locked score count differs for {row["source_plot"]}.')
            artifact_candidates = artifact['candidate_indices']
            candidate_mismatches += int(np.count_nonzero(
                locked['candidate_index'][score_indices] !=
                artifact_candidates))
            for target_name, locked_name in (
                    ('target_reliable', 'target_reliable'),
                    ('target_coverage_critical',
                     'target_coverage_critical'),
                    ('target_utility', 'target_utility'),
                    ('target_tree_id', 'target_tree_id')):
                source = artifact[target_name]
                observed = locked[locked_name][score_indices]
                if np.issubdtype(source.dtype, np.floating):
                    score_target_mismatches += int(np.count_nonzero(
                        ~np.isclose(source, observed, atol=1e-7, rtol=0)))
                else:
                    score_target_mismatches += int(np.count_nonzero(
                        source != observed))

            for name in full:
                if name == 'plot_id':
                    full[name].append(np.full(count, plot_id, dtype=np.int16))
                else:
                    full[name].append(artifact[name].copy())

            generator = np.random.default_rng(sample_seed + 1009 * plot_id)
            take = min(count, sample_count)
            chosen = np.sort(generator.choice(
                count, size=take, replace=False))
            sampled_features.append(np.concatenate([
                artifact['backbone_features'][chosen].astype(np.float32),
                artifact['scalar_features'][chosen].astype(np.float32),
            ], axis=1))
            sampled_reliable.append(
                artifact['target_reliable'][chosen].astype(bool))
            sampled_is_tree.append(
                artifact['target_is_tree'][chosen].astype(bool))
            sampled_utility.append(
                artifact['target_utility'][chosen].astype(np.float32))
            sampled_scores.append(
                locked['reliability_scores'][score_indices[chosen]])
        plot_names[int(plot_id)] = row['source_plot']
        offset += count

    if offset != len(locked['reliability_scores']):
        raise ValueError('Validation artifacts do not cover all locked scores.')
    output = {
        name: np.concatenate(values) for name, values in full.items()}
    output.update({
        'locked': locked,
        'sampled_features': np.concatenate(sampled_features),
        'sampled_reliable': np.concatenate(sampled_reliable),
        'sampled_is_tree': np.concatenate(sampled_is_tree),
        'sampled_utility': np.concatenate(sampled_utility),
        'sampled_scores': np.concatenate(sampled_scores),
        'feature_names': feature_names,
        'plot_names': plot_names,
        'candidate_mismatches': int(candidate_mismatches),
        'score_target_mismatches': int(score_target_mismatches),
    })
    return output


def top_univariate_features(features, targets, feature_names, limit):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=bool)
    if features.shape != (len(targets), len(feature_names)):
        raise ValueError('Sampled feature matrix has an invalid shape.')
    rows = []
    for index, name in enumerate(feature_names):
        values = features[:, index]
        if float(values.max()) == float(values.min()):
            continue
        metrics = finite_binary_metrics(targets, values)
        auc = float(metrics['roc_auc'])
        direction = 'higher' if auc >= 0.5 else 'lower'
        separability = max(auc, 1.0 - auc)
        rows.append({
            'feature': name,
            'group': 'backbone' if name.startswith('backbone_') else 'scalar',
            'separability_auc': float(separability),
            'direction': direction,
        })
    rows.sort(key=lambda row: (-row['separability_auc'], row['feature']))
    return rows[:int(limit)], rows


def build_report(settings, data):
    summary_path = Path(settings['e1b_summary_path'])
    if not summary_path.is_file():
        raise FileNotFoundError(f'Missing E1b summary: {summary_path}')
    with summary_path.open(encoding='utf-8') as file:
        e1b = json.load(file)
    locked_seed = int(settings['locked_seed'])
    seed_row = next(
        (row for row in e1b['per_seed'] if int(row['seed']) == locked_seed),
        None)
    if seed_row is None:
        raise ValueError('The locked seed is missing from E1b summary.')

    scores = data['locked']['reliability_scores']
    reliable = data['target_reliable'].astype(bool)
    is_tree = data['target_is_tree'].astype(bool)
    utility = data['target_utility'].astype(np.float32)
    plot_ids = data['plot_id'].astype(np.int64)
    threshold = float(settings['reliability_threshold'])
    expected_reliable = is_tree & (utility >= threshold)
    label_mismatches = int(np.count_nonzero(reliable != expected_reliable))

    overall = finite_binary_metrics(reliable, scores)
    tree_detection = finite_binary_metrics(is_tree, scores)
    tree_only = finite_binary_metrics(reliable[is_tree], scores[is_tree])
    per_plot = []
    for plot_id in sorted(np.unique(plot_ids).tolist()):
        mask = plot_ids == plot_id
        tree_mask = mask & is_tree
        metrics = finite_binary_metrics(reliable[mask], scores[mask])
        tree_metrics = finite_binary_metrics(
            reliable[tree_mask], scores[tree_mask])
        per_plot.append({
            'plot': data['plot_names'][plot_id],
            'num_candidates': int(mask.sum()),
            'reliable_prevalence': float(reliable[mask].mean()),
            'reliability_roc_auc': (
                None if metrics is None else metrics['roc_auc']),
            'tree_only_roc_auc': (
                None if tree_metrics is None else tree_metrics['roc_auc']),
        })
    plot_aucs = [
        row['reliability_roc_auc'] for row in per_plot
        if row['reliability_roc_auc'] is not None]
    macro_plot_auc = float(np.mean(plot_aucs))

    threshold_rows = []
    for value in settings['diagnostic']['utility_thresholds']:
        value = float(value)
        targets = is_tree & (utility >= value)
        metrics = finite_binary_metrics(targets, scores)
        threshold_rows.append({
            'threshold': value,
            'prevalence': float(targets.mean()),
            'roc_auc': None if metrics is None else metrics['roc_auc'],
            'average_precision': (
                None if metrics is None else metrics['average_precision']),
        })
    finite_threshold_rows = [
        row for row in threshold_rows if row['roc_auc'] is not None]
    best_threshold = max(
        finite_threshold_rows, key=lambda row: row['roc_auc'])

    tree_utility = utility[is_tree]
    ambiguity = {}
    for half_width in settings['diagnostic']['ambiguity_half_widths']:
        half_width = float(half_width)
        ambiguity[f'{half_width:g}'] = float(np.mean(
            np.abs(tree_utility - threshold) <= half_width))

    top_features, all_features = top_univariate_features(
        data['sampled_features'], data['sampled_reliable'],
        data['feature_names'],
        int(settings['diagnostic']['top_features_to_report']))
    best_univariate = max(
        (row['separability_auc'] for row in all_features), default=0.5)
    sampled_spearman = spearman_correlation(
        data['sampled_scores'], data['sampled_utility'])
    score_report_difference = abs(
        float(overall['roc_auc']) -
        float(seed_row['reliability_roc_auc']))
    false_gate_items = sorted(
        name for name, passed in e1b['gate'].items()
        if name != 'passed' and not bool(passed))

    decision_cfg = settings['decision']
    current_auc = float(overall['roc_auc'])
    macro_gap = macro_plot_auc - current_auc
    threshold_gain = float(best_threshold['roc_auc']) - current_auc
    near_rate = ambiguity.get('0.1', max(ambiguity.values()))
    tree_only_auc = (
        0.5 if tree_only is None else float(tree_only['roc_auc']))
    only_reliability_failed = false_gate_items == ['reliability_auc_passed']
    integrity = {
        'candidate_alignment_passed': data['candidate_mismatches'] == 0,
        'score_target_alignment_passed': (
            data['score_target_mismatches'] == 0),
        'label_definition_passed': label_mismatches == 0,
        'locked_score_metric_reproduced': score_report_difference <= 1e-7,
        'only_reliability_gate_failed': only_reliability_failed,
        'expected_validation_plots_present': (
            len(per_plot) == int(settings['expected_validation_plots'])),
    }
    integrity['passed'] = bool(all(integrity.values()))

    if not integrity['passed']:
        recommendation = 'repair_alignment'
    elif current_auc >= float(decision_cfg['target_auc']):
        recommendation = 'recheck_e1b_gate'
    elif macro_gap >= float(
            decision_cfg['macro_auc_gap_for_calibration']):
        recommendation = 'forest_calibrated_ranking'
    elif (
            threshold_gain >= float(
                decision_cfg['threshold_auc_gain_for_target_redesign']) or
            near_rate > float(
                decision_cfg['max_near_threshold_tree_rate'])):
        recommendation = 'continuous_utility_ranking'
    elif best_univariate >= float(
            decision_cfg['min_univariate_auc_for_capacity_signal']):
        recommendation = 'capacity_optimization_probe'
    elif tree_only_auc < float(
            decision_cfg['min_tree_only_auc_for_pointwise_signal']):
        recommendation = 'relational_attention_candidate'
    else:
        recommendation = 'parameter_matched_capacity_probe'

    return {
        'num_validation_candidates': int(len(scores)),
        'num_sampled_for_feature_audit': int(
            len(data['sampled_reliable'])),
        'locked_seed': locked_seed,
        'reliability_threshold': threshold,
        'overall_reliability': overall,
        'tree_detection': tree_detection,
        'tree_only_reliability': tree_only,
        'sampled_score_utility_spearman': sampled_spearman,
        'macro_plot_reliability_auc': macro_plot_auc,
        'macro_minus_global_auc': macro_gap,
        'per_plot': per_plot,
        'threshold_sensitivity': threshold_rows,
        'best_diagnostic_threshold': best_threshold,
        'best_threshold_auc_gain': threshold_gain,
        'tree_utility_ambiguity': ambiguity,
        'top_univariate_features': top_features,
        'best_univariate_auc': float(best_univariate),
        'alignment': {
            'candidate_mismatches': data['candidate_mismatches'],
            'score_target_mismatches': data['score_target_mismatches'],
            'label_definition_mismatches': label_mismatches,
            'locked_score_metric_difference': score_report_difference,
        },
        'e1b_false_gate_items': false_gate_items,
        'integrity_gate': integrity,
        'recommendation': recommendation,
    }


def format_markdown(report):
    def percent(value):
        return 'n/a' if value is None else f'{100 * value:.3f}%'

    lines = [
        '# E1b2 Seed Reliability 瓶颈审计', '',
        f'- 验证候选：{report["num_validation_candidates"]:,}',
        f'- 特征审计采样：{report["num_sampled_for_feature_audit"]:,}',
        f'- 锁定 seed：{report["locked_seed"]}',
        f'- 原 Reliability 阈值：{report["reliability_threshold"]:.2f}',
        f'- 自动建议：**{report["recommendation"]}**', '',
        '## 分解指标', '',
        '| Metric | ROC-AUC | AP / prevalence |',
        '|---|---:|---:|',
    ]
    for name, metrics in (
            ('all reliability', report['overall_reliability']),
            ('tree vs non-tree', report['tree_detection']),
            ('within-tree reliability', report['tree_only_reliability'])):
        if metrics is None:
            lines.append(f'| {name} | n/a | n/a |')
        else:
            lines.append(
                f'| {name} | {metrics["roc_auc"]:.6f} | '
                f'{metrics["average_precision"]:.6f} / '
                f'{metrics["prevalence"]:.6f} |')
    lines.extend([
        '',
        f'- Score/utility Spearman（固定采样）：'
        f'{report["sampled_score_utility_spearman"]:.6f}',
        f'- 每森林 macro AUC：'
        f'{report["macro_plot_reliability_auc"]:.6f}',
        f'- macro − global：{report["macro_minus_global_auc"]:+.6f}',
        f'- 最佳单变量 separability AUC：'
        f'{report["best_univariate_auc"]:.6f}', '',
        '## 每森林结果', '',
        '| Plot | Candidates | Prevalence | AUC | Tree-only AUC |',
        '|---|---:|---:|---:|---:|',
    ])
    for row in report['per_plot']:
        lines.append(
            f'| {row["plot"]} | {row["num_candidates"]:,} | '
            f'{percent(row["reliable_prevalence"])} | '
            f'{percent(row["reliability_roc_auc"])} | '
            f'{percent(row["tree_only_roc_auc"])} |')
    lines.extend([
        '', '## Utility 阈值敏感性（只作诊断）', '',
        '| Threshold | Prevalence | ROC-AUC | AP |',
        '|---:|---:|---:|---:|',
    ])
    for row in report['threshold_sensitivity']:
        lines.append(
            f'| {row["threshold"]:.2f} | {percent(row["prevalence"])} | '
            f'{percent(row["roc_auc"])} | '
            f'{percent(row["average_precision"])} |')
    lines.extend([
        '',
        f'- 最佳诊断阈值 AUC 增益：'
        f'{report["best_threshold_auc_gain"]:+.6f}',
        f'- 树点 utility 位于 0.50±0.05：'
        f'{percent(report["tree_utility_ambiguity"].get("0.05"))}',
        f'- 树点 utility 位于 0.50±0.10：'
        f'{percent(report["tree_utility_ambiguity"].get("0.1"))}', '',
        '## 最强单变量特征', '',
        '| Feature | Group | Separability AUC | Direction |',
        '|---|---|---:|---|',
    ])
    for row in report['top_univariate_features']:
        lines.append(
            f'| {row["feature"]} | {row["group"]} | '
            f'{row["separability_auc"]:.6f} | {row["direction"]} |')
    lines.extend(['', '## 完整性 Gate', ''])
    for name, passed in report['integrity_gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend([
        '', '## 下一步解释', '',
        '- `repair_alignment`：先修复 artifact/score/标签错位。',
        '- `forest_calibrated_ranking`：先解决跨森林校准，不加注意力。',
        '- `continuous_utility_ranking`：二值边界不稳定，改为连续排序目标。',
        '- `capacity_optimization_probe`：已有强单变量信号，先排查 MLP 优化。',
        '- `parameter_matched_capacity_probe`：先运行参数匹配的更强逐点网络。',
        '- `relational_attention_candidate`：标签稳定但树内逐点信号弱，'
        '进入显式邻域注意力对照。', '',
        '本阶段不改变 E1b Gate，也不读取 Wytham。',
    ])
    return '\n'.join(lines) + '\n'


def run(config_path):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    data = load_validation_diagnostic_data(settings)
    report = build_report(settings, data)
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not report['integrity_gate']['passed']:
        raise RuntimeError('E1b2 integrity gate failed; repair alignment first.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose the E1b seed-reliability bottleneck.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
