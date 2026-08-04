import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_yaml(path):
    import yaml

    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_evaluation(path):
    import torch

    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_quality_scores(path):
    instance_ids = []
    scores = []
    with open(path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        required = {'instance_id', 'quality_score'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f'Quality score CSV misses fields: {sorted(missing)}')
        for row in reader:
            instance_ids.append(int(row['instance_id']))
            scores.append(float(row['quality_score']))
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(instance_ids) == 0:
        raise ValueError('Quality score CSV is empty.')
    if len(np.unique(instance_ids)) != len(instance_ids):
        raise ValueError('Quality score instance IDs must be unique.')
    if not np.isfinite(scores).all():
        raise ValueError('Quality scores must be finite.')
    return instance_ids, scores


def validate_score_only_metadata(path, expected_instances):
    with open(path, encoding='utf-8') as file:
        metadata = json.load(file)
    checks = {
        'filter_disabled': not bool(metadata.get('filter_enabled', True)),
        'labels_identical': bool(
            metadata.get('score_only_labels_identical', False)),
        'instance_count_matches': (
            int(metadata.get('num_instances', -1)) ==
            int(expected_instances)),
    }
    if not all(checks.values()):
        raise ValueError(
            f'Quality scores are not a valid score-only control: {checks}')
    return metadata, checks


def build_detection_status(evaluation, quality_ids, quality_scores):
    detection = evaluation['detection_results']
    matched = np.asarray(
        detection['matched_preds'], dtype=np.int64).reshape(-1)
    non_matched = np.asarray(
        detection['non_matched_preds'], dtype=np.int64).reshape(-1)
    counted_false = np.asarray(
        detection['non_matched_preds_filtered'],
        dtype=np.int64).reshape(-1)
    if set(matched.tolist()) & set(non_matched.tolist()):
        raise ValueError('Matched and non-matched prediction IDs overlap.')
    if not set(counted_false.tolist()) <= set(non_matched.tolist()):
        raise ValueError('Counted false predictions must be non-matched.')
    all_predictions = np.concatenate([matched, non_matched])
    if len(np.unique(all_predictions)) != len(all_predictions):
        raise ValueError('Evaluation prediction IDs must be unique.')
    if set(all_predictions.tolist()) != set(quality_ids.tolist()):
        missing = sorted(
            set(all_predictions.tolist()) - set(quality_ids.tolist()))
        extra = sorted(
            set(quality_ids.tolist()) - set(all_predictions.tolist()))
        raise ValueError(
            'Quality IDs do not match evaluation IDs: '
            f'missing={missing[:10]}, extra={extra[:10]}')
    score_order = np.argsort(quality_ids)
    sorted_ids = quality_ids[score_order]
    score_positions = np.searchsorted(sorted_ids, all_predictions)
    aligned_scores = quality_scores[score_order][score_positions]
    tp_ids = set(matched.tolist())
    fp_ids = set(counted_false.tolist())
    status = np.asarray([
        1 if int(value) in tp_ids else
        -1 if int(value) in fp_ids else 0
        for value in all_predictions
    ], dtype=np.int8)
    num_gt = int(
        len(np.asarray(detection['matched_gts']).reshape(-1)) +
        len(np.asarray(detection['non_matched_gts']).reshape(-1)))
    if num_gt <= 0:
        raise ValueError('Evaluation contains no GT trees.')
    return {
        'instance_ids': all_predictions,
        'scores': aligned_scores,
        'status': status,
        'num_gt': num_gt,
        'baseline_tp': int(len(matched)),
        'baseline_fp': int(len(counted_false)),
        'baseline_ignored': int(len(non_matched) - len(counted_false)),
    }


def metrics_from_counts(tp, fp, num_gt, retained, total):
    tp = int(tp)
    fp = int(fp)
    fn = int(num_gt - tp)
    precision = tp / max(tp + fp, 1)
    completeness = tp / max(num_gt, 1)
    f1 = (
        2.0 * precision * completeness /
        (precision + completeness)
        if precision + completeness else 0.0)
    return {
        'retained_predictions': int(retained),
        'total_predictions': int(total),
        'retention_ratio_actual': float(retained / max(total, 1)),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': float(precision),
        'completeness': float(completeness),
        'commission': float(1.0 - precision),
        'f1': float(f1),
    }


def retention_counts(total, ratios):
    counts = [
        max(1, int(math.ceil(float(ratio) * total)))
        for ratio in ratios
    ]
    return np.asarray(counts, dtype=np.int64)


def curve_for_order(status, order, ratios, num_gt):
    status = np.asarray(status, dtype=np.int8)
    order = np.asarray(order, dtype=np.int64)
    if len(order) != len(status) or set(order.tolist()) != set(
            range(len(status))):
        raise ValueError('Order must be a permutation of all predictions.')
    ordered = status[order]
    cumulative_tp = np.cumsum(ordered == 1)
    cumulative_fp = np.cumsum(ordered == -1)
    counts = retention_counts(len(status), ratios)
    rows = []
    for requested_ratio, count in zip(ratios, counts):
        metrics = metrics_from_counts(
            cumulative_tp[count - 1], cumulative_fp[count - 1],
            num_gt, count, len(status))
        metrics['retention_ratio_requested'] = float(requested_ratio)
        rows.append(metrics)
    return rows


def score_curve(instance_ids, scores, status, ratios, num_gt):
    order = np.lexsort((instance_ids, -scores))
    return curve_for_order(status, order, ratios, num_gt)


def random_curves(status, ratios, num_gt, repeats, seed):
    rng = np.random.default_rng(int(seed))
    all_curves = []
    for _ in range(int(repeats)):
        order = rng.permutation(len(status))
        all_curves.append(curve_for_order(
            status, order, ratios, num_gt))
    rows = []
    for index, ratio in enumerate(ratios):
        row = {'retention_ratio_requested': float(ratio)}
        for metric in (
                'retention_ratio_actual', 'precision', 'completeness',
                'commission', 'f1'):
            values = np.asarray([
                curve[index][metric] for curve in all_curves],
                dtype=np.float64)
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(
                values.std(ddof=1) if len(values) > 1 else 0.0)
        rows.append(row)
    return rows


def normalized_area(rows, metric):
    x_values = np.asarray([
        row['retention_ratio_actual'] for row in rows],
        dtype=np.float64)
    y_values = np.asarray([row[metric] for row in rows], dtype=np.float64)
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    span = float(x_values[-1] - x_values[0])
    if span <= 0:
        return float(y_values.mean())
    integrate = (
        np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)
    return float(integrate(y_values, x_values) / span)


def summarize_curve(score_rows, random_rows):
    random_as_curve = [{
        'retention_ratio_actual': row['retention_ratio_actual_mean'],
        'commission': row['commission_mean'],
        'f1': row['f1_mean'],
    } for row in random_rows]
    score_commission_area = normalized_area(score_rows, 'commission')
    random_commission_area = normalized_area(
        random_as_curve, 'commission')
    score_f1_area = normalized_area(score_rows, 'f1')
    random_f1_area = normalized_area(random_as_curve, 'f1')
    return {
        'score_commission_area': score_commission_area,
        'random_commission_area': random_commission_area,
        'relative_commission_area_reduction': float(
            (random_commission_area - score_commission_area) /
            max(random_commission_area, 1e-12)),
        'score_f1_area': score_f1_area,
        'random_f1_area': random_f1_area,
        'f1_area_gain_pp': float(
            100.0 * (score_f1_area - random_f1_area)),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty CSV.')
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, name, score_rows, random_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x_score = [100 * row['retention_ratio_actual'] for row in score_rows]
    x_random = [
        100 * row['retention_ratio_actual_mean'] for row in random_rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        x_score, [100 * row['commission'] for row in score_rows],
        marker='o', markersize=3, label='Vertical-MLP ranking')
    axes[0].plot(
        x_random, [100 * row['commission_mean'] for row in random_rows],
        linestyle='--', label='Random ranking mean')
    axes[0].set_ylabel('Commission error rate (%)')
    axes[1].plot(
        x_score, [100 * row['f1'] for row in score_rows],
        marker='o', markersize=3, label='Vertical-MLP ranking')
    axes[1].plot(
        x_random, [100 * row['f1_mean'] for row in random_rows],
        linestyle='--', label='Random ranking mean')
    axes[1].set_ylabel('Detection F1 (%)')
    for axis in axes:
        axis.set_xlabel('Retained predicted instances (%)')
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(f'Selective instance segmentation: {name}')
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def format_percent(value):
    return f'{100.0 * float(value):.3f}%'


def format_markdown(report):
    lines = [
        f"# E9 选择性实例分割：{report['name']}",
        '',
        f"- 数据角色：**{report['dataset_role']}**",
        '- 原始实例标签保持不变；本阶段只评估质量排序。',
        f"- 实例数：{report['num_predictions']}（TP "
        f"{report['baseline_counts']['tp']}，counted FP "
        f"{report['baseline_counts']['fp']}，ignored "
        f"{report['baseline_counts']['ignored']}）",
        f"- 随机排序对照：{report['random_repeats']} 次，seed "
        f"{report['random_seed']}",
        '',
        '## 面积指标',
        '',
        '| Metric | Vertical-MLP | Random | Delta |',
        '|---|---:|---:|---:|',
        '| Commission area（越低越好） | '
        f"{format_percent(report['summary']['score_commission_area'])} | "
        f"{format_percent(report['summary']['random_commission_area'])} | "
        f"{format_percent(report['summary']['relative_commission_area_reduction'])} relative reduction |",
        '| Detection F1 area（越高越好） | '
        f"{format_percent(report['summary']['score_f1_area'])} | "
        f"{format_percent(report['summary']['random_f1_area'])} | "
        f"{report['summary']['f1_area_gain_pp']:+.3f} pp |",
        '',
        '## 关键保留率',
        '',
        '| Retained | Score F1 | Random F1 | Score Commission | Random Commission | Score Completeness |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    selected_indices = sorted(set([
        0, len(report['score_curve']) // 2,
        len(report['score_curve']) - 1,
    ] + [
        index for index, row in enumerate(report['score_curve'])
        if abs(row['retention_ratio_requested'] - 0.85) < 1e-9
    ]))
    for index in selected_indices:
        score = report['score_curve'][index]
        random = report['random_curve'][index]
        lines.append(
            f"| {format_percent(score['retention_ratio_actual'])} | "
            f"{format_percent(score['f1'])} | "
            f"{format_percent(random['f1_mean'])} | "
            f"{format_percent(score['commission'])} | "
            f"{format_percent(random['commission_mean'])} | "
            f"{format_percent(score['completeness'])} |")
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    lines.extend([''])
    if report['gate']['passed']:
        lines.append('PASS：质量分数提供了优于随机排序的选择性风险控制。')
    else:
        lines.append('STOP：垂直质量分数没有形成稳定的选择性风险优势。')
    return '\n'.join(lines) + '\n'


def run_dataset(spec, settings):
    paths = {
        key: Path(spec[key])
        for key in ('evaluation', 'quality_scores', 'quality_metadata')
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing E9 inputs: ' + ', '.join(missing))
    evaluation = load_evaluation(paths['evaluation'])
    quality_ids, quality_scores = load_quality_scores(
        paths['quality_scores'])
    status_data = build_detection_status(
        evaluation, quality_ids, quality_scores)
    metadata, metadata_checks = validate_score_only_metadata(
        paths['quality_metadata'], len(quality_ids))
    ratios = np.round(np.arange(
        float(settings['ratio_grid']['start']),
        float(settings['ratio_grid']['stop']) +
        0.5 * float(settings['ratio_grid']['step']),
        float(settings['ratio_grid']['step'])), 10)
    score_rows = score_curve(
        status_data['instance_ids'], status_data['scores'],
        status_data['status'], ratios, status_data['num_gt'])
    random_rows = random_curves(
        status_data['status'], ratios, status_data['num_gt'],
        settings['random_control']['repeats'],
        settings['random_control']['seed'])
    summary = summarize_curve(score_rows, random_rows)
    baseline = score_rows[-1]
    baseline_counts_match = bool(
        baseline['tp'] == status_data['baseline_tp'] and
        baseline['fp'] == status_data['baseline_fp'] and
        baseline['fn'] == status_data['num_gt'] -
        status_data['baseline_tp'])
    gate = {
        'score_only_metadata_passed': bool(all(metadata_checks.values())),
        'baseline_counts_reproduced': baseline_counts_match,
        'commission_area_improved': bool(
            summary['relative_commission_area_reduction'] >=
            float(settings['gate'][
                'min_relative_commission_area_reduction'])),
        'f1_area_improved': bool(
            summary['f1_area_gain_pp'] >=
            float(settings['gate']['min_f1_area_gain_pp'])),
    }
    gate['passed'] = bool(all(gate.values()))
    report = {
        'name': str(spec['name']),
        'dataset_role': str(spec['dataset_role']),
        'evaluation': str(paths['evaluation']),
        'quality_scores': str(paths['quality_scores']),
        'quality_metadata': str(paths['quality_metadata']),
        'num_predictions': int(len(quality_ids)),
        'num_gt': int(status_data['num_gt']),
        'baseline_counts': {
            'tp': int(status_data['baseline_tp']),
            'fp': int(status_data['baseline_fp']),
            'ignored': int(status_data['baseline_ignored']),
        },
        'checkpoint_seed': metadata.get('checkpoint_seed'),
        'random_repeats': int(settings['random_control']['repeats']),
        'random_seed': int(settings['random_control']['seed']),
        'summary': summary,
        'score_curve': score_rows,
        'random_curve': random_rows,
        'gate': gate,
    }
    output_dir = Path(spec['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'score_curve.csv', score_rows)
    write_csv(output_dir / 'random_curve.csv', random_rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8')
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    save_plot(
        output_dir / 'selective_curve.png', report['name'],
        score_rows, random_rows)
    print(markdown, flush=True)
    return report


def run(config_path):
    settings = load_yaml(config_path)
    reports = [run_dataset(spec, settings) for spec in settings['runs']]
    primary_name = str(settings['primary_run'])
    primary = next(
        report for report in reports if report['name'] == primary_name)
    summary = {
        'config': str(config_path),
        'primary_run': primary_name,
        'reports': {
            report['name']: {
                'dataset_role': report['dataset_role'],
                'summary': report['summary'],
                'gate': report['gate'],
            }
            for report in reports
        },
        'gate': {
            'all_runs_passed': bool(all(
                report['gate']['passed'] for report in reports)),
            'primary_run_passed': bool(primary['gate']['passed']),
        },
    }
    summary['gate']['passed'] = bool(all(summary['gate'].values()))
    output_dir = Path(settings['summary_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8')
    lines = ['# E9 选择性实例分割汇总', '']
    for report in reports:
        lines.append(
            f"- {report['name']}: Commission area reduction="
            f"{100 * report['summary']['relative_commission_area_reduction']:.3f}%，"
            f"F1 area gain={report['summary']['f1_area_gain_pp']:+.3f} pp，"
            f"Gate={report['gate']['passed']}")
    lines.extend([
        '', f"- Primary Gate：**{summary['gate']['passed']}**", '',
        '注意：Wytham 曲线只用于最终评价，不得从曲线上重新选择保留率。',
    ])
    markdown = '\n'.join(lines) + '\n'
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError(
            'E9 selective-quality gate failed; do not claim risk control.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate E9 selective instance-quality curves.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
