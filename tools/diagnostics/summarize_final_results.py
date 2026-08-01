import argparse
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_RUNS = {
    'l1w_base_r100': (
        'data/pipeline/L1W/results_seed_ratio_r100/full_forest/'
        'evaluation/evaluation_results.pt'),
    'l1w_random_r078_s42': (
        'data/pipeline/L1W/results_seed_random_r078_s42/full_forest/'
        'evaluation/evaluation_results.pt'),
    'l1w_random_r078_s43': (
        'data/pipeline/L1W/results_seed_random_r078_s43/full_forest/'
        'evaluation/evaluation_results.pt'),
    'l1w_random_r078_s44': (
        'data/pipeline/L1W/results_seed_random_r078_s44/full_forest/'
        'evaluation/evaluation_results.pt'),
    'l1w_mlp_r078_s42': (
        'data/pipeline/L1W/results_seed_ratio_r078/full_forest/'
        'evaluation/evaluation_results.pt'),
    'l1w_mlp_r078_s43': (
        'data/pipeline/L1W/results_seed_ratio_mlp_s43_r078/full_forest/'
        'evaluation/evaluation_results.pt'),
    'l1w_mlp_r078_s44': (
        'data/pipeline/L1W/results_seed_ratio_mlp_s44_r078/full_forest/'
        'evaluation/evaluation_results.pt'),
    'l1w_pt_r078': (
        'data/pipeline/L1W/results_seed_ratio_pt_r078/full_forest/'
        'evaluation/evaluation_results.pt'),
    'wytham_base': (
        'data/pipeline/wytham/results_seed_conf_t000_control/full_forest/'
        'evaluation/evaluation_results.pt'),
    'wytham_mlp_r078_s42': (
        'data/pipeline/wytham/results_seed_ratio_r078_development/'
        'full_forest/evaluation/evaluation_results.pt'),
    'wytham_mlp_r078_s43_locked': (
        'data/pipeline/wytham/results_seed_ratio_mlp_s43_r078_locked/'
        'full_forest/evaluation/evaluation_results.pt'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recompute exact final metrics from evaluation artifacts.')
    parser.add_argument(
        '--run', action='append', default=[], metavar='NAME=PATH',
        help='Override or add a run. May be specified more than once.')
    parser.add_argument(
        '--allow_missing', action='store_true',
        help='Skip missing artifacts instead of failing.')
    parser.add_argument(
        '--output_json', default='logs/final_exact_metrics.json')
    parser.add_argument(
        '--output_md', default='logs/final_exact_metrics.md')
    return parser.parse_args()


def load_results(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _length(values):
    return int(len(values))


def calculate_exact_metrics(results):
    detection = results['detection_results']
    segmentation = results['segmentation_results']

    matched_gt_count = _length(detection['matched_gts'])
    missed_gt_count = _length(detection['non_matched_gts'])
    matched_pred_count = _length(detection['matched_preds'])
    false_pred_count = _length(
        detection['non_matched_preds_filtered'])

    gt_denominator = matched_gt_count + missed_gt_count
    pred_denominator = matched_pred_count + false_pred_count
    if gt_denominator == 0 or pred_denominator == 0:
        raise ValueError('Detection metric denominator must be non-zero.')

    completeness = matched_gt_count / gt_denominator
    commission = false_pred_count / pred_denominator
    omission = 1.0 - completeness
    f1_score = (
        2.0 * (1.0 - commission) * (1.0 - omission) /
        (2.0 - commission - omission))

    no_partition = segmentation['no_partition']
    precision = float(np.asarray(no_partition['prec'], dtype=float).mean())
    recall = float(np.asarray(no_partition['rec'], dtype=float).mean())
    coverage = float(np.asarray(no_partition['iou'], dtype=float).mean())

    return {
        'completeness': 100.0 * completeness,
        'omission_error_rate': 100.0 * omission,
        'commission_error_rate': 100.0 * commission,
        'f1_score': 100.0 * f1_score,
        'precision': 100.0 * precision,
        'recall': 100.0 * recall,
        'coverage': 100.0 * coverage,
        'counts': {
            'matched_gt': matched_gt_count,
            'missed_gt': missed_gt_count,
            'matched_pred': matched_pred_count,
            'false_pred': false_pred_count,
        },
    }


def summarize_group(runs, names):
    selected = [runs[name] for name in names if name in runs]
    if len(selected) != len(names):
        return None
    metrics = [
        'completeness', 'commission_error_rate', 'f1_score',
        'precision', 'recall', 'coverage']
    summary = {'run_names': names, 'count': len(selected)}
    for metric in metrics:
        values = np.asarray([run[metric] for run in selected], dtype=float)
        summary[metric] = {
            'mean': float(values.mean()),
            'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def format_markdown(runs, groups):
    headers = [
        'Run', 'Completeness', 'Commission', 'F1',
        'Precision', 'Recall', 'Coverage']
    rows = [
        '# Final exact evaluation metrics', '',
        '| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join(['---'] + ['---:'] * 6) + '|']
    for name, result in runs.items():
        values = [
            name,
            f"{result['completeness']:.6f}%",
            f"{result['commission_error_rate']:.6f}%",
            f"{result['f1_score']:.6f}%",
            f"{result['precision']:.6f}%",
            f"{result['recall']:.6f}%",
            f"{result['coverage']:.6f}%",
        ]
        rows.append('| ' + ' | '.join(values) + ' |')

    rows.extend(['', '## Multi-seed summaries', ''])
    for group_name, summary in groups.items():
        if summary is None:
            continue
        rows.append(f'### {group_name}')
        rows.append('')
        rows.append('| Metric | Mean | Sample std |')
        rows.append('|---|---:|---:|')
        for metric, stats in summary.items():
            if metric in ('run_names', 'count'):
                continue
            rows.append(
                f"| {metric} | {stats['mean']:.6f}% | "
                f"{stats['sample_std']:.6f}% |")
        rows.append('')
    return '\n'.join(rows) + '\n'


def main():
    args = parse_args()
    run_paths = dict(DEFAULT_RUNS)
    for item in args.run:
        if '=' not in item:
            raise ValueError('--run must use NAME=PATH format.')
        name, path = item.split('=', 1)
        run_paths[name] = path

    runs = {}
    missing = {}
    for name, path_string in run_paths.items():
        path = Path(path_string)
        if not path.is_file():
            missing[name] = str(path)
            continue
        runs[name] = calculate_exact_metrics(load_results(path))

    if missing and not args.allow_missing:
        details = '\n'.join(
            f'  {name}: {path}' for name, path in missing.items())
        raise FileNotFoundError(
            'Missing evaluation artifacts:\n' + details +
            '\nUse --allow_missing only for an intentionally partial summary.')

    groups = {
        'l1w_random_r078': summarize_group(runs, [
            'l1w_random_r078_s42', 'l1w_random_r078_s43',
            'l1w_random_r078_s44']),
        'l1w_mlp_r078': summarize_group(runs, [
            'l1w_mlp_r078_s42', 'l1w_mlp_r078_s43',
            'l1w_mlp_r078_s44']),
    }
    output = {
        'runs': runs,
        'groups': groups,
        'missing': missing,
        'units': 'percent',
        'std_definition': 'sample standard deviation (ddof=1)',
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    output_md.write_text(format_markdown(runs, groups), encoding='utf-8')

    print(format_markdown(runs, groups), end='')
    if missing:
        print('Skipped missing runs:')
        for name, path in missing.items():
            print(f'  {name}: {path}')
    print(f'Saved JSON: {output_json}')
    print(f'Saved Markdown: {output_md}')


if __name__ == '__main__':
    main()
