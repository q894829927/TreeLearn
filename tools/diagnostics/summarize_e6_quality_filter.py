import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch



def load_results(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def exact_metrics(results):
    detection = results['detection_results']
    segmentation = results['segmentation_results']
    matched_gt = len(detection['matched_gts'])
    missed_gt = len(detection['non_matched_gts'])
    matched_pred = len(detection['matched_preds'])
    false_pred = len(detection['non_matched_preds_filtered'])
    completeness = matched_gt / (matched_gt + missed_gt)
    commission = false_pred / (matched_pred + false_pred)
    precision_detection = 1.0 - commission
    f1_score = (
        2.0 * precision_detection * completeness /
        (precision_detection + completeness))
    no_partition = segmentation['no_partition']
    return {
        'completeness': 100.0 * completeness,
        'commission_error_rate': 100.0 * commission,
        'f1_score': 100.0 * f1_score,
        'precision': 100.0 * float(np.mean(no_partition['prec'])),
        'recall': 100.0 * float(np.mean(no_partition['rec'])),
        'coverage': 100.0 * float(np.mean(no_partition['iou'])),
        'counts': {
            'matched_gt': matched_gt,
            'missed_gt': missed_gt,
            'matched_pred': matched_pred,
            'false_pred': false_pred,
        },
    }


def read_pipeline_seconds(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    values = re.findall(r'pipeline finished in ([0-9.]+)s', text)
    if not values:
        raise ValueError(f'Pipeline total time is missing from {path}.')
    return float(values[-1])


def metrics_identical(first, second, tolerance):
    names = (
        'completeness', 'commission_error_rate', 'f1_score',
        'precision', 'recall', 'coverage')
    return (
        first['counts'] == second['counts'] and
        all(abs(first[name] - second[name]) <= tolerance for name in names))


def format_markdown(report):
    rows = [
        '# E6 Vertical-MLP pipeline 集成',
        '',
        f"- 锁定 checkpoint：`{report['checkpoint']}`",
        f"- 锁定阈值：{report['threshold']:.4f}",
        f"- Control 标签摘要一致："
        f"**{report['control_metadata']['score_only_labels_identical']}**",
        f"- 最大质量评分耗时占比："
        f"{100.0 * report['maximum_quality_runtime_fraction']:.3f}%",
        '',
        '| Run | Completeness | Commission | F1 | Precision | Recall | Coverage |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('baseline', 'control', 'filtered'):
        metrics = report[name]
        rows.append(
            f"| {name} | {metrics['completeness']:.6f}% | "
            f"{metrics['commission_error_rate']:.6f}% | "
            f"{metrics['f1_score']:.6f}% | "
            f"{metrics['precision']:.6f}% | "
            f"{metrics['recall']:.6f}% | "
            f"{metrics['coverage']:.6f}% |")
    rows.extend([
        '',
        '## Delta: filtered minus baseline',
        '',
        f"- F1：{report['delta']['f1_gain_pp']:+.6f} 个百分点",
        f"- Commission 降低："
        f"{report['delta']['commission_reduction_pp']:+.6f} 个百分点",
        f"- Completeness 下降："
        f"{report['delta']['completeness_drop_pp']:+.6f} 个百分点",
        '',
        '## Gate',
        '',
    ])
    for name, passed in report['gate'].items():
        rows.append(f"- {name}: **{passed}**")
    rows.extend([
        '',
        'PASS：进入 E7。' if report['gate']['passed']
        else 'STOP：修复失败项，不进入 E7。',
        '',
    ])
    return '\n'.join(rows)


def main():
    from tree_learn.util import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = get_config(args.config)

    runs = {}
    for name in ('baseline', 'control', 'filtered'):
        path = Path(config.evaluations[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        runs[name] = exact_metrics(load_results(path))

    with open(config.metadata.control, encoding='utf-8') as file:
        control_metadata = json.load(file)
    with open(config.metadata.filtered, encoding='utf-8') as file:
        filtered_metadata = json.load(file)

    control_total = read_pipeline_seconds(config.pipeline_logs.control)
    filtered_total = read_pipeline_seconds(config.pipeline_logs.filtered)
    runtime_fractions = [
        control_metadata['scoring_seconds'] / control_total,
        filtered_metadata['scoring_seconds'] / filtered_total,
    ]
    threshold = float(config.locked.threshold)
    tolerance = float(config.gate.control_metric_tolerance)
    delta = {
        'f1_gain_pp': (
            runs['filtered']['f1_score'] - runs['baseline']['f1_score']),
        'commission_reduction_pp': (
            runs['baseline']['commission_error_rate'] -
            runs['filtered']['commission_error_rate']),
        'completeness_drop_pp': (
            runs['baseline']['completeness'] -
            runs['filtered']['completeness']),
    }
    gate = {
        'score_only_digest_identical': bool(
            control_metadata['score_only_labels_identical']),
        'control_metrics_identical': metrics_identical(
            runs['baseline'], runs['control'], tolerance),
        'checkpoint_seed_locked': (
            int(control_metadata['checkpoint_seed']) ==
            int(config.locked.checkpoint_seed) and
            int(filtered_metadata['checkpoint_seed']) ==
            int(config.locked.checkpoint_seed)),
        'threshold_locked': (
            abs(float(control_metadata['threshold']) - threshold) <= 1e-12 and
            abs(float(filtered_metadata['threshold']) - threshold) <= 1e-12),
        'filter_modes_correct': (
            not bool(control_metadata['filter_enabled']) and
            bool(filtered_metadata['filter_enabled'])),
        'completeness_drop_passed': (
            delta['completeness_drop_pp'] <=
            float(config.gate.max_completeness_drop_pp) + 1e-12),
        'f1_not_below_baseline': (
            delta['f1_gain_pp'] >=
            float(config.gate.min_f1_gain_pp) - 1e-12),
        'runtime_overhead_passed': (
            max(runtime_fractions) <=
            float(config.gate.max_quality_runtime_fraction) + 1e-12),
    }
    gate['passed'] = bool(all(gate.values()))
    report = {
        'checkpoint': str(config.locked.checkpoint),
        'checkpoint_seed': int(config.locked.checkpoint_seed),
        'threshold': threshold,
        **runs,
        'delta': delta,
        'control_metadata': control_metadata,
        'filtered_metadata': filtered_metadata,
        'pipeline_seconds': {
            'control': control_total,
            'filtered': filtered_total,
        },
        'quality_runtime_fractions': runtime_fractions,
        'maximum_quality_runtime_fraction': max(runtime_fractions),
        'gate': gate,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = format_markdown(report)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    if not gate['passed']:
        raise RuntimeError('E6 gate failed; do not proceed to E7.')


if __name__ == '__main__':
    main()
