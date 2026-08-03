import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes')


def load_predictions(path):
    rows = []
    with Path(path).open(newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            rows.append({
                'seed': int(row['seed']),
                'source_plot': str(row['source_plot']),
                'instance_id': int(row['instance_id']),
                'score': float(row['score']),
                'target_true': parse_bool(row['target_true']),
                'classification_valid': parse_bool(
                    row['classification_valid']),
            })
    if not rows:
        raise ValueError('Validation prediction CSV is empty.')
    return rows


def top_ratio_mask(scores, instance_ids, plots, keep_ratio):
    scores = np.asarray(scores, dtype=np.float64)
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    plots = np.asarray(plots, dtype=str)
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError('keep_ratio must be in (0, 1].')
    retained = np.zeros(len(scores), dtype=bool)
    for plot in np.unique(plots):
        indices = np.flatnonzero(plots == plot)
        keep_count = max(1, int(math.ceil(float(keep_ratio) * len(indices))))
        order = np.lexsort((instance_ids[indices], -scores[indices]))
        retained[indices[order[:keep_count]]] = True
    return retained


def detection_metrics(targets, classification_valid, retained):
    targets = np.asarray(targets, dtype=bool)
    valid = np.asarray(classification_valid, dtype=bool)
    retained = np.asarray(retained, dtype=bool)
    targets = targets[valid]
    retained = retained[valid]
    positives = int(np.count_nonzero(targets))
    if positives == 0:
        raise ValueError('A seed has no valid positive validation instances.')
    tp = int(np.count_nonzero(retained & targets))
    fp = int(np.count_nonzero(retained & ~targets))
    fn = positives - tp
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'completeness': tp / positives,
        'commission': fp / max(tp + fp, 1),
        'f1': 2 * tp / max(2 * tp + fp + fn, 1),
    }


def evaluate_keep_ratios(rows, seeds, ratios, max_completeness_drop):
    by_seed = defaultdict(list)
    for row in rows:
        if row['seed'] in seeds:
            by_seed[row['seed']].append(row)
    missing = set(seeds) - set(by_seed)
    if missing:
        raise ValueError(f'Missing validation predictions for seeds {missing}.')

    summaries = []
    details = []
    for ratio in ratios:
        seed_metrics = []
        for seed in seeds:
            group = by_seed[seed]
            retained = top_ratio_mask(
                [row['score'] for row in group],
                [row['instance_id'] for row in group],
                [row['source_plot'] for row in group],
                ratio)
            metrics = detection_metrics(
                [row['target_true'] for row in group],
                [row['classification_valid'] for row in group],
                retained)
            metrics.update({
                'seed': int(seed),
                'keep_ratio': float(ratio),
                'retained_instances': int(retained.sum()),
                'total_instances': len(retained),
            })
            seed_metrics.append(metrics)
            details.append(metrics)
        completeness = np.asarray(
            [item['completeness'] for item in seed_metrics])
        f1_values = np.asarray([item['f1'] for item in seed_metrics])
        commission = np.asarray(
            [item['commission'] for item in seed_metrics])
        summaries.append({
            'keep_ratio': float(ratio),
            'mean_f1': float(f1_values.mean()),
            'min_f1': float(f1_values.min()),
            'mean_completeness': float(completeness.mean()),
            'min_completeness': float(completeness.min()),
            'mean_commission': float(commission.mean()),
            'constraint_passed': bool(np.all(
                completeness >= 1.0 - max_completeness_drop - 1e-12)),
        })
    candidates = [row for row in summaries if row['constraint_passed']]
    if not candidates:
        raise RuntimeError('No keep ratio satisfies every seed constraint.')
    selected = max(
        candidates,
        key=lambda row: (
            row['mean_f1'], row['min_f1'], -row['keep_ratio']))
    return selected, summaries, details


def format_markdown(report):
    selected = report['selected']
    rows = [
        '# E7b 域稳健质量排序：验证集保留比例选择',
        '',
        '- 数据来源：仅 E4 固定 validation forests；未使用 Wytham 标签。',
        f"- 随机种子：{report['seeds']}",
        f"- 最大 Completeness 下降："
        f"{100 * report['max_completeness_drop']:.2f}%",
        f"- 选中 keep ratio：**{selected['keep_ratio']:.4f}**",
        f"- 平均候选 F1：{selected['mean_f1']:.6f}",
        f"- 最低 Completeness：{selected['min_completeness']:.6f}",
        '',
        '| Ratio | Mean F1 | Min F1 | Mean completeness | '
        'Min completeness | Mean commission | Pass |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in report['ratios']:
        rows.append(
            f"| {item['keep_ratio']:.4f} | {item['mean_f1']:.6f} | "
            f"{item['min_f1']:.6f} | "
            f"{item['mean_completeness']:.6f} | "
            f"{item['min_completeness']:.6f} | "
            f"{item['mean_commission']:.6f} | "
            f"{item['constraint_passed']} |")
    return '\n'.join(rows) + '\n'


def main():
    from tree_learn.util import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    start = float(config.ratio_grid.start)
    stop = float(config.ratio_grid.stop)
    step = float(config.ratio_grid.step)
    ratios = np.round(
        np.arange(start, stop + step / 2.0, step), decimals=10)
    rows = load_predictions(config.validation_predictions)
    selected, summaries, details = evaluate_keep_ratios(
        rows,
        [int(seed) for seed in config.seeds],
        ratios,
        float(config.max_completeness_drop))
    report = {
        'validation_predictions': str(config.validation_predictions),
        'seeds': [int(seed) for seed in config.seeds],
        'max_completeness_drop': float(config.max_completeness_drop),
        'selected': selected,
        'ratios': summaries,
        'per_seed': details,
        'selection_rule': (
            'all seeds satisfy completeness; maximize mean F1, then min F1, '
            'then choose the smaller ratio'),
        'wytham_labels_used': False,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = format_markdown(report)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown)


if __name__ == '__main__':
    main()
