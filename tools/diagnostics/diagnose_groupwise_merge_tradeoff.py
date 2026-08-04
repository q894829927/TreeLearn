import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_yaml(path):
    import yaml

    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'true', '1', 'yes'}:
        return True
    if normalized in {'false', '0', 'no'}:
        return False
    raise ValueError(f'Invalid boolean value: {value!r}')


def load_prediction_records(path, expected_seeds):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records_by_seed = {int(seed): [] for seed in expected_seeds}
    seen = set()
    with path.open('r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        required = {
            'seed', 'source_key', 'target_instance_id', 'confidence',
            'no_merge_probability', 'chosen_correct',
            'has_positive_target',
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f'Prediction CSV is missing fields: {sorted(missing)}')
        for raw in reader:
            seed = int(raw['seed'])
            if seed not in records_by_seed:
                raise ValueError(f'Unexpected seed in predictions: {seed}')
            source_key = str(raw['source_key'])
            identity = (seed, source_key)
            if identity in seen:
                raise ValueError(
                    f'Duplicate source record for seed {seed}: {source_key}')
            seen.add(identity)
            confidence = float(raw['confidence'])
            no_merge_probability = float(raw['no_merge_probability'])
            if (
                    not math.isfinite(confidence) or
                    not 0.0 <= confidence <= 1.0 or
                    not math.isfinite(no_merge_probability) or
                    not 0.0 <= no_merge_probability <= 1.0):
                raise ValueError(
                    f'Invalid probabilities for {seed}:{source_key}')
            chosen_correct = parse_bool(raw['chosen_correct'])
            has_positive = parse_bool(raw['has_positive_target'])
            if chosen_correct and not has_positive:
                raise ValueError(
                    'A correct chosen target requires a positive target.')
            records_by_seed[seed].append({
                'seed': seed,
                'source_key': source_key,
                'target_instance_id': int(raw['target_instance_id']),
                'confidence': confidence,
                'no_merge_probability': no_merge_probability,
                'chosen_correct': chosen_correct,
                'has_positive_target': has_positive,
            })
    if any(not rows for rows in records_by_seed.values()):
        raise ValueError('Every expected seed must contain predictions.')
    reference_seed = int(expected_seeds[0])
    reference = {
        row['source_key']: row['has_positive_target']
        for row in records_by_seed[reference_seed]
    }
    for seed, rows in records_by_seed.items():
        current = {
            row['source_key']: row['has_positive_target']
            for row in rows
        }
        if current != reference:
            raise ValueError(
                f'Source keys or labels do not align for seed {seed}.')
    return records_by_seed


def source_metrics(records, threshold):
    selected = [
        row for row in records
        if row['confidence'] >= float(threshold)]
    correct = int(sum(row['chosen_correct'] for row in selected))
    unsafe = int(len(selected) - correct)
    positives = int(sum(
        row['has_positive_target'] for row in records))
    return {
        'threshold': float(threshold),
        'num_evaluable_sources': int(len(records)),
        'num_positive_sources': positives,
        'num_proposed_merges': int(len(selected)),
        'num_correct_merges': correct,
        'num_unsafe_merges': unsafe,
        'pair_precision': float(
            correct / len(selected) if selected else 1.0),
        'positive_source_recall': float(
            correct / max(positives, 1)),
        'unsafe_merge_rate': float(
            unsafe / max(len(records), 1)),
    }


def build_threshold_frontier(records):
    thresholds = [float('inf')]
    thresholds.extend(sorted({
        float(row['confidence']) for row in records
    }, reverse=True))
    return [source_metrics(records, threshold) for threshold in thresholds]


def select_frontier_row(frontier, min_precision, max_unsafe_rate):
    feasible = [
        row for row in frontier
        if row['pair_precision'] >= float(min_precision)
        and row['unsafe_merge_rate'] <= float(max_unsafe_rate)
    ]
    if not feasible:
        raise RuntimeError('No threshold satisfies the safety constraints.')
    return dict(max(
        feasible,
        key=lambda row: (
            row['positive_source_recall'],
            row['pair_precision'],
            -row['unsafe_merge_rate'],
            row['threshold'],
        ),
    ))


def sample_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot write an empty CSV.')
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_sensitivity(
        records_by_seed, precision_levels, unsafe_rate_levels,
        primary, locked_seed, minimum_pass_seeds):
    frontiers = {
        seed: build_threshold_frontier(records)
        for seed, records in records_by_seed.items()
    }
    frontier_rows = []
    for seed, rows in frontiers.items():
        frontier_rows.extend({'seed': seed, **row} for row in rows)

    sensitivity_rows = []
    for min_precision in precision_levels:
        for max_unsafe_rate in unsafe_rate_levels:
            for seed, frontier in frontiers.items():
                selected = select_frontier_row(
                    frontier, min_precision, max_unsafe_rate)
                selected.update({
                    'seed': int(seed),
                    'min_pair_precision': float(min_precision),
                    'max_unsafe_merge_rate': float(max_unsafe_rate),
                })
                sensitivity_rows.append(selected)

    primary_precision = float(primary['min_pair_precision'])
    primary_unsafe = float(primary['max_unsafe_merge_rate'])
    minimum_recall = float(primary['min_positive_source_recall'])
    primary_rows = [
        row for row in sensitivity_rows
        if row['min_pair_precision'] == primary_precision
        and row['max_unsafe_merge_rate'] == primary_unsafe
    ]
    for row in primary_rows:
        row['recall_passed'] = bool(
            row['positive_source_recall'] >= minimum_recall)
    recall_values = [
        row['positive_source_recall'] for row in primary_rows]
    recall_std = sample_std(recall_values)
    locked_row = next(
        row for row in primary_rows
        if int(row['seed']) == int(locked_seed))
    num_passed = int(sum(row['recall_passed'] for row in primary_rows))
    gate = {
        'locked_seed_passed': bool(locked_row['recall_passed']),
        'enough_seed_passes': bool(num_passed >= int(minimum_pass_seeds)),
        'recall_stability_passed': bool(
            recall_std <= float(primary['max_recall_sample_std'])),
    }
    gate['passed'] = bool(all(gate.values()))
    top1_rates = {}
    for seed, records in records_by_seed.items():
        positive_sources = sum(
            row['has_positive_target'] for row in records)
        top1_rates[str(seed)] = float(
            sum(row['chosen_correct'] for row in records) /
            max(positive_sources, 1))
    return {
        'frontier_rows': frontier_rows,
        'sensitivity_rows': sensitivity_rows,
        'primary_rows': primary_rows,
        'primary_recall_sample_std': recall_std,
        'primary_num_passed_seeds': num_passed,
        'locked_seed_primary': locked_row,
        'top1_correct_rate_on_positive_sources': top1_rates,
        'gate': gate,
    }


def format_percent(value):
    return f'{100.0 * float(value):.3f}%'


def format_markdown(summary):
    lines = [
        '# E8c3 Groupwise 安全合并敏感性审计',
        '',
        '- 数据：仅固定 validation forests；未读取 Wytham。',
        '- 主决策条件：pair precision ≥ '
        f"{format_percent(summary['primary']['min_pair_precision'])}，"
        'unsafe merge rate ≤ '
        f"{format_percent(summary['primary']['max_unsafe_merge_rate'])}。",
        '- 90%/85% precision 结果只用于诊断，不得据此进入 pipeline。',
        '',
        '## Top-1 排序上限',
        '',
        '| Seed | Positive-source Top-1 correct rate |',
        '|---:|---:|',
    ]
    for seed, value in summary[
            'top1_correct_rate_on_positive_sources'].items():
        lines.append(f'| {seed} | {format_percent(value)} |')
    lines.extend([
        '',
        '## 主决策：95% precision + 1% unsafe rate',
        '',
        '| Seed | Threshold | Proposed | Correct | Unsafe | Precision | Recall | Unsafe rate | Pass |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ])
    for row in summary['primary_rows']:
        lines.append(
            f"| {row['seed']} | {row['threshold']:.6f} | "
            f"{row['num_proposed_merges']} | {row['num_correct_merges']} | "
            f"{row['num_unsafe_merges']} | "
            f"{format_percent(row['pair_precision'])} | "
            f"{format_percent(row['positive_source_recall'])} | "
            f"{format_percent(row['unsafe_merge_rate'])} | "
            f"{row['recall_passed']} |")
    lines.extend([
        '',
        f"- Recall sample std：{summary['primary_recall_sample_std']:.6f}",
        f"- Pass seeds：{summary['primary_num_passed_seeds']} / "
        f"{len(summary['seeds'])}",
        '',
        '## 完整敏感性摘要',
        '',
        '| Min precision | Max unsafe | Mean recall | Min recall | Mean proposed |',
        '|---:|---:|---:|---:|---:|',
    ])
    grouped = {}
    for row in summary['sensitivity_rows']:
        key = (row['min_pair_precision'], row['max_unsafe_merge_rate'])
        grouped.setdefault(key, []).append(row)
    for (precision, unsafe), rows in sorted(
            grouped.items(), reverse=True):
        recalls = [row['positive_source_recall'] for row in rows]
        proposed = [row['num_proposed_merges'] for row in rows]
        lines.append(
            f'| {format_percent(precision)} | {format_percent(unsafe)} | '
            f'{format_percent(np.mean(recalls))} | '
            f'{format_percent(np.min(recalls))} | '
            f'{np.mean(proposed):.2f} |')
    lines.extend([
        '',
        '## Gate',
        '',
    ])
    for name, passed in summary['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    lines.extend([''])
    if summary['gate']['passed']:
        lines.append('PASS：锁定 seed42 与主决策阈值，进入 E8d L1W 合并集成。')
    else:
        lines.append(
            'STOP：1% unsafe-rate 敏感性仍未达到 30% recall；关闭学习式合并路线。')
    return '\n'.join(lines) + '\n'


def run(config_path):
    settings = load_yaml(config_path)
    seeds = [int(seed) for seed in settings['seeds']]
    locked_seed = int(settings['selection']['locked_seed'])
    if locked_seed not in seeds:
        raise ValueError('The locked seed must be present in seeds.')
    records_by_seed = load_prediction_records(
        settings['predictions_path'], seeds)
    result = evaluate_sensitivity(
        records_by_seed,
        settings['diagnostic']['precision_levels'],
        settings['diagnostic']['unsafe_rate_levels'],
        settings['primary'],
        locked_seed,
        settings['selection']['min_gate_pass_seeds'],
    )
    summary = {
        'config': str(config_path),
        'predictions_path': settings['predictions_path'],
        'seeds': seeds,
        'locked_seed': locked_seed,
        'num_sources_per_seed': int(len(records_by_seed[locked_seed])),
        'primary': settings['primary'],
        'diagnostic': settings['diagnostic'],
        **{key: value for key, value in result.items()
           if key not in {'frontier_rows'}},
    }
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'frontier.csv', result['frontier_rows'])
    write_csv(output_dir / 'sensitivity.csv', result['sensitivity_rows'])
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8')
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(
        markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError(
            'E8c3 sensitivity gate failed; close the learned merge route.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Audit the E8c2 source-level safety/recall frontier.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
