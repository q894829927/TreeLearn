"""Apply the fixed T1/T2/T3 gates to validation-matrix records."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True)
    parser.add_argument('--stage', choices=('t1', 't2', 't3'), required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--winner')
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    numeric = (
        'completeness', 'commission_error_rate', 'f1_score',
        'precision', 'recall', 'coverage',
    )
    for row in rows:
        row['seed'] = int(row['seed'])
        for key in numeric:
            row[key] = float(row[key])
    return rows


def group(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['seed'])].append(row)
    return grouped


def macro(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def attention_variance_passed(rows):
    logs = sorted({
        row.get('training_log', '') for row in rows
        if row.get('training_log', '')
    })
    values = []
    for log in logs:
        path = Path(log)
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        values.extend(float(value) for value in re.findall(
            r'attention[^=,]*_std=([0-9.eE+-]+)', text))
    return bool(values) and all(math.isfinite(v) for v in values) and max(values) > 0


def t1(rows, baseline):
    grouped = group(rows)
    baseline_rows = grouped[(baseline, 42)]
    results = {}
    for (model, seed), model_rows in grouped.items():
        if model in (baseline, 'b0_official') or seed != 42:
            continue
        baseline_by_plot = {row['plot']: row for row in baseline_rows}
        deltas = [
            row['f1_score'] - baseline_by_plot[row['plot']]['f1_score']
            for row in model_rows
        ]
        result = {
            'macro_f1': macro(model_rows, 'f1_score'),
            'f1_gain_pp': (
                macro(model_rows, 'f1_score') -
                macro(baseline_rows, 'f1_score')),
            'completeness_drop_pp': (
                macro(baseline_rows, 'completeness') -
                macro(model_rows, 'completeness')),
            'commission_increase_pp': (
                macro(model_rows, 'commission_error_rate') -
                macro(baseline_rows, 'commission_error_rate')),
            'nonnegative_plots': sum(delta >= 0 for delta in deltas),
            'worst_plot_f1_gain_pp': min(deltas),
            'attention_variance_passed': attention_variance_passed(model_rows),
        }
        result['passed'] = bool(
            result['f1_gain_pp'] >= 0.2 and
            result['completeness_drop_pp'] <= 0.5 and
            result['commission_increase_pp'] <= 0.5 and
            result['nonnegative_plots'] >= 3 and
            result['worst_plot_f1_gain_pp'] >= -1.0 and
            result['attention_variance_passed'])
        results[model] = result
    finalists = sorted(
        (model for model, result in results.items() if result['passed']),
        key=lambda model: results[model]['macro_f1'],
        reverse=True)[:2]
    return {'models': results, 'finalists': finalists,
            'passed': bool(finalists)}


def t2(rows, baseline):
    grouped = group(rows)
    models = sorted({row['model'] for row in rows if row['model'] != baseline})
    results = {}
    for model in models:
        seed_results = []
        forest_deltas = defaultdict(list)
        for seed in (42, 43, 44):
            base_rows = grouped.get((baseline, seed), [])
            model_rows = grouped.get((model, seed), [])
            if len(base_rows) != 5 or len(model_rows) != 5:
                continue
            base_by_plot = {row['plot']: row for row in base_rows}
            for row in model_rows:
                forest_deltas[row['plot']].append(
                    row['f1_score'] -
                    base_by_plot[row['plot']]['f1_score'])
            seed_results.append({
                'seed': seed,
                'f1_gain_pp': (
                    macro(model_rows, 'f1_score') -
                    macro(base_rows, 'f1_score')),
                'f1': macro(model_rows, 'f1_score'),
                'completeness_drop_pp': (
                    macro(base_rows, 'completeness') -
                    macro(model_rows, 'completeness')),
                'commission_increase_pp': (
                    macro(model_rows, 'commission_error_rate') -
                    macro(base_rows, 'commission_error_rate')),
            })
        if len(seed_results) != 3:
            results[model] = {'passed': False, 'reason': 'missing seeds'}
            continue
        f1_values = [item['f1'] for item in seed_results]
        mean_f1 = sum(f1_values) / 3
        std_f1 = (
            sum((value - mean_f1) ** 2 for value in f1_values) / 2
        ) ** 0.5
        result = {
            'seed_results': seed_results,
            'mean_f1_gain_pp': sum(
                item['f1_gain_pp'] for item in seed_results) / 3,
            'seed_wins': sum(
                item['f1_gain_pp'] > 0 for item in seed_results),
            'forest_wins': sum(
                sum(values) / len(values) > 0
                for values in forest_deltas.values()),
            'mean_completeness_drop_pp': sum(
                item['completeness_drop_pp']
                for item in seed_results) / 3,
            'mean_commission_increase_pp': sum(
                item['commission_increase_pp']
                for item in seed_results) / 3,
            'f1_seed_std_pp': std_f1,
            'mean_macro_f1': mean_f1,
        }
        result['passed'] = bool(
            result['mean_f1_gain_pp'] >= 0.5 and
            result['seed_wins'] >= 2 and
            result['forest_wins'] >= 3 and
            result['mean_completeness_drop_pp'] <= 0.5 and
            result['mean_commission_increase_pp'] <= 0.0 and
            result['f1_seed_std_pp'] <= 0.5)
        results[model] = result
    eligible = [
        model for model, result in results.items() if result['passed']]
    eligible.sort(
        key=lambda model: (
            results[model]['mean_macro_f1'],
            -results[model]['mean_completeness_drop_pp'],
            -results[model]['mean_commission_increase_pp']),
        reverse=True)
    return {
        'models': results,
        'winner': eligible[0] if eligible else None,
        'passed': bool(eligible),
    }


def t3(rows, baseline, winner):
    if not winner:
        raise ValueError('--winner is required for T3.')
    grouped = group(rows)
    seed_results = []
    parameter_differences = []
    for seed in (42, 43, 44):
        base_rows = grouped.get((baseline, seed), [])
        winner_rows = grouped.get((winner, seed), [])
        if len(base_rows) != 5 or len(winner_rows) != 5:
            continue
        seed_results.append({
            'seed': seed,
            'f1_gain_pp': (
                macro(winner_rows, 'f1_score') -
                macro(base_rows, 'f1_score')),
        })
        try:
            base_params = float(base_rows[0]['parameter_count'])
            winner_params = float(winner_rows[0]['parameter_count'])
            parameter_differences.append(
                abs(base_params - winner_params) / max(winner_params, 1))
        except (KeyError, ValueError):
            pass
    mean_gain = (
        sum(item['f1_gain_pp'] for item in seed_results) /
        max(len(seed_results), 1))
    result = {
        'seed_results': seed_results,
        'mean_f1_gain_pp': mean_gain,
        'seed_wins': sum(item['f1_gain_pp'] > 0 for item in seed_results),
        'max_parameter_difference': (
            max(parameter_differences) if parameter_differences else None),
    }
    result['passed'] = bool(
        len(seed_results) == 3 and mean_gain >= 0.3 and
        result['seed_wins'] >= 2 and parameter_differences and
        result['max_parameter_difference'] <= 0.05)
    return result


def markdown(report, stage):
    lines = [f'# Sparse U-Net tournament {stage.upper()}', '']
    lines.extend(
        '    ' + line for line in
        json.dumps(report, indent=2, ensure_ascii=False).splitlines())
    lines += ['', f'- Gate: **{report["passed"]}**', '']
    return '\n'.join(lines)


def main():
    args = parse_args()
    rows = read_rows(args.records)
    if args.stage == 't1':
        report = t1(rows, args.baseline)
    elif args.stage == 't2':
        report = t2(rows, args.baseline)
    else:
        report = t3(rows, args.baseline, args.winner)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    (output / 'summary.md').write_text(
        markdown(report, args.stage), encoding='utf-8')
    print(markdown(report, args.stage))
    if not report['passed']:
        raise RuntimeError(
            f'{args.stage.upper()} gate failed; stop the tournament route.')


if __name__ == '__main__':
    main()
