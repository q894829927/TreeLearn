import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


E9_PATH = Path(__file__).resolve().parent / 'evaluate_selective_instance_quality.py'
E9_SPEC = importlib.util.spec_from_file_location(
    'selective_instance_quality_e9', E9_PATH)
E9 = importlib.util.module_from_spec(E9_SPEC)
E9_SPEC.loader.exec_module(E9)


def load_yaml(path):
    import yaml

    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {'true', '1', 'yes'}:
        return True
    if normalized in {'false', '0', 'no'}:
        return False
    raise ValueError(f'Invalid boolean value: {value!r}')


def load_validation_predictions(path, expected_model, expected_seeds):
    records = {}
    with open(path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        required = {
            'model', 'seed', 'source_plot', 'instance_id', 'score',
            'target_true', 'classification_valid',
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f'Validation prediction CSV misses: {sorted(missing)}')
        for row in reader:
            if str(row['model']) != str(expected_model):
                continue
            seed = int(row['seed'])
            if seed not in expected_seeds:
                continue
            key = (seed, str(row['source_plot']), int(row['instance_id']))
            if key in records:
                raise ValueError(f'Duplicate validation prediction: {key}')
            score = float(row['score'])
            if not math.isfinite(score):
                raise ValueError(f'Non-finite validation score: {key}')
            records[key] = {
                'seed': seed,
                'source_plot': key[1],
                'instance_id': key[2],
                'score': score,
                'target_true': parse_bool(row['target_true']),
                'classification_valid': parse_bool(
                    row['classification_valid']),
            }
    observed_seeds = {key[0] for key in records}
    if observed_seeds != set(expected_seeds):
        raise ValueError(
            f'Expected seeds {expected_seeds}, found {sorted(observed_seeds)}')
    return records


def align_validation_predictions(global_records, vertical_records):
    if set(global_records) != set(vertical_records):
        missing = sorted(set(global_records) - set(vertical_records))
        extra = sorted(set(vertical_records) - set(global_records))
        raise ValueError(
            'Global and vertical prediction keys do not align: '
            f'missing={missing[:5]}, extra={extra[:5]}')
    for key in global_records:
        global_row = global_records[key]
        vertical_row = vertical_records[key]
        labels = ('target_true', 'classification_valid')
        if any(global_row[name] != vertical_row[name] for name in labels):
            raise ValueError(f'Validation targets differ at {key}.')
    return sorted(global_records)


def build_group_curves(records, ratios):
    groups = {}
    for row in records.values():
        key = (int(row['seed']), str(row['source_plot']))
        groups.setdefault(key, []).append(row)
    curves = {}
    for key, rows in groups.items():
        instance_ids = np.asarray([
            row['instance_id'] for row in rows], dtype=np.int64)
        scores = np.asarray([row['score'] for row in rows], dtype=np.float64)
        status = np.asarray([
            1 if row['classification_valid'] and row['target_true'] else
            -1 if row['classification_valid'] else 0
            for row in rows
        ], dtype=np.int8)
        positives = int(np.count_nonzero(status == 1))
        if positives == 0:
            raise ValueError(f'Validation group {key} has no positives.')
        order = np.lexsort((instance_ids, -scores))
        curves[key] = E9.curve_for_order(
            status, order, ratios, positives)
    return curves


def aggregate_group_curves(curves, group_keys, ratios):
    rows = []
    for index, ratio in enumerate(ratios):
        selected = [curves[key][index] for key in group_keys]
        tp = sum(row['tp'] for row in selected)
        fp = sum(row['fp'] for row in selected)
        num_gt = sum(row['tp'] + row['fn'] for row in selected)
        retained = sum(row['retained_predictions'] for row in selected)
        total = sum(row['total_predictions'] for row in selected)
        metrics = E9.metrics_from_counts(
            tp, fp, num_gt, retained, total)
        metrics['retention_ratio_requested'] = float(ratio)
        rows.append(metrics)
    return rows


def curve_areas(rows):
    return {
        'commission_area': E9.normalized_area(rows, 'commission'),
        'f1_area': E9.normalized_area(rows, 'f1'),
    }


def percentile_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        'mean': float(values.mean()),
        'lower': float(lower),
        'upper': float(upper),
    }


def validation_bootstrap(
        global_curves, vertical_curves, ratios, seeds, plots,
        repeats, random_seed):
    expected_keys = {(int(seed), str(plot)) for seed in seeds for plot in plots}
    if set(global_curves) != expected_keys or set(vertical_curves) != expected_keys:
        raise ValueError('Validation curves do not form the seed/plot grid.')
    all_keys = sorted(expected_keys)
    global_point = curve_areas(aggregate_group_curves(
        global_curves, all_keys, ratios))
    vertical_point = curve_areas(aggregate_group_curves(
        vertical_curves, all_keys, ratios))

    seed_rows = []
    for seed in seeds:
        keys = [(int(seed), str(plot)) for plot in plots]
        global_seed = curve_areas(aggregate_group_curves(
            global_curves, keys, ratios))
        vertical_seed = curve_areas(aggregate_group_curves(
            vertical_curves, keys, ratios))
        seed_rows.append({
            'seed': int(seed),
            'commission_area_reduction_pp': float(100.0 * (
                global_seed['commission_area'] -
                vertical_seed['commission_area'])),
            'f1_area_gain_pp': float(100.0 * (
                vertical_seed['f1_area'] - global_seed['f1_area'])),
        })

    rng = np.random.default_rng(int(random_seed))
    commission_differences = []
    f1_differences = []
    bootstrap_rows = []
    for replicate in range(int(repeats)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_plots = rng.choice(plots, size=len(plots), replace=True)
        sampled_keys = [
            (int(seed), str(plot))
            for seed in sampled_seeds for plot in sampled_plots
        ]
        global_area = curve_areas(aggregate_group_curves(
            global_curves, sampled_keys, ratios))
        vertical_area = curve_areas(aggregate_group_curves(
            vertical_curves, sampled_keys, ratios))
        commission_difference = float(100.0 * (
            global_area['commission_area'] -
            vertical_area['commission_area']))
        f1_difference = float(100.0 * (
            vertical_area['f1_area'] - global_area['f1_area']))
        commission_differences.append(commission_difference)
        f1_differences.append(f1_difference)
        bootstrap_rows.append({
            'replicate': int(replicate),
            'commission_area_reduction_pp': commission_difference,
            'f1_area_gain_pp': f1_difference,
        })
    return {
        'global_point': global_point,
        'vertical_point': vertical_point,
        'point_delta': {
            'commission_area_reduction_pp': float(100.0 * (
                global_point['commission_area'] -
                vertical_point['commission_area'])),
            'f1_area_gain_pp': float(100.0 * (
                vertical_point['f1_area'] - global_point['f1_area'])),
        },
        'commission_delta_ci': percentile_interval(
            commission_differences),
        'f1_delta_ci': percentile_interval(f1_differences),
        'seed_rows': seed_rows,
        'bootstrap_rows': bootstrap_rows,
    }


def external_randomization(
        status_data, ratios, repeats, random_seed):
    score_rows = E9.score_curve(
        status_data['instance_ids'], status_data['scores'],
        status_data['status'], ratios, status_data['num_gt'])
    observed = curve_areas(score_rows)
    rng = np.random.default_rng(int(random_seed))
    permutation_rows = []
    commission_null = []
    f1_null = []
    for replicate in range(int(repeats)):
        rows = E9.curve_for_order(
            status_data['status'],
            rng.permutation(len(status_data['status'])),
            ratios, status_data['num_gt'])
        areas = curve_areas(rows)
        commission_null.append(areas['commission_area'])
        f1_null.append(areas['f1_area'])
        permutation_rows.append({
            'replicate': int(replicate),
            'commission_area': float(areas['commission_area']),
            'f1_area': float(areas['f1_area']),
        })
    commission_null = np.asarray(commission_null)
    f1_null = np.asarray(f1_null)
    baseline = score_rows[-1]
    baseline_counts_reproduced = bool(
        baseline['tp'] == status_data['baseline_tp'] and
        baseline['fp'] == status_data['baseline_fp'] and
        baseline['fn'] == status_data['num_gt'] -
        status_data['baseline_tp'])
    return {
        'observed': observed,
        'baseline_counts_reproduced': baseline_counts_reproduced,
        'commission_null_ci': percentile_interval(commission_null),
        'f1_null_ci': percentile_interval(f1_null),
        'commission_one_sided_p': float(
            (1 + np.count_nonzero(
                commission_null <= observed['commission_area'])) /
            (len(commission_null) + 1)),
        'f1_one_sided_p': float(
            (1 + np.count_nonzero(f1_null >= observed['f1_area'])) /
            (len(f1_null) + 1)),
        'permutation_rows': permutation_rows,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write empty CSV output.')
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_validation_markdown(report):
    delta = report['point_delta']
    commission_ci = report['commission_delta_ci']
    f1_ci = report['f1_delta_ci']
    lines = [
        '# E10 Validation 配对消融：Vertical-MLP vs Global-MLP',
        '',
        f"- Bootstrap：{report['bootstrap_repeats']} 次；同时重采样森林和训练 seed。",
        '- 所有重采样均为配对重采样，两个模型使用相同森林和 seed。',
        '',
        '| Metric | Global-MLP | Vertical-MLP | Paired delta | 95% CI |',
        '|---|---:|---:|---:|---:|',
        '| Commission area | '
        f"{100 * report['global_point']['commission_area']:.3f}% | "
        f"{100 * report['vertical_point']['commission_area']:.3f}% | "
        f"{delta['commission_area_reduction_pp']:+.3f} pp reduction | "
        f"[{commission_ci['lower']:+.3f}, {commission_ci['upper']:+.3f}] |",
        '| Detection F1 area | '
        f"{100 * report['global_point']['f1_area']:.3f}% | "
        f"{100 * report['vertical_point']['f1_area']:.3f}% | "
        f"{delta['f1_area_gain_pp']:+.3f} pp | "
        f"[{f1_ci['lower']:+.3f}, {f1_ci['upper']:+.3f}] |",
        '',
        '## 每个训练 seed 的配对差值',
        '',
        '| Seed | Commission area reduction | F1 area gain |',
        '|---:|---:|---:|',
    ]
    for row in report['seed_rows']:
        lines.append(
            f"| {row['seed']} | "
            f"{row['commission_area_reduction_pp']:+.3f} pp | "
            f"{row['f1_area_gain_pp']:+.3f} pp |")
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    return '\n'.join(lines) + '\n'


def format_external_markdown(report):
    observed = report['observed']
    commission_ci = report['commission_null_ci']
    f1_ci = report['f1_null_ci']
    lines = [
        '# E10 Wytham 锁定随机排序置换检验',
        '',
        '- Wytham 不参与模型、比例网格或 Gate 参数选择。',
        f"- 随机排序置换：{report['permutation_repeats']} 次。",
        '',
        '| Metric | Vertical-MLP observed | Random null mean | Random 95% interval | One-sided p |',
        '|---|---:|---:|---:|---:|',
        '| Commission area（越低越好） | '
        f"{100 * observed['commission_area']:.3f}% | "
        f"{100 * commission_ci['mean']:.3f}% | "
        f"[{100 * commission_ci['lower']:.3f}%, "
        f"{100 * commission_ci['upper']:.3f}%] | "
        f"{report['commission_one_sided_p']:.6f} |",
        '| Detection F1 area（越高越好） | '
        f"{100 * observed['f1_area']:.3f}% | "
        f"{100 * f1_ci['mean']:.3f}% | "
        f"[{100 * f1_ci['lower']:.3f}%, "
        f"{100 * f1_ci['upper']:.3f}%] | "
        f"{report['f1_one_sided_p']:.6f} |",
        '',
        '## Gate',
        '',
    ]
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    return '\n'.join(lines) + '\n'


def run(config_path):
    settings = load_yaml(config_path)
    seeds = [int(seed) for seed in settings['validation']['seeds']]
    global_records = load_validation_predictions(
        settings['validation']['global_predictions'],
        settings['validation']['global_model'], seeds)
    vertical_records = load_validation_predictions(
        settings['validation']['vertical_predictions'],
        settings['validation']['vertical_model'], seeds)
    keys = align_validation_predictions(global_records, vertical_records)
    plots = sorted({key[1] for key in keys})
    expected_plots = sorted(str(value) for value in settings[
        'validation']['plots'])
    if plots != expected_plots:
        raise ValueError(
            f'Validation plots differ: expected={expected_plots}, got={plots}')

    grid = settings['ratio_grid']
    ratios = np.round(np.arange(
        float(grid['start']),
        float(grid['stop']) + 0.5 * float(grid['step']),
        float(grid['step'])), 10)
    global_curves = build_group_curves(global_records, ratios)
    vertical_curves = build_group_curves(vertical_records, ratios)
    validation = validation_bootstrap(
        global_curves, vertical_curves, ratios, seeds, plots,
        settings['bootstrap']['repeats'], settings['bootstrap']['seed'])
    validation['bootstrap_repeats'] = int(
        settings['bootstrap']['repeats'])
    validation['num_validation_instances'] = int(len(keys))
    validation['seeds'] = seeds
    validation['plots'] = plots
    validation['gate'] = {
        'commission_mean_improved': bool(
            validation['point_delta']['commission_area_reduction_pp'] > 0),
        'f1_mean_improved': bool(
            validation['point_delta']['f1_area_gain_pp'] > 0),
        'commission_seed_wins': bool(sum(
            row['commission_area_reduction_pp'] > 0
            for row in validation['seed_rows']) >=
            int(settings['gate']['min_validation_seed_wins'])),
        'f1_seed_wins': bool(sum(
            row['f1_area_gain_pp'] > 0
            for row in validation['seed_rows']) >=
            int(settings['gate']['min_validation_seed_wins'])),
    }
    validation['gate']['passed'] = bool(all(validation['gate'].values()))

    external_spec = settings['external']
    evaluation = E9.load_evaluation(external_spec['evaluation'])
    quality_ids, quality_scores = E9.load_quality_scores(
        external_spec['quality_scores'])
    status_data = E9.build_detection_status(
        evaluation, quality_ids, quality_scores)
    _, metadata_checks = E9.validate_score_only_metadata(
        external_spec['quality_metadata'], len(quality_ids))
    external = external_randomization(
        status_data, ratios,
        settings['permutation']['repeats'],
        settings['permutation']['seed'])
    external['permutation_repeats'] = int(
        settings['permutation']['repeats'])
    external['metadata_checks'] = metadata_checks
    maximum_p = float(settings['gate']['max_external_one_sided_p'])
    external['gate'] = {
        'score_only_metadata_passed': bool(all(metadata_checks.values())),
        'baseline_counts_reproduced': bool(
            external['baseline_counts_reproduced']),
        'commission_permutation_passed': bool(
            external['commission_one_sided_p'] <= maximum_p),
        'f1_permutation_passed': bool(
            external['f1_one_sided_p'] <= maximum_p),
    }
    external['gate']['passed'] = bool(all(external['gate'].values()))

    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / 'validation_bootstrap.csv',
        validation.pop('bootstrap_rows'))
    write_csv(
        output_dir / 'external_permutations.csv',
        external.pop('permutation_rows'))
    validation_markdown = format_validation_markdown(validation)
    external_markdown = format_external_markdown(external)
    (output_dir / 'validation_ablation.md').write_text(
        validation_markdown, encoding='utf-8')
    (output_dir / 'wytham_permutation.md').write_text(
        external_markdown, encoding='utf-8')
    print(validation_markdown, flush=True)
    print(external_markdown, flush=True)

    summary = {
        'config': str(config_path),
        'validation': validation,
        'external': external,
        'gate': {
            'validation_ablation_passed': bool(
                validation['gate']['passed']),
            'external_permutation_passed': bool(
                external['gate']['passed']),
        },
    }
    summary['gate']['passed'] = bool(all(summary['gate'].values()))
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8')
    summary_lines = [
        '# E10 统计证据汇总', '',
        f"- Validation Vertical-vs-Global Commission-area reduction: "
        f"{validation['point_delta']['commission_area_reduction_pp']:+.3f} pp",
        f"- Validation Vertical-vs-Global F1-area gain: "
        f"{validation['point_delta']['f1_area_gain_pp']:+.3f} pp",
        f"- Wytham Commission permutation p: "
        f"{external['commission_one_sided_p']:.6f}",
        f"- Wytham F1 permutation p: {external['f1_one_sided_p']:.6f}",
        '', f"- Gate：**{summary['gate']['passed']}**", '',
    ]
    if summary['gate']['passed']:
        summary_lines.append('PASS：可以整理论文主表、消融表和统计显著性描述。')
    else:
        summary_lines.append('STOP：不得声称垂直结构优于全局特征或具有跨域显著性。')
    summary_markdown = '\n'.join(summary_lines) + '\n'
    (output_dir / 'summary.md').write_text(
        summary_markdown, encoding='utf-8')
    print(summary_markdown, flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError('E10 statistical evidence gate failed.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate E10 quality-score statistical evidence.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
