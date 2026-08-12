"""Generate and aggregate the fixed-validation Q5a seed-completion Oracle."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tree_learn.util.omission_diagnostics import detection_metrics
from tree_learn.util.seed_completion_diagnostics import SEED_COMPLETION_MODES


def _read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def locate_forest(source_root, plot_name):
    root = Path(source_root)
    matches = sorted(
        path for path in root.glob(f'{plot_name}.*')
        if path.is_file() and path.suffix.lower() in {
            '.las', '.laz', '.npy', '.npz', '.txt'} and
        path.stem == plot_name)
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected one forest for {plot_name}, found {matches}.')
    return matches[0]


def artifact_path(output_root, plot_name):
    return Path(output_root) / 'validation' / plot_name / 'summary.json'


def q3_target_tree_ids(q3_root, plot_name):
    path = Path(q3_root) / 'validation' / plot_name / 'gt_trees.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing fixed Q3 artifact: {path}')
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows or {'gt_tree_id', 'category'} - set(rows[0]):
        raise ValueError(f'Invalid fixed Q3 artifact: {path}')
    return sorted(
        int(row['gt_tree_id']) for row in rows
        if row['category'] == 'base_seed_support_failure')


def validate_artifact(path, plot_name, q3_root):
    report = _read_json(path)
    required = {
        'baseline', 'num_target_trees', 'expected_target_tree_ids',
        'num_baseline_seeds', 'per_tree_topup', 'modes',
        'source_plot', 'split'}
    if required - set(report):
        raise ValueError(f'Incomplete Q5a artifact for {plot_name}.')
    if report['source_plot'] != plot_name or report['split'] != 'validation':
        raise ValueError('Q5a source plot or split is invalid.')
    if set(report['modes']) != set(SEED_COMPLETION_MODES):
        raise ValueError('Q5a modes differ from preregistration.')
    q3_summary = _read_json(
        Path(q3_root) / 'validation' / plot_name / 'summary.json')
    expected_baseline = {
        key: int(q3_summary['baseline'][key])
        for key in ('tp', 'fp', 'fn')}
    observed_baseline = {
        key: int(report['baseline'][key]) for key in ('tp', 'fp', 'fn')}
    if observed_baseline != expected_baseline:
        raise ValueError(
            f'Q5a baseline differs from Q3 for {plot_name}.')
    expected_ids = q3_target_tree_ids(q3_root, plot_name)
    if sorted(map(int, report['expected_target_tree_ids'])) != expected_ids:
        raise ValueError(f'Q5a targets differ from Q3 for {plot_name}.')
    if int(report['num_target_trees']) != len(expected_ids) or int(
            q3_summary['category_counts'][
                'base_seed_support_failure']) != len(expected_ids):
        raise ValueError(f'Q5a target count differs for {plot_name}.')
    if len(report['per_tree_topup']) != len(expected_ids):
        raise ValueError(f'Q5a per-tree top-up rows differ for {plot_name}.')
    topup_ids = sorted(
        int(row['gt_tree_id']) for row in report['per_tree_topup'])
    if topup_ids != expected_ids:
        raise ValueError(f'Q5a per-tree top-up IDs differ for {plot_name}.')
    for row in report['per_tree_topup']:
        baseline_count = int(row['baseline_seed_count'])
        added_count = int(row['added_seed_count'])
        if not (baseline_count < 50 and baseline_count + added_count == 50):
            raise ValueError(
                f'Q5a did not top up tree {row["gt_tree_id"]} exactly '
                f'to 50 seeds for {plot_name}.')
    return report


def _safe_remove(path, root):
    target = Path(path).resolve()
    allowed = Path(root).resolve()
    if target == allowed or allowed not in target.parents:
        raise ValueError(f'Refusing unsafe cleanup: {target}')
    if target.exists():
        shutil.rmtree(target)


def write_resolved_config(settings, plot_name, forest_path, destination):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(settings['pipeline_template']))
    runtime_dir = Path(settings['output_root']) / 'runtime' / plot_name
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(runtime_dir)
    config.seed_completion_output_dir = str(destination)
    config.pretrain = str(settings['checkpoint'])
    diagnostic = dict(settings['seed_completion'])
    diagnostic['expected_target_tree_ids'] = q3_target_tree_ids(
        settings['q3_reference_root'], plot_name)
    config.seed_completion = diagnostic
    config_path = (
        Path(settings['output_root']) / 'runtime_configs' /
        f'{plot_name}.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            munch_to_dict(config), file, sort_keys=False,
            allow_unicode=True)
    return config_path, runtime_dir


def run_plot(settings, plot_name, force=False):
    if 'wytham' in plot_name.lower():
        raise ValueError('Q5a must not read Wytham.')
    path = artifact_path(settings['output_root'], plot_name)
    destination = path.parent
    if path.is_file() and not force:
        try:
            validate_artifact(
                path, plot_name, settings['q3_reference_root'])
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(
                f'STALE {plot_name}: {error} Rebuilding artifact.',
                flush=True)
            _safe_remove(destination, settings['output_root'])
        else:
            print(
                f'SKIP {plot_name}: validated existing Q5a artifact.',
                flush=True)
            return
    elif destination.exists() and not force:
        _safe_remove(destination, settings['output_root'])
    if force:
        _safe_remove(destination, settings['output_root'])
    forest = locate_forest(settings['source_root'], plot_name)
    config_path, runtime_dir = write_resolved_config(
        settings, plot_name, forest, destination)
    log_path = Path(settings['output_root']) / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-u',
        'tools/diagnostics/run_seed_completion_oracle_pipeline.py',
        '--config', str(config_path)]
    print(f'RUN {plot_name}; log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q5a pipeline failed for {plot_name}; inspect {log_path}.')
    validate_artifact(path, plot_name, settings['q3_reference_root'])
    _safe_remove(runtime_dir, Path(settings['output_root']) / 'runtime')
    print(f'DONE {plot_name}', flush=True)


def aggregate_detection(reports, mode=None):
    rows = [
        report['baseline'] if mode is None else report['modes'][mode]
        for report in reports]
    return detection_metrics(
        sum(int(row['tp']) for row in rows),
        sum(int(row['fp']) for row in rows),
        sum(int(row['fn']) for row in rows))


def aggregate_mode(reports, mode, baseline):
    metrics = aggregate_detection(reports, mode)
    plot_rows = []
    for report in reports:
        row = report['modes'][mode]
        base = report['baseline']
        plot_rows.append({
            'source_plot': report['source_plot'],
            'recovered_target_trees': int(row['recovered_target_trees']),
            'lost_baseline_trees': int(row['lost_baseline_trees']),
            'f1_gain_pp': 100.0 * (row['f1'] - base['f1']),
            'commission_increase_pp': 100.0 * (
                row['commission'] - base['commission']),
            'completeness_drop_pp': 100.0 * (
                base['completeness'] - row['completeness']),
        })
    metrics.update({
        'recovered_target_trees': int(sum(
            report['modes'][mode]['recovered_target_trees']
            for report in reports)),
        'lost_baseline_trees': int(sum(
            report['modes'][mode]['lost_baseline_trees']
            for report in reports)),
        'num_added_seed_points': int(sum(
            report['modes'][mode]['num_added_seed_points']
            for report in reports)),
        'f1_gain_pp': 100.0 * (metrics['f1'] - baseline['f1']),
        'commission_increase_pp': 100.0 * (
            metrics['commission'] - baseline['commission']),
        'completeness_drop_pp': 100.0 * (
            baseline['completeness'] - metrics['completeness']),
        'nonnegative_plots': int(sum(
            row['f1_gain_pp'] >= -1e-9 for row in plot_rows)),
        'plot_metrics': plot_rows,
    })
    return metrics


def mode_gate(metrics, rule):
    gate = {
        'recovered_tree_count_passed': (
            metrics['recovered_target_trees'] >=
            int(rule['min_recovered_target_trees'])),
        'f1_gain_passed': (
            metrics['f1_gain_pp'] >= float(rule['min_f1_gain_pp'])),
        'lost_tree_count_passed': (
            metrics['lost_baseline_trees'] <=
            int(rule['max_lost_baseline_trees'])),
        'commission_preserved': (
            metrics['commission_increase_pp'] <=
            float(rule['max_commission_increase_pp'])),
        'plot_consistency_passed': (
            metrics['nonnegative_plots'] >=
            int(rule['min_nonnegative_plots'])),
    }
    gate['passed'] = all(gate.values())
    return gate


def summarize(settings, reports):
    baseline = aggregate_detection(reports)
    modes = {
        mode: aggregate_mode(reports, mode, baseline)
        for mode in SEED_COMPLETION_MODES}
    gate = {
        'expected_validation_plots': (
            len(reports) == int(
                settings['gate']['expected_validation_plots'])),
        'baseline_reproduced': True,
        'target_count_reproduced': (
            sum(report['num_target_trees'] for report in reports) ==
            int(settings['gate']['expected_target_trees'])),
    }
    mode_gates = {
        mode: mode_gate(modes[mode], settings['gate'][mode])
        for mode in SEED_COMPLETION_MODES}
    gate['xy_oracle_ceiling_passed'] = mode_gates[
        'xy_oracle_topup']['passed']
    gate['passed'] = all(gate.values())
    recommendation = (
        'coverage_seed_completion_head'
        if mode_gates['margin_topup']['passed'] else
        ('seed_candidate_ranking_redesign'
         if mode_gates['xy_oracle_topup']['passed'] else
         'close_seed_completion_route'))
    result = {
        'baseline': baseline,
        'num_target_trees': int(sum(
            report['num_target_trees'] for report in reports)),
        'modes': modes,
        'mode_gates': mode_gates,
        'gate': gate,
        'recommendation': recommendation,
    }
    output = Path(settings['output_root'])
    (output / 'summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# Q5a Coverage-Seed Completion Oracle', '',
        '- 数据：固定五个 validation forests；未读取 Wytham。',
        '- GT 仅选择 Q3 base-seed support failure 目标树。',
        '- 两个模式都只补足到 tau_min=50，不修改 vote 或 HDBSCAN 参数。',
        f'- 目标树：{result["num_target_trees"]}', '',
        '| Mode | Recovered | Lost | Added seeds | Completeness | Commission | F1 | F1 gain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode in SEED_COMPLETION_MODES:
        row = modes[mode]
        lines.append(
            f'| {mode} | {row["recovered_target_trees"]} | '
            f'{row["lost_baseline_trees"]} | '
            f'{row["num_added_seed_points"]:,} | '
            f'{100 * row["completeness"]:.3f}% | '
            f'{100 * row["commission"]:.3f}% | '
            f'{100 * row["f1"]:.3f}% | '
            f'{row["f1_gain_pp"]:+.3f} pp |')
    lines.extend(['', '## 每森林', ''])
    for mode in SEED_COMPLETION_MODES:
        lines.extend([
            f'### {mode}', '',
            '| Plot | Recovered | Lost | F1 gain | Commission increase |',
            '|---|---:|---:|---:|---:|'])
        for row in modes[mode]['plot_metrics']:
            lines.append(
                f'| {row["source_plot"]} | '
                f'{row["recovered_target_trees"]} | '
                f'{row["lost_baseline_trees"]} | '
                f'{row["f1_gain_pp"]:+.3f} pp | '
                f'{row["commission_increase_pp"]:+.3f} pp |')
        lines.append('')
    for mode in SEED_COMPLETION_MODES:
        lines.extend([f'## {mode} Gate', ''])
        lines.extend(
            f'- {key}: **{value}**'
            for key, value in mode_gates[mode].items())
        lines.append('')
    lines.extend([
        '## 主 Gate', '',
        *(f'- {key}: **{value}**' for key, value in gate.items()),
        '', f'- 推荐路线：**{recommendation}**', '',
        ('PASS：按推荐路线进入 Q5b；仍不得使用 Wytham。'
         if gate['passed'] else
         'STOP：Seed-Completion 上限不足，关闭该路线。')])
    (output / 'summary.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result


def load_settings(config_path):
    import yaml
    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    if 'wytham' in json.dumps(settings).lower():
        raise ValueError('Q5a must not reference Wytham.')
    if len(settings['validation_plots']) != len(set(
            settings['validation_plots'])):
        raise ValueError('Q5a validation plots contain duplicates.')
    return settings


def run(config_path, force=False):
    settings = load_settings(config_path)
    for index, plot in enumerate(settings['validation_plots'], start=1):
        print(f'[{index}/{len(settings["validation_plots"])}] {plot}',
              flush=True)
        run_plot(settings, plot, force=force)
    reports = [
        validate_artifact(
            artifact_path(settings['output_root'], plot), plot,
            settings['q3_reference_root'])
        for plot in settings['validation_plots']]
    result = summarize(settings, reports)
    if not result['gate']['passed']:
        raise RuntimeError(
            'Q5a seed-completion Oracle gate failed; close the route.')


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed-validation Q5a seed-completion Oracle.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
