"""Run and aggregate the fixed-validation Q5b0 proposal Oracle."""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tree_learn.util.omission_diagnostics import detection_metrics


Q5A_PATH = Path(__file__).with_name('diagnose_seed_completion_oracle.py')
Q5A_SPEC = importlib.util.spec_from_file_location(
    'q5a_helpers_for_q5b0', Q5A_PATH)
Q5A = importlib.util.module_from_spec(Q5A_SPEC)
Q5A_SPEC.loader.exec_module(Q5A)


def _read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def artifact_path(output_root, plot_name):
    return Path(output_root) / 'validation' / plot_name / 'summary.json'


def validate_artifact(path, plot_name, settings):
    report = _read_json(path)
    required = {
        'baseline', 'num_target_trees', 'num_proposals',
        'num_activated_proposals', 'num_baseline_seeds',
        'num_oracle_seeds', 'proposal_covered_target_trees',
        'per_target', 'oracle_metrics', 'proposal_parameters',
        'source_plot', 'split'}
    if required - set(report):
        raise ValueError(f'Incomplete Q5b0 artifact for {plot_name}.')
    if report['source_plot'] != plot_name or report['split'] != 'validation':
        raise ValueError('Q5b0 source plot or split is invalid.')
    proposal_csv = Path(path).with_name('proposals.csv')
    if not proposal_csv.is_file():
        raise FileNotFoundError(f'Missing Q5b0 proposal table: {proposal_csv}')
    with proposal_csv.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if len(rows) != int(report['num_proposals']):
        raise ValueError(f'Q5b0 proposal count differs for {plot_name}.')

    q3_summary = _read_json(
        Path(settings['q3_reference_root']) / 'validation' /
        plot_name / 'summary.json')
    expected_baseline = {
        key: int(q3_summary['baseline'][key])
        for key in ('tp', 'fp', 'fn')}
    observed_baseline = {
        key: int(report['baseline'][key]) for key in ('tp', 'fp', 'fn')}
    if observed_baseline != expected_baseline:
        raise ValueError(f'Q5b0 baseline differs from Q3 for {plot_name}.')
    expected_ids = Q5A.q3_target_tree_ids(
        settings['q3_reference_root'], plot_name)
    observed_ids = sorted(
        int(row['gt_tree_id']) for row in report['per_target'])
    if observed_ids != expected_ids:
        raise ValueError(f'Q5b0 targets differ from Q3 for {plot_name}.')
    if int(report['num_target_trees']) != len(expected_ids):
        raise ValueError(f'Q5b0 target count differs for {plot_name}.')
    parameters = report['proposal_parameters']
    if list(map(float, parameters['scales'])) != list(map(
            float, settings['seed_completion_proposal']['scales'])):
        raise ValueError('Q5b0 proposal scales differ from preregistration.')
    if list(map(float, parameters['shift_fractions'])) != list(map(
            float,
            settings['seed_completion_proposal']['shift_fractions'])):
        raise ValueError('Q5b0 shifts differ from preregistration.')
    return report


def write_resolved_config(settings, plot_name, forest_path, destination):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(settings['pipeline_template']))
    runtime_dir = Path(settings['output_root']) / 'runtime' / plot_name
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(runtime_dir)
    config.seed_completion_proposal_output_dir = str(destination)
    config.pretrain = str(settings['checkpoint'])
    diagnostic = dict(settings['seed_completion_proposal'])
    diagnostic['expected_target_tree_ids'] = Q5A.q3_target_tree_ids(
        settings['q3_reference_root'], plot_name)
    config.seed_completion_proposal = diagnostic
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
        raise ValueError('Q5b0 must not read Wytham.')
    path = artifact_path(settings['output_root'], plot_name)
    destination = path.parent
    if path.is_file() and not force:
        try:
            validate_artifact(path, plot_name, settings)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(
                f'STALE {plot_name}: {error} Rebuilding artifact.',
                flush=True)
            Q5A._safe_remove(destination, settings['output_root'])
        else:
            print(
                f'SKIP {plot_name}: validated existing Q5b0 artifact.',
                flush=True)
            return
    elif destination.exists() and not force:
        Q5A._safe_remove(destination, settings['output_root'])
    if force:
        Q5A._safe_remove(destination, settings['output_root'])
    forest = Q5A.locate_forest(settings['source_root'], plot_name)
    config_path, runtime_dir = write_resolved_config(
        settings, plot_name, forest, destination)
    log_path = Path(settings['output_root']) / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-u',
        'tools/diagnostics/run_seed_completion_proposal_pipeline.py',
        '--config', str(config_path)]
    print(f'RUN {plot_name}; log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q5b0 pipeline failed for {plot_name}; inspect {log_path}.')
    validate_artifact(path, plot_name, settings)
    Q5A._safe_remove(
        runtime_dir, Path(settings['output_root']) / 'runtime')
    print(f'DONE {plot_name}', flush=True)


def aggregate_detection(reports, key):
    rows = [report[key] for report in reports]
    return detection_metrics(
        sum(int(row['tp']) for row in rows),
        sum(int(row['fp']) for row in rows),
        sum(int(row['fn']) for row in rows))


def summarize(settings, reports):
    baseline = aggregate_detection(reports, 'baseline')
    oracle = aggregate_detection(reports, 'oracle_metrics')
    plot_rows = []
    for report in reports:
        base = report['baseline']
        row = report['oracle_metrics']
        plot_rows.append({
            'source_plot': report['source_plot'],
            'num_proposals': int(report['num_proposals']),
            'covered': int(report['proposal_covered_target_trees']),
            'recovered': int(row['recovered_target_trees']),
            'lost': int(row['lost_baseline_trees']),
            'f1_gain_pp': 100.0 * (row['f1'] - base['f1']),
            'commission_increase_pp': 100.0 * (
                row['commission'] - base['commission']),
        })
    metrics = {
        **oracle,
        'num_proposals': int(sum(
            report['num_proposals'] for report in reports)),
        'num_activated_proposals': int(sum(
            report['num_activated_proposals'] for report in reports)),
        'proposal_covered_target_trees': int(sum(
            report['proposal_covered_target_trees'] for report in reports)),
        'recovered_target_trees': int(sum(
            report['oracle_metrics']['recovered_target_trees']
            for report in reports)),
        'lost_baseline_trees': int(sum(
            report['oracle_metrics']['lost_baseline_trees']
            for report in reports)),
        'added_seeds': int(sum(
            report['num_oracle_seeds'] - report['num_baseline_seeds']
            for report in reports)),
        'f1_gain_pp': 100.0 * (oracle['f1'] - baseline['f1']),
        'commission_increase_pp': 100.0 * (
            oracle['commission'] - baseline['commission']),
        'completeness_drop_pp': 100.0 * (
            baseline['completeness'] - oracle['completeness']),
        'nonnegative_plots': int(sum(
            row['f1_gain_pp'] >= -1e-9 for row in plot_rows)),
        'plot_metrics': plot_rows,
    }
    rule = settings['gate']
    gate = {
        'expected_validation_plots': (
            len(reports) == int(rule['expected_validation_plots'])),
        'baseline_reproduced': True,
        'target_count_reproduced': (
            sum(report['num_target_trees'] for report in reports) ==
            int(rule['expected_target_trees'])),
        'proposal_coverage_passed': (
            metrics['proposal_covered_target_trees'] >=
            int(rule['min_proposal_covered_target_trees'])),
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
    result = {'baseline': baseline, 'oracle': metrics, 'gate': gate}
    output = Path(settings['output_root'])
    (output / 'summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# Q5b0 GT-free Seed-Completion Proposal Oracle', '',
        '- 数据：固定五个 validation forests；未读取 Wytham。',
        '- 候选区域完全由预测 vote、semantic、verticality 和 offset 生成。',
        '- GT 只在评估端激活候选，用于测量候选生成上限。',
        f'- GT-free proposals：{metrics["num_proposals"]:,}',
        f'- Oracle 激活：{metrics["num_activated_proposals"]:,}',
        f'- Proposal 覆盖目标树：'
        f'{metrics["proposal_covered_target_trees"]}/'
        f'{rule["expected_target_trees"]}', '',
        '| Mode | Completeness | Commission | F1 | F1 gain | Recovered | Lost |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f'| baseline | {100 * baseline["completeness"]:.3f}% | '
        f'{100 * baseline["commission"]:.3f}% | '
        f'{100 * baseline["f1"]:.3f}% | - | - | - |',
        f'| proposal_oracle | {100 * metrics["completeness"]:.3f}% | '
        f'{100 * metrics["commission"]:.3f}% | '
        f'{100 * metrics["f1"]:.3f}% | '
        f'{metrics["f1_gain_pp"]:+.3f} pp | '
        f'{metrics["recovered_target_trees"]} | '
        f'{metrics["lost_baseline_trees"]} |', '',
        '## 每森林', '',
        '| Plot | Proposals | Covered | Recovered | Lost | F1 gain |',
        '|---|---:|---:|---:|---:|---:|']
    for row in plot_rows:
        lines.append(
            f'| {row["source_plot"]} | {row["num_proposals"]:,} | '
            f'{row["covered"]} | {row["recovered"]} | '
            f'{row["lost"]} | {row["f1_gain_pp"]:+.3f} pp |')
    lines.extend(['', '## Gate', ''])
    lines.extend(f'- {key}: **{value}**' for key, value in gate.items())
    lines.extend(['', (
        'PASS：进入 Q5b1，生成 train/validation proposal 数据并训练激活头。'
        if gate['passed'] else
        'STOP：GT-free proposal 上限不足，关闭 Seed-Completion 路线。')])
    (output / 'summary.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result


def load_settings(config_path):
    import yaml
    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    if 'wytham' in json.dumps(settings).lower():
        raise ValueError('Q5b0 must not reference Wytham.')
    if len(settings['validation_plots']) != len(set(
            settings['validation_plots'])):
        raise ValueError('Q5b0 validation plots contain duplicates.')
    return settings


def run(config_path, force=False):
    settings = load_settings(config_path)
    for index, plot in enumerate(settings['validation_plots'], start=1):
        print(f'[{index}/{len(settings["validation_plots"])}] {plot}',
              flush=True)
        run_plot(settings, plot, force=force)
    reports = [
        validate_artifact(
            artifact_path(settings['output_root'], plot), plot, settings)
        for plot in settings['validation_plots']]
    result = summarize(settings, reports)
    if not result['gate']['passed']:
        raise RuntimeError(
            'Q5b0 GT-free proposal Oracle gate failed; close the route.')


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed-validation Q5b0 proposal Oracle.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
