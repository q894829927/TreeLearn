"""Generate and aggregate fixed-validation Q4b0 split controls."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tree_learn.util.instance_split_diagnostics import (
    SPLIT_DEPLOYABILITY_MODES,
)
from tree_learn.util.omission_diagnostics import detection_metrics


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


def q3_undersegmented_gt_ids(q3_reference_root, plot_name):
    csv_path = (
        Path(q3_reference_root) / 'validation' / plot_name / 'gt_trees.csv')
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing Q3 per-tree artifact: {csv_path}')
    with csv_path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows or {'gt_tree_id', 'category'} - set(rows[0]):
        raise ValueError(f'Invalid Q3 per-tree artifact for {plot_name}.')
    ids = sorted(
        int(row['gt_tree_id']) for row in rows
        if row['category'] == 'undersegmentation')
    if len(ids) != len(set(ids)):
        raise ValueError(
            f'Q3 undersegmentation IDs are duplicated for {plot_name}.')
    return ids


def _safe_remove(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Refusing unsafe cleanup: {path}')
    if path.exists():
        shutil.rmtree(path)


def artifact_path(output_root, plot_name):
    return Path(output_root) / 'validation' / plot_name / 'summary.json'


def _read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def validate_artifact(path, plot_name, q4a_reference_root):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing Q4b0 artifact: {path}')
    report = _read_json(path)
    required = {
        'baseline', 'num_split_target_predictions',
        'num_undersegmented_gt_trees', 'expected_undersegmented_gt_ids',
        'modes', 'source_plot', 'split'}
    if required - set(report):
        raise ValueError(f'Incomplete Q4b0 artifact for {plot_name}.')
    if report['source_plot'] != plot_name or report['split'] != 'validation':
        raise ValueError('Q4b0 source plot or split is invalid.')
    if set(report['modes']) != set(SPLIT_DEPLOYABILITY_MODES):
        raise ValueError('Q4b0 modes differ from preregistration.')

    q4a_path = (
        Path(q4a_reference_root) / 'validation' / plot_name / 'summary.json')
    q4a = _read_json(q4a_path)
    for key in ('tp', 'fp', 'fn'):
        if int(report['baseline'][key]) != int(q4a['baseline'][key]):
            raise ValueError(
                f'Q4b0 baseline differs from Q4a for {plot_name}.')
    expected_ids = sorted(map(
        int, q4a['expected_undersegmented_gt_ids']))
    observed_ids = sorted(map(
        int, report['expected_undersegmented_gt_ids']))
    if observed_ids != expected_ids:
        raise ValueError(
            f'Q4b0 target GT IDs differ from Q4a for {plot_name}.')
    if int(report['num_undersegmented_gt_trees']) != len(expected_ids):
        raise ValueError(
            f'Q4b0 target count differs from Q4a for {plot_name}.')
    if int(report['num_split_target_predictions']) != int(
            q4a['num_split_target_predictions']):
        raise ValueError(
            f'Q4b0 parent count differs from Q4a for {plot_name}.')

    reference = q4a['modes']['raw_xy_kmeans_oracle']
    reproduced = report['modes']['known_k_oracle_accept']
    for key in ('tp', 'fp', 'fn', 'recovered_undersegmented_trees',
                'lost_baseline_trees', 'proposed_splits',
                'accepted_splits'):
        if int(reproduced.get(key, 0)) != int(reference.get(key, 0)):
            raise ValueError(
                f'Q4b0 known-K Oracle does not reproduce Q4a {key} '
                f'for {plot_name}.')
    # HDBSCAN prediction IDs are run-local labels and may be permuted across
    # otherwise identical reruns. Reproduction is therefore established by
    # detection counts, proposal/acceptance counts, recovered trees and lost
    # baseline trees above, never by raw prediction-ID equality.
    return report


def write_resolved_config(settings, plot_name, forest_path, destination):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(settings['pipeline_template']))
    runtime_dir = Path(settings['output_root']) / 'runtime' / plot_name
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(runtime_dir)
    config.split_deployability_output_dir = str(destination)
    config.pretrain = str(settings['checkpoint'])
    diagnostic = dict(settings['split_deployability'])
    diagnostic['expected_undersegmented_gt_ids'] = (
        q3_undersegmented_gt_ids(
            settings['q3_reference_root'], plot_name))
    config.split_deployability = diagnostic
    payload = munch_to_dict(config)
    config_path = (
        Path(settings['output_root']) / 'runtime_configs' /
        f'{plot_name}.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return config_path, runtime_dir


def run_plot(settings, plot_name, force=False):
    if 'wytham' in plot_name.lower():
        raise ValueError('Q4b0 must not read Wytham.')
    path = artifact_path(settings['output_root'], plot_name)
    destination = path.parent
    if path.is_file() and not force:
        try:
            validate_artifact(
                path, plot_name, settings['q4a_reference_root'])
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(
                f'STALE {plot_name}: {error} Rebuilding artifact.',
                flush=True)
            _safe_remove(destination, settings['output_root'])
        else:
            print(
                f'SKIP {plot_name}: validated existing Q4b0 artifact.',
                flush=True)
            return
    elif destination.exists() and not force:
        print(
            f'STALE {plot_name}: incomplete Q4b0 artifact. Rebuilding.',
            flush=True)
        _safe_remove(destination, settings['output_root'])
    if force:
        _safe_remove(destination, settings['output_root'])

    forest_path = locate_forest(settings['source_root'], plot_name)
    config_path, runtime_dir = write_resolved_config(
        settings, plot_name, forest_path, destination)
    log_path = Path(settings['output_root']) / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-u',
        'tools/diagnostics/run_instance_split_deployability_pipeline.py',
        '--config', str(config_path),
    ]
    print(f'RUN {plot_name}; log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q4b0 pipeline failed for {plot_name}; inspect {log_path}.')
    validate_artifact(path, plot_name, settings['q4a_reference_root'])
    _safe_remove(runtime_dir, Path(settings['output_root']) / 'runtime')
    print(f'DONE {plot_name}', flush=True)


def _sum_detection(reports, mode=None):
    rows = [
        report['baseline'] if mode is None else report['modes'][mode]
        for report in reports]
    return detection_metrics(
        sum(int(row['tp']) for row in rows),
        sum(int(row['fp']) for row in rows),
        sum(int(row['fn']) for row in rows))


def _aggregate_mode(reports, mode, baseline):
    metrics = _sum_detection(reports, mode)
    gains = [float(report['modes'][mode]['f1_gain_pp'])
             for report in reports]
    return {
        **metrics,
        'recovered_undersegmented_trees': int(sum(
            report['modes'][mode]['recovered_undersegmented_trees']
            for report in reports)),
        'lost_baseline_trees': int(sum(
            report['modes'][mode]['lost_baseline_trees']
            for report in reports)),
        'proposed_splits': int(sum(
            report['modes'][mode]['proposed_splits']
            for report in reports)),
        'accepted_splits': int(sum(
            report['modes'][mode]['accepted_splits']
            for report in reports)),
        'f1_gain_pp': float(100 * (metrics['f1'] - baseline['f1'])),
        'completeness_gain_pp': float(
            100 * (metrics['completeness'] - baseline['completeness'])),
        'commission_reduction_pp': float(
            100 * (baseline['commission'] - metrics['commission'])),
        'nonnegative_f1_plots': int(sum(value >= -1e-9 for value in gains)),
        'worst_plot_f1_gain_pp': float(min(gains)),
    }


def evaluate_mode_gate(metrics, gate_cfg):
    gate = {
        'f1_gain_passed': metrics['f1_gain_pp'] >= float(
            gate_cfg['min_f1_gain_pp']),
        'completeness_preserved': metrics['completeness_gain_pp'] >= -float(
            gate_cfg['max_completeness_drop_pp']),
        'commission_preserved': metrics['commission_reduction_pp'] >= -float(
            gate_cfg['max_commission_increase_pp']),
        'plot_consistency_passed': metrics['nonnegative_f1_plots'] >= int(
            gate_cfg['min_nonnegative_plots']),
    }
    gate['passed'] = bool(all(gate.values()))
    return gate


def aggregate(settings):
    plots = list(settings['validation_plots'])
    reports = [validate_artifact(
        artifact_path(settings['output_root'], plot), plot,
        settings['q4a_reference_root']) for plot in plots]
    baseline = _sum_detection(reports)
    modes = {
        mode: _aggregate_mode(reports, mode, baseline)
        for mode in SPLIT_DEPLOYABILITY_MODES}

    q4a = _read_json(Path(settings['q4a_reference_root']) / 'summary.json')
    reference_baseline = {
        key: int(q4a['baseline'][key]) for key in ('tp', 'fp', 'fn')}
    observed_baseline = {
        key: int(baseline[key]) for key in ('tp', 'fp', 'fn')}
    reference = q4a['modes']['raw_xy_kmeans_oracle']
    reproduced = modes['known_k_oracle_accept']
    integrity_gate = {
        'expected_validation_plots': (
            len(reports) == int(settings['gate']['expected_validation_plots'])),
        'q4a_main_gate_passed': bool(q4a['gate']['passed']),
        'q4a_recommended_simple_geometry': (
            q4a['recommendation'] == 'simple_geometry_split'),
        'baseline_reproduced': observed_baseline == reference_baseline,
        'known_k_oracle_reproduced': all(
            int(reproduced[key]) == int(reference[key])
            for key in ('tp', 'fp', 'fn',
                        'recovered_undersegmented_trees',
                        'lost_baseline_trees')),
    }
    integrity_gate['passed'] = bool(all(integrity_gate.values()))
    deployment_gates = {
        'known_k_accept_all': evaluate_mode_gate(
            modes['known_k_accept_all'], settings['gate']),
        'fixed_k2_accept_all': evaluate_mode_gate(
            modes['fixed_k2_accept_all'], settings['gate']),
    }
    if deployment_gates['fixed_k2_accept_all']['passed']:
        recommendation = 'candidate_classifier_only'
    elif deployment_gates['known_k_accept_all']['passed']:
        recommendation = 'candidate_and_child_count_heads'
    else:
        recommendation = 'candidate_child_count_and_safety_heads'

    result = {
        'validation_plots': plots,
        'num_undersegmented_gt_trees': int(sum(
            report['num_undersegmented_gt_trees'] for report in reports)),
        'num_split_target_predictions': int(sum(
            report['num_split_target_predictions'] for report in reports)),
        'baseline': baseline,
        'modes': modes,
        'per_plot': reports,
        'integrity_gate': integrity_gate,
        'deployment_gates': deployment_gates,
        'recommendation': recommendation,
    }
    output_root = Path(settings['output_root'])
    (output_root / 'summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = render_markdown(result)
    (output_root / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    return result


def render_markdown(result):
    baseline = result['baseline']
    lines = [
        '# Q4b0 Raw-XY 实例拆分可部署性分解', '',
        '- 数据：固定五个 validation forests；未读取 Wytham。',
        '- 候选父实例仍由 Q3 GT 类别给出，本阶段不代表完整可部署系统。',
        '- 目的：分别移除 GT 验收与 GT 子树数量 K。',
        f'- 欠分割 GT：{result["num_undersegmented_gt_trees"]}',
        f'- 候选父实例：{result["num_split_target_predictions"]}', '',
        '## 汇总指标', '',
        '| Mode | Proposed | Accepted | Recovered | Lost | Completeness | '
        'Commission | F1 | F1 gain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        f'| baseline | - | - | - | - | '
        f'{100 * baseline["completeness"]:.3f}% | '
        f'{100 * baseline["commission"]:.3f}% | '
        f'{100 * baseline["f1"]:.3f}% | - |',
    ]
    for mode in SPLIT_DEPLOYABILITY_MODES:
        row = result['modes'][mode]
        lines.append(
            f'| {mode} | {row["proposed_splits"]} | '
            f'{row["accepted_splits"]} | '
            f'{row["recovered_undersegmented_trees"]} | '
            f'{row["lost_baseline_trees"]} | '
            f'{100 * row["completeness"]:.3f}% | '
            f'{100 * row["commission"]:.3f}% | '
            f'{100 * row["f1"]:.3f}% | '
            f'{row["f1_gain_pp"]:+.3f} pp |')
    lines.extend(['', '## 每森林 F1 差值', '',
                  '| Plot | Known-K accept-all | Fixed-K2 accept-all |',
                  '|---|---:|---:|'])
    for report in result['per_plot']:
        known = report['modes']['known_k_accept_all']['f1_gain_pp']
        fixed = report['modes']['fixed_k2_accept_all']['f1_gain_pp']
        lines.append(
            f'| {report["source_plot"]} | {known:+.3f} pp | '
            f'{fixed:+.3f} pp |')
    lines.extend(['', '## 完整性 Gate', ''])
    for name, passed in result['integrity_gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    for mode, gate in result['deployment_gates'].items():
        lines.extend(['', f'## {mode} Gate', ''])
        for name, passed in gate.items():
            lines.append(f'- {name}: **{bool(passed)}**')
    descriptions = {
        'candidate_classifier_only': (
            '固定 K=2 且全部接受已通过；Q4b1 只训练欠分割候选分类头。'),
        'candidate_and_child_count_heads': (
            '已知 K 全部接受通过，但固定 K=2 未通过；Q4b1 训练候选分类头与 K 预测头。'),
        'candidate_child_count_and_safety_heads': (
            '全部接受未通过；Q4b1 必须同时训练候选、K 与安全验收头。'),
    }
    lines.extend([
        '', '## 决策', '',
        f'- 推荐路线：**{result["recommendation"]}**',
        f'- {descriptions[result["recommendation"]]}', '',
        '下一阶段仍只使用固定 train/validation forests，不得读取 Wytham。',
        '',
    ])
    return '\n'.join(lines)


def load_settings(config_path):
    raw_text = Path(config_path).read_text(encoding='utf-8')
    active_text = '\n'.join(
        line.split('#', 1)[0] for line in raw_text.splitlines())
    if 'wytham' in active_text.lower():
        raise ValueError('Q4b0 configuration must not mention Wytham.')
    import yaml
    settings = yaml.safe_load(raw_text)
    required = {
        'output_root', 'q3_reference_root', 'q4a_reference_root',
        'source_root', 'pipeline_template', 'checkpoint',
        'validation_plots', 'split_deployability', 'gate'}
    missing = required - set(settings)
    if missing:
        raise ValueError(f'Missing Q4b0 fields: {sorted(missing)}')
    if 'wytham' in json.dumps(settings, ensure_ascii=False).lower():
        raise ValueError('Q4b0 configuration must not mention Wytham.')
    plots = list(settings['validation_plots'])
    if len(plots) != len(set(plots)):
        raise ValueError('Q4b0 validation plots contain duplicates.')
    return settings


def run(config_path, force=False):
    settings = load_settings(config_path)
    Path(settings['output_root']).mkdir(parents=True, exist_ok=True)
    for index, plot_name in enumerate(settings['validation_plots'], 1):
        print(
            f'[{index}/{len(settings["validation_plots"])}] {plot_name}',
            flush=True)
        run_plot(settings, plot_name, force=force)
    result = aggregate(settings)
    if not result['integrity_gate']['passed']:
        raise RuntimeError(
            'Q4b0 integrity gate failed; do not use the decomposition.')
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed-validation Q4b0 split decomposition.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
