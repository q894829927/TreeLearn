"""Generate and aggregate the fixed-validation Q4a instance-split Oracle."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tree_learn.util.omission_diagnostics import detection_metrics
from tree_learn.util.instance_split_diagnostics import SPLIT_MODES


GEOMETRY_MODES = SPLIT_MODES[1:]


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


def _read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def validate_artifact(path, plot_name, q3_reference_root):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing Q4a artifact: {path}')
    report = _read_json(path)
    required = {
        'baseline', 'num_split_target_predictions',
        'num_undersegmented_gt_trees', 'modes', 'source_plot', 'split'}
    if required - set(report):
        raise ValueError(f'Incomplete Q4a artifact for {plot_name}.')
    if report['source_plot'] != plot_name or report['split'] != 'validation':
        raise ValueError('Q4a source plot or split is invalid.')
    if set(report['modes']) != set(SPLIT_MODES):
        raise ValueError('Q4a split modes differ from preregistration.')
    q3_path = (
        Path(q3_reference_root) / 'validation' / plot_name / 'summary.json')
    q3 = _read_json(q3_path)
    expected_baseline = {
        key: int(q3['baseline'][key]) for key in ('tp', 'fp', 'fn')}
    observed_baseline = {
        key: int(report['baseline'][key]) for key in ('tp', 'fp', 'fn')}
    if observed_baseline != expected_baseline:
        raise ValueError(
            f'Q4a baseline differs from Q3 for {plot_name}: '
            f'{observed_baseline} != {expected_baseline}.')
    expected_underseg = int(q3['category_counts']['undersegmentation'])
    if int(report['num_undersegmented_gt_trees']) != expected_underseg:
        raise ValueError(
            f'Q4a undersegmentation count differs from Q3 for {plot_name}.')
    return report


def write_resolved_config(settings, plot_name, forest_path, destination):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(settings['pipeline_template']))
    runtime_dir = Path(settings['output_root']) / 'runtime' / plot_name
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(runtime_dir)
    config.split_oracle_output_dir = str(destination)
    config.pretrain = str(settings['checkpoint'])
    config.split_oracle = settings['split_oracle']
    payload = munch_to_dict(config)
    config_path = (
        Path(settings['output_root']) / 'runtime_configs' /
        f'{plot_name}.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return config_path, runtime_dir


def _safe_remove(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Refusing unsafe cleanup: {path}')
    if path.exists():
        shutil.rmtree(path)


def run_plot(settings, plot_name, force=False):
    if 'wytham' in plot_name.lower():
        raise ValueError('Q4a must not read Wytham.')
    path = artifact_path(settings['output_root'], plot_name)
    if path.is_file() and not force:
        validate_artifact(
            path, plot_name, settings['q3_reference_root'])
        print(f'SKIP {plot_name}: validated existing Q4a artifact.', flush=True)
        return
    destination = path.parent
    if force:
        _safe_remove(destination, settings['output_root'])
    forest_path = locate_forest(settings['source_root'], plot_name)
    config_path, runtime_dir = write_resolved_config(
        settings, plot_name, forest_path, destination)
    log_path = Path(settings['output_root']) / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-u',
        'tools/diagnostics/run_instance_split_oracle_pipeline.py',
        '--config', str(config_path),
    ]
    print(f'RUN {plot_name}; log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q4a pipeline failed for {plot_name}; inspect {log_path}.')
    validate_artifact(path, plot_name, settings['q3_reference_root'])
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
    return {
        **metrics,
        'recovered_undersegmented_trees': int(sum(
            report['modes'][mode]['recovered_undersegmented_trees']
            for report in reports)),
        'lost_baseline_trees': int(sum(
            report['modes'][mode]['lost_baseline_trees']
            for report in reports)),
        'proposed_splits': int(sum(
            report['modes'][mode].get('proposed_splits', 0)
            for report in reports)),
        'accepted_splits': int(sum(
            report['modes'][mode].get('accepted_splits', 0)
            for report in reports)),
        'f1_gain_pp': float(100 * (metrics['f1'] - baseline['f1'])),
        'completeness_gain_pp': float(
            100 * (metrics['completeness'] - baseline['completeness'])),
        'commission_reduction_pp': float(
            100 * (baseline['commission'] - metrics['commission'])),
        'plot_wins': int(sum(
            report['modes'][mode]['f1_gain_pp'] > 0
            for report in reports)),
    }


def aggregate(settings):
    plots = list(settings['validation_plots'])
    reports = [validate_artifact(
        artifact_path(settings['output_root'], plot), plot,
        settings['q3_reference_root']) for plot in plots]
    baseline = _sum_detection(reports)
    modes = {
        mode: _aggregate_mode(reports, mode, baseline)
        for mode in SPLIT_MODES}
    best_geometry = max(
        GEOMETRY_MODES,
        key=lambda mode: (
            modes[mode]['f1'],
            modes[mode]['recovered_undersegmented_trees'],
            -modes[mode]['lost_baseline_trees']))
    best = modes[best_geometry]
    vertical = modes['vertical_axis_kmeans_oracle']
    vote = modes['base_vote_kmeans_oracle']
    vertical_comparison = {
        'f1_gain_over_base_vote_pp': float(
            100 * (vertical['f1'] - vote['f1'])),
        'additional_recovered_trees': int(
            vertical['recovered_undersegmented_trees'] -
            vote['recovered_undersegmented_trees']),
    }
    gate_cfg = settings['gate']
    ceiling = modes['gt_extraction_ceiling']
    geometry_effect = (
        best['recovered_undersegmented_trees'] >= int(
            gate_cfg['min_geometry_recovered_trees']) or
        best['f1_gain_pp'] >= float(
            gate_cfg['min_geometry_f1_gain_pp']))
    gate = {
        'expected_validation_plots': (
            len(reports) == int(gate_cfg['expected_validation_plots'])),
        'baseline_reproduced': all(
            report['baseline']['tp'] >= 0 for report in reports),
        'gt_ceiling_f1_passed': (
            ceiling['f1_gain_pp'] >= float(
                gate_cfg['min_gt_ceiling_f1_gain_pp'])),
        'gt_ceiling_completeness_passed': (
            ceiling['completeness_gain_pp'] >= float(
                gate_cfg['min_gt_ceiling_completeness_gain_pp'])),
        'gt_ceiling_safe': ceiling['lost_baseline_trees'] == 0,
        'geometry_effect_passed': bool(geometry_effect),
        'geometry_commission_passed': (
            best['commission_reduction_pp'] >=
            -float(gate_cfg['max_geometry_commission_increase_pp'])),
        'geometry_completeness_passed': (
            best['completeness_gain_pp'] >=
            -float(gate_cfg['max_geometry_completeness_drop_pp'])),
        'geometry_plot_consistency_passed': (
            best['plot_wins'] >= int(gate_cfg['min_geometry_plot_wins'])),
    }
    vertical_gate = {
        'vertical_is_best_geometry': best_geometry ==
        'vertical_axis_kmeans_oracle',
        'vertical_complementarity_passed': (
            vertical_comparison['f1_gain_over_base_vote_pp'] >= float(
                gate_cfg['min_vertical_gain_over_base_vote_pp']) or
            vertical_comparison['additional_recovered_trees'] >= int(
                gate_cfg['min_vertical_additional_recovered_trees'])),
    }
    gate['passed'] = bool(all(gate.values()))
    vertical_gate['passed'] = bool(
        gate['passed'] and all(vertical_gate.values()))
    recommendation = (
        'vertical_topology_split_attention'
        if vertical_gate['passed'] else
        ('simple_geometry_split' if gate['passed'] else 'close_split_route'))
    result = {
        'validation_plots': plots,
        'num_undersegmented_gt_trees': int(sum(
            report['num_undersegmented_gt_trees'] for report in reports)),
        'num_split_target_predictions': int(sum(
            report['num_split_target_predictions'] for report in reports)),
        'baseline': baseline,
        'modes': modes,
        'best_geometry_mode': best_geometry,
        'vertical_comparison': vertical_comparison,
        'per_plot': reports,
        'gate': gate,
        'vertical_attention_gate': vertical_gate,
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
    base = result['baseline']
    lines = [
        '# Q4a 拓扑感知实例拆分 Oracle', '',
        '- 数据：固定五个 validation forests；未读取 Wytham。',
        '- 所有几何方法只对 Q3 欠分割父实例生成 known-K proposal。',
        '- K 和是否接受 proposal 由 GT Oracle 决定；结果不是可部署方法。',
        f'- 欠分割 GT：{result["num_undersegmented_gt_trees"]}',
        f'- 待拆父实例：{result["num_split_target_predictions"]}', '',
        '## 汇总指标', '',
        '| Mode | TP | FP | FN | Completeness | Commission | F1 | '
        'Recovered | Lost | F1 gain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        f'| baseline | {base["tp"]} | {base["fp"]} | {base["fn"]} | '
        f'{100 * base["completeness"]:.3f}% | '
        f'{100 * base["commission"]:.3f}% | {100 * base["f1"]:.3f}% | '
        '- | - | - |',
    ]
    for mode in SPLIT_MODES:
        row = result['modes'][mode]
        lines.append(
            f'| {mode} | {row["tp"]} | {row["fp"]} | {row["fn"]} | '
            f'{100 * row["completeness"]:.3f}% | '
            f'{100 * row["commission"]:.3f}% | '
            f'{100 * row["f1"]:.3f}% | '
            f'{row["recovered_undersegmented_trees"]} | '
            f'{row["lost_baseline_trees"]} | '
            f'{row["f1_gain_pp"]:+.3f} pp |')
    comparison = result['vertical_comparison']
    lines.extend([
        '', '## 决策', '',
        f'- 最佳几何模式：**{result["best_geometry_mode"]}**',
        f'- Vertical 相对 Base-vote F1：'
        f'{comparison["f1_gain_over_base_vote_pp"]:+.3f} pp',
        f'- Vertical 额外恢复：'
        f'{comparison["additional_recovered_trees"]:+d} 棵',
        f'- 推荐路线：**{result["recommendation"]}**', '',
        '## 每森林最佳几何结果', '',
        '| Plot | Underseg GT | Baseline F1 | Best-geometry F1 | Gain |',
        '|---|---:|---:|---:|---:|',
    ])
    mode = result['best_geometry_mode']
    for report in result['per_plot']:
        row = report['modes'][mode]
        lines.append(
            f'| {report["source_plot"]} | '
            f'{report["num_undersegmented_gt_trees"]} | '
            f'{100 * report["baseline"]["f1"]:.3f}% | '
            f'{100 * row["f1"]:.3f}% | {row["f1_gain_pp"]:+.3f} pp |')
    lines.extend(['', '## 主 Gate', ''])
    for name, passed in result['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend(['', '## 垂直注意力 Gate', ''])
    for name, passed in result['vertical_attention_gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend(['', (
        'PASS：按推荐路线进入 Q4b；仍不得使用 Wytham 调参。'
        if result['gate']['passed'] else
        'STOP：实例拆分几何上限不足，关闭拆分网络路线。'), ''])
    return '\n'.join(lines)


def load_settings(config_path):
    raw_text = Path(config_path).read_text(encoding='utf-8')
    active_text = '\n'.join(
        line.split('#', 1)[0] for line in raw_text.splitlines())
    if 'wytham' in active_text.lower():
        raise ValueError('Q4a configuration must not mention Wytham.')

    import yaml
    settings = yaml.safe_load(raw_text)
    required = {
        'output_root', 'q3_reference_root', 'source_root',
        'pipeline_template', 'checkpoint', 'validation_plots',
        'split_oracle', 'gate'}
    missing = required - set(settings)
    if missing:
        raise ValueError(f'Missing Q4a fields: {sorted(missing)}')
    if 'wytham' in json.dumps(settings, ensure_ascii=False).lower():
        raise ValueError('Q4a configuration must not mention Wytham.')
    plots = list(settings['validation_plots'])
    if len(plots) != len(set(plots)):
        raise ValueError('Q4a validation plots contain duplicates.')
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
    if not result['gate']['passed']:
        raise RuntimeError(
            'Q4a instance-split Oracle gate failed; close split route.')
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed-validation Q4a instance-split Oracle.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
