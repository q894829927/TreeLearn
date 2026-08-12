"""Generate and aggregate the fixed-validation Q6a fragment-graph Oracle."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tree_learn.util.omission_diagnostics import detection_metrics
from tree_learn.util.fragment_graph_diagnostics import MERGE_MODES


class BaselineReferenceDriftError(ValueError):
    """Paired baseline drift exceeded the preregistered Q3 audit bound."""


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


def q3_fragmentation_gt_ids(q3_reference_root, plot_name):
    path = Path(q3_reference_root) / 'validation' / plot_name / 'gt_trees.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing Q3 per-tree artifact: {path}')
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows or {'gt_tree_id', 'category'} - set(rows[0]):
        raise ValueError(f'Invalid Q3 per-tree artifact for {plot_name}.')
    identifiers = sorted(
        int(row['gt_tree_id']) for row in rows
        if row['category'] == 'fragmentation')
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f'Q3 fragmentation IDs repeat for {plot_name}.')
    return identifiers


def audit_baseline_reference(observed, expected, gate):
    observed = {key: int(observed[key]) for key in ('tp', 'fp', 'fn')}
    expected = {key: int(expected[key]) for key in ('tp', 'fp', 'fn')}
    drift = {key: observed[key] - expected[key] for key in observed}
    audit = {
        'expected': expected,
        'observed': observed,
        'drift': drift,
        'gt_count_preserved': (
            observed['tp'] + observed['fn'] ==
            expected['tp'] + expected['fn']),
        'tp_drift_passed': abs(drift['tp']) <= int(
            gate['max_reference_tp_drift_per_plot']),
        'fp_drift_passed': abs(drift['fp']) <= int(
            gate['max_reference_fp_drift_per_plot']),
        'fn_drift_passed': abs(drift['fn']) <= int(
            gate['max_reference_tp_drift_per_plot']),
    }
    audit['exact'] = not any(drift.values())
    audit['passed'] = all((
        audit['gt_count_preserved'], audit['tp_drift_passed'],
        audit['fp_drift_passed'], audit['fn_drift_passed']))
    return audit


def validate_artifact(path, plot_name, settings):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing Q6a artifact: {path}')
    report = _read_json(path)
    required = {
        'baseline', 'num_fragmentation_gt_trees',
        'expected_fragmentation_gt_ids', 'num_candidate_edges',
        'num_graph_proposals', 'graph_covered_target_trees', 'per_target',
        'modes', 'adjacency_settings', 'source_plot', 'split'}
    missing = required - set(report)
    if missing:
        raise ValueError(f'Q6a artifact misses {sorted(missing)}.')
    if report['source_plot'] != plot_name or report['split'] != 'validation':
        raise ValueError('Q6a source plot or split is invalid.')
    if set(report['modes']) != set(MERGE_MODES):
        raise ValueError('Q6a merge modes differ from preregistration.')

    q3_dir = Path(settings['q3_reference_root']) / 'validation' / plot_name
    q3 = _read_json(q3_dir / 'summary.json')
    expected_ids = q3_fragmentation_gt_ids(
        settings['q3_reference_root'], plot_name)
    if int(q3['category_counts']['fragmentation']) != len(expected_ids):
        raise ValueError(f'Q3 fragmentation CSV/JSON differ for {plot_name}.')
    if sorted(map(int, report['expected_fragmentation_gt_ids'])) != expected_ids:
        raise ValueError(f'Q6a fragmentation IDs differ from Q3 for {plot_name}.')
    if int(report['num_fragmentation_gt_trees']) != len(expected_ids):
        raise ValueError(f'Q6a fragmentation count differs for {plot_name}.')

    expected_baseline = {
        key: int(q3['baseline'][key]) for key in ('tp', 'fp', 'fn')}
    baseline_audit = audit_baseline_reference(
        report['baseline'], expected_baseline, settings['gate'])
    if not baseline_audit['passed']:
        raise BaselineReferenceDriftError(
            f'Q6a baseline drift exceeds tolerance for {plot_name}: '
            f'{baseline_audit["drift"]}.')
    report['baseline_reference_audit'] = baseline_audit
    if not baseline_audit['exact']:
        print(
            f'WARNING {plot_name}: accepted paired/Q3 baseline drift '
            f'{baseline_audit["drift"]}; effects remain paired.', flush=True)
    observed_adjacency = report['adjacency_settings']
    expected_adjacency = settings['fragment_graph_oracle']['adjacency']
    for name, expected in expected_adjacency.items():
        if float(observed_adjacency[name]) != float(expected):
            raise ValueError(f'Q6a adjacency setting {name} differs.')
    return report


def write_resolved_config(settings, plot_name, forest_path, destination):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(settings['pipeline_template']))
    runtime_dir = Path(settings['output_root']) / 'runtime' / plot_name
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(runtime_dir)
    config.fragment_graph_output_dir = str(destination)
    config.pretrain = str(settings['checkpoint'])
    diagnostic = dict(settings['fragment_graph_oracle'])
    adjacency = diagnostic.pop('adjacency')
    diagnostic.update(adjacency)
    diagnostic['expected_fragmentation_gt_ids'] = q3_fragmentation_gt_ids(
        settings['q3_reference_root'], plot_name)
    config.fragment_graph_oracle = diagnostic
    path = Path(settings['output_root']) / 'runtime_configs' / f'{plot_name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            munch_to_dict(config), file, sort_keys=False, allow_unicode=True)
    return path, runtime_dir


def _safe_remove(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Refusing unsafe cleanup: {path}')
    if path.exists():
        shutil.rmtree(path)


def run_plot(settings, plot_name, force=False):
    if 'wytham' in plot_name.lower():
        raise ValueError('Q6a must not read Wytham.')
    path = artifact_path(settings['output_root'], plot_name)
    destination = path.parent
    if path.is_file() and not force:
        try:
            validate_artifact(path, plot_name, settings)
        except BaselineReferenceDriftError:
            raise
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(f'STALE {plot_name}: {error} Rebuilding.', flush=True)
            _safe_remove(destination, settings['output_root'])
        else:
            print(f'SKIP {plot_name}: validated existing Q6a artifact.', flush=True)
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
        'tools/diagnostics/run_fragment_graph_oracle_pipeline.py',
        '--config', str(config_path)]
    print(f'RUN {plot_name}; log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q6a pipeline failed for {plot_name}; inspect {log_path}.')
    validate_artifact(path, plot_name, settings)
    _safe_remove(runtime_dir, Path(settings['output_root']) / 'runtime')
    print(f'DONE {plot_name}', flush=True)


def _aggregate_detection(reports, key):
    rows = [report[key] if key == 'baseline' else report['modes'][key]
            for report in reports]
    return detection_metrics(
        sum(int(row['tp']) for row in rows),
        sum(int(row['fp']) for row in rows),
        sum(int(row['fn']) for row in rows))


def _aggregate_mode(reports, mode, baseline):
    metrics = _aggregate_detection(reports, mode)
    plot_rows = []
    for report in reports:
        base = report['baseline']
        row = report['modes'][mode]
        plot_rows.append({
            'source_plot': report['source_plot'],
            'targets': int(report['num_fragmentation_gt_trees']),
            'graph_covered': int(report['graph_covered_target_trees']),
            'recovered': int(row['recovered_fragmentation_trees']),
            'lost': int(row['lost_baseline_trees']),
            'f1_gain_pp': float(100 * (row['f1'] - base['f1'])),
            'commission_increase_pp': float(
                100 * (row['commission'] - base['commission'])),
        })
    return {
        **metrics,
        'accepted_groups': int(sum(
            report['modes'][mode]['accepted_groups'] for report in reports)),
        'recovered_fragmentation_trees': int(sum(
            report['modes'][mode]['recovered_fragmentation_trees']
            for report in reports)),
        'lost_baseline_trees': int(sum(
            report['modes'][mode]['lost_baseline_trees']
            for report in reports)),
        'f1_gain_pp': float(100 * (metrics['f1'] - baseline['f1'])),
        'commission_increase_pp': float(
            100 * (metrics['commission'] - baseline['commission'])),
        'completeness_drop_pp': float(
            100 * (baseline['completeness'] - metrics['completeness'])),
        'nonnegative_plots': int(sum(
            row['f1_gain_pp'] >= -1e-9 for row in plot_rows)),
        'plot_metrics': plot_rows,
    }


def aggregate(settings):
    reports = [validate_artifact(
        artifact_path(settings['output_root'], plot), plot, settings)
        for plot in settings['validation_plots']]
    baseline = _aggregate_detection(reports, 'baseline')
    modes = {
        mode: _aggregate_mode(reports, mode, baseline)
        for mode in MERGE_MODES}
    graph = modes['fragment_graph_oracle']
    ceiling = modes['fragment_union_ceiling']
    rule = settings['gate']
    reference_audits = [report['baseline_reference_audit'] for report in reports]
    total_abs_tp_drift = sum(abs(row['drift']['tp']) for row in reference_audits)
    total_abs_fp_drift = sum(abs(row['drift']['fp']) for row in reference_audits)
    expected_targets = int(rule['expected_fragmentation_trees'])
    observed_targets = sum(
        int(report['num_fragmentation_gt_trees']) for report in reports)
    graph_covered = sum(
        int(report['graph_covered_target_trees']) for report in reports)
    gate = {
        'expected_validation_plots': len(reports) == int(
            rule['expected_validation_plots']),
        'baseline_reference_per_plot_compatible': all(
            row['passed'] for row in reference_audits),
        'baseline_reference_total_tp_drift_passed': (
            total_abs_tp_drift <= int(rule['max_reference_tp_drift_total'])),
        'baseline_reference_total_fp_drift_passed': (
            total_abs_fp_drift <= int(rule['max_reference_fp_drift_total'])),
        'fragmentation_targets_reproduced': observed_targets == expected_targets,
        'union_ceiling_passed': (
            ceiling['recovered_fragmentation_trees'] >= int(
                rule['min_union_ceiling_recovered_trees']) and
            ceiling['f1_gain_pp'] >= float(
                rule['min_union_ceiling_f1_gain_pp'])),
        'graph_target_coverage_passed': graph_covered >= int(
            rule['min_graph_covered_target_trees']),
        'graph_recovery_passed': (
            graph['recovered_fragmentation_trees'] >= int(
                rule['min_graph_recovered_trees'])),
        'graph_f1_gain_passed': graph['f1_gain_pp'] >= float(
            rule['min_graph_f1_gain_pp']),
        'graph_lost_tree_passed': graph['lost_baseline_trees'] <= int(
            rule['max_lost_baseline_trees']),
        'graph_commission_passed': graph['commission_increase_pp'] <= float(
            rule['max_commission_increase_pp']),
        'graph_plot_consistency_passed': graph['nonnegative_plots'] >= int(
            rule['min_nonnegative_plots']),
    }
    gate['passed'] = bool(all(gate.values()))
    result = {
        'validation_plots': list(settings['validation_plots']),
        'num_fragmentation_gt_trees': int(observed_targets),
        'num_candidate_edges': int(sum(
            report['num_candidate_edges'] for report in reports)),
        'num_graph_proposals': int(sum(
            report['num_graph_proposals'] for report in reports)),
        'graph_covered_target_trees': int(graph_covered),
        'baseline': baseline,
        'modes': modes,
        'per_plot': reports,
        'gate': gate,
        'recommendation': (
            'fragment_graph_ranking_head' if gate['passed'] else
            'close_fragment_merge_route'),
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
        '# Q6a 欠分割碎片受限合并 Oracle', '',
        '- 数据：固定五个 validation forests；未读取 Wytham。',
        '- 邻接图只由预测结果生成；固定邻接门限位于实例几何与 base-vote 空间。',
        '- GT 只用于固定 Q3 fragmentation 目标并执行安全 Oracle 验收。',
        f'- Fragmentation 目标树：{result["num_fragmentation_gt_trees"]}',
        f'- GT-free 候选边：{result["num_candidate_edges"]:,}',
        f'- 图覆盖目标树：{result["graph_covered_target_trees"]}', '',
        '## 汇总指标', '',
        '| Mode | TP | FP | FN | Completeness | Commission | F1 | '
        'Recovered | Lost | F1 gain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        f'| baseline | {baseline["tp"]} | {baseline["fp"]} | '
        f'{baseline["fn"]} | {100 * baseline["completeness"]:.3f}% | '
        f'{100 * baseline["commission"]:.3f}% | '
        f'{100 * baseline["f1"]:.3f}% | - | - | - |',
    ]
    for mode in MERGE_MODES:
        row = result['modes'][mode]
        lines.append(
            f'| {mode} | {row["tp"]} | {row["fp"]} | {row["fn"]} | '
            f'{100 * row["completeness"]:.3f}% | '
            f'{100 * row["commission"]:.3f}% | {100 * row["f1"]:.3f}% | '
            f'{row["recovered_fragmentation_trees"]} | '
            f'{row["lost_baseline_trees"]} | {row["f1_gain_pp"]:+.3f} pp |')
    graph_rows = result['modes']['fragment_graph_oracle']['plot_metrics']
    lines.extend([
        '', '## 每森林图 Oracle', '',
        '| Plot | Targets | Covered | Recovered | Lost | F1 gain |',
        '|---|---:|---:|---:|---:|---:|'])
    for row in graph_rows:
        lines.append(
            f'| {row["source_plot"]} | {row["targets"]} | '
            f'{row["graph_covered"]} | {row["recovered"]} | '
            f'{row["lost"]} | {row["f1_gain_pp"]:+.3f} pp |')
    lines.extend(['', '## Gate', ''])
    for name, passed in result['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend([
        '', f'- 推荐路线：**{result["recommendation"]}**', '',
        ('PASS：进入 Q6b，只在固定 train/validation forests 生成碎片图数据并训练排序头。'
         if result['gate']['passed'] else
         'STOP：关闭 fragmentation 合并路线，不使用 Wytham 调参。'), ''])
    return '\n'.join(lines)


def load_settings(config_path):
    raw_text = Path(config_path).read_text(encoding='utf-8')
    active_text = '\n'.join(
        line.split('#', 1)[0] for line in raw_text.splitlines())
    if 'wytham' in active_text.lower():
        raise ValueError('Q6a configuration must not mention Wytham.')
    import yaml
    settings = yaml.safe_load(raw_text)
    required = {
        'output_root', 'q3_reference_root', 'source_root',
        'pipeline_template', 'checkpoint', 'validation_plots',
        'fragment_graph_oracle', 'gate'}
    missing = required - set(settings)
    if missing:
        raise ValueError(f'Q6a config misses {sorted(missing)}.')
    if 'wytham' in json.dumps(settings, ensure_ascii=False).lower():
        raise ValueError('Q6a configuration must not mention Wytham.')
    plots = list(settings['validation_plots'])
    if len(plots) != len(set(plots)):
        raise ValueError('Q6a validation plots contain duplicates.')
    adjacency = settings['fragment_graph_oracle']['adjacency']
    required_adjacency = {
        'max_vote_center_distance', 'max_raw_center_distance',
        'max_xy_bbox_gap', 'max_vertical_gap', 'max_candidate_edges'}
    if required_adjacency - set(adjacency):
        raise ValueError('Q6a adjacency settings are incomplete.')
    if any(float(adjacency[name]) <= 0 for name in required_adjacency):
        raise ValueError('Q6a adjacency settings must be positive.')
    return settings


def run(config_path, force=False):
    settings = load_settings(config_path)
    Path(settings['output_root']).mkdir(parents=True, exist_ok=True)
    for index, plot in enumerate(settings['validation_plots'], 1):
        print(f'[{index}/{len(settings["validation_plots"])}] {plot}', flush=True)
        run_plot(settings, plot, force=force)
    result = aggregate(settings)
    if not result['gate']['passed']:
        raise RuntimeError(
            'Q6a fragment-graph Oracle gate failed; close merge route.')
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed-validation Q6a fragment-graph Oracle.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
