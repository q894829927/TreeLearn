"""Generate and aggregate fixed-validation GT omission diagnostics (Q3)."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tree_learn.util.omission_diagnostics import (
    OMISSION_CATEGORIES,
    detection_metrics,
)


ROUTE_BY_CATEGORY = {
    'semantic_support_failure': 'semantic_recovery',
    'base_seed_support_failure': 'seed_recovery',
    'density_clustering_failure': 'density_aware_grouping',
    'undersegmentation': 'instance_split',
    'fragmentation': 'instance_merge',
    'partial_or_localization': 'proposal_localization',
}


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


def artifact_paths(output_root, plot_name):
    directory = Path(output_root) / 'validation' / plot_name
    return {
        'directory': directory,
        'csv': directory / 'gt_trees.csv',
        'json': directory / 'summary.json',
    }


def validate_artifact(paths, expected_plot):
    if not paths['csv'].is_file() or not paths['json'].is_file():
        raise FileNotFoundError(f'Incomplete Q3 artifact for {expected_plot}.')
    with paths['json'].open(encoding='utf-8') as file:
        summary = json.load(file)
    with paths['csv'].open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    required = {
        'gt_tree_id', 'detected', 'category', 'semantic_tree_points',
        'base_seed_points', 'clustered_seed_points', 'best_iou',
        'best_precision', 'best_recall'}
    if not rows or required - set(rows[0]):
        raise ValueError(f'Invalid Q3 rows for {expected_plot}.')
    if summary['source_plot'] != expected_plot:
        raise ValueError('Q3 source_plot does not match its destination.')
    if summary['split'] != 'validation':
        raise ValueError('Q3 accepts validation artifacts only.')
    if int(summary['num_gt_trees']) != len(rows):
        raise ValueError('Q3 GT count differs between CSV and JSON.')
    counts = {
        category: sum(row['category'] == category for row in rows)
        for category in ('detected',) + OMISSION_CATEGORIES}
    if counts != summary['category_counts']:
        raise ValueError('Q3 category counts differ between CSV and JSON.')
    if len({int(row['gt_tree_id']) for row in rows}) != len(rows):
        raise ValueError('Q3 contains duplicate GT tree IDs.')
    return summary, rows


def write_resolved_config(settings, plot_name, forest_path, destination_dir):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(settings['pipeline_template']))
    runtime_dir = Path(settings['output_root']) / 'runtime' / plot_name
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(runtime_dir)
    config.omission_output_dir = str(destination_dir)
    config.omission_split = 'validation'
    config.pretrain = str(settings['checkpoint'])
    payload = munch_to_dict(config)
    config_path = (
        Path(settings['output_root']) / 'runtime_configs' /
        f'{plot_name}.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            payload, file, sort_keys=False, allow_unicode=True)
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
        raise ValueError('Q3 must not read Wytham.')
    paths = artifact_paths(settings['output_root'], plot_name)
    if paths['csv'].is_file() and paths['json'].is_file() and not force:
        validate_artifact(paths, plot_name)
        print(f'SKIP {plot_name}: validated existing artifact.', flush=True)
        return
    if force:
        _safe_remove(paths['directory'], settings['output_root'])
    forest_path = locate_forest(settings['source_root'], plot_name)
    config_path, runtime_dir = write_resolved_config(
        settings, plot_name, forest_path, paths['directory'])
    log_path = Path(settings['output_root']) / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-u',
        'tools/diagnostics/run_gt_omission_pipeline.py',
        '--config', str(config_path),
    ]
    print(f'RUN {plot_name}; log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q3 pipeline failed for {plot_name}; inspect {log_path}.')
    validate_artifact(paths, plot_name)
    _safe_remove(runtime_dir, Path(settings['output_root']) / 'runtime')
    print(f'DONE {plot_name}', flush=True)


def aggregate_artifacts(settings):
    plot_names = list(settings['validation_plots'])
    reports = []
    all_rows = []
    for plot_name in plot_names:
        summary, rows = validate_artifact(
            artifact_paths(settings['output_root'], plot_name), plot_name)
        reports.append(summary)
        for row in rows:
            row = dict(row)
            row['source_plot'] = plot_name
            all_rows.append(row)

    tp = sum(int(row['baseline']['tp']) for row in reports)
    fp = sum(int(row['baseline']['fp']) for row in reports)
    fn = sum(int(row['baseline']['fn']) for row in reports)
    baseline = detection_metrics(tp, fp, fn)
    category_counts = {
        category: sum(
            int(row['category_counts'][category]) for row in reports)
        for category in ('detected',) + OMISSION_CATEGORIES}
    category_oracles = {}
    for category in OMISSION_CATEGORIES:
        recovered = category_counts[category]
        oracle = detection_metrics(tp + recovered, fp, fn - recovered)
        affected_plots = sum(
            int(row['category_counts'][category]) > 0 for row in reports)
        category_oracles[category] = {
            'route': ROUTE_BY_CATEGORY[category],
            'recoverable_trees': int(recovered),
            'affected_plots': int(affected_plots),
            'oracle': oracle,
            'completeness_gain_pp': float(
                100 * (oracle['completeness'] - baseline['completeness'])),
            'f1_gain_pp': float(100 * (oracle['f1'] - baseline['f1'])),
        }
    dominant = max(
        OMISSION_CATEGORIES,
        key=lambda category: (
            category_oracles[category]['f1_gain_pp'],
            category_oracles[category]['recoverable_trees']))
    gate_cfg = settings['gate']
    actionable = [
        category for category in OMISSION_CATEGORIES
        if category_oracles[category]['recoverable_trees'] >= int(
            gate_cfg['min_actionable_omissions']) and
        category_oracles[category]['affected_plots'] >= int(
            gate_cfg['min_actionable_plots']) and
        category_oracles[category]['f1_gain_pp'] >= float(
            gate_cfg['min_actionable_f1_gain_pp'])]
    gate = {
        'expected_validation_plots': (
            len(reports) == int(gate_cfg['expected_validation_plots'])),
        'no_wytham': not any(
            'wytham' in str(value).lower()
            for value in (plot_names, settings['source_root'])),
        'all_gt_accounted': (
            sum(category_counts.values()) ==
            sum(int(row['num_gt_trees']) for row in reports)),
        'enough_omissions': fn >= int(gate_cfg['min_total_omissions']),
        'actionable_route_identified': bool(actionable),
    }
    gate['passed'] = all(gate.values())
    result = {
        'validation_plots': plot_names,
        'num_gt_trees': int(tp + fn),
        'num_omissions': int(fn),
        'baseline': baseline,
        'category_counts': category_counts,
        'category_oracles': category_oracles,
        'actionable_categories': actionable,
        'recommended_category': dominant,
        'recommended_route': ROUTE_BY_CATEGORY[dominant],
        'per_plot': reports,
        'gate': gate,
    }
    output_root = Path(settings['output_root'])
    with (output_root / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    with (output_root / 'gt_trees_all.csv').open(
            'w', newline='', encoding='utf-8') as file:
        fieldnames = list(all_rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    markdown = render_markdown(result)
    (output_root / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    return result


def render_markdown(result):
    baseline = result['baseline']
    lines = [
        '# Q3 复杂林分漏检树归因 Oracle', '',
        '- 数据：固定 5 个 validation forests；未读取 Wytham。',
        '- 诊断层级：语义 → base seed → HDBSCAN 初始簇 → 最终实例。',
        '- 各类别 Oracle 仅表示完美恢复该类 FN 的检测上限。', '',
        '## Baseline', '',
        '| GT trees | TP | FP | FN | Completeness | Commission | F1 |',
        '|---:|---:|---:|---:|---:|---:|---:|',
        f'| {result["num_gt_trees"]} | {baseline["tp"]} | '
        f'{baseline["fp"]} | {baseline["fn"]} | '
        f'{100 * baseline["completeness"]:.3f}% | '
        f'{100 * baseline["commission"]:.3f}% | '
        f'{100 * baseline["f1"]:.3f}% |', '',
        '## 漏检原因与单类恢复上限', '',
        '| Category | Route | FN | Plots | Completeness gain | F1 gain |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for category in OMISSION_CATEGORIES:
        row = result['category_oracles'][category]
        lines.append(
            f'| {category} | {row["route"]} | '
            f'{row["recoverable_trees"]} | {row["affected_plots"]} | '
            f'+{row["completeness_gain_pp"]:.3f} pp | '
            f'+{row["f1_gain_pp"]:.3f} pp |')
    lines.extend([
        '', '## 每森林', '',
        '| Plot | GT | TP | FP | FN | Dominant omission |',
        '|---|---:|---:|---:|---:|---|',
    ])
    for row in result['per_plot']:
        omissions = {
            category: int(row['category_counts'][category])
            for category in OMISSION_CATEGORIES}
        dominant = max(omissions, key=omissions.get)
        lines.append(
            f'| {row["source_plot"]} | {row["num_gt_trees"]} | '
            f'{row["baseline"]["tp"]} | {row["baseline"]["fp"]} | '
            f'{row["baseline"]["fn"]} | {dominant} '
            f'({omissions[dominant]}) |')
    lines.extend([
        '', '## 决策', '',
        f'- 推荐类别：**{result["recommended_category"]}**',
        f'- 推荐路线：**{result["recommended_route"]}**',
        f'- 可执行类别：{", ".join(result["actionable_categories"]) or "无"}',
        '', '## Gate', '',
    ])
    for name, passed in result['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    lines.extend([
        '',
        ('PASS：只实现推荐路线的最小控制实验，不接触 Wytham。'
         if result['gate']['passed'] else
         'STOP：验证集未识别出足够大的可恢复瓶颈，不新增网络。'),
        '',
    ])
    return '\n'.join(lines)


def load_settings(config_path):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    serialized = json.dumps(settings, ensure_ascii=False).lower()
    if 'wytham' in serialized:
        raise ValueError('Q3 configuration must not mention Wytham.')
    plots = list(settings['validation_plots'])
    if len(plots) != len(set(plots)):
        raise ValueError('Q3 validation plots contain duplicates.')
    return settings


def run(config_path, force=False):
    settings = load_settings(config_path)
    Path(settings['output_root']).mkdir(parents=True, exist_ok=True)
    for index, plot_name in enumerate(settings['validation_plots'], 1):
        print(
            f'[{index}/{len(settings["validation_plots"])}] {plot_name}',
            flush=True)
        run_plot(settings, plot_name, force=force)
    result = aggregate_artifacts(settings)
    if not result['gate']['passed']:
        raise RuntimeError('Q3 omission bottleneck gate failed.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate fixed-validation GT omission diagnostics.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
