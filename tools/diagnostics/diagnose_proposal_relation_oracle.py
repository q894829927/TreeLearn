"""Run and aggregate the fixed-validation P0 micro-proposal Oracle."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from tree_learn.util import get_config, munch_to_dict
from tree_learn.util.omission_diagnostics import detection_metrics
from tree_learn.util.proposal_relation import (
    build_proposal_graph, build_vertical_micro_proposals,
    coordinate_sha256, evaluate_graph_oracle,
    load_inference_graph, save_inference_graph)


class DiagnosticComplete(RuntimeError):
    pass


def _pipeline_module():
    path = Path(__file__).resolve().parents[1]/'pipeline'/'pipeline.py'
    spec = importlib.util.spec_from_file_location(
        'proposal_oracle_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forest(root, plot):
    matches = [path for suffix in ('.las', '.laz')
               for path in Path(root).glob(f'{plot}{suffix}')]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one forest for {plot}: {matches}')
    return matches[0]


def _safe_remove(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Refusing unsafe cleanup: {path}')
    if path.exists():
        shutil.rmtree(path)


def _artifact(root, plot):
    return Path(root)/'validation'/plot/'summary.json'


def _validate(path, plot, settings):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {
        'source_plot', 'split', 'baseline', 'oracle', 'recovered_trees',
        'missed_trees', 'graph_covered_missed_trees', 'graph_coverage',
        'num_nodes', 'num_edges', 'num_baseline_instances',
        'average_out_degree'}
    if required-set(report):
        raise ValueError(
            f'P0 artifact misses {sorted(required-set(report))}.')
    if report['source_plot'] != plot or report['split'] != 'validation':
        raise ValueError('P0 artifact source is invalid.')
    graph, metadata = load_inference_graph(Path(path).with_name('graph.npz'))
    if metadata['source_plot'] != plot or metadata['split'] != 'validation':
        raise ValueError('P0 graph metadata is invalid.')
    if graph['edge_index'].shape[1] != int(report['num_edges']):
        raise ValueError('P0 graph and summary edge counts differ.')
    if int(report['num_nodes']) > int(settings['gate']['max_nodes_absolute']):
        raise ValueError(f'P0 graph node safety bound exceeded for {plot}.')
    return report


def _run_worker(config_path):
    config = get_config(config_path)
    settings = config.proposal_relation_oracle
    output = Path(str(config.proposal_relation_output_dir))
    plot = Path(str(config.forest_path)).stem
    pipeline = _pipeline_module()
    original_savez = np.savez_compressed
    completed = {'value': False}

    def intercept(file, *args, **kwargs):
        if Path(str(file)).name != 'pointwise_results.npz':
            return original_savez(file, *args, **kwargs)
        required = {
            'coords', 'semantic_prediction_logits', 'instance_labels',
            'instance_preds', 'input_feats'}
        missing = required-set(kwargs)
        if missing:
            raise ValueError(f'P0 pointwise payload misses {sorted(missing)}.')
        coords = kwargs['coords']
        proposals = build_vertical_micro_proposals(
            coords, kwargs['semantic_prediction_logits'],
            verticality=kwargs['input_feats'][:, -1],
            tree_probability_threshold=float(
                settings.tree_probability_threshold),
            xy_cell_size=float(settings.xy_cell_size),
            z_bin_size=float(settings.z_bin_size),
            vertical_split_gap=float(settings.vertical_split_gap),
            tree_class_index=int(settings.tree_class_index))
        graph = build_proposal_graph(
            proposals, max_xy_distance=float(settings.max_xy_distance),
            max_xy_bbox_gap=float(settings.max_xy_bbox_gap),
            max_vertical_gap=float(settings.max_vertical_gap),
            max_neighbors=int(settings.max_neighbors))
        result = evaluate_graph_oracle(
            graph, kwargs['instance_labels'], kwargs['instance_preds'],
            match_iou_threshold=float(settings.match_iou_threshold),
            min_precision_for_counted_fp=float(
                settings.min_precision_for_counted_fp))
        targets = result.pop('targets')
        result.pop('predictions')
        result.update({'source_plot': plot, 'split': 'validation'})
        output.mkdir(parents=True, exist_ok=True)
        save_inference_graph(output/'graph.npz', graph, {
            'source_plot': plot, 'split': 'validation',
            'coordinate_sha256': coordinate_sha256(coords),
            'checkpoint': str(config.pretrain),
            'tree_probability_threshold': float(
                settings.tree_probability_threshold),
            'xy_cell_size': float(settings.xy_cell_size),
            'z_bin_size': float(settings.z_bin_size),
            'vertical_split_gap': float(settings.vertical_split_gap),
            'max_xy_distance': float(settings.max_xy_distance),
            'max_xy_bbox_gap': float(settings.max_xy_bbox_gap),
            'max_vertical_gap': float(settings.max_vertical_gap),
            'max_neighbors': int(settings.max_neighbors)})
        original_savez(
            output/'learning.npz',
            node_gt=targets['node_gt'], node_purity=targets['node_purity'],
            edge_label=targets['edge_label'], edge_valid=targets['edge_valid'],
            covered_gt_ids=targets['covered_gt_ids'],
            point_gt=np.asarray(kwargs['instance_labels'], np.int64),
            baseline_predictions=np.asarray(kwargs['instance_preds'], np.int64))
        (output/'summary.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        completed['value'] = True
        print(
            f'P0 saved {plot}: {result["num_nodes"]:,} nodes, '
            f'{result["num_edges"]:,} edges, '
            f'{result["recovered_trees"]} recovered.', flush=True)
        raise DiagnosticComplete()

    np.savez_compressed = intercept
    try:
        pipeline.run_treelearn_pipeline(config, config_path)
    except DiagnosticComplete:
        pass
    finally:
        np.savez_compressed = original_savez
    if not completed['value']:
        raise RuntimeError('Pipeline did not emit the intercepted P0 payload.')


def _runtime_config(settings, plot, destination):
    config = get_config(settings['pipeline_template'])
    train = get_config(settings['train_config'])
    runtime = Path(settings['output_root'])/'runtime'/plot
    config.forest_path = str(_forest(settings['source_root'], plot))
    config.pipeline_base_dir = str(runtime)
    config.pretrain = str(settings['checkpoint'])
    config.model = train.model
    config.proposal_relation_output_dir = str(destination)
    config.proposal_relation_oracle = settings['proposal_relation_oracle']
    config.save_cfg.results_dir = f'results_p0_{plot}'
    config.save_cfg.save_pointwise = True
    config.save_cfg.save_full_forest = False
    config.save_cfg.save_formats = ['npz']
    config.save_cfg.return_type = 'voxelized_and_filtered'
    config.tile_generation = not (runtime/'tiles'/'npz').is_dir()
    path = Path(settings['output_root'])/'runtime_configs'/f'{plot}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(
        munch_to_dict(config), sort_keys=False, allow_unicode=True),
        encoding='utf-8')
    return path, runtime


def _run_plot(settings, plot, force=False):
    if 'wytham' in plot.lower():
        raise ValueError('P0 must not read Wytham.')
    path = _artifact(settings['output_root'], plot)
    destination = path.parent
    if path.is_file() and not force:
        try:
            _validate(path, plot, settings)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            print(f'STALE {plot}: {error} Rebuilding.', flush=True)
            _safe_remove(destination, settings['output_root'])
        else:
            print(f'SKIP {plot}: validated existing P0 artifact.', flush=True)
            return
    if force:
        _safe_remove(destination, settings['output_root'])
    config, runtime = _runtime_config(settings, plot, destination)
    log = Path(settings['output_root'])/'logs'/f'{plot}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'RUN {plot}; log={log}', flush=True)
    with log.open('w', encoding='utf-8') as stream:
        completed = subprocess.run([
            sys.executable, '-u', __file__, '--worker-config', str(config)],
            stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f'P0 failed for {plot}; inspect {log}.')
    _validate(path, plot, settings)
    _safe_remove(runtime, Path(settings['output_root'])/'runtime')
    print(f'DONE {plot}', flush=True)


def _aggregate_metrics(reports, name):
    return detection_metrics(
        sum(int(report[name]['tp']) for report in reports),
        sum(int(report[name]['fp']) for report in reports),
        sum(int(report[name]['fn']) for report in reports))


def _render(result):
    base, oracle = result['baseline'], result['oracle']
    lines = [
        '# P0 垂直微提议关系图 Oracle', '',
        '- 固定五个 validation forests；未读取 Wytham。',
        '- 锁定 B1 Partial Fine-tune seed 42。',
        '- 实例形成不使用 offset、base seed 或 HDBSCAN。', '',
        '| Mode | TP | FP | FN | Completeness | Commission | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f'| B1-HDBSCAN | {base["tp"]} | {base["fp"]} | {base["fn"]} | '
        f'{100*base["completeness"]:.3f}% | '
        f'{100*base["commission"]:.3f}% | {100*base["f1"]:.3f}% |',
        f'| micro-proposal Oracle | {oracle["tp"]} | {oracle["fp"]} | '
        f'{oracle["fn"]} | {100*oracle["completeness"]:.3f}% | '
        f'{100*oracle["commission"]:.3f}% | {100*oracle["f1"]:.3f}% |',
        '', '## 效果', '',
        f'- F1 gain：{result["f1_gain_pp"]:+.3f} pp',
        f'- Completeness gain：{result["completeness_gain_pp"]:+.3f} pp',
        f'- Recovered trees：{result["recovered_trees"]}',
        f'- Missed-tree graph coverage：{100*result["graph_coverage"]:.3f}%',
        f'- Nodes / B1 instances：{result["node_inflation"]:.3f}x',
        f'- Average out degree：{result["average_out_degree"]:.3f}',
        '', '## 每森林', '',
        '| Plot | Nodes | Edges | Recovered | Coverage | F1 gain |',
        '|---|---:|---:|---:|---:|---:|']
    for row in result['per_plot']:
        lines.append(
            f'| {row["source_plot"]} | {row["num_nodes"]:,} | '
            f'{row["num_edges"]:,} | {row["recovered_trees"]} | '
            f'{100*row["graph_coverage"]:.2f}% | '
            f'{row["f1_gain_pp"]:+.3f} pp |')
    lines.extend(['', '## Gate', ''])
    lines.extend(f'- {key}: **{value}**'
                 for key, value in result['gate'].items())
    lines.extend(['', 'PASS：进入 P1/P2。' if result['gate']['passed']
                  else 'STOP：微提议上限不足。', ''])
    return '\n'.join(lines)


def _aggregate(settings):
    reports = [_validate(
        _artifact(settings['output_root'], plot), plot, settings)
        for plot in settings['validation_plots']]
    baseline = _aggregate_metrics(reports, 'baseline')
    oracle = _aggregate_metrics(reports, 'oracle')
    for report in reports:
        report['f1_gain_pp'] = 100*(
            report['oracle']['f1']-report['baseline']['f1'])
    baseline_instances = sum(
        int(report['num_baseline_instances']) for report in reports)
    nodes = sum(int(report['num_nodes']) for report in reports)
    rule = settings['gate']
    result = {
        'baseline': baseline, 'oracle': oracle,
        'f1_gain_pp': 100*(oracle['f1']-baseline['f1']),
        'completeness_gain_pp': 100*(
            oracle['completeness']-baseline['completeness']),
        'commission_increase_pp': 100*(
            oracle['commission']-baseline['commission']),
        'recovered_trees': sum(int(row['recovered_trees'])
                               for row in reports),
        'graph_coverage': sum(int(row['graph_covered_missed_trees'])
                              for row in reports) /
                          max(sum(int(row['missed_trees'])
                                  for row in reports), 1),
        'num_nodes': nodes,
        'num_edges': sum(int(row['num_edges']) for row in reports),
        'node_inflation': nodes/max(baseline_instances, 1),
        'average_out_degree': sum(int(row['num_edges']) for row in reports) /
                              max(nodes, 1),
        'per_plot': reports}
    result['gate'] = {
        'expected_validation_plots':
            len(reports) == int(rule['expected_validation_plots']),
        'oracle_f1_gain_passed':
            result['f1_gain_pp'] >= float(rule['min_f1_gain_pp']),
        'oracle_completeness_gain_passed':
            result['completeness_gain_pp'] >=
            float(rule['min_completeness_gain_pp']),
        'recovered_tree_count_passed':
            result['recovered_trees'] >= int(rule['min_recovered_trees']),
        'commission_not_worse':
            result['commission_increase_pp'] <=
            float(rule['max_commission_increase_pp']),
        'plot_consistency_passed':
            sum(row['f1_gain_pp'] >= -1e-9 for row in reports) >=
            int(rule['min_nonnegative_plots']),
        'missed_tree_coverage_passed':
            result['graph_coverage'] >= float(rule['min_graph_coverage']),
        'proposal_inflation_passed':
            result['node_inflation'] <= float(rule['max_node_inflation']),
        'degree_bound_passed':
            result['average_out_degree'] <=
            int(settings['proposal_relation_oracle']['max_neighbors'])}
    result['gate']['passed'] = bool(all(result['gate'].values()))
    output = Path(settings['output_root'])
    (output/'summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = _render(result)
    (output/'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    return result


def run(config_path, force=False):
    settings = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if 'wytham' in json.dumps(settings).lower():
        raise ValueError('P0 configuration must not mention Wytham.')
    Path(settings['output_root']).mkdir(parents=True, exist_ok=True)
    for index, plot in enumerate(settings['validation_plots'], 1):
        print(f'[{index}/{len(settings["validation_plots"])}] {plot}',
              flush=True)
        _run_plot(settings, plot, force)
    result = _aggregate(settings)
    if not result['gate']['passed']:
        raise RuntimeError('P0 Oracle gate failed; do not train attention.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--worker-config')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.worker_config:
        _run_worker(args.worker_config)
    elif args.config:
        run(args.config, args.force)
    else:
        parser.error('--config or --worker-config is required')


if __name__ == '__main__':
    main()
