"""Evaluate label-free hierarchical super-proposals on passed P0 artifacts."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import yaml

from tree_learn.util.omission_diagnostics import detection_metrics
from tree_learn.util.proposal_relation import (
    evaluate_instances, load_inference_graph, point_instances_from_nodes,
    save_inference_graph)
from tree_learn.util.proposal_relation_coarsening import (
    build_hierarchical_graph)


class _Union:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value):
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _fast_targets(graph, point_gt, minimum_node_purity=.5):
    """Vectorized equivalent of build_learning_targets for large forests."""
    point_node = np.asarray(graph['point_node_id'], np.int64)
    point_gt = np.asarray(point_gt, np.int64).reshape(-1)
    if len(point_node) != len(point_gt):
        raise ValueError('P0b point labels and proposal mapping are not aligned.')
    node_count = len(graph['node_features'])
    valid_node = point_node >= 0
    node_sizes = np.bincount(
        point_node[valid_node], minlength=node_count).astype(np.int64)
    positive = valid_node & (point_gt > 0)
    node_gt = np.zeros(node_count, np.int64)
    node_purity = np.zeros(node_count, np.float32)
    if np.any(positive):
        gt_base = int(point_gt[positive].max())+1
        encoded = point_node[positive]*gt_base + point_gt[positive]
        pairs, counts = np.unique(encoded, return_counts=True)
        pair_nodes, pair_gt = pairs//gt_base, pairs%gt_base
        order = np.lexsort((-counts, pair_nodes))
        ordered_nodes = pair_nodes[order]
        first = np.r_[True, ordered_nodes[1:] != ordered_nodes[:-1]]
        chosen = order[first]
        chosen_nodes = pair_nodes[chosen].astype(np.int64)
        node_gt[chosen_nodes] = pair_gt[chosen]
        node_purity[chosen_nodes] = (
            counts[chosen]/np.maximum(node_sizes[chosen_nodes], 1))
    source, target = np.asarray(graph['edge_index'], np.int64)
    clean = node_purity >= float(minimum_node_purity)
    edge_valid = clean[source] & clean[target]
    edge_label = (
        edge_valid & (node_gt[source] > 0) &
        (node_gt[source] == node_gt[target]))
    return {
        'node_gt': node_gt, 'node_purity': node_purity,
        'edge_label': edge_label.astype(np.int8),
        'edge_valid': edge_valid,
        'covered_gt_ids': np.unique(node_gt[node_gt > 0]),
    }


def _evaluate_oracle(graph, point_gt, baseline_predictions,
                     match_iou_threshold=.5,
                     min_precision_for_counted_fp=.5):
    targets = _fast_targets(graph, point_gt)
    union = _Union(len(graph['node_features']))
    for source, target in graph['edge_index'][:, targets['edge_label'].astype(bool)].T:
        union.union(source, target)
    roots = np.asarray(
        [union.find(node) for node in range(len(graph['node_features']))])
    _, node_instances = np.unique(roots, return_inverse=True)
    predictions = point_instances_from_nodes(
        graph['point_node_id'], node_instances.astype(np.int64)+1)
    baseline, baseline_matches, baseline_table = evaluate_instances(
        point_gt, baseline_predictions, match_iou_threshold,
        min_precision_for_counted_fp)
    oracle, oracle_matches, _ = evaluate_instances(
        point_gt, predictions, match_iou_threshold,
        min_precision_for_counted_fp)
    missed = set(range(len(baseline_table['gt_ids'])))-set(baseline_matches)
    missed_ids = {int(baseline_table['gt_ids'][index]) for index in missed}
    covered = missed_ids & set(map(int, targets['covered_gt_ids']))
    recovered = missed & set(oracle_matches)
    return {
        'baseline': baseline, 'oracle': oracle,
        'recovered_trees': int(len(recovered)),
        'missed_trees': int(len(missed)),
        'graph_covered_missed_trees': int(len(covered)),
        'graph_coverage': float(len(covered)/max(len(missed), 1)),
        'num_nodes': int(len(graph['node_features'])),
        'num_edges': int(graph['edge_index'].shape[1]),
        'num_baseline_instances': int(len(baseline_table['pred_ids'])),
        'average_out_degree': float(
            graph['edge_index'].shape[1]/max(len(graph['node_features']), 1)),
        'targets': targets,
    }


def _paths(settings, method, plot):
    source = Path(settings['input_root'])/plot
    output = Path(settings['output_root'])/method/plot
    return source, output


def _validate_source(source, plot):
    graph_path = source/'graph.npz'
    learning_path = source/'learning.npz'
    summary_path = source/'summary.json'
    if not all(path.is_file() for path in (
            graph_path, learning_path, summary_path)):
        raise FileNotFoundError(f'Incomplete P0 source artifact for {plot}.')
    graph, metadata = load_inference_graph(graph_path)
    report = json.loads(summary_path.read_text(encoding='utf-8'))
    if metadata.get('source_plot') != plot or report.get('source_plot') != plot:
        raise ValueError(f'P0 source plot mismatch for {plot}.')
    with np.load(learning_path, allow_pickle=False) as payload:
        point_gt = payload['point_gt']
        baseline = payload['baseline_predictions']
    if len(point_gt) != len(graph['point_node_id']) or len(baseline) != len(point_gt):
        raise ValueError(f'P0 source arrays are not aligned for {plot}.')
    return graph, metadata, point_gt, baseline


def _validate_output(output, method, plot):
    summary = output/'summary.json'
    if not summary.is_file():
        raise FileNotFoundError(summary)
    report = json.loads(summary.read_text(encoding='utf-8'))
    required = {
        'source_plot', 'method', 'baseline', 'oracle', 'num_nodes',
        'num_edges', 'recovered_trees', 'graph_coverage',
        'num_baseline_instances'}
    if required-set(report):
        raise ValueError(f'P0b artifact misses {sorted(required-set(report))}.')
    if report['source_plot'] != plot or report['method'] != method:
        raise ValueError('P0b artifact identity mismatch.')
    graph, metadata = load_inference_graph(output/'graph.npz')
    if metadata.get('source_plot') != plot or metadata.get('method') != method:
        raise ValueError('P0b graph metadata mismatch.')
    if len(graph['node_features']) != int(report['num_nodes']):
        raise ValueError('P0b node count mismatch.')
    return report


def _run_one(settings, method, plot, force=False):
    source, output = _paths(settings, method, plot)
    if not force:
        try:
            report = _validate_output(output, method, plot)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            pass
        else:
            print(f'SKIP {method}/{plot}: validated existing artifact.', flush=True)
            return report
    graph, metadata, point_gt, baseline = _validate_source(source, plot)
    supergraph, assignments = build_hierarchical_graph(
        graph, method, settings['coarsening'])
    report = _evaluate_oracle(
        supergraph, point_gt, baseline,
        float(settings['match_iou_threshold']),
        float(settings['min_precision_for_counted_fp']))
    targets = report.pop('targets')
    report.update({
        'source_plot': plot, 'split': 'validation', 'method': method,
        'source_num_nodes': int(len(graph['node_features'])),
        'compression_ratio': float(
            len(graph['node_features'])/max(len(supergraph['node_features']), 1)),
    })
    output.mkdir(parents=True, exist_ok=True)
    save_inference_graph(output/'graph.npz', supergraph, {
        'source_plot': plot, 'split': 'validation', 'method': method,
        'source_coordinate_sha256': metadata.get('coordinate_sha256', ''),
        'source_checkpoint': metadata.get('checkpoint', ''),
        'source_graph_version': metadata.get('graph_version', -1),
    })
    np.savez_compressed(
        output/'learning.npz',
        node_gt=targets['node_gt'], node_purity=targets['node_purity'],
        edge_label=targets['edge_label'], edge_valid=targets['edge_valid'],
        covered_gt_ids=targets['covered_gt_ids'],
        point_gt=np.asarray(point_gt, np.int64),
        baseline_predictions=np.asarray(baseline, np.int64),
        source_node_to_super=np.asarray(assignments, np.int64))
    (output/'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(
        f'DONE {method}/{plot}: {report["source_num_nodes"]:,} -> '
        f'{report["num_nodes"]:,} nodes, recovered={report["recovered_trees"]}, '
        f'F1={100*report["oracle"]["f1"]:.3f}%.', flush=True)
    return report


def _aggregate_metrics(rows, name):
    return detection_metrics(
        sum(int(row[name]['tp']) for row in rows),
        sum(int(row[name]['fp']) for row in rows),
        sum(int(row[name]['fn']) for row in rows))


def _method_result(rows, gate):
    baseline = _aggregate_metrics(rows, 'baseline')
    oracle = _aggregate_metrics(rows, 'oracle')
    for row in rows:
        row['f1_gain_pp'] = 100*(row['oracle']['f1']-row['baseline']['f1'])
    nodes = sum(int(row['num_nodes']) for row in rows)
    edges = sum(int(row['num_edges']) for row in rows)
    baseline_instances = sum(int(row['num_baseline_instances']) for row in rows)
    result = {
        'baseline': baseline, 'oracle': oracle,
        'f1_gain_pp': 100*(oracle['f1']-baseline['f1']),
        'completeness_gain_pp': 100*(
            oracle['completeness']-baseline['completeness']),
        'commission_increase_pp': 100*(
            oracle['commission']-baseline['commission']),
        'recovered_trees': sum(int(row['recovered_trees']) for row in rows),
        'graph_coverage': (
            sum(int(row['graph_covered_missed_trees']) for row in rows) /
            max(sum(int(row['missed_trees']) for row in rows), 1)),
        'num_nodes': nodes, 'num_edges': edges,
        'node_inflation': nodes/max(baseline_instances, 1),
        'average_out_degree': edges/max(nodes, 1),
        'mean_compression_ratio': float(np.mean([
            row['compression_ratio'] for row in rows])),
        'per_plot': rows,
    }
    result['gate'] = {
        'expected_validation_plots':
            len(rows) == int(gate['expected_validation_plots']),
        'oracle_f1_gain_passed':
            result['f1_gain_pp'] >= float(gate['min_f1_gain_pp']),
        'oracle_completeness_gain_passed':
            result['completeness_gain_pp'] >=
            float(gate['min_completeness_gain_pp']),
        'recovered_tree_count_passed':
            result['recovered_trees'] >= int(gate['min_recovered_trees']),
        'commission_not_worse':
            result['commission_increase_pp'] <=
            float(gate['max_commission_increase_pp']),
        'plot_consistency_passed':
            sum(row['f1_gain_pp'] >= -1e-9 for row in rows) >=
            int(gate['min_nonnegative_plots']),
        'missed_tree_coverage_passed':
            result['graph_coverage'] >= float(gate['min_graph_coverage']),
        'proposal_inflation_passed':
            result['node_inflation'] <= float(gate['max_node_inflation']),
        'degree_bound_passed':
            result['average_out_degree'] <=
            float(gate['max_average_out_degree']),
    }
    result['gate']['passed'] = bool(all(result['gate'].values()))
    return result


def _render(result):
    lines = [
        '# P0b 层次化 Superproposal Oracle', '',
        '- 直接复用 P0 固定验证集 artifacts；未重新运行 B1，未读取 Wytham。',
        '- 三种预聚合均不读取 GT；GT 只用于预聚合完成后的 Oracle 验收。',
        '- P0 的失败结论保持不变，本阶段检验新的层次化实例形成假设。', '',
        '| Method | Nodes | Compression | Inflation | Recall | Commission | F1 | F1 gain | Gate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for method, row in result['methods'].items():
        lines.append(
            f'| {method} | {row["num_nodes"]:,} | '
            f'{row["mean_compression_ratio"]:.2f}x | '
            f'{row["node_inflation"]:.2f}x | '
            f'{100*row["oracle"]["completeness"]:.3f}% | '
            f'{100*row["oracle"]["commission"]:.3f}% | '
            f'{100*row["oracle"]["f1"]:.3f}% | '
            f'{row["f1_gain_pp"]:+.3f} pp | {row["gate"]["passed"]} |')
    lines.extend(['', '## 每森林 F1 gain', '',
                  '| Method | G4N | G4W | L1N | O1N | O1W |',
                  '|---|---:|---:|---:|---:|---:|'])
    for method, row in result['methods'].items():
        gains = {item['source_plot']: item['f1_gain_pp']
                 for item in row['per_plot']}
        lines.append(
            f'| {method} | ' + ' | '.join(
                f'{gains[plot]:+.3f} pp'
                for plot in ('G4N', 'G4W', 'L1N', 'O1N', 'O1W')) + ' |')
    lines.extend(['', '## 决策', '',
                  f'- 推荐方法：**{result["recommended_method"]}**',
                  f'- Primary Gate：**{result["passed"]}**', ''])
    if result['passed']:
        lines.append('PASS：只对推荐的层次化图生成 P1 数据，再进入关系模型比较。')
    else:
        lines.append('STOP：层次化预聚合仍无足够上限，关闭 proposal-relation 路线。')
    lines.append('')
    return '\n'.join(lines)


def run(config_path, force=False):
    settings = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if 'wytham' in json.dumps(settings).lower():
        raise ValueError('P0b must not mention or read Wytham.')
    methods = list(settings['methods'])
    plots = list(settings['validation_plots'])
    output = Path(settings['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for method_index, method in enumerate(methods, 1):
        rows = []
        for plot_index, plot in enumerate(plots, 1):
            print(
                f'[{method_index}/{len(methods)}][{plot_index}/{len(plots)}] '
                f'{method}/{plot}', flush=True)
            rows.append(_run_one(settings, method, plot, force))
            gc.collect()
        all_results[method] = _method_result(rows, settings['gate'])
    eligible = [
        (method, row) for method, row in all_results.items()
        if row['gate']['passed']]
    eligible.sort(key=lambda item: (
        -item[1]['oracle']['f1'], -item[1]['oracle']['completeness'],
        item[1]['oracle']['commission'], item[1]['num_nodes'], item[0]))
    recommended = eligible[0][0] if eligible else None
    result = {
        'methods': all_results, 'recommended_method': recommended,
        'passed': recommended is not None,
    }
    (output/'summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = _render(result)
    (output/'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not result['passed']:
        raise RuntimeError(
            'P0b hierarchical proposal gate failed; do not run P1/P2.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, args.force)


if __name__ == '__main__':
    main()
