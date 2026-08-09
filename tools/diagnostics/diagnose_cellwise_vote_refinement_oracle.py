"""Evaluate a GT-only cellwise XY vote-refinement upper bound.

E2a keeps every TreeLearn base seed and all original clustering settings.  It
uses the exact offset targets exported by the pipeline to test whether one
shared XY residual per 0.6 m vote cell can improve the geometry seen by
HDBSCAN.  Wytham is explicitly forbidden in this diagnostic.
"""

import argparse
import csv
import gc
import itertools
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def vote_cells(votes_xy, cell_size):
    votes = np.asarray(votes_xy, dtype=np.float64)
    size = float(cell_size)
    if votes.ndim != 2 or votes.shape[1] != 2 or len(votes) == 0:
        raise ValueError('votes_xy must have non-empty shape [N, 2].')
    if not np.isfinite(votes).all() or size <= 0:
        raise ValueError('Vote cells require finite votes and positive size.')
    keys = np.floor(votes / size).astype(np.int64)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    return unique, inverse.astype(np.int64), counts.astype(np.int64)


def cellwise_shared_residual(
        votes_xy, target_residual_xy, tree_mask, cell_size):
    """Return one robust GT residual shared by every candidate in a cell."""
    votes = np.asarray(votes_xy, dtype=np.float64)
    residual = np.asarray(target_residual_xy, dtype=np.float64)
    tree = np.asarray(tree_mask, dtype=bool).reshape(-1)
    if votes.shape != residual.shape or votes.shape != (len(tree), 2):
        raise ValueError('Cell residual arrays are not aligned.')
    if not np.isfinite(residual).all() or not tree.any():
        raise ValueError('Cell residuals require finite supervised tree data.')
    _, inverse, counts = vote_cells(votes, cell_size)
    tree_indices = np.flatnonzero(tree)
    tree_cells = inverse[tree_indices]
    order = np.argsort(tree_cells, kind='stable')
    ordered_indices = tree_indices[order]
    ordered_cells = tree_cells[order]
    starts = np.r_[
        0, np.flatnonzero(ordered_cells[1:] != ordered_cells[:-1]) + 1]
    ends = np.r_[starts[1:], len(ordered_cells)]
    cell_residuals = np.zeros((len(counts), 2), dtype=np.float64)
    support_counts = np.zeros(len(counts), dtype=np.int64)
    for start, end in zip(starts, ends):
        cell_id = int(ordered_cells[start])
        indices = ordered_indices[start:end]
        cell_residuals[cell_id] = np.median(residual[indices], axis=0)
        support_counts[cell_id] = len(indices)
    corrections = cell_residuals[inverse]
    return corrections, {
        'num_cells': int(len(counts)),
        'num_tree_supported_cells': int(np.count_nonzero(support_counts)),
        'tree_supported_cell_rate': float(
            np.count_nonzero(support_counts) / len(counts)),
        'median_tree_support': float(np.median(
            support_counts[support_counts > 0])),
    }


def vote_error_metrics(votes_xy, target_votes_xy, tree_mask):
    votes = np.asarray(votes_xy, dtype=np.float64)
    targets = np.asarray(target_votes_xy, dtype=np.float64)
    tree = np.asarray(tree_mask, dtype=bool).reshape(-1)
    if votes.shape != targets.shape or votes.shape != (len(tree), 2):
        raise ValueError('Vote error arrays are not aligned.')
    if not tree.any():
        raise ValueError('Vote error metrics require supervised tree seeds.')
    errors = np.linalg.norm(votes[tree] - targets[tree], axis=1)
    return {
        'tree_seed_count': int(tree.sum()),
        'mean_error_xy': float(errors.mean()),
        'median_error_xy': float(np.median(errors)),
        'p90_error_xy': float(np.quantile(errors, 0.90)),
        'within_0p30m_rate': float(np.mean(errors <= 0.30)),
        'within_0p60m_rate': float(np.mean(errors <= 0.60)),
    }


def vote_topology_metrics(votes_xy, labels, cell_size):
    votes = np.asarray(votes_xy, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if votes.shape != (len(labels), 2):
        raise ValueError('Vote topology arrays are not aligned.')
    tree = labels > 0
    if not tree.any():
        raise ValueError('Vote topology requires supervised tree seeds.')
    _, inverse, counts = vote_cells(votes, cell_size)
    num_cells = len(counts)
    tree_indices = np.flatnonzero(tree)
    tree_pairs = np.column_stack([inverse[tree], labels[tree]])
    unique_pairs, pair_inverse, pair_counts = np.unique(
        tree_pairs, axis=0, return_inverse=True, return_counts=True)
    tree_ids_per_cell = np.bincount(
        unique_pairs[:, 0], minlength=num_cells)
    tree_cells = tree_ids_per_cell > 0
    own_tree_purity = (
        pair_counts[pair_inverse] / counts[inverse[tree_indices]])
    _, cells_per_tree = np.unique(
        unique_pairs[:, 1], return_counts=True)
    candidate_in_tree_cell = tree_cells[inverse]
    return {
        'num_vote_cells': int(num_cells),
        'tree_occupied_cells': int(tree_cells.sum()),
        'tree_point_cell_purity_mean': float(own_tree_purity.mean()),
        'multi_tree_collision_cell_rate': float(
            np.mean(tree_ids_per_cell[tree_cells] >= 2)),
        'non_tree_rate_in_tree_cells': float(np.mean(
            labels[candidate_in_tree_cell] == 0)),
        'mean_cells_per_tree': float(cells_per_tree.mean()),
        'p90_cells_per_tree': float(np.quantile(cells_per_tree, 0.90)),
    }


def maximize_assignment(values):
    """Use SciPy Hungarian, with an exact tiny-matrix test fallback."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError('Assignment values must be a matrix.')
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(matrix, maximize=True)
    except ModuleNotFoundError:
        rows, columns = matrix.shape
        matched = min(rows, columns)
        if matched > 8:
            raise ModuleNotFoundError(
                'SciPy is required for production Hungarian matching.')
        best_score = -np.inf
        best_rows = best_columns = None
        if rows <= columns:
            fixed_rows = np.arange(rows, dtype=np.int64)
            for permutation in itertools.permutations(range(columns), rows):
                selected_columns = np.asarray(permutation, dtype=np.int64)
                score = float(matrix[fixed_rows, selected_columns].sum())
                if score > best_score:
                    best_score = score
                    best_rows = fixed_rows.copy()
                    best_columns = selected_columns
        else:
            fixed_columns = np.arange(columns, dtype=np.int64)
            for permutation in itertools.permutations(range(rows), columns):
                selected_rows = np.asarray(permutation, dtype=np.int64)
                score = float(matrix[selected_rows, fixed_columns].sum())
                if score > best_score:
                    best_score = score
                    best_rows = selected_rows
                    best_columns = fixed_columns.copy()
        return best_rows, best_columns


def seed_cluster_detection_metrics(
        ground_truth_labels, predicted_labels, min_iou=0.50):
    """Hungarian seed-cluster proxy; unmatched clusters count as false."""
    gt = np.asarray(ground_truth_labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
    if len(gt) != len(pred) or not 0 <= float(min_iou) <= 1:
        raise ValueError('Seed cluster metric inputs are invalid.')
    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred >= 0])
    num_gt, num_pred = len(gt_ids), len(pred_ids)
    if num_gt == 0:
        raise ValueError('Seed cluster metrics require GT trees.')
    if num_pred == 0:
        return {
            'num_gt_trees': int(num_gt), 'num_pred_clusters': 0,
            'tp': 0, 'fp': 0, 'fn': int(num_gt),
            'tree_recall': 0.0, 'commission': 0.0, 'f1': 0.0,
            'mean_matched_iou': 0.0,
            'noise_rate': float(np.mean(pred < 0)),
        }
    gt_position = np.searchsorted(gt_ids, gt)
    pred_position = np.searchsorted(pred_ids, pred)
    valid = (gt > 0) & (pred >= 0)
    flat = (
        pred_position[valid].astype(np.int64) * num_gt +
        gt_position[valid].astype(np.int64))
    intersection = np.bincount(
        flat, minlength=num_pred * num_gt).reshape(num_pred, num_gt)
    pred_sizes = np.bincount(
        pred_position[pred >= 0], minlength=num_pred).reshape(-1, 1)
    gt_sizes = np.bincount(
        gt_position[gt > 0], minlength=num_gt).reshape(1, -1)
    union = pred_sizes + gt_sizes - intersection
    iou = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)
    matched_pred, matched_gt = maximize_assignment(iou)
    matched_iou = iou[matched_pred, matched_gt]
    accepted = matched_iou > float(min_iou)
    accepted_iou = matched_iou[accepted]
    tp = int(accepted.sum())
    fp = int(num_pred - tp)
    fn = int(num_gt - tp)
    denominator = 2 * tp + fp + fn
    return {
        'num_gt_trees': int(num_gt),
        'num_pred_clusters': int(num_pred),
        'tp': tp, 'fp': fp, 'fn': fn,
        'tree_recall': float(tp / num_gt),
        'commission': float(fp / max(tp + fp, 1)),
        'f1': float(2 * tp / max(denominator, 1)),
        'mean_matched_iou': float(
            accepted_iou.mean() if len(accepted_iou) else 0.0),
        'noise_rate': float(np.mean(pred < 0)),
    }


def cluster_votes(votes_xy, min_cluster_size, n_jobs=-1):
    votes = np.asarray(votes_xy, dtype=np.float64)
    if votes.ndim != 2 or votes.shape[1] != 2:
        raise ValueError('HDBSCAN votes must have shape [N, 2].')
    if len(votes) < int(min_cluster_size):
        return np.full(len(votes), -1, dtype=np.int64)
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        n_jobs=int(n_jobs)).fit_predict(votes).astype(np.int64)


def _array_digest(values):
    return hashlib.sha256(
        np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def load_validation_rows(manifest_path, expected_plots):
    with Path(manifest_path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    validation = [row for row in rows if row['split'] == 'validation']
    names = [row['source_plot'] for row in validation]
    if names != list(expected_plots):
        raise ValueError(
            f'Validation forests differ from preregistration: {names}.')
    if any('wytham' in str(row).lower() for row in validation):
        raise ValueError('E2a must not read Wytham artifacts.')
    return validation


def load_vote_artifact(row, data_root):
    path = Path(row['artifact_path']).resolve()
    root = Path(data_root).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f'Invalid E2a artifact path: {path}')
    required = {
        'base_votes_xy', 'target_base_vote_xy',
        'target_vote_residual_xy', 'target_vote_error_xy',
        'target_is_tree', 'target_tree_id', 'target_valid',
        'target_coverage_critical',
        'candidate_indices',
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f'{path} lacks exact E2a targets: {sorted(missing)}. '
                'Regenerate data/seed_vote_refinement first.')
        arrays = {name: data[name] for name in required}
    count = len(arrays['base_votes_xy'])
    if count != int(row['num_candidates']):
        raise ValueError(f'Manifest count differs for {path}.')
    candidate_indices = np.asarray(arrays['candidate_indices'])
    if (candidate_indices.shape != (count,) or
            len(np.unique(candidate_indices)) != count or
            np.any(np.diff(candidate_indices) <= 0)):
        raise ValueError(
            'candidate_indices must be unique and strictly increasing.')
    for name in ('base_votes_xy', 'target_base_vote_xy',
                 'target_vote_residual_xy'):
        if arrays[name].shape != (count, 2):
            raise ValueError(f'{name} must have shape [N, 2].')
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'{name} contains non-finite values.')
    tree = arrays['target_is_tree'].astype(bool)
    valid = arrays['target_valid'].astype(bool)
    if not valid.all() or not tree.any():
        raise ValueError('E2a requires fully labeled validation artifacts.')
    reconstructed = (
        arrays['base_votes_xy'][tree] +
        arrays['target_vote_residual_xy'][tree])
    if not np.allclose(
            reconstructed, arrays['target_base_vote_xy'][tree], atol=1e-5):
        raise ValueError('Exact vector vote targets are inconsistent.')
    observed_error = np.linalg.norm(
        arrays['target_vote_residual_xy'][tree], axis=1)
    if not np.allclose(
            observed_error, arrays['target_vote_error_xy'][tree], atol=1e-5):
        raise ValueError('Vector and scalar vote targets disagree.')
    stat = path.stat()
    return arrays, {
        'path': str(path), 'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def analyze_plot(row, settings, logger=print):
    arrays, artifact_identity = load_vote_artifact(
        row, settings['data_root'])
    base = arrays['base_votes_xy'].astype(np.float64)
    targets = arrays['target_base_vote_xy'].astype(np.float64)
    residual = arrays['target_vote_residual_xy'].astype(np.float64)
    tree = arrays['target_is_tree'].astype(bool)
    labels = arrays['target_tree_id'].astype(np.int64)
    global_residual = np.median(residual[tree], axis=0)
    global_votes = base + global_residual
    cell_corrections, cell_support = cellwise_shared_residual(
        base, residual, tree, settings['cell_size'])
    cell_votes = base + cell_corrections
    point_votes = base.copy()
    point_votes[tree] += residual[tree]
    modes = {
        'base': base,
        'global_shared': global_votes,
        'cellwise_shared': cell_votes,
        'pointwise_oracle': point_votes,
    }
    mode_metrics = {}
    for name, votes in modes.items():
        mode_metrics[name] = {
            'error': vote_error_metrics(votes, targets, tree),
            'topology': vote_topology_metrics(
                votes, labels, settings['topology_cell_size']),
        }

    cluster_digests = {}
    if settings['run_hdbscan']:
        clustered = {}
        for name in ('base', 'cellwise_shared', 'pointwise_oracle'):
            start = time.time()
            logger(
                f'  HDBSCAN {name}: {len(base):,} seeds...', flush=True)
            predictions = cluster_votes(
                modes[name], settings['min_cluster_size'],
                settings['n_jobs'])
            mode_metrics[name]['cluster'] = seed_cluster_detection_metrics(
                labels, predictions, settings['min_iou'])
            mode_metrics[name]['cluster']['seconds'] = float(
                time.time() - start)
            cluster_digests[name] = _array_digest(predictions)
            clustered[name] = predictions
            logger(
                f'    F1={100 * mode_metrics[name]["cluster"]["f1"]:.3f}%, '
                f'recall={100 * mode_metrics[name]["cluster"]["tree_recall"]:.3f}%, '
                f'time={mode_metrics[name]["cluster"]["seconds"]:.1f}s',
                flush=True)
        mode_metrics['global_shared']['cluster'] = dict(
            mode_metrics['base']['cluster'])
        mode_metrics['global_shared']['cluster']['reused_translation'] = True
        cluster_digests['global_shared'] = cluster_digests['base']
        del clustered
        gc.collect()

    base_error = mode_metrics['base']['error']
    cell_error = mode_metrics['cellwise_shared']['error']
    point_error = mode_metrics['pointwise_oracle']['error']
    base_topology = mode_metrics['base']['topology']
    cell_topology = mode_metrics['cellwise_shared']['topology']
    point_improvement = (
        base_error['mean_error_xy'] - point_error['mean_error_xy'])
    effects = {
        'mean_error_reduction_relative': float(
            (base_error['mean_error_xy'] - cell_error['mean_error_xy']) /
            max(base_error['mean_error_xy'], 1e-12)),
        'p90_error_reduction_relative': float(
            (base_error['p90_error_xy'] - cell_error['p90_error_xy']) /
            max(base_error['p90_error_xy'], 1e-12)),
        'pointwise_improvement_retained': float(
            (base_error['mean_error_xy'] - cell_error['mean_error_xy']) /
            max(point_improvement, 1e-12)),
        'cell_purity_gain': float(
            cell_topology['tree_point_cell_purity_mean'] -
            base_topology['tree_point_cell_purity_mean']),
        'multi_tree_collision_increase': float(
            cell_topology['multi_tree_collision_cell_rate'] -
            base_topology['multi_tree_collision_cell_rate']),
    }
    if settings['run_hdbscan']:
        base_cluster = mode_metrics['base']['cluster']
        cell_cluster = mode_metrics['cellwise_shared']['cluster']
        effects.update({
            'cluster_f1_gain_pp': float(
                100 * (cell_cluster['f1'] - base_cluster['f1'])),
            'cluster_tree_recall_drop_pp': float(
                100 * (base_cluster['tree_recall'] -
                       cell_cluster['tree_recall'])),
            'cluster_commission_reduction_pp': float(
                100 * (base_cluster['commission'] -
                       cell_cluster['commission'])),
        })
    effects['plot_win'] = bool(
        effects['mean_error_reduction_relative'] > 0 and
        effects['p90_error_reduction_relative'] >= 0 and
        (not settings['run_hdbscan'] or (
            effects['cluster_f1_gain_pp'] >= 0 and
            effects['cluster_tree_recall_drop_pp'] <= 0)))
    return {
        'source_plot': row['source_plot'],
        'num_candidates': int(len(base)),
        'num_tree_candidates': int(tree.sum()),
        'num_critical_candidates': int(
            arrays['target_coverage_critical'].sum()),
        'candidate_set_sha256': _array_digest(
            arrays['candidate_indices'].astype(np.int64)),
        'artifact_identity': artifact_identity,
        'cell_support': cell_support,
        'global_residual_xy': global_residual.tolist(),
        'modes': mode_metrics,
        'cluster_digests': cluster_digests,
        'effects': effects,
    }


def _macro(reports, mode, group, metric):
    return float(np.mean([
        report['modes'][mode][group][metric] for report in reports]))


def aggregate_reports(reports, settings):
    modes = ('base', 'global_shared', 'cellwise_shared', 'pointwise_oracle')
    aggregate = {
        'num_validation_plots': int(len(reports)),
        'num_candidates': int(sum(
            report['num_candidates'] for report in reports)),
        'num_tree_candidates': int(sum(
            report['num_tree_candidates'] for report in reports)),
        'modes': {},
    }
    for mode in modes:
        aggregate['modes'][mode] = {
            'mean_error_xy': _macro(
                reports, mode, 'error', 'mean_error_xy'),
            'median_error_xy': _macro(
                reports, mode, 'error', 'median_error_xy'),
            'p90_error_xy': _macro(
                reports, mode, 'error', 'p90_error_xy'),
            'tree_point_cell_purity_mean': _macro(
                reports, mode, 'topology',
                'tree_point_cell_purity_mean'),
            'multi_tree_collision_cell_rate': _macro(
                reports, mode, 'topology',
                'multi_tree_collision_cell_rate'),
        }
        if settings['run_hdbscan']:
            aggregate['modes'][mode].update({
                'cluster_f1': _macro(
                    reports, mode, 'cluster', 'f1'),
                'cluster_tree_recall': _macro(
                    reports, mode, 'cluster', 'tree_recall'),
                'cluster_commission': _macro(
                    reports, mode, 'cluster', 'commission'),
            })
    base = aggregate['modes']['base']
    cell = aggregate['modes']['cellwise_shared']
    point = aggregate['modes']['pointwise_oracle']
    point_improvement = base['mean_error_xy'] - point['mean_error_xy']
    effects = {
        'mean_error_reduction_relative': float(
            (base['mean_error_xy'] - cell['mean_error_xy']) /
            max(base['mean_error_xy'], 1e-12)),
        'p90_error_reduction_relative': float(
            (base['p90_error_xy'] - cell['p90_error_xy']) /
            max(base['p90_error_xy'], 1e-12)),
        'pointwise_improvement_retained': float(
            (base['mean_error_xy'] - cell['mean_error_xy']) /
            max(point_improvement, 1e-12)),
        'cell_purity_gain': float(
            cell['tree_point_cell_purity_mean'] -
            base['tree_point_cell_purity_mean']),
        'multi_tree_collision_increase': float(
            cell['multi_tree_collision_cell_rate'] -
            base['multi_tree_collision_cell_rate']),
        'plot_wins': int(sum(
            report['effects']['plot_win'] for report in reports)),
        'max_pointwise_oracle_mean_error': float(max(
            report['modes']['pointwise_oracle']['error']['mean_error_xy']
            for report in reports)),
    }
    if settings['run_hdbscan']:
        effects.update({
            'cluster_f1_gain_pp': float(
                100 * (cell['cluster_f1'] - base['cluster_f1'])),
            'cluster_tree_recall_drop_pp': float(
                100 * (base['cluster_tree_recall'] -
                       cell['cluster_tree_recall'])),
            'cluster_commission_reduction_pp': float(
                100 * (base['cluster_commission'] -
                       cell['cluster_commission'])),
        })
    gate_cfg = settings['gate']
    gate = {
        'expected_validation_plots_present': (
            len(reports) == int(gate_cfg['expected_validation_plots'])),
        'exact_pointwise_ceiling_passed': (
            effects['max_pointwise_oracle_mean_error'] <=
            float(gate_cfg['max_pointwise_oracle_mean_error'])),
        'mean_error_reduction_passed': (
            effects['mean_error_reduction_relative'] >=
            float(gate_cfg['min_mean_error_reduction_relative'])),
        'p90_error_reduction_passed': (
            effects['p90_error_reduction_relative'] >=
            float(gate_cfg['min_p90_error_reduction_relative'])),
        'pointwise_improvement_retained': (
            effects['pointwise_improvement_retained'] >=
            float(gate_cfg['min_pointwise_improvement_retained'])),
        'cell_purity_not_worse': (
            effects['cell_purity_gain'] >=
            -float(gate_cfg['max_cell_purity_drop'])),
        'collision_not_worse': (
            effects['multi_tree_collision_increase'] <=
            float(gate_cfg['max_multi_tree_collision_increase'])),
        'per_plot_consistency_passed': (
            effects['plot_wins'] >= int(gate_cfg['min_plot_wins'])),
    }
    if settings['run_hdbscan']:
        gate.update({
            'cluster_f1_gain_passed': (
                effects['cluster_f1_gain_pp'] >=
                float(gate_cfg['min_cluster_f1_gain_pp'])),
            'cluster_tree_recall_preserved': (
                effects['cluster_tree_recall_drop_pp'] <=
                float(gate_cfg['max_cluster_tree_recall_drop_pp'])),
        })
    gate['passed'] = bool(all(gate.values()))
    return aggregate, effects, gate


def format_markdown(report):
    aggregate = report['aggregate']
    effects = report['effects']
    lines = [
        '# E2a Cellwise Vote-Refinement Oracle', '',
        '- 数据：新生成的固定 validation artifacts；未读取 Wytham。',
        '- 不删除种子，不修改 HDBSCAN 参数。',
        f'- Vote cell：{report["settings"]["cell_size"]:.3f} m',
        f'- 候选种子：{aggregate["num_candidates"]:,}',
        f'- 树种子：{aggregate["num_tree_candidates"]:,}', '',
        '## Macro 指标', '',
        '| Mode | Mean error | Median error | P90 error | Cell purity | '
        'Collision | Cluster F1 | Tree recall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, row in aggregate['modes'].items():
        cluster_f1 = row.get('cluster_f1', float('nan'))
        cluster_recall = row.get('cluster_tree_recall', float('nan'))
        lines.append(
            f'| {name} | {row["mean_error_xy"]:.6f} m | '
            f'{row["median_error_xy"]:.6f} m | '
            f'{row["p90_error_xy"]:.6f} m | '
            f'{100 * row["tree_point_cell_purity_mean"]:.3f}% | '
            f'{100 * row["multi_tree_collision_cell_rate"]:.3f}% | '
            f'{100 * cluster_f1:.3f}% | {100 * cluster_recall:.3f}% |')
    lines.extend(['', '## 效果', ''])
    for name, value in effects.items():
        lines.append(f'- {name}: {value:+.6f}')
    lines.extend([
        '', '## 每森林', '',
        '| Plot | Seeds | Mean reduction | P90 reduction | '
        'F1 gain | Recall drop | Win |',
        '|---|---:|---:|---:|---:|---:|---|',
    ])
    for row in report['plots']:
        effect = row['effects']
        lines.append(
            f'| {row["source_plot"]} | {row["num_candidates"]:,} | '
            f'{100 * effect["mean_error_reduction_relative"]:+.3f}% | '
            f'{100 * effect["p90_error_reduction_relative"]:+.3f}% | '
            f'{effect.get("cluster_f1_gain_pp", float("nan")):+.3f} pp | '
            f'{effect.get("cluster_tree_recall_drop_pp", float("nan")):+.3f} pp | '
            f'{effect["plot_win"]} |')
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend(['', (
        'PASS：实现参数匹配的 Cell-MLP residual 与 Cell-Query residual。'
        if report['gate']['passed'] else
        'STOP：关闭 Cellwise vote-refinement，不读取 Wytham 调参。')])
    return '\n'.join(lines) + '\n'


def write_per_plot_csv(path, reports):
    rows = []
    for report in reports:
        row = {
            'source_plot': report['source_plot'],
            'num_candidates': report['num_candidates'],
            **report['effects'],
        }
        for mode in ('base', 'cellwise_shared', 'pointwise_oracle'):
            row[f'{mode}_mean_error_xy'] = report[
                'modes'][mode]['error']['mean_error_xy']
            row[f'{mode}_p90_error_xy'] = report[
                'modes'][mode]['error']['p90_error_xy']
            if 'cluster' in report['modes'][mode]:
                row[f'{mode}_cluster_f1'] = report[
                    'modes'][mode]['cluster']['f1']
                row[f'{mode}_cluster_tree_recall'] = report[
                    'modes'][mode]['cluster']['tree_recall']
        rows.append(row)
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config_fingerprint(settings):
    relevant = {
        name: settings[name] for name in (
            'cell_size', 'topology_cell_size', 'run_hdbscan',
            'min_cluster_size', 'min_iou', 'n_jobs', 'gate')}
    return hashlib.sha256(json.dumps(
        relevant, sort_keys=True).encode('utf-8')).hexdigest()


def run_settings(raw, config_path='<memory>', force=False):
    required = {
        'data_root', 'manifest_path', 'output_dir',
        'expected_validation_plots', 'cell_size', 'topology_cell_size',
        'run_hdbscan', 'min_cluster_size', 'min_iou', 'n_jobs', 'gate'}
    missing = required - set(raw)
    if missing:
        raise ValueError(f'Missing E2a config fields: {sorted(missing)}')
    settings = {
        'data_root': raw['data_root'],
        'cell_size': float(raw['cell_size']),
        'topology_cell_size': float(raw['topology_cell_size']),
        'run_hdbscan': bool(raw['run_hdbscan']),
        'min_cluster_size': int(raw['min_cluster_size']),
        'min_iou': float(raw['min_iou']),
        'n_jobs': int(raw['n_jobs']),
        'gate': raw['gate'],
    }
    if settings['cell_size'] <= 0 or settings['topology_cell_size'] <= 0:
        raise ValueError('E2a cell sizes must be positive.')
    if settings['min_cluster_size'] < 2:
        raise ValueError('min_cluster_size must be at least two.')
    expected = list(raw['expected_validation_plots'])
    rows = load_validation_rows(raw['manifest_path'], expected)
    output_dir = Path(raw['output_dir'])
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _config_fingerprint(settings)
    reports = []
    for position, row in enumerate(rows, start=1):
        cache_path = plot_dir / f'{row["source_plot"]}.json'
        artifact_path = Path(row['artifact_path']).resolve()
        stat = artifact_path.stat()
        identity = {
            'path': str(artifact_path), 'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns)}
        if cache_path.is_file() and not force:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            if (cached.get('config_fingerprint') == fingerprint and
                    cached.get('artifact_identity') == identity):
                print(
                    f'[{position}/{len(rows)}] SKIP '
                    f'{row["source_plot"]}: validated cache.', flush=True)
                reports.append(cached)
                continue
        print(
            f'[{position}/{len(rows)}] Analyzing {row["source_plot"]}...',
            flush=True)
        report = analyze_plot(row, settings)
        report['config_fingerprint'] = fingerprint
        cache_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding='utf-8')
        reports.append(report)
        print(
            f'  mean reduction='
            f'{100 * report["effects"]["mean_error_reduction_relative"]:+.2f}%, '
            f'cluster F1 gain='
            f'{report["effects"].get("cluster_f1_gain_pp", float("nan")):+.2f} pp, '
            f'win={report["effects"]["plot_win"]}', flush=True)
    aggregate, effects, gate = aggregate_reports(reports, settings)
    report = {
        'config': str(config_path),
        'settings': {
            name: settings[name] for name in (
                'cell_size', 'topology_cell_size', 'run_hdbscan',
                'min_cluster_size', 'min_iou', 'n_jobs')},
        'plots': reports,
        'aggregate': aggregate,
        'effects': effects,
        'gate': gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    write_per_plot_csv(output_dir / 'per_plot_metrics.csv', reports)
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'E2a vote-refinement gate failed; close this route.')


def run(config_path, force=False):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        raw = yaml.safe_load(file)
    return run_settings(raw, config_path=config_path, force=force)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the fixed-validation E2a vote-refinement Oracle.')
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--force', action='store_true',
        help='Ignore completed per-plot caches and recompute all plots.')
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
