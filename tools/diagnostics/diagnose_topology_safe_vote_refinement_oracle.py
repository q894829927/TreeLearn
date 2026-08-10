"""Evaluate a topology-safe gate for cellwise TreeLearn vote refinement.

E2b is a new, preregistered diagnostic after the unconditional E2a cellwise
refinement failed its topology gate.  It keeps the fixed 0.6 m cells, the
candidate set, and HDBSCAN settings unchanged.  Ground truth is used only to
construct an Oracle binary gate on the five fixed validation forests.  Wytham
is explicitly forbidden.
"""

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.diagnostics.diagnose_cellwise_vote_refinement_oracle import (  # noqa: E402
    _array_digest,
    cellwise_shared_residual,
    cluster_votes,
    load_validation_rows,
    load_vote_artifact,
    seed_cluster_detection_metrics,
    vote_cells,
    vote_error_metrics,
    vote_topology_metrics,
)


def damping_mode_name(value):
    return f'damped_{float(value):.2f}'.replace('.', 'p')


def _positive_label_stats(inverse, labels, num_cells):
    positive = labels > 0
    minimum = np.full(num_cells, np.iinfo(np.int64).max, dtype=np.int64)
    maximum = np.full(num_cells, -1, dtype=np.int64)
    np.minimum.at(minimum, inverse[positive], labels[positive])
    np.maximum.at(maximum, inverse[positive], labels[positive])
    return minimum, maximum


def topology_safe_cell_gate(
        base_votes_xy, cell_corrections_xy, target_residual_xy,
        labels, cell_size):
    """Return a conservative GT-only binary gate shared by each source cell.

    A source cell is corrected only when all its candidates belong to one
    positive tree, the shared correction lowers its vote error, every proposed
    destination is empty or already contains only that tree, and no different
    source trees propose the same destination.  This is an upper-bound target,
    not a deployable inference rule.
    """
    base = np.asarray(base_votes_xy, dtype=np.float64)
    corrections = np.asarray(cell_corrections_xy, dtype=np.float64)
    residual = np.asarray(target_residual_xy, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if (base.shape != corrections.shape or base.shape != residual.shape or
            base.shape != (len(labels), 2)):
        raise ValueError('Topology-safe gate arrays are not aligned.')
    if not np.isfinite(base).all() or not np.isfinite(corrections).all():
        raise ValueError('Topology-safe gate requires finite votes.')
    if np.any(labels < 0) or not np.any(labels > 0):
        raise ValueError('Topology-safe gate requires complete GT labels.')

    base_keys, source_inverse, source_counts = vote_cells(base, cell_size)
    num_source_cells = len(source_counts)
    minimum_all = np.full(
        num_source_cells, np.iinfo(np.int64).max, dtype=np.int64)
    maximum_all = np.full(num_source_cells, -1, dtype=np.int64)
    np.minimum.at(minimum_all, source_inverse, labels)
    np.maximum.at(maximum_all, source_inverse, labels)
    source_pure_tree = (minimum_all == maximum_all) & (minimum_all > 0)
    source_tree = minimum_all.copy()

    base_error = np.linalg.norm(residual, axis=1)
    corrected_error = np.linalg.norm(residual - corrections, axis=1)
    base_error_sum = np.bincount(
        source_inverse, weights=base_error, minlength=num_source_cells)
    corrected_error_sum = np.bincount(
        source_inverse, weights=corrected_error,
        minlength=num_source_cells)
    improvement_tolerance = np.maximum(1e-9, 1e-7 * base_error_sum)
    error_improves = (
        corrected_error_sum < base_error_sum - improvement_tolerance)
    nonzero_correction = np.bincount(
        source_inverse,
        weights=np.linalg.norm(corrections, axis=1),
        minlength=num_source_cells) > 0
    eligible = source_pure_tree & error_improves & nonzero_correction

    eligible_points = eligible[source_inverse]
    safe_cells = eligible.copy()
    if np.any(eligible_points):
        proposed_votes = (
            base[eligible_points] + corrections[eligible_points])
        proposed_keys = np.floor(
            proposed_votes / float(cell_size)).astype(np.int64)
        combined_keys = np.vstack([base_keys, proposed_keys])
        _, combined_inverse = np.unique(
            combined_keys, axis=0, return_inverse=True)
        base_global = combined_inverse[:num_source_cells]
        proposed_global = combined_inverse[num_source_cells:]
        num_global_cells = int(combined_inverse.max()) + 1

        positive_min, positive_max = _positive_label_stats(
            source_inverse, labels, num_source_cells)
        base_positive_min = np.full(
            num_global_cells, np.iinfo(np.int64).max, dtype=np.int64)
        base_positive_max = np.full(
            num_global_cells, -1, dtype=np.int64)
        base_positive_min[base_global] = positive_min
        base_positive_max[base_global] = positive_max
        source_has_non_tree = np.bincount(
            source_inverse, weights=(labels == 0).astype(np.int64),
            minlength=num_source_cells) > 0
        base_has_non_tree = np.zeros(num_global_cells, dtype=bool)
        base_has_non_tree[base_global] = source_has_non_tree

        proposed_source_cells = source_inverse[eligible_points]
        proposed_tree_ids = source_tree[proposed_source_cells]
        proposed_min = np.full(
            num_global_cells, np.iinfo(np.int64).max, dtype=np.int64)
        proposed_max = np.full(num_global_cells, -1, dtype=np.int64)
        np.minimum.at(proposed_min, proposed_global, proposed_tree_ids)
        np.maximum.at(proposed_max, proposed_global, proposed_tree_ids)
        proposal_is_single_tree = (
            proposed_min[proposed_global] == proposed_max[proposed_global])

        existing_is_empty = base_positive_max[proposed_global] < 0
        existing_is_same_tree = (
            (base_positive_min[proposed_global] == proposed_tree_ids) &
            (base_positive_max[proposed_global] == proposed_tree_ids))
        destination_tree_safe = existing_is_empty | existing_is_same_tree
        destination_non_tree_safe = (
            ~base_has_non_tree[proposed_global] | existing_is_same_tree)
        point_is_safe = (
            proposal_is_single_tree & destination_tree_safe &
            destination_non_tree_safe)
        unsafe_source_cells = np.unique(
            proposed_source_cells[~point_is_safe])
        safe_cells[unsafe_source_cells] = False

    initial_safe_cells = safe_cells.copy()
    point_gate = safe_cells[source_inverse].astype(np.float32)
    initial_refined = apply_gated_correction(base, corrections, point_gate)
    base_topology = vote_topology_metrics(base, labels, cell_size)
    initial_topology = vote_topology_metrics(
        initial_refined, labels, cell_size)
    strict_fallback = bool(
        initial_topology['tree_point_cell_purity_mean'] + 1e-12 <
        base_topology['tree_point_cell_purity_mean'] or
        initial_topology['multi_tree_collision_cell_rate'] >
        base_topology['multi_tree_collision_cell_rate'] + 1e-12)
    if strict_fallback:
        proposed_keys = np.floor(
            (base + corrections) / float(cell_size)).astype(np.int64)
        base_point_keys = np.floor(
            base / float(cell_size)).astype(np.int64)
        point_stays_in_cell = np.all(
            proposed_keys == base_point_keys, axis=1)
        cell_stays_in_place = np.ones(num_source_cells, dtype=bool)
        np.logical_and.at(
            cell_stays_in_place, source_inverse, point_stays_in_cell)
        safe_cells &= cell_stays_in_place
        point_gate = safe_cells[source_inverse].astype(np.float32)
        strict_refined = apply_gated_correction(
            base, corrections, point_gate)
        strict_topology = vote_topology_metrics(
            strict_refined, labels, cell_size)
        if (strict_topology['tree_point_cell_purity_mean'] + 1e-12 <
                base_topology['tree_point_cell_purity_mean'] or
                strict_topology['multi_tree_collision_cell_rate'] >
                base_topology['multi_tree_collision_cell_rate'] + 1e-12):
            raise AssertionError(
                'Strict topology fallback failed to preserve cell metrics.')
    tree = labels > 0
    return point_gate, {
        'num_source_cells': int(num_source_cells),
        'num_pure_tree_cells': int(source_pure_tree.sum()),
        'num_error_improving_cells': int(
            (source_pure_tree & error_improves).sum()),
        'num_destination_safe_cells_before_fallback': int(
            initial_safe_cells.sum()),
        'num_destination_safe_cells': int(safe_cells.sum()),
        'rejected_mixed_or_non_tree_cells': int(
            (~source_pure_tree).sum()),
        'rejected_non_improving_cells': int(
            (source_pure_tree & ~error_improves).sum()),
        'rejected_destination_cells': int(
            (eligible & ~initial_safe_cells).sum()),
        'rejected_topology_fallback_cells': int(
            (initial_safe_cells & ~safe_cells).sum()),
        'strict_fallback_applied': strict_fallback,
        'safe_cell_rate': float(safe_cells.mean()),
        'corrected_candidate_rate': float(point_gate.mean()),
        'corrected_tree_candidate_rate': float(point_gate[tree].mean()),
        'gate_sha256': _array_digest(point_gate.astype(np.uint8)),
    }


def apply_gated_correction(base_votes, corrections, gate):
    base = np.asarray(base_votes, dtype=np.float64)
    correction = np.asarray(corrections, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64).reshape(-1)
    if base.shape != correction.shape or base.shape != (len(gate), 2):
        raise ValueError('Gated correction arrays are not aligned.')
    if np.any((gate < 0) | (gate > 1)):
        raise ValueError('Vote gate must be within [0, 1].')
    return base + correction * gate[:, None]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_e2a_reference(settings, expected_plots):
    summary_path = Path(settings['e2a_summary_path'])
    plot_dir = Path(settings['e2a_plot_dir'])
    if not summary_path.is_file() or not plot_dir.is_dir():
        raise FileNotFoundError(
            'E2a summary/plot caches are required before E2b.')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected_settings = {
        'cell_size': settings['cell_size'],
        'topology_cell_size': settings['topology_cell_size'],
        'run_hdbscan': settings['run_hdbscan'],
        'min_cluster_size': settings['min_cluster_size'],
        'min_iou': settings['min_iou'],
        'n_jobs': settings['n_jobs'],
    }
    for name, expected in expected_settings.items():
        observed = summary.get('settings', {}).get(name)
        if isinstance(expected, float):
            matches = observed is not None and np.isclose(
                float(observed), expected)
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f'E2a cache setting {name}={observed!r} does not match '
                f'E2b value {expected!r}.')
    references = {}
    for plot in expected_plots:
        path = plot_dir / f'{plot}.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing E2a plot cache: {path}')
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('source_plot') != plot:
            raise ValueError(f'E2a plot cache identity differs for {plot}.')
        clusters = {}
        for source_name, target_name in (
                ('base', 'base'),
                ('cellwise_shared', damping_mode_name(1.0)),
                ('pointwise_oracle', 'pointwise_oracle')):
            cluster = report.get('modes', {}).get(
                source_name, {}).get('cluster')
            if settings['run_hdbscan'] and cluster is None:
                raise ValueError(
                    f'E2a cache lacks {source_name} cluster metrics for {plot}.')
            if cluster is not None:
                clusters[target_name] = cluster
        references[plot] = {
            'artifact_identity': report.get('artifact_identity'),
            'clusters': clusters,
            'cache_sha256': _file_sha256(path),
        }
    reference_digest = hashlib.sha256(json.dumps({
        'summary_sha256': _file_sha256(summary_path),
        'plot_sha256': {
            name: row['cache_sha256'] for name, row in references.items()},
    }, sort_keys=True).encode('utf-8')).hexdigest()
    return references, reference_digest


def analyze_plot(row, settings, e2a_reference, logger=print):
    arrays, artifact_identity = load_vote_artifact(
        row, settings['data_root'])
    if e2a_reference.get('artifact_identity') != artifact_identity:
        raise ValueError(
            f'E2a artifact identity differs for {row["source_plot"]}.')
    base = arrays['base_votes_xy'].astype(np.float64)
    residual = arrays['target_vote_residual_xy'].astype(np.float64)
    targets = arrays['target_base_vote_xy'].astype(np.float64)
    labels = arrays['target_tree_id'].astype(np.int64)
    tree = arrays['target_is_tree'].astype(bool)
    corrections, cell_support = cellwise_shared_residual(
        base, residual, tree, settings['cell_size'])
    gate, gate_stats = topology_safe_cell_gate(
        base, corrections, residual, labels, settings['cell_size'])

    modes = {'base': base}
    for damping in settings['damping_factors']:
        modes[damping_mode_name(damping)] = (
            base + float(damping) * corrections)
    modes['topology_safe'] = apply_gated_correction(
        base, corrections, gate)
    pointwise = base.copy()
    pointwise[tree] += residual[tree]
    modes['pointwise_oracle'] = pointwise

    mode_metrics = {}
    cluster_digests = {}
    cached_clusters = e2a_reference.get('clusters', {})
    for name, votes in modes.items():
        mode_metrics[name] = {
            'error': vote_error_metrics(votes, targets, tree),
            'topology': vote_topology_metrics(
                votes, labels, settings['topology_cell_size']),
        }
        if settings['run_hdbscan']:
            if name in cached_clusters:
                mode_metrics[name]['cluster'] = dict(cached_clusters[name])
                mode_metrics[name]['cluster']['reused_e2a_cache'] = True
                continue
            start = time.time()
            logger(
                f'  HDBSCAN {name}: {len(base):,} seeds...', flush=True)
            predictions = cluster_votes(
                votes, settings['min_cluster_size'], settings['n_jobs'])
            cluster = seed_cluster_detection_metrics(
                labels, predictions, settings['min_iou'])
            cluster['seconds'] = float(time.time() - start)
            mode_metrics[name]['cluster'] = cluster
            cluster_digests[name] = _array_digest(predictions)
            logger(
                f'    F1={100 * cluster["f1"]:.3f}%, '
                f'recall={100 * cluster["tree_recall"]:.3f}%, '
                f'time={cluster["seconds"]:.1f}s', flush=True)
            del predictions
            gc.collect()

    base_metrics = mode_metrics['base']
    safe_metrics = mode_metrics['topology_safe']
    effects = {
        'mean_error_reduction_relative': float(
            (base_metrics['error']['mean_error_xy'] -
             safe_metrics['error']['mean_error_xy']) /
            max(base_metrics['error']['mean_error_xy'], 1e-12)),
        'p90_error_reduction_relative': float(
            (base_metrics['error']['p90_error_xy'] -
             safe_metrics['error']['p90_error_xy']) /
            max(base_metrics['error']['p90_error_xy'], 1e-12)),
        'cell_purity_gain': float(
            safe_metrics['topology']['tree_point_cell_purity_mean'] -
            base_metrics['topology']['tree_point_cell_purity_mean']),
        'multi_tree_collision_increase': float(
            safe_metrics['topology']['multi_tree_collision_cell_rate'] -
            base_metrics['topology']['multi_tree_collision_cell_rate']),
    }
    if settings['run_hdbscan']:
        base_cluster = base_metrics['cluster']
        safe_cluster = safe_metrics['cluster']
        effects.update({
            'cluster_f1_gain_pp': float(
                100 * (safe_cluster['f1'] - base_cluster['f1'])),
            'cluster_tree_recall_drop_pp': float(
                100 * (base_cluster['tree_recall'] -
                       safe_cluster['tree_recall'])),
            'cluster_commission_reduction_pp': float(
                100 * (base_cluster['commission'] -
                       safe_cluster['commission'])),
        })
        effects['plot_f1_nonnegative'] = bool(
            effects['cluster_f1_gain_pp'] >= 0)
    return {
        'source_plot': row['source_plot'],
        'num_candidates': int(len(base)),
        'num_tree_candidates': int(tree.sum()),
        'candidate_set_sha256': _array_digest(
            arrays['candidate_indices'].astype(np.int64)),
        'artifact_identity': artifact_identity,
        'e2a_cache_sha256': e2a_reference['cache_sha256'],
        'cell_support': cell_support,
        'gate_stats': gate_stats,
        'modes': mode_metrics,
        'cluster_digests': cluster_digests,
        'effects': effects,
    }


def _macro(reports, mode, group, metric):
    return float(np.mean([
        report['modes'][mode][group][metric] for report in reports]))


def aggregate_reports(reports, settings):
    mode_names = ['base'] + [
        damping_mode_name(value) for value in settings['damping_factors']]
    mode_names += ['topology_safe', 'pointwise_oracle']
    aggregate = {
        'num_validation_plots': int(len(reports)),
        'num_candidates': int(sum(
            report['num_candidates'] for report in reports)),
        'num_tree_candidates': int(sum(
            report['num_tree_candidates'] for report in reports)),
        'safe_cell_rate': float(np.mean([
            report['gate_stats']['safe_cell_rate'] for report in reports])),
        'corrected_tree_candidate_rate': float(np.mean([
            report['gate_stats']['corrected_tree_candidate_rate']
            for report in reports])),
        'modes': {},
    }
    for mode in mode_names:
        row = {
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
            row.update({
                'cluster_f1': _macro(reports, mode, 'cluster', 'f1'),
                'cluster_tree_recall': _macro(
                    reports, mode, 'cluster', 'tree_recall'),
                'cluster_commission': _macro(
                    reports, mode, 'cluster', 'commission'),
            })
        aggregate['modes'][mode] = row

    base = aggregate['modes']['base']
    safe = aggregate['modes']['topology_safe']
    effects = {
        'mean_error_reduction_relative': float(
            (base['mean_error_xy'] - safe['mean_error_xy']) /
            max(base['mean_error_xy'], 1e-12)),
        'p90_error_reduction_relative': float(
            (base['p90_error_xy'] - safe['p90_error_xy']) /
            max(base['p90_error_xy'], 1e-12)),
        'cell_purity_gain': float(
            safe['tree_point_cell_purity_mean'] -
            base['tree_point_cell_purity_mean']),
        'multi_tree_collision_increase': float(
            safe['multi_tree_collision_cell_rate'] -
            base['multi_tree_collision_cell_rate']),
        'max_pointwise_oracle_mean_error': float(max(
            report['modes']['pointwise_oracle']['error']['mean_error_xy']
            for report in reports)),
    }
    if settings['run_hdbscan']:
        effects.update({
            'cluster_f1_gain_pp': float(
                100 * (safe['cluster_f1'] - base['cluster_f1'])),
            'cluster_tree_recall_drop_pp': float(
                100 * (base['cluster_tree_recall'] -
                       safe['cluster_tree_recall'])),
            'cluster_commission_reduction_pp': float(
                100 * (base['cluster_commission'] -
                       safe['cluster_commission'])),
            'nonnegative_f1_plots': int(sum(
                report['effects']['cluster_f1_gain_pp'] >= 0
                for report in reports)),
            'worst_plot_f1_gain_pp': float(min(
                report['effects']['cluster_f1_gain_pp']
                for report in reports)),
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
        'cell_purity_preserved': (
            effects['cell_purity_gain'] >=
            -float(gate_cfg['max_cell_purity_drop'])),
        'collision_preserved': (
            effects['multi_tree_collision_increase'] <=
            float(gate_cfg['max_multi_tree_collision_increase'])),
    }
    if settings['run_hdbscan']:
        gate.update({
            'cluster_f1_gain_passed': (
                effects['cluster_f1_gain_pp'] >=
                float(gate_cfg['min_cluster_f1_gain_pp'])),
            'commission_reduction_passed': (
                effects['cluster_commission_reduction_pp'] >=
                float(gate_cfg['min_cluster_commission_reduction_pp'])),
            'tree_recall_preserved': (
                effects['cluster_tree_recall_drop_pp'] <=
                float(gate_cfg['max_cluster_tree_recall_drop_pp'])),
            'per_plot_nonnegative_passed': (
                effects['nonnegative_f1_plots'] >=
                int(gate_cfg['min_nonnegative_f1_plots'])),
            'worst_plot_drop_passed': (
                effects['worst_plot_f1_gain_pp'] >=
                -float(gate_cfg['max_single_plot_f1_drop_pp'])),
        })
    gate['passed'] = bool(all(gate.values()))
    return aggregate, effects, gate


def format_markdown(report):
    aggregate = report['aggregate']
    effects = report['effects']
    lines = [
        '# E2b 拓扑安全门控 Vote-Refinement Oracle', '',
        '- 数据：固定五个 validation forests；未读取 Wytham。',
        '- Cell size、候选种子和 HDBSCAN 参数与 E2a 完全一致。',
        '- E2a 的失败结论保持不变；本阶段是新的安全门控假设。',
        f'- 候选种子：{aggregate["num_candidates"]:,}',
        f'- 树种子：{aggregate["num_tree_candidates"]:,}',
        f'- 安全 Cell 比例：{100 * aggregate["safe_cell_rate"]:.3f}%',
        f'- 被修正树种子比例：'
        f'{100 * aggregate["corrected_tree_candidate_rate"]:.3f}%', '',
        '## Macro 指标', '',
        '| Mode | Mean error | P90 error | Cell purity | Collision | '
        'Cluster F1 | Commission | Tree recall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, row in aggregate['modes'].items():
        lines.append(
            f'| {name} | {row["mean_error_xy"]:.6f} m | '
            f'{row["p90_error_xy"]:.6f} m | '
            f'{100 * row["tree_point_cell_purity_mean"]:.3f}% | '
            f'{100 * row["multi_tree_collision_cell_rate"]:.3f}% | '
            f'{100 * row.get("cluster_f1", float("nan")):.3f}% | '
            f'{100 * row.get("cluster_commission", float("nan")):.3f}% | '
            f'{100 * row.get("cluster_tree_recall", float("nan")):.3f}% |')
    lines.extend(['', '## 拓扑安全模式效果', ''])
    for name, value in effects.items():
        lines.append(f'- {name}: {value:+.6f}')
    lines.extend([
        '', '## 每森林', '',
        '| Plot | Safe cells | Corrected trees | Mean reduction | '
        'F1 gain | Commission reduction | Recall drop |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in report['plots']:
        effect = row['effects']
        stats = row['gate_stats']
        lines.append(
            f'| {row["source_plot"]} | '
            f'{100 * stats["safe_cell_rate"]:.3f}% | '
            f'{100 * stats["corrected_tree_candidate_rate"]:.3f}% | '
            f'{100 * effect["mean_error_reduction_relative"]:+.3f}% | '
            f'{effect.get("cluster_f1_gain_pp", float("nan")):+.3f} pp | '
            f'{effect.get("cluster_commission_reduction_pp", float("nan")):+.3f} pp | '
            f'{effect.get("cluster_tree_recall_drop_pp", float("nan")):+.3f} pp |')
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend(['', (
        'PASS：实现 Cell-MLP Gate 控制组与 Cell-Query Attention Gate。'
        if report['gate']['passed'] else
        'STOP：关闭 vote-refinement/seed-attention 路线，不使用 Wytham 调参。')])
    return '\n'.join(lines) + '\n'


def write_per_plot_csv(path, reports):
    rows = []
    for report in reports:
        row = {
            'source_plot': report['source_plot'],
            'num_candidates': report['num_candidates'],
            **report['gate_stats'],
            **report['effects'],
        }
        rows.append(row)
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config_fingerprint(settings):
    relevant = {
        name: settings[name] for name in (
            'cell_size', 'topology_cell_size', 'damping_factors',
            'run_hdbscan', 'min_cluster_size', 'min_iou', 'n_jobs',
            'gate', 'e2a_reference_digest')}
    return hashlib.sha256(json.dumps(
        relevant, sort_keys=True).encode('utf-8')).hexdigest()


def run_settings(raw, config_path='<memory>', force=False):
    required = {
        'data_root', 'manifest_path', 'output_dir',
        'e2a_summary_path', 'e2a_plot_dir',
        'expected_validation_plots', 'cell_size', 'topology_cell_size',
        'damping_factors', 'run_hdbscan', 'min_cluster_size',
        'min_iou', 'n_jobs', 'gate'}
    missing = required - set(raw)
    if missing:
        raise ValueError(f'Missing E2b config fields: {sorted(missing)}')
    damping = [float(value) for value in raw['damping_factors']]
    if damping != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError(
            'E2b damping factors are preregistered as 0.25/0.5/0.75/1.0.')
    settings = {
        'data_root': raw['data_root'],
        'e2a_summary_path': raw['e2a_summary_path'],
        'e2a_plot_dir': raw['e2a_plot_dir'],
        'cell_size': float(raw['cell_size']),
        'topology_cell_size': float(raw['topology_cell_size']),
        'damping_factors': damping,
        'run_hdbscan': bool(raw['run_hdbscan']),
        'min_cluster_size': int(raw['min_cluster_size']),
        'min_iou': float(raw['min_iou']),
        'n_jobs': int(raw['n_jobs']),
        'gate': raw['gate'],
    }
    if not np.isclose(settings['cell_size'], 0.6):
        raise ValueError('E2b cell size is locked to 0.6 m.')
    if not np.isclose(
            settings['topology_cell_size'], settings['cell_size']):
        raise ValueError('E2b topology cell size must equal vote cell size.')
    if settings['min_cluster_size'] < 2:
        raise ValueError('min_cluster_size must be at least two.')
    expected = list(raw['expected_validation_plots'])
    rows = load_validation_rows(raw['manifest_path'], expected)
    references, reference_digest = load_e2a_reference(settings, expected)
    settings['e2a_reference_digest'] = reference_digest

    output_dir = Path(raw['output_dir'])
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _config_fingerprint(settings)
    reports = []
    for position, row in enumerate(rows, start=1):
        plot = row['source_plot']
        cache_path = plot_dir / f'{plot}.json'
        artifact_path = Path(row['artifact_path']).resolve()
        stat = artifact_path.stat()
        identity = {
            'path': str(artifact_path), 'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns)}
        reference = references[plot]
        if cache_path.is_file() and not force:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            if (cached.get('config_fingerprint') == fingerprint and
                    cached.get('artifact_identity') == identity and
                    cached.get('e2a_cache_sha256') ==
                    reference['cache_sha256']):
                print(
                    f'[{position}/{len(rows)}] SKIP {plot}: validated cache.',
                    flush=True)
                reports.append(cached)
                continue
        print(f'[{position}/{len(rows)}] Analyzing {plot}...', flush=True)
        report = analyze_plot(row, settings, reference)
        report['config_fingerprint'] = fingerprint
        cache_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding='utf-8')
        reports.append(report)
        print(
            f'  safe cells={100 * report["gate_stats"]["safe_cell_rate"]:.2f}%, '
            f'F1 gain={report["effects"].get("cluster_f1_gain_pp", float("nan")):+.2f} pp',
            flush=True)

    aggregate, effects, gate = aggregate_reports(reports, settings)
    report = {
        'config': str(config_path),
        'settings': {
            name: settings[name] for name in (
                'cell_size', 'topology_cell_size', 'damping_factors',
                'run_hdbscan', 'min_cluster_size', 'min_iou', 'n_jobs',
                'e2a_reference_digest')},
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
            'E2b topology-safe gate failed; close vote refinement.')


def run(config_path, force=False):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        raw = yaml.safe_load(file)
    return run_settings(raw, config_path=config_path, force=force)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the E2b topology-safe vote-refinement Oracle.')
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
