"""Diagnose whether vote-cell granularity can improve seed selection.

E1d is an offline, GT-only Oracle over the five fixed E1a validation forests.
It never runs on Wytham and never changes TreeLearn predictions.  The purpose is
to decide whether a cell/superpoint query is a scientifically justified next
representation after pointwise relation attention failed E1c.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def vote_cells(base_votes_xy, cell_size):
    votes = np.asarray(base_votes_xy, dtype=np.float64)
    size = float(cell_size)
    if votes.ndim != 2 or votes.shape[1] != 2 or len(votes) == 0:
        raise ValueError('base_votes_xy must have non-empty shape [N, 2].')
    if size <= 0:
        raise ValueError('cell_size must be positive.')
    cells = np.floor(votes / size).astype(np.int64)
    unique, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True)
    return unique, inverse.astype(np.int64), counts.astype(np.int64)


def allocate_weighted_cell_quotas(
        counts, keep_count, mandatory_counts=None, density_exponent=0.5,
        preserve_nonempty=True):
    """Allocate an exact capped budget with deterministic weighted waterfill."""
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if len(counts) == 0 or np.any(counts < 1):
        raise ValueError('Cell counts must be non-empty and positive.')
    keep_count = int(keep_count)
    total = int(counts.sum())
    if not 1 <= keep_count <= total:
        raise ValueError('keep_count must be within the candidate count.')
    exponent = float(density_exponent)
    if not 0 <= exponent <= 1:
        raise ValueError('density_exponent must be in [0, 1].')
    if mandatory_counts is None:
        mandatory = np.zeros(len(counts), dtype=np.int64)
    else:
        mandatory = np.asarray(mandatory_counts, dtype=np.int64).reshape(-1)
    if mandatory.shape != counts.shape:
        raise ValueError('mandatory_counts and counts are not aligned.')
    if np.any((mandatory < 0) | (mandatory > counts)):
        raise ValueError('mandatory_counts are outside cell capacities.')

    minimum = mandatory.copy()
    if preserve_nonempty:
        if keep_count < len(counts):
            raise RuntimeError(
                'The locked budget cannot preserve every occupied cell.')
        minimum = np.maximum(minimum, 1)
    if int(minimum.sum()) > keep_count:
        raise RuntimeError(
            'Mandatory seed-cell constraints exceed the locked budget.')

    quotas = minimum.copy()
    remaining = keep_count - int(quotas.sum())
    weights = np.power(counts.astype(np.float64), exponent)
    # Cells hitting capacity are removed and the remaining mass is redistributed.
    while remaining:
        capacity = counts - quotas
        active = np.flatnonzero(capacity > 0)
        if len(active) == 0:
            raise RuntimeError('Cell quota waterfill exhausted all capacity.')
        active_weights = weights[active]
        raw = remaining * active_weights / active_weights.sum()
        additions = np.minimum(
            np.floor(raw).astype(np.int64), capacity[active])
        added = int(additions.sum())
        if added:
            quotas[active] += additions
            remaining -= added
            continue
        # The final remainder is smaller than the active set.  Stable cell id
        # breaks equal-weight ties so repeated runs are bitwise deterministic.
        order = np.lexsort((active, -raw))
        take = min(remaining, len(active))
        quotas[active[order[:take]]] += 1
        remaining -= take

    if int(quotas.sum()) != keep_count or np.any(quotas > counts):
        raise RuntimeError('Exact cell quota allocation failed.')
    if np.any(quotas < mandatory):
        raise RuntimeError('Cell allocation dropped mandatory candidates.')
    return quotas


def select_with_cell_quotas(
        cell_inverse, quotas, scores, mandatory_mask=None):
    """Select the highest score within every cell while preserving mandatory."""
    inverse = np.asarray(cell_inverse, dtype=np.int64).reshape(-1)
    quotas = np.asarray(quotas, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(inverse) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Cell indices and finite scores must be aligned.')
    if np.any((inverse < 0) | (inverse >= len(quotas))):
        raise ValueError('cell_inverse contains invalid cell ids.')
    mandatory = (
        np.zeros(len(scores), dtype=bool) if mandatory_mask is None else
        np.asarray(mandatory_mask, dtype=bool).reshape(-1))
    if len(mandatory) != len(scores):
        raise ValueError('mandatory_mask is not aligned.')
    indices = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((indices, -scores, ~mandatory, inverse))
    ordered_cells = inverse[order]
    starts = np.r_[0, np.flatnonzero(
        ordered_cells[1:] != ordered_cells[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    ranks = np.arange(len(order)) - np.repeat(starts, lengths)
    selected_order = ranks < quotas[ordered_cells]
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[selected_order]] = True
    if not np.all(selected[mandatory]):
        raise RuntimeError('Cell selection dropped mandatory candidates.')
    if int(selected.sum()) != int(quotas.sum()):
        raise RuntimeError('Cell selection did not reproduce its quota budget.')
    return selected


def select_global(scores, keep_count, mandatory_mask=None):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mandatory = (
        np.zeros(len(scores), dtype=bool) if mandatory_mask is None else
        np.asarray(mandatory_mask, dtype=bool).reshape(-1))
    if len(mandatory) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Global score inputs are invalid.')
    keep_count = int(keep_count)
    if int(mandatory.sum()) > keep_count:
        raise RuntimeError('Mandatory candidates exceed the global budget.')
    indices = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((indices, -scores, ~mandatory))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:keep_count]] = True
    if not np.all(selected[mandatory]):
        raise RuntimeError('Global selection dropped mandatory candidates.')
    return selected


def _coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean()) if len(values) else 0.0
    return float(values.std() / mean) if mean > 0 else 0.0


def cell_structure_metrics(inverse, counts, labels, utility):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    if len(inverse) != len(labels) or len(labels) != len(utility):
        raise ValueError('Cell structure arrays are not aligned.')
    num_cells = len(counts)
    pairs = np.column_stack([inverse, labels])
    unique_pairs, pair_counts = np.unique(
        pairs, axis=0, return_counts=True)
    dominant = np.zeros(num_cells, dtype=np.int64)
    np.maximum.at(dominant, unique_pairs[:, 0], pair_counts)
    purity = dominant / counts
    positive_pairs = unique_pairs[unique_pairs[:, 1] > 0]
    tree_ids_per_cell = np.bincount(
        positive_pairs[:, 0], minlength=num_cells)

    sums = np.bincount(inverse, weights=utility, minlength=num_cells)
    means = sums / counts
    global_mean = float(utility.mean())
    total_variance = float(np.square(utility - global_mean).sum())
    between_variance = float(
        (counts * np.square(means - global_mean)).sum())
    return {
        'num_cells': int(num_cells),
        'mean_candidates_per_cell': float(counts.mean()),
        'p90_candidates_per_cell': float(np.quantile(counts, 0.90)),
        'max_candidates_per_cell': int(counts.max()),
        'impure_cell_rate_below_0_90': float(np.mean(purity < 0.90)),
        'multi_tree_collision_cell_rate': float(
            np.mean(tree_ids_per_cell >= 2)),
        'between_cell_utility_variance_ratio': float(
            between_variance / max(total_variance, 1e-12)),
    }


def selection_metrics(
        selected, inverse, labels, utility, vote_error, critical):
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    vote_error = np.asarray(vote_error, dtype=np.float64).reshape(-1)
    critical = np.asarray(critical, dtype=bool).reshape(-1)
    if not all(
            len(values) == len(selected) for values in (
                inverse, labels, utility, vote_error, critical)):
        raise ValueError('Selection metric arrays are not aligned.')
    tree = labels > 0
    selected_tree = selected & tree
    selected_counts = np.bincount(
        np.asarray(inverse)[selected], minlength=int(np.max(inverse)) + 1)
    all_critical_trees = np.unique(labels[critical & tree])
    kept_critical_trees = np.unique(labels[selected & critical & tree])
    return {
        'selected_count': int(selected.sum()),
        'selected_mean_utility': float(utility[selected].mean()),
        'selected_non_tree_rate': float(
            np.mean(labels[selected] == 0)),
        'selected_reliable_rate': float(
            np.mean(utility[selected] >= 0.5)),
        'selected_tree_mean_vote_error_xy': float(
            vote_error[selected_tree].mean()) if selected_tree.any() else 0.0,
        'critical_recall': float(
            np.count_nonzero(selected & critical) /
            max(np.count_nonzero(critical), 1)),
        'critical_tree_coverage': float(
            len(kept_critical_trees) / max(len(all_critical_trees), 1)),
        'occupied_cell_coverage': float(
            np.count_nonzero(selected_counts) / len(selected_counts)),
        'selected_cell_count_cv': _coefficient_of_variation(
            selected_counts),
        'selection_sha256': hashlib.sha256(
            np.flatnonzero(selected).astype(np.int32).tobytes()).hexdigest(),
    }


def load_validation_rows(manifest_path, expected_plots):
    with Path(manifest_path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    validation = [row for row in rows if row['split'] == 'validation']
    names = [row['source_plot'] for row in validation]
    if names != list(expected_plots):
        raise ValueError(
            f'Validation forests differ from the preregistration: {names}.')
    return validation


def analyze_plot(row, settings):
    artifact_path = Path(row['artifact_path']).resolve()
    data_root = Path(settings['data_root']).resolve()
    if data_root not in artifact_path.parents or not artifact_path.is_file():
        raise ValueError(f'Invalid E1a artifact path: {artifact_path}')
    with np.load(artifact_path, allow_pickle=False) as data:
        votes = data['base_votes_xy'].astype(np.float64)
        labels = data['target_tree_id'].astype(np.int64)
        utility = data['target_utility'].astype(np.float64)
        vote_error = data['target_vote_error_xy'].astype(np.float64)
        critical = data['target_coverage_critical'].astype(bool)
        valid = data['target_valid'].astype(bool)
    if not np.all(valid):
        raise ValueError(
            f'E1d requires complete validation labels: {artifact_path}.')
    if len(votes) != int(row['num_candidates']):
        raise ValueError(f'Manifest count differs for {artifact_path}.')
    _, inverse, counts = vote_cells(votes, settings['cell_size'])
    keep_count = int(np.ceil(len(votes) * float(settings['keep_ratio'])))
    mandatory_counts = np.bincount(
        inverse, weights=critical.astype(np.int64),
        minlength=len(counts)).astype(np.int64)
    quotas = allocate_weighted_cell_quotas(
        counts, keep_count, mandatory_counts,
        settings['density_exponent'], preserve_nonempty=True)
    modes = {
        'point_oracle': select_global(utility, keep_count, critical),
        'cell_oracle': select_with_cell_quotas(
            inverse, quotas, utility, critical),
    }
    for seed in settings['random_seeds']:
        random_scores = np.random.default_rng(int(seed)).random(len(votes))
        modes[f'cell_random_s{int(seed)}'] = select_with_cell_quotas(
            inverse, quotas, random_scores, critical)
    mode_metrics = {
        name: selection_metrics(
            mask, inverse, labels, utility, vote_error, critical)
        for name, mask in modes.items()
    }
    random_rows = [
        mode_metrics[f'cell_random_s{int(seed)}']
        for seed in settings['random_seeds']]
    random_mean = {
        name: float(np.mean([row[name] for row in random_rows]))
        for name in (
            'selected_mean_utility', 'selected_non_tree_rate',
            'critical_recall', 'critical_tree_coverage',
            'selected_cell_count_cv')
    }
    cell = mode_metrics['cell_oracle']
    point = mode_metrics['point_oracle']
    effects = {
        'utility_gain_over_random_relative': float(
            (cell['selected_mean_utility'] -
             random_mean['selected_mean_utility']) /
            max(abs(random_mean['selected_mean_utility']), 1e-12)),
        'non_tree_reduction_over_random_relative': float(
            (random_mean['selected_non_tree_rate'] -
             cell['selected_non_tree_rate']) /
            max(random_mean['selected_non_tree_rate'], 1e-12)),
        'utility_retention_vs_point_oracle': float(
            cell['selected_mean_utility'] /
            max(point['selected_mean_utility'], 1e-12)),
        'critical_recall_drop_vs_point_oracle': float(
            point['critical_recall'] - cell['critical_recall']),
        'density_cv_reduction_vs_point_oracle_relative': float(
            (point['selected_cell_count_cv'] -
             cell['selected_cell_count_cv']) /
            max(point['selected_cell_count_cv'], 1e-12)),
    }
    effects['plot_win_over_random'] = bool(
        effects['utility_gain_over_random_relative'] > 0 and
        effects['non_tree_reduction_over_random_relative'] > 0 and
        cell['critical_tree_coverage'] >= 0.99)
    return {
        'source_plot': row['source_plot'],
        'artifact_path': str(artifact_path),
        'num_candidates': int(len(votes)),
        'keep_count': keep_count,
        'structure': cell_structure_metrics(
            inverse, counts, labels, utility),
        'modes': mode_metrics,
        'random_mean': random_mean,
        'effects': effects,
    }


def aggregate_reports(reports, settings):
    plots = len(reports)
    cell_rows = [report['modes']['cell_oracle'] for report in reports]
    point_rows = [report['modes']['point_oracle'] for report in reports]
    random_rows = [report['random_mean'] for report in reports]
    aggregate = {
        'num_validation_plots': plots,
        'num_candidates': int(sum(
            report['num_candidates'] for report in reports)),
        'num_cells': int(sum(
            report['structure']['num_cells'] for report in reports)),
        'cell_oracle': {},
        'point_oracle': {},
        'cell_random_mean': {},
    }
    metric_names = (
        'selected_mean_utility', 'selected_non_tree_rate',
        'critical_recall', 'critical_tree_coverage',
        'selected_cell_count_cv')
    for name in metric_names:
        aggregate['cell_oracle'][name] = float(np.mean([
            row[name] for row in cell_rows]))
        aggregate['point_oracle'][name] = float(np.mean([
            row[name] for row in point_rows]))
        aggregate['cell_random_mean'][name] = float(np.mean([
            row[name] for row in random_rows]))
    cell = aggregate['cell_oracle']
    point = aggregate['point_oracle']
    random_mean = aggregate['cell_random_mean']
    effects = {
        'utility_gain_over_random_relative': float(
            (cell['selected_mean_utility'] -
             random_mean['selected_mean_utility']) /
            max(abs(random_mean['selected_mean_utility']), 1e-12)),
        'non_tree_reduction_over_random_relative': float(
            (random_mean['selected_non_tree_rate'] -
             cell['selected_non_tree_rate']) /
            max(random_mean['selected_non_tree_rate'], 1e-12)),
        'utility_retention_vs_point_oracle': float(
            cell['selected_mean_utility'] /
            max(point['selected_mean_utility'], 1e-12)),
        'critical_recall_drop_vs_point_oracle': float(
            point['critical_recall'] - cell['critical_recall']),
        'density_cv_reduction_vs_point_oracle_relative': float(
            (point['selected_cell_count_cv'] -
             cell['selected_cell_count_cv']) /
            max(point['selected_cell_count_cv'], 1e-12)),
        'plot_wins': int(sum(
            report['effects']['plot_win_over_random']
            for report in reports)),
        'mean_between_cell_utility_variance_ratio': float(np.mean([
            report['structure']['between_cell_utility_variance_ratio']
            for report in reports])),
    }
    gate_cfg = settings['gate']
    gate = {
        'expected_validation_plots_present': (
            plots == int(gate_cfg['expected_validation_plots'])),
        'enough_seed_cells': (
            aggregate['num_cells'] >= int(gate_cfg['min_total_cells'])),
        'cell_structure_signal_passed': (
            effects['mean_between_cell_utility_variance_ratio'] >=
            float(gate_cfg['min_between_cell_utility_variance_ratio'])),
        'utility_gain_over_random_passed': (
            effects['utility_gain_over_random_relative'] >=
            float(gate_cfg['min_utility_gain_over_random_relative'])),
        'non_tree_reduction_passed': (
            effects['non_tree_reduction_over_random_relative'] >=
            float(gate_cfg[
                'min_non_tree_reduction_over_random_relative'])),
        'utility_retention_vs_point_passed': (
            effects['utility_retention_vs_point_oracle'] >=
            float(gate_cfg['min_utility_retention_vs_point_oracle'])),
        'critical_recall_preserved': (
            effects['critical_recall_drop_vs_point_oracle'] <=
            float(gate_cfg['max_critical_recall_drop'])),
        'critical_tree_coverage_passed': (
            min(row['critical_tree_coverage'] for row in cell_rows) >=
            float(gate_cfg['min_critical_tree_coverage'])),
        'density_balance_improved': (
            effects['density_cv_reduction_vs_point_oracle_relative'] >=
            float(gate_cfg[
                'min_density_cv_reduction_vs_point_relative'])),
        'per_plot_consistency_passed': (
            effects['plot_wins'] >= int(gate_cfg['min_plot_wins'])),
    }
    gate['passed'] = bool(all(gate.values()))
    return aggregate, effects, gate


def format_markdown(report):
    aggregate = report['aggregate']
    effects = report['effects']
    lines = [
        '# E1d Seed-Cell 粒度 Oracle', '',
        '- 数据：固定 E1a validation forests；未读取 Wytham。',
        f'- Vote cell：{report["settings"]["cell_size"]:.3f} m',
        f'- 保留比例：{report["settings"]["keep_ratio"]:.3f}',
        f'- 密度指数：{report["settings"]["density_exponent"]:.3f}',
        f'- 验证森林：{aggregate["num_validation_plots"]}',
        f'- 候选种子：{aggregate["num_candidates"]:,}',
        f'- Seed cells：{aggregate["num_cells"]:,}', '',
        '## Macro 指标', '',
        '| Mode | Utility | Non-tree rate | Critical recall | '
        'Critical-tree coverage | Cell-count CV |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name in ('point_oracle', 'cell_oracle', 'cell_random_mean'):
        row = aggregate[name]
        lines.append(
            f'| {name} | {row["selected_mean_utility"]:.6f} | '
            f'{100 * row["selected_non_tree_rate"]:.3f}% | '
            f'{100 * row["critical_recall"]:.3f}% | '
            f'{100 * row["critical_tree_coverage"]:.3f}% | '
            f'{row["selected_cell_count_cv"]:.6f} |')
    lines.extend(['', '## 效果', ''])
    for name, value in effects.items():
        if name == 'plot_wins':
            lines.append(
                f'- plot_wins: {int(value)}/'
                f'{aggregate["num_validation_plots"]}')
        else:
            lines.append(f'- {name}: {value:+.6f}')
    lines.extend(['', '## 每森林', '',
                  '| Plot | Candidates | Cells | Impure cells | '
                  'Multi-tree cells | Utility gain | Non-tree reduction | Win |',
                  '|---|---:|---:|---:|---:|---:|---:|---|'])
    for row in report['plots']:
        structure = row['structure']
        effect = row['effects']
        lines.append(
            f'| {row["source_plot"]} | {row["num_candidates"]:,} | '
            f'{structure["num_cells"]:,} | '
            f'{100 * structure["impure_cell_rate_below_0_90"]:.3f}% | '
            f'{100 * structure["multi_tree_collision_cell_rate"]:.3f}% | '
            f'{100 * effect["utility_gain_over_random_relative"]:+.3f}% | '
            f'{100 * effect["non_tree_reduction_over_random_relative"]:+.3f}% | '
            f'{bool(effect["plot_win_over_random"])} |')
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend([
        '',
        ('PASS：进入参数匹配的 Seed-Cell MLP 与 Superpoint Query Adapter。'
         if report['gate']['passed'] else
         'STOP：Cell 粒度上限不足，关闭种子注意力路线，不使用 Wytham 调参。'),
    ])
    return '\n'.join(lines) + '\n'


def write_per_plot_csv(path, reports):
    rows = []
    for report in reports:
        row = {
            'source_plot': report['source_plot'],
            'num_candidates': report['num_candidates'],
            'keep_count': report['keep_count'],
            **report['structure'],
            **report['effects'],
        }
        for mode_name in ('point_oracle', 'cell_oracle'):
            for metric, value in report['modes'][mode_name].items():
                if metric != 'selection_sha256':
                    row[f'{mode_name}_{metric}'] = value
        rows.append(row)
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    expected = settings['expected_validation_plots']
    rows = load_validation_rows(settings['manifest_path'], expected)
    normalized = {
        'data_root': settings['data_root'],
        'cell_size': float(settings['cell_size']),
        'keep_ratio': float(settings['keep_ratio']),
        'density_exponent': float(settings['density_exponent']),
        'random_seeds': [int(seed) for seed in settings['random_seeds']],
        'gate': settings['gate'],
    }
    if not 0 < normalized['keep_ratio'] <= 1:
        raise ValueError('keep_ratio must be in (0, 1].')
    if len(normalized['random_seeds']) < 3:
        raise ValueError('E1d requires at least three random controls.')
    reports = []
    for position, row in enumerate(rows, start=1):
        print(
            f'[{position}/{len(rows)}] Analyzing {row["source_plot"]}...',
            flush=True)
        report = analyze_plot(row, normalized)
        reports.append(report)
        print(
            f'  cells={report["structure"]["num_cells"]:,}, '
            f'utility_gain='
            f'{100 * report["effects"]["utility_gain_over_random_relative"]:+.2f}%, '
            f'win={report["effects"]["plot_win_over_random"]}', flush=True)
    aggregate, effects, gate = aggregate_reports(
        reports, {**normalized, 'gate': settings['gate']})
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'config': str(config_path),
        'settings': {
            'cell_size': normalized['cell_size'],
            'keep_ratio': normalized['keep_ratio'],
            'density_exponent': normalized['density_exponent'],
            'random_seeds': normalized['random_seeds'],
            'expected_validation_plots': list(expected),
        },
        'plots': reports,
        'aggregate': aggregate,
        'effects': effects,
        'gate': gate,
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    write_per_plot_csv(output_dir / 'per_plot_metrics.csv', reports)
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'E1d Seed-Cell Oracle gate failed; close the seed-attention route.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the fixed-validation E1d Seed-Cell Oracle.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
