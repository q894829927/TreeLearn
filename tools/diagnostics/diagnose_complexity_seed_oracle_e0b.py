"""Run the E0b unknown-protected, spatially constrained seed Oracle.

This diagnostic corrects the original E0 protocol for partially annotated
validation forests.  Unclassified points (label -1) are protected instead of
being treated as negative supervision.  The spatial Oracle additionally keeps
a GT-only per-tree/per-vote-cell support set before ranking the remaining seed
budget.  All random controls share the same mandatory support set.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import diagnose_complexity_seed_oracle as e0  # noqa: E402


def label_composition(mask, labels):
    """Count supervised tree, supervised non-tree, and unknown points."""
    selected = np.asarray(mask, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(selected) != len(labels):
        raise ValueError('mask and labels must have identical lengths.')
    return {
        'total': int(selected.sum()),
        'tree': int(np.count_nonzero(selected & (labels > 0))),
        'non_tree': int(np.count_nonzero(selected & (labels == 0))),
        'unknown': int(np.count_nonzero(selected & (labels < 0))),
    }


def _tree_cell_keys(coords, offsets, labels, candidate_mask, voxel_size):
    coords = np.asarray(coords, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    candidates = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    if coords.shape != offsets.shape or coords.shape != (len(labels), 3):
        raise ValueError('coords, offsets, and labels are not aligned.')
    if len(candidates) != len(labels):
        raise ValueError('candidate_mask and labels are not aligned.')
    if float(voxel_size) <= 0:
        raise ValueError('spatial_vote_cell_size must be positive.')
    indices = np.flatnonzero(candidates & (labels > 0))
    votes = coords[indices, :2] + offsets[indices, :2]
    cells = np.floor(votes / float(voxel_size)).astype(np.int64)
    return indices, labels[indices], cells


def spatial_mandatory_mask(
        coords, offsets, labels, candidate_mask, utility,
        voxel_size=0.60, min_seeds_per_tree=50):
    """Build a GT-only support set preserving trees and vote-space cells."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    candidates = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    if len(utility) != len(labels) or len(candidates) != len(labels):
        raise ValueError('utility, labels, and candidate_mask are not aligned.')
    if int(min_seeds_per_tree) < 1:
        raise ValueError('min_seeds_per_tree must be at least one.')
    mandatory = np.zeros(len(labels), dtype=bool)
    indices, tree_labels, cells = _tree_cell_keys(
        coords, offsets, labels, candidates, voxel_size)
    if len(indices) == 0:
        return mandatory

    # Keep the highest-utility representative of every occupied GT tree/cell.
    order = np.lexsort((
        indices,
        -utility[indices],
        cells[:, 1],
        cells[:, 0],
        tree_labels,
    ))
    ordered_indices = indices[order]
    ordered_labels = tree_labels[order]
    ordered_cells = cells[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (
        (ordered_labels[1:] != ordered_labels[:-1]) |
        (ordered_cells[1:, 0] != ordered_cells[:-1, 0]) |
        (ordered_cells[1:, 1] != ordered_cells[:-1, 1]))
    mandatory[ordered_indices[first]] = True

    # Keep at least tau_min high-utility seeds for every supervised tree.
    for tree_id in np.unique(tree_labels):
        tree_indices = indices[tree_labels == tree_id]
        quota = min(int(min_seeds_per_tree), len(tree_indices))
        tree_order = np.lexsort((
            tree_indices, -utility[tree_indices]))
        mandatory[tree_indices[tree_order[:quota]]] = True
    return mandatory


def effective_ratio_for_count(keep_count, candidate_count):
    """Return a ratio whose ceil(candidate_count * ratio) is keep_count."""
    keep_count = int(keep_count)
    candidate_count = int(candidate_count)
    if not 1 <= keep_count <= candidate_count:
        raise ValueError('keep_count must be within the candidate count.')
    if keep_count == candidate_count:
        return 1.0
    ratio = float(keep_count / candidate_count)
    while int(np.ceil(candidate_count * ratio)) > keep_count:
        ratio = float(np.nextafter(ratio, 0.0))
    while int(np.ceil(candidate_count * ratio)) < keep_count:
        ratio = float(np.nextafter(ratio, 1.0))
    return ratio


def compose_rank_scores(
        base_scores, candidate_mask, protected_mask, mandatory_mask=None,
        random_seed=None):
    """Compose stable scores with protected > mandatory > fill priority."""
    candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    protected_mask = np.asarray(protected_mask, dtype=bool).reshape(-1)
    base_scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
    if (
            len(candidate_mask) != len(protected_mask) or
            len(candidate_mask) != len(base_scores)):
        raise ValueError('score masks and arrays are not aligned.')
    if np.any(protected_mask & ~candidate_mask):
        raise ValueError('protected_mask must be a candidate subset.')
    if mandatory_mask is None:
        mandatory_mask = np.zeros(len(candidate_mask), dtype=bool)
    mandatory_mask = np.asarray(mandatory_mask, dtype=bool).reshape(-1)
    if len(mandatory_mask) != len(candidate_mask):
        raise ValueError('mandatory_mask is not aligned.')
    if np.any(mandatory_mask & ~candidate_mask):
        raise ValueError('mandatory_mask must be a candidate subset.')

    scores = np.zeros(len(candidate_mask), dtype=np.float32)
    candidate_indices = np.flatnonzero(candidate_mask)
    if random_seed is None:
        fill_scores = np.clip(base_scores[candidate_indices], 0.0, 1.0)
    else:
        generator = np.random.default_rng(int(random_seed))
        fill_scores = generator.random(len(candidate_indices))
    scores[candidate_indices] = fill_scores.astype(np.float32)
    scores[mandatory_mask] = 2.0
    scores[protected_mask] = 3.0
    return scores


def tree_retention_summary(selected_mask, candidate_mask, labels):
    selected = np.asarray(selected_mask, dtype=bool)
    candidates = np.asarray(candidate_mask, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    tree_ids = np.unique(labels[candidates & (labels > 0)])
    retained_counts = []
    retained_ratios = []
    for tree_id in tree_ids:
        tree_candidates = candidates & (labels == tree_id)
        candidate_count = int(tree_candidates.sum())
        retained_count = int(np.count_nonzero(selected & tree_candidates))
        retained_counts.append(retained_count)
        retained_ratios.append(retained_count / max(candidate_count, 1))
    if not retained_counts:
        return {
            'num_trees': 0,
            'min_candidate_count': 0,
            'min_count': 0,
            'p10_count': 0.0,
            'mean_count': 0.0,
            'min_ratio': 0.0,
            'p10_ratio': 0.0,
            'mean_ratio': 0.0,
        }
    return {
        'num_trees': int(len(retained_counts)),
        'min_candidate_count': int(min(
            np.count_nonzero(candidates & (labels == tree_id))
            for tree_id in tree_ids)),
        'min_count': int(np.min(retained_counts)),
        'p10_count': float(np.quantile(retained_counts, 0.10)),
        'mean_count': float(np.mean(retained_counts)),
        'min_ratio': float(np.min(retained_ratios)),
        'p10_ratio': float(np.quantile(retained_ratios, 0.10)),
        'mean_ratio': float(np.mean(retained_ratios)),
    }


def spatial_coverage(
        selected_mask, coords, offsets, labels, candidate_mask, voxel_size):
    indices, tree_labels, cells = _tree_cell_keys(
        coords, offsets, labels, candidate_mask, voxel_size)
    if len(indices) == 0:
        return {'total_cells': 0, 'retained_cells': 0, 'coverage': 1.0}
    keys = np.column_stack([tree_labels, cells])
    total_cells = len(np.unique(keys, axis=0))
    selected_rows = np.asarray(selected_mask, dtype=bool)[indices]
    retained_cells = len(np.unique(keys[selected_rows], axis=0))
    return {
        'total_cells': int(total_cells),
        'retained_cells': int(retained_cells),
        'coverage': float(retained_cells / max(total_cells, 1)),
    }


def selection_jaccard(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / max(union, 1))


def selection_digest(mask):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int32)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def build_selection_plan(arrays, candidate_mask, utility, settings):
    """Construct and validate all E0b masks before expensive clustering."""
    labels = np.asarray(arrays['instance_labels'], dtype=np.int64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    protected = candidate_mask & (labels < int(settings['known_label_min']))
    mandatory = spatial_mandatory_mask(
        arrays['coords'], arrays['offset_predictions'], labels,
        candidate_mask, utility,
        voxel_size=float(settings['spatial_vote_cell_size']),
        min_seeds_per_tree=int(settings['min_seeds_per_tree']))
    candidate_count = int(candidate_mask.sum())
    requested_count = int(np.ceil(
        candidate_count * float(settings['keep_ratio'])))
    required_count = int(np.count_nonzero(protected | mandatory))
    if required_count > requested_count:
        raise RuntimeError(
            'E0b constraints are infeasible at the locked keep ratio: '
            f'required={required_count:,}, budget={requested_count:,}.')
    effective_ratio = effective_ratio_for_count(
        requested_count, candidate_count)

    score_arrays = {
        'oracle_masked': compose_rank_scores(
            utility, candidate_mask, protected),
        'oracle_spatial': compose_rank_scores(
            utility, candidate_mask, protected, mandatory),
    }
    for seed in settings['random_seeds']:
        score_arrays[f'random_spatial_s{int(seed)}'] = compose_rank_scores(
            utility, candidate_mask, protected, mandatory,
            random_seed=int(seed))
    masks = {
        name: e0.exact_top_ratio_mask(
            candidate_mask, scores, effective_ratio)
        for name, scores in score_arrays.items()
    }
    for name, mask in masks.items():
        if int(mask.sum()) != requested_count:
            raise RuntimeError(
                f'{name} retained {int(mask.sum()):,}, expected '
                f'{requested_count:,}.')
        if not np.all(mask[protected]):
            raise RuntimeError(f'{name} removed protected unknown seeds.')
    for name, mask in masks.items():
        if name == 'oracle_masked':
            continue
        if not np.all(mask[mandatory]):
            raise RuntimeError(f'{name} removed mandatory spatial seeds.')
    return {
        'protected_mask': protected,
        'mandatory_mask': mandatory,
        'score_arrays': score_arrays,
        'masks': masks,
        'candidate_count': candidate_count,
        'requested_count': requested_count,
        'required_count': required_count,
        'effective_keep_ratio': effective_ratio,
    }


def selection_diagnostics(
        plan, arrays, candidate_mask, utility_result, settings):
    labels = arrays['instance_labels']
    result = {
        'candidate_composition': label_composition(candidate_mask, labels),
        'protected_unknown_count': int(plan['protected_mask'].sum()),
        'mandatory_spatial_count': int(plan['mandatory_mask'].sum()),
        'requested_count': int(plan['requested_count']),
        'required_count': int(plan['required_count']),
        'effective_keep_ratio': float(plan['effective_keep_ratio']),
        'modes': {},
        'oracle_masked_spatial_jaccard': selection_jaccard(
            plan['masks']['oracle_masked'],
            plan['masks']['oracle_spatial']),
    }
    for name, mask in plan['masks'].items():
        metrics = e0.seed_set_metrics(
            mask, candidate_mask, labels,
            utility_result['vote_error_xy'], utility_result['utility'],
            settings['min_seeds_per_tree'])
        metrics['composition'] = label_composition(mask, labels)
        metrics['tree_retention'] = tree_retention_summary(
            mask, candidate_mask, labels)
        metrics['spatial_coverage'] = spatial_coverage(
            mask, arrays['coords'], arrays['offset_predictions'], labels,
            candidate_mask, settings['spatial_vote_cell_size'])
        metrics['selection_sha256'] = selection_digest(mask)
        result['modes'][name] = metrics
    return result


def assess_gate(mode_metrics, random_metrics, settings):
    baseline = mode_metrics['baseline']
    oracle = mode_metrics['oracle_spatial']
    random_mean = float(np.mean([row['f1'] for row in random_metrics]))
    effects = {
        'f1_gain_pp': 100 * (oracle['f1'] - baseline['f1']),
        'commission_reduction_pp': 100 * (
            baseline['commission'] - oracle['commission']),
        'completeness_drop_pp': 100 * (
            baseline['completeness'] - oracle['completeness']),
        'f1_gain_over_random_mean_pp': 100 * (
            oracle['f1'] - random_mean),
        'random_mean_f1': random_mean,
    }
    gate = {
        'minimum_f1_gain_pp': bool(
            effects['f1_gain_pp'] >= float(settings['min_f1_gain_pp'])),
        'minimum_commission_reduction_pp': bool(
            effects['commission_reduction_pp'] >= float(
                settings['min_commission_reduction_pp'])),
        'maximum_completeness_drop_pp': bool(
            effects['completeness_drop_pp'] <= float(
                settings['max_completeness_drop_pp'])),
        'minimum_gain_over_random_pp': bool(
            effects['f1_gain_over_random_mean_pp'] >= float(
                settings['min_gain_over_random_pp'])),
    }
    gate['passed'] = bool(all(gate.values()))
    return gate, {name: float(value) for name, value in effects.items()}


def format_markdown(report):
    selection = report['selection']
    lines = [
        '# E0b 未标注保护的空间约束 Seed Oracle',
        '',
        f"- 候选种子：{report['num_candidate_seeds']:,}",
        f"- 固定保留数：{selection['requested_count']:,}",
        f"- 未标注保护种子：{selection['protected_unknown_count']:,}",
        f"- 空间/树级保底种子：{selection['mandatory_spatial_count']:,}",
        f"- Masked/Spatial Jaccard："
        f"{selection['oracle_masked_spatial_jaccard']:.6f}",
        '',
        '## 检测结果',
        '',
        '| Mode | TP | FP | FN | Completeness | Commission | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, metrics in report['mode_metrics'].items():
        lines.append(
            f"| {name} | {metrics['tp']} | {metrics['fp']} | "
            f"{metrics['fn']} | {100 * metrics['completeness']:.3f}% | "
            f"{100 * metrics['commission']:.3f}% | "
            f"{100 * metrics['f1']:.3f}% |")
    lines.extend(['', '## 选择集合诊断', ''])
    lines.extend([
        '| Mode | Retained | Tree | Non-tree | Unknown | Cell coverage | '
        'Min tree count | Mean error |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for name, metrics in selection['modes'].items():
        composition = metrics['composition']
        tree = metrics['tree_retention']
        coverage = metrics['spatial_coverage']['coverage']
        lines.append(
            f"| {name} | {metrics['selected_count']:,} | "
            f"{composition['tree']:,} | {composition['non_tree']:,} | "
            f"{composition['unknown']:,} | {100 * coverage:.3f}% | "
            f"{tree['min_count']} | {metrics['mean_vote_error_xy']:.4f} m |")
    lines.extend(['', '## 效果', ''])
    for name, value in report['effects'].items():
        lines.append(f'- {name}: {value:+.6f}')
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    lines.append('')
    if report['gate']['passed']:
        lines.append('PASS：进入冻结主干的 Seed-MLP 控制实验。')
    else:
        lines.append('STOP：关闭种子删减注意力路线，不得使用 Wytham 调参。')
    return '\n'.join(lines) + '\n'


def write_metrics_csv(path, mode_metrics):
    rows = []
    for name, metrics in mode_metrics.items():
        rows.append({'mode': name, **metrics})
    fieldnames = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    import yaml
    from tree_learn.util import get_config, get_root_logger

    with open(config_path, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    required = {
        'pipeline_config', 'ground_truth_path', 'output_dir', 'keep_ratio',
        'random_seeds', 'utility', 'selection', 'evaluation_thresholds',
        'gate', 'baseline_reference',
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f'Missing E0b config fields: {missing}')
    selection_settings = {
        'keep_ratio': float(settings['keep_ratio']),
        'random_seeds': [int(seed) for seed in settings['random_seeds']],
        'known_label_min': int(settings['selection']['known_label_min']),
        'spatial_vote_cell_size': float(
            settings['selection']['spatial_vote_cell_size']),
        'min_seeds_per_tree': int(
            settings['selection']['min_seeds_per_tree']),
    }
    if not 0 < selection_settings['keep_ratio'] <= 1:
        raise ValueError('keep_ratio must be in (0, 1].')
    if not selection_settings['random_seeds']:
        raise ValueError('At least one random control seed is required.')

    pipeline_config = str(settings['pipeline_config'])
    config = get_config(pipeline_config)
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger(str(output_dir / 'oracle_e0b.log'))
    logger.info('Getting shared pointwise predictions and ensemble...')
    arrays = e0.pointwise_and_ensemble(config, logger)
    oracle_gt = e0.prepare_ground_truth(config, settings, arrays, logger)
    arrays['instance_labels'] = oracle_gt['labels_at_ensemble']
    arrays['offset_labels'] = oracle_gt['offset_targets_at_ensemble']

    candidate_mask = e0.candidate_base_seed_mask(
        arrays['semantic_logits'], arrays['verticality'],
        arrays['offset_predictions'], config.grouping.tree_conf_thresh,
        config.grouping.tau_vert, config.grouping.tau_off)
    utility_result = e0.compute_seed_utility(
        arrays['coords'], arrays['offset_predictions'],
        arrays['offset_labels'], arrays['instance_labels'], candidate_mask,
        sigma_m=float(settings['utility']['sigma_m']),
        purity_voxel_size=float(
            settings['utility']['purity_voxel_size']),
        purity_power=float(settings['utility']['purity_power']))
    plan = build_selection_plan(
        arrays, candidate_mask, utility_result['utility'],
        selection_settings)
    selection_report = selection_diagnostics(
        plan, arrays, candidate_mask, utility_result,
        selection_settings)

    for name, diagnostics in selection_report['modes'].items():
        if diagnostics['selected_count'] != plan['requested_count']:
            raise RuntimeError(f'{name} failed the exact-count preflight.')
        if diagnostics['composition']['unknown'] != \
                selection_report['protected_unknown_count']:
            raise RuntimeError(f'{name} failed unknown protection preflight.')
        if name != 'oracle_masked':
            if diagnostics['spatial_coverage']['coverage'] != 1.0:
                raise RuntimeError(f'{name} failed spatial coverage preflight.')
            required_minimum = min(
                selection_settings['min_seeds_per_tree'],
                diagnostics['tree_retention']['min_candidate_count'])
            if diagnostics['tree_retention']['min_count'] < required_minimum:
                raise RuntimeError(f'{name} failed tree quota preflight.')
    logger.info(
        'Selection preflight PASS: '
        f"candidates={plan['candidate_count']:,}, "
        f"kept={plan['requested_count']:,}, "
        f"unknown={int(plan['protected_mask'].sum()):,}, "
        f"mandatory={int(plan['mandatory_mask'].sum()):,}.")

    mode_metrics = {}
    logger.info('===== START baseline clustering =====')
    baseline_predictions = e0.cluster_mode(
        'baseline', config, arrays, None,
        plan['effective_keep_ratio'], 0, logger)
    baseline_eval = e0.predictions_on_official_gt(
        baseline_predictions, oracle_gt['propagation'])
    mode_metrics['baseline'] = e0.detection_metrics(
        oracle_gt['ground_truth_labels'], baseline_eval,
        settings['evaluation_thresholds'])
    del baseline_predictions, baseline_eval
    if not e0.baseline_matches_reference(
            mode_metrics['baseline'], settings['baseline_reference']):
        raise RuntimeError(
            'E0b baseline reproduction failed: '
            f"observed={mode_metrics['baseline']}, "
            f"expected={settings['baseline_reference']}")

    for mode in ('oracle_masked', 'oracle_spatial'):
        logger.info(f'===== START {mode} clustering =====')
        predictions = e0.cluster_mode(
            mode, config, arrays, plan['score_arrays'][mode],
            plan['effective_keep_ratio'], 0, logger)
        evaluation_predictions = e0.predictions_on_official_gt(
            predictions, oracle_gt['propagation'])
        mode_metrics[mode] = e0.detection_metrics(
            oracle_gt['ground_truth_labels'], evaluation_predictions,
            settings['evaluation_thresholds'])
        logger.info(
            f"DONE {mode}: F1={100 * mode_metrics[mode]['f1']:.3f}%, "
            f"Commission={100 * mode_metrics[mode]['commission']:.3f}%")
        del predictions, evaluation_predictions

    random_metrics = []
    for seed in selection_settings['random_seeds']:
        mode = f'random_spatial_s{seed}'
        logger.info(f'===== START {mode} clustering =====')
        predictions = e0.cluster_mode(
            mode, config, arrays, plan['score_arrays'][mode],
            plan['effective_keep_ratio'], seed, logger)
        evaluation_predictions = e0.predictions_on_official_gt(
            predictions, oracle_gt['propagation'])
        metrics = e0.detection_metrics(
            oracle_gt['ground_truth_labels'], evaluation_predictions,
            settings['evaluation_thresholds'])
        metrics['seed'] = int(seed)
        mode_metrics[mode] = metrics
        random_metrics.append(metrics)
        logger.info(
            f"DONE {mode}: F1={100 * metrics['f1']:.3f}%, "
            f"Commission={100 * metrics['commission']:.3f}%")
        del predictions, evaluation_predictions

    gate, effects = assess_gate(
        mode_metrics, random_metrics, settings['gate'])
    report = {
        'config': str(config_path),
        'pipeline_config': pipeline_config,
        'ground_truth_path': str(settings['ground_truth_path']),
        'num_points': int(len(arrays['coords'])),
        'num_candidate_seeds': int(candidate_mask.sum()),
        'selection': selection_report,
        'mode_metrics': mode_metrics,
        'effects': effects,
        'gate': gate,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    write_metrics_csv(output_dir / 'metrics.csv', mode_metrics)
    np.savez_compressed(
        output_dir / 'selected_indices.npz',
        candidate=np.flatnonzero(candidate_mask).astype(np.int32),
        **{
            name: np.flatnonzero(mask).astype(np.int32)
            for name, mask in plan['masks'].items()
        })
    print(markdown)
    if not gate['passed']:
        raise RuntimeError(
            'E0b seed Oracle gate failed; close the seed-pruning attention route.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
