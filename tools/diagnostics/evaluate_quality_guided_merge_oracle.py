import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_instance_quality_oracle import (
    contingency_from_accumulator,
    evaluate_threshold,
    evaluation_artifact_counts,
    json_default,
    load_evaluation,
    resolve_propagated_path,
    scan_aligned_las,
    validate_baseline,
)


REQUIRED_RUN_FIELDS = (
    'name', 'dataset_role', 'ground_truth', 'evaluation',
    'quality_scores', 'quality_metadata', 'output_dir')


def load_yaml(path):
    import yaml

    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_quality_scores(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            rows.append({
                'instance_id': int(row['instance_id']),
                'quality_score': float(row['quality_score']),
            })
    if not rows:
        raise ValueError(f'No quality scores found in {path}.')
    instance_ids = np.asarray(
        [row['instance_id'] for row in rows], dtype=np.int64)
    scores = np.asarray(
        [row['quality_score'] for row in rows], dtype=np.float64)
    if len(np.unique(instance_ids)) != len(instance_ids):
        raise ValueError('Quality score instance IDs must be unique.')
    if not np.isfinite(scores).all():
        raise ValueError('Quality scores must be finite.')
    return instance_ids, scores


def align_quality_scores(pred_ids, quality_ids, quality_scores):
    pred_ids = np.asarray(pred_ids, dtype=np.int64)
    quality_ids = np.asarray(quality_ids, dtype=np.int64)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)
    if set(pred_ids.tolist()) != set(quality_ids.tolist()):
        missing = sorted(set(pred_ids.tolist()) - set(quality_ids.tolist()))
        extra = sorted(set(quality_ids.tolist()) - set(pred_ids.tolist()))
        raise ValueError(
            'Quality IDs do not match prediction IDs: '
            f'missing={missing[:10]}, extra={extra[:10]}.')
    order = np.argsort(quality_ids)
    sorted_ids = quality_ids[order]
    positions = np.searchsorted(sorted_ids, pred_ids)
    return quality_scores[order][positions]


def validate_score_only_metadata(metadata, expected_instances):
    checks = {
        'filter_disabled': not bool(metadata.get('filter_enabled', True)),
        'labels_identical': bool(
            metadata.get('score_only_labels_identical', False)),
        'instance_count_matches': (
            int(metadata.get('num_instances', -1)) == int(expected_instances)),
    }
    if not all(checks.values()):
        raise ValueError(
            'Quality scores must come from an unchanged score-only control: '
            f'{checks}.')
    return checks


def top_ratio_masks(scores, instance_ids, keep_ratio):
    scores = np.asarray(scores, dtype=np.float64)
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    keep_ratio = float(keep_ratio)
    if scores.shape != instance_ids.shape or scores.ndim != 1:
        raise ValueError('Scores and instance IDs must be aligned 1D arrays.')
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError('keep_ratio must be in (0, 1].')
    keep_count = max(1, int(np.ceil(keep_ratio * len(scores))))
    order = np.lexsort((instance_ids, -scores))
    high_mask = np.zeros(len(scores), dtype=bool)
    high_mask[order[:keep_count]] = True
    return high_mask, ~high_mask


def build_oracle_merge_map(tables, scores, keep_ratio):
    high_mask, low_mask = top_ratio_masks(
        scores, tables['pred_ids'], keep_ratio)
    intersections = np.asarray(tables['intersections'])
    iou = np.asarray(tables['iou'])
    dominant_gt = np.argmax(iou, axis=1)
    dominant_overlap = intersections[
        np.arange(len(tables['pred_ids'])), dominant_gt]
    high_indices = np.flatnonzero(high_mask)
    merge_map = {}
    for source in np.flatnonzero(low_mask):
        gt_index = dominant_gt[source]
        if dominant_overlap[source] <= 0:
            continue
        candidates = high_indices[
            (dominant_gt[high_indices] == gt_index) &
            (intersections[high_indices, gt_index] > 0)]
        if len(candidates) == 0:
            continue
        candidate_iou = iou[candidates, gt_index]
        candidate_ids = tables['pred_ids'][candidates]
        order = np.lexsort((candidate_ids, -candidate_iou))
        merge_map[int(source)] = int(candidates[order[0]])
    return merge_map, high_mask, dominant_gt


def merge_contingency_rows(tables, merge_map):
    num_predictions = len(tables['pred_ids'])
    roots = np.arange(num_predictions, dtype=np.int64)
    for source, target in merge_map.items():
        if source == target:
            raise ValueError('A prediction cannot merge into itself.')
        if target in merge_map:
            raise ValueError('Merge targets must not be merge sources.')
        roots[int(source)] = int(target)
    retained_roots = np.unique(roots)
    root_positions = np.searchsorted(retained_roots, roots)
    intersections = np.zeros(
        (len(retained_roots), len(tables['gt_ids'])), dtype=np.int64)
    pred_sizes = np.zeros(len(retained_roots), dtype=np.int64)
    np.add.at(intersections, root_positions, tables['intersections'])
    np.add.at(pred_sizes, root_positions, tables['pred_sizes'])
    gt_sizes = np.asarray(tables['gt_sizes'], dtype=np.int64)
    precision = np.divide(
        intersections, pred_sizes[:, None], dtype=np.float64)
    recall = np.divide(
        intersections, gt_sizes[None, :], dtype=np.float64)
    unions = pred_sizes[:, None] + gt_sizes[None, :] - intersections
    iou = np.divide(
        intersections, unions,
        out=np.zeros_like(intersections, dtype=np.float64), where=unions > 0)
    return {
        'gt_ids': np.asarray(tables['gt_ids'], dtype=np.int64),
        'pred_ids': np.asarray(tables['pred_ids'], dtype=np.int64)[
            retained_roots],
        'gt_sizes': gt_sizes,
        'pred_sizes': pred_sizes,
        'intersections': intersections,
        'precision': precision,
        'recall': recall,
        'iou': iou,
    }


def detection_status_by_id(evaluation):
    detection = evaluation['detection_results']
    matched = set(np.asarray(
        detection['matched_preds'], dtype=np.int64).tolist())
    counted_false = set(np.asarray(
        detection['non_matched_preds_filtered'], dtype=np.int64).tolist())
    all_unmatched = set(np.asarray(
        detection['non_matched_preds'], dtype=np.int64).tolist())
    return {
        int(instance_id): (
            'tp' if int(instance_id) in matched else
            'fp_counted' if int(instance_id) in counted_false else
            'unmatched_ignored' if int(instance_id) in all_unmatched else
            'not_referenced')
        for instance_id in matched | all_unmatched
    }


def summarize_statuses(instance_ids, indices, statuses):
    counts = {
        'tp': 0,
        'fp_counted': 0,
        'unmatched_ignored': 0,
        'not_referenced': 0,
    }
    for index in indices:
        status = statuses.get(int(instance_ids[index]), 'not_referenced')
        counts[status] += 1
    return counts


def build_gate(baseline, oracle, settings):
    f1_gain = oracle['f1_score'] - baseline['f1_score']
    commission_reduction = (
        baseline['commission_error_rate'] -
        oracle['commission_error_rate'])
    completeness_drop = (
        baseline['completeness'] - oracle['completeness'])
    gate = {
        'minimum_f1_gain_pp': float(settings['min_f1_gain_pp']),
        'minimum_commission_reduction_pp': float(
            settings['min_commission_reduction_pp']),
        'maximum_completeness_drop_pp': float(
            settings['max_completeness_drop_pp']),
        'f1_gain_pp': float(f1_gain),
        'commission_reduction_pp': float(commission_reduction),
        'completeness_drop_pp': float(completeness_drop),
        'f1_gain_passed': bool(
            f1_gain >= float(settings['min_f1_gain_pp']) - 1e-12),
        'commission_reduction_passed': bool(
            commission_reduction >=
            float(settings['min_commission_reduction_pp']) - 1e-12),
        'completeness_passed': bool(
            completeness_drop <=
            float(settings['max_completeness_drop_pp']) + 1e-12),
    }
    gate['passed'] = bool(
        gate['f1_gain_passed'] and
        gate['commission_reduction_passed'] and
        gate['completeness_passed'])
    return gate


def format_markdown(report):
    baseline = report['baseline']
    oracle = report['merge_oracle']
    gate = report['gate']
    return '\n'.join([
        f"# E8a 质量引导合并 Oracle：{report['name']}",
        '',
        f"- 数据角色：**{report['dataset_role']}**",
        f"- 锁定低质量比例：{1.0 - report['keep_ratio']:.4f}",
        f"- 低质量实例：{report['num_low_quality']} / "
        f"{report['num_predictions']}",
        f"- Oracle 可合并实例：{report['num_oracle_merged']}",
        f"- 未找到安全目标并保留：{report['num_low_retained']}",
        '',
        '| Run | TP | FP | FN | Completeness | Commission | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| baseline | {baseline['tp']} | {baseline['fp']} | "
        f"{baseline['fn']} | {baseline['completeness']:.6f}% | "
        f"{baseline['commission_error_rate']:.6f}% | "
        f"{baseline['f1_score']:.6f}% |",
        f"| merge_oracle | {oracle['tp']} | {oracle['fp']} | "
        f"{oracle['fn']} | {oracle['completeness']:.6f}% | "
        f"{oracle['commission_error_rate']:.6f}% | "
        f"{oracle['f1_score']:.6f}% |",
        '',
        '## 低质量实例构成',
        '',
        f"- TP：{report['low_quality_status']['tp']}",
        f"- counted FP：{report['low_quality_status']['fp_counted']}",
        f"- ignored unmatched："
        f"{report['low_quality_status']['unmatched_ignored']}",
        '',
        '## Oracle 合并来源构成',
        '',
        f"- TP：{report['merged_source_status']['tp']}",
        f"- counted FP：{report['merged_source_status']['fp_counted']}",
        f"- ignored unmatched："
        f"{report['merged_source_status']['unmatched_ignored']}",
        '',
        '## Gate',
        '',
        *[f'- {key}: **{value}**' for key, value in gate.items()],
        '',
        ('PASS：进入 E8b validation 邻接对数据生成与无标签合并规则。'
         if gate['passed'] else
         'STOP：合并 Oracle 上限不足，停止实例修复路线。'),
        '',
    ])


def write_assignments(path, tables, scores, merge_map, dominant_gt, statuses):
    fields = [
        'source_instance_id', 'target_instance_id', 'quality_score',
        'dominant_gt_id', 'source_status']
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for source, target in sorted(merge_map.items()):
            writer.writerow({
                'source_instance_id': int(tables['pred_ids'][source]),
                'target_instance_id': int(tables['pred_ids'][target]),
                'quality_score': float(scores[source]),
                'dominant_gt_id': int(
                    tables['gt_ids'][dominant_gt[source]]),
                'source_status': statuses.get(
                    int(tables['pred_ids'][source]), 'not_referenced'),
            })


def validate_config_runs(config):
    runs = config.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('E8a config requires a non-empty runs list.')
    errors = []
    names = []
    for index, spec in enumerate(runs):
        if not isinstance(spec, dict):
            errors.append(f'runs[{index}] is not a mapping')
            continue
        missing = [name for name in REQUIRED_RUN_FIELDS if not spec.get(name)]
        if missing:
            errors.append(f'runs[{index}] missing {missing}')
        names.append(spec.get('name'))
    valid_names = [name for name in names if name]
    if len(valid_names) != len(set(valid_names)):
        errors.append('run names must be unique')
    if not any(
            isinstance(spec, dict) and spec.get('primary_gate', False)
            for spec in runs):
        errors.append('at least one run must set primary_gate: true')
    if errors:
        raise ValueError('Invalid E8a config: ' + '; '.join(errors))
    return runs


def run_one(spec, settings):
    evaluation_path = Path(spec['evaluation'])
    paths = {
        'ground_truth': Path(spec['ground_truth']),
        'evaluation': evaluation_path,
        'quality_scores': Path(spec['quality_scores']),
        'quality_metadata': Path(spec['quality_metadata']),
        'propagated_predictions': resolve_propagated_path(spec),
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            'Required E8a inputs are missing: ' + ', '.join(missing_paths))

    print(f"===== E8a merge Oracle: {spec['name']} =====", flush=True)
    accumulator = scan_aligned_las(
        paths['ground_truth'], paths['propagated_predictions'],
        int(settings['chunk_size']),
        float(settings['coordinate_tolerance_m']))
    tables = contingency_from_accumulator(accumulator)
    evaluation = load_evaluation(paths['evaluation'])
    baseline = evaluate_threshold(
        tables, np.ones(len(tables['pred_ids'])), 0.0,
        float(settings['min_iou_for_match']),
        float(settings['min_precision_for_pred']))
    validate_baseline(baseline, evaluation_artifact_counts(evaluation))

    quality_ids, quality_values = load_quality_scores(paths['quality_scores'])
    with paths['quality_metadata'].open(encoding='utf-8') as file:
        quality_metadata = json.load(file)
    score_only_checks = validate_score_only_metadata(
        quality_metadata, len(quality_ids))
    scores = align_quality_scores(
        tables['pred_ids'], quality_ids, quality_values)
    merge_map, high_mask, dominant_gt = build_oracle_merge_map(
        tables, scores, float(settings['keep_ratio']))
    merged_tables = merge_contingency_rows(tables, merge_map)
    oracle = evaluate_threshold(
        merged_tables, np.ones(len(merged_tables['pred_ids'])), 0.0,
        float(settings['min_iou_for_match']),
        float(settings['min_precision_for_pred']))
    statuses = detection_status_by_id(evaluation)
    low_indices = np.flatnonzero(~high_mask)
    merged_indices = np.asarray(sorted(merge_map), dtype=np.int64)
    gate = build_gate(baseline, oracle, settings)
    report = {
        'name': str(spec['name']),
        'dataset_role': str(spec['dataset_role']),
        'keep_ratio': float(settings['keep_ratio']),
        'num_predictions': int(len(tables['pred_ids'])),
        'num_low_quality': int(len(low_indices)),
        'num_oracle_merged': int(len(merge_map)),
        'num_low_retained': int(len(low_indices) - len(merge_map)),
        'low_quality_status': summarize_statuses(
            tables['pred_ids'], low_indices, statuses),
        'merged_source_status': summarize_statuses(
            tables['pred_ids'], merged_indices, statuses),
        'baseline': baseline,
        'merge_oracle': oracle,
        'gate': gate,
        'score_only_checks': score_only_checks,
        'provenance': {key: str(value) for key, value in paths.items()},
        'max_coordinate_difference_m': float(
            accumulator['max_coordinate_difference_m']),
    }
    output_dir = Path(spec['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_assignments(
        output_dir / 'oracle_merge_assignments.csv',
        tables, scores, merge_map, dominant_gt, statuses)
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    (output_dir / 'results.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_default),
        encoding='utf-8')
    print(markdown)
    return report


def run_config(path):
    config = load_yaml(path)
    settings = {
        key: config[key] for key in [
            'keep_ratio', 'min_iou_for_match', 'min_precision_for_pred',
            'min_f1_gain_pp', 'min_commission_reduction_pp',
            'max_completeness_drop_pp', 'chunk_size',
            'coordinate_tolerance_m']}
    runs = validate_config_runs(config)
    reports = {spec['name']: run_one(spec, settings) for spec in runs}
    primary = [
        spec['name'] for spec in runs if spec.get('primary_gate', False)]
    if not primary:
        raise ValueError('At least one E8a run must set primary_gate: true.')
    passed = bool(all(reports[name]['gate']['passed'] for name in primary))
    summary = {
        'config': str(path),
        'primary_runs': primary,
        'runs': reports,
        'passed': passed,
    }
    output_dir = Path(config['summary_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'e8a_merge_oracle_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding='utf-8')
    lines = ['# E8a 质量引导合并 Oracle 汇总', '']
    for name, report in reports.items():
        lines.append(
            f"- {name}: F1 {report['baseline']['f1_score']:.6f}% → "
            f"{report['merge_oracle']['f1_score']:.6f}%，"
            f"Commission {report['baseline']['commission_error_rate']:.6f}% → "
            f"{report['merge_oracle']['commission_error_rate']:.6f}%，"
            f"Gate={report['gate']['passed']}")
    lines.extend(['', f'- Primary Gate：**{passed}**', ''])
    text = '\n'.join(lines)
    (output_dir / 'e8a_merge_oracle_summary.md').write_text(
        text, encoding='utf-8')
    print(text)
    if not passed:
        raise RuntimeError('E8a merge Oracle gate failed.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the upper bound of quality-guided merging.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run_config(args.config)


if __name__ == '__main__':
    main()