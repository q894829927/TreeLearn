import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


DEFAULT_THRESHOLDS = [value / 20 for value in range(21)]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the upper bound of oracle instance filtering.')
    parser.add_argument('--config')
    parser.add_argument('--predictions')
    parser.add_argument('--ground_truth')
    parser.add_argument('--evaluation')
    parser.add_argument('--propagated_predictions')
    parser.add_argument('--output_dir')
    parser.add_argument('--thresholds', nargs='*', type=float)
    parser.add_argument('--min_iou_for_match', type=float, default=0.5)
    parser.add_argument('--min_precision_for_pred', type=float, default=0.5)
    parser.add_argument('--max_completeness_drop_pp', type=float, default=1.0)
    parser.add_argument('--min_f1_gain_pp', type=float, default=1.0)
    parser.add_argument('--chunk_size', type=int, default=2_000_000)
    parser.add_argument(
        '--coordinate_tolerance_m', type=float, default=0.002)
    return parser.parse_args()


def load_yaml(path):
    import yaml

    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_evaluation(path):
    import torch

    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f'Object of type {value.__class__.__name__} is not JSON serializable')


def git_value(*args):
    try:
        return subprocess.check_output(
            ['git', *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _update_counts(target, values):
    if len(values) == 0:
        return
    unique, counts = np.unique(values, return_counts=True)
    for value, count in zip(unique, counts):
        key = int(value)
        target[key] = target.get(key, 0) + int(count)


def _update_pair_counts(target, gt_labels, pred_labels):
    mask = (gt_labels > 0) & (pred_labels > 0)
    if not np.any(mask):
        return
    encoded = (
        (pred_labels[mask].astype(np.uint64) << np.uint64(32)) |
        gt_labels[mask].astype(np.uint64)
    )
    unique, counts = np.unique(encoded, return_counts=True)
    for value, count in zip(unique, counts):
        integer = int(value)
        pair = (integer >> 32, integer & 0xffffffff)
        target[pair] = target.get(pair, 0) + int(count)


def new_accumulator():
    return {
        'point_count': 0,
        'max_coordinate_difference_m': 0.0,
        'gt_counts': {},
        'pred_counts': {},
        'pair_counts': {},
    }


def accumulate_label_chunk(accumulator, gt_labels, pred_labels):
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    pred_labels = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
    if gt_labels.shape != pred_labels.shape:
        raise ValueError('Ground-truth and prediction chunks differ in size.')
    accumulator['point_count'] += int(len(gt_labels))
    _update_counts(accumulator['gt_counts'], gt_labels[gt_labels > 0])
    _update_counts(accumulator['pred_counts'], pred_labels[pred_labels > 0])
    _update_pair_counts(
        accumulator['pair_counts'], gt_labels, pred_labels)


def _label_dimensions(reader):
    names = [str(name) for name in reader.header.point_format.dimension_names]
    lookup = {name.lower(): name for name in names}
    tree_dimension = lookup.get('treeid')
    class_dimension = lookup.get('classification')
    if tree_dimension is None or class_dimension is None:
        raise ValueError(
            f'{reader.source.name} must contain treeID and classification.')
    return tree_dimension, class_dimension


def _labels_from_las_points(points, tree_dimension, class_dimension):
    tree_ids = np.asarray(points[tree_dimension], dtype=np.int64)
    classes = np.asarray(points[class_dimension])
    labels = -np.ones(len(points), dtype=np.int64)
    tree_mask = tree_ids != 0
    labels[tree_mask] = tree_ids[tree_mask]
    labels[(~tree_mask) & np.isin(classes, [1, 2])] = 0
    return labels


def scan_aligned_las(
        ground_truth, propagated_predictions, chunk_size,
        coordinate_tolerance_m=0.002):
    import laspy

    accumulator = new_accumulator()
    with laspy.open(ground_truth) as gt_reader, laspy.open(
            propagated_predictions) as pred_reader:
        if gt_reader.header.point_count != pred_reader.header.point_count:
            raise ValueError(
                'Propagated predictions and ground truth have different '
                'point counts.')
        total_points = int(gt_reader.header.point_count)
        print(
            f'Scanning {total_points:,} aligned GT/prediction points...',
            flush=True)
        gt_dimensions = _label_dimensions(gt_reader)
        pred_dimensions = _label_dimensions(pred_reader)
        gt_iterator = gt_reader.chunk_iterator(chunk_size)
        pred_iterator = pred_reader.chunk_iterator(chunk_size)
        chunk_index = 0
        while True:
            gt_points = next(gt_iterator, None)
            pred_points = next(pred_iterator, None)
            if gt_points is None or pred_points is None:
                if gt_points is not None or pred_points is not None:
                    raise ValueError('LAS chunk iterators ended differently.')
                break
            chunk_index += 1
            if len(gt_points) != len(pred_points):
                raise ValueError(f'Chunk {chunk_index} differs in size.')
            for axis in ('x', 'y', 'z'):
                gt_axis = np.asarray(getattr(gt_points, axis))
                pred_axis = np.asarray(getattr(pred_points, axis))
                maximum_difference = float(np.max(np.abs(
                    gt_axis - pred_axis))) if len(gt_axis) else 0.0
                accumulator['max_coordinate_difference_m'] = max(
                    accumulator['max_coordinate_difference_m'],
                    maximum_difference)
                if maximum_difference > coordinate_tolerance_m:
                    raise ValueError(
                        f'Coordinates differ in chunk {chunk_index}, axis {axis}: '
                        f'max difference {maximum_difference:.6f} m exceeds '
                        f'{coordinate_tolerance_m:.6f} m.')
            gt_labels = _labels_from_las_points(
                gt_points, *gt_dimensions)
            pred_labels = _labels_from_las_points(
                pred_points, *pred_dimensions)
            accumulate_label_chunk(accumulator, gt_labels, pred_labels)
            processed = accumulator['point_count']
            if chunk_index == 1 or chunk_index % 5 == 0 or processed == total_points:
                print(
                    f'  processed {processed:,}/{total_points:,} points '
                    f'({100 * processed / total_points:.1f}%)',
                    flush=True)
    return accumulator


def contingency_from_accumulator(accumulator):
    gt_ids = np.asarray(
        sorted(accumulator['gt_counts']), dtype=np.int64)
    pred_ids = np.asarray(
        sorted(accumulator['pred_counts']), dtype=np.int64)
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        raise ValueError('At least one GT and predicted instance are required.')
    gt_sizes = np.asarray(
        [accumulator['gt_counts'][int(value)] for value in gt_ids],
        dtype=np.int64)
    pred_sizes = np.asarray(
        [accumulator['pred_counts'][int(value)] for value in pred_ids],
        dtype=np.int64)
    intersections = np.zeros(
        (len(pred_ids), len(gt_ids)), dtype=np.int64)
    for (pred_id, gt_id), count in accumulator['pair_counts'].items():
        pred_index = int(np.searchsorted(pred_ids, pred_id))
        gt_index = int(np.searchsorted(gt_ids, gt_id))
        intersections[pred_index, gt_index] = count

    precision = np.divide(
        intersections, pred_sizes[:, None], dtype=np.float64)
    recall = np.divide(
        intersections, gt_sizes[None, :], dtype=np.float64)
    unions = pred_sizes[:, None] + gt_sizes[None, :] - intersections
    iou = np.divide(
        intersections, unions,
        out=np.zeros_like(intersections, dtype=np.float64), where=unions > 0)
    return {
        'gt_ids': gt_ids,
        'pred_ids': pred_ids,
        'gt_sizes': gt_sizes,
        'pred_sizes': pred_sizes,
        'intersections': intersections,
        'precision': precision,
        'recall': recall,
        'iou': iou,
    }


def contingency_from_arrays(gt_labels, pred_labels):
    accumulator = new_accumulator()
    accumulate_label_chunk(accumulator, gt_labels, pred_labels)
    return contingency_from_accumulator(accumulator)


def evaluate_threshold(
        tables, qualities, threshold,
        min_iou_for_match=0.5, min_precision_for_pred=0.5):
    from scipy.optimize import linear_sum_assignment

    keep = np.flatnonzero(qualities >= threshold)
    num_gt = len(tables['gt_ids'])
    if len(keep):
        subset = tables['iou'][keep]
        matched_rows, matched_gt = linear_sum_assignment(
            subset, maximize=True)
        valid = subset[matched_rows, matched_gt] > min_iou_for_match
        matched_kept_rows = matched_rows[valid]
        matched_count = int(valid.sum())
        matched_mask = np.zeros(len(keep), dtype=bool)
        matched_mask[matched_kept_rows] = True
        labeled_precision = tables['precision'][keep].sum(axis=1)
        counted_false = int(np.count_nonzero(
            (~matched_mask) &
            (labeled_precision >= min_precision_for_pred)))
    else:
        matched_count = 0
        counted_false = 0

    missed_count = num_gt - matched_count
    detection_denominator = matched_count + counted_false
    detection_precision = (
        matched_count / detection_denominator
        if detection_denominator else 0.0)
    completeness = matched_count / num_gt
    commission = (
        counted_false / detection_denominator
        if detection_denominator else 0.0)
    f1_score = (
        2 * detection_precision * completeness /
        (detection_precision + completeness)
        if detection_precision + completeness else 0.0)
    return {
        'threshold': float(threshold),
        'retained_predictions': int(len(keep)),
        'removed_predictions': int(len(qualities) - len(keep)),
        'tp': matched_count,
        'fp': counted_false,
        'fn': missed_count,
        'completeness': 100.0 * completeness,
        'commission_error_rate': 100.0 * commission,
        'f1_score': 100.0 * f1_score,
    }


def evaluation_artifact_counts(evaluation):
    detection = evaluation['detection_results']
    return {
        'tp': int(len(detection['matched_preds'])),
        'fp': int(len(detection['non_matched_preds_filtered'])),
        'fn': int(len(detection['non_matched_gts'])),
        'predictions': int(
            len(detection['matched_preds']) +
            len(detection['non_matched_preds'])),
    }


def validate_baseline(baseline, artifact_counts):
    observed = {
        key: baseline[key] for key in ('tp', 'fp', 'fn')}
    observed['predictions'] = baseline['retained_predictions']
    if observed != artifact_counts:
        raise ValueError(
            'Oracle baseline does not reproduce the evaluation artifact. '
            f'computed={observed}, artifact={artifact_counts}')


def select_oracle(rows, max_completeness_drop_pp):
    baseline = rows[0]
    minimum_completeness = (
        baseline['completeness'] - max_completeness_drop_pp)
    eligible = [
        row for row in rows
        if row['completeness'] >= minimum_completeness - 1e-12]
    return max(
        eligible,
        key=lambda row: (
            row['f1_score'], row['completeness'], -row['threshold']))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_curve(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    completeness = [row['completeness'] for row in rows]
    f1_score = [row['f1_score'] for row in rows]
    thresholds = [row['threshold'] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(completeness, f1_score, marker='o', linewidth=1.5)
    for x_value, y_value, threshold in zip(
            completeness, f1_score, thresholds):
        axis.annotate(
            f'{threshold:.2f}', (x_value, y_value),
            fontsize=7, xytext=(3, 3), textcoords='offset points')
    axis.set_xlabel('Completeness (%)')
    axis.set_ylabel('Detection F1 (%)')
    axis.set_title('Oracle IoU filtering upper bound')
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def format_markdown(report):
    baseline = report['baseline']
    best = report['best_oracle']
    rows = [
        f"# E1 Oracle IoU 上限：{report['name']}", '',
        f"- baseline F1：{baseline['f1_score']:.6f}%",
        f"- best Oracle F1：{best['f1_score']:.6f}%",
        f"- F1 提升：{report['gate']['f1_gain_pp']:.6f} 个百分点",
        f"- baseline Completeness：{baseline['completeness']:.6f}%",
        f"- best Completeness：{best['completeness']:.6f}%",
        f"- Oracle threshold：{best['threshold']:.2f}",
        f"- Gate：**{report['gate']['passed']}**", '',
        '| Threshold | Retained | TP | FP | FN | Completeness | Commission | F1 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in report['threshold_results']:
        rows.append(
            f"| {item['threshold']:.2f} | {item['retained_predictions']} | "
            f"{item['tp']} | {item['fp']} | {item['fn']} | "
            f"{item['completeness']:.6f}% | "
            f"{item['commission_error_rate']:.6f}% | "
            f"{item['f1_score']:.6f}% |")
    rows.extend(['', (
        'PASS：可以进入 E2。' if report['gate']['passed'] else
        'STOP：Oracle 上限不足，不进入 E2。'), ''])
    return '\n'.join(rows)


def resolve_propagated_path(spec):
    explicit = spec.get('propagated_predictions')
    if explicit:
        return Path(explicit)
    return (
        Path(spec['evaluation']).parent /
        'pred_forest_propagated_to_gt_pointcloud.laz')


def run_oracle(name, spec, settings):
    required = ['predictions', 'ground_truth', 'evaluation', 'output_dir']
    missing_fields = [key for key in required if not spec.get(key)]
    if missing_fields:
        raise ValueError(f'Missing run fields: {missing_fields}')
    paths = {key: Path(spec[key]) for key in required[:3]}
    paths['propagated_predictions'] = resolve_propagated_path(spec)
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            'Required E1 inputs are missing: ' + ', '.join(missing_paths))

    print(f'===== E1 Oracle: {name} =====', flush=True)
    accumulator = scan_aligned_las(
        paths['ground_truth'], paths['propagated_predictions'],
        int(settings['chunk_size']),
        float(settings['coordinate_tolerance_m']))
    tables = contingency_from_accumulator(accumulator)
    quality = tables['iou'].max(axis=1)
    print(
        f"Built IoU table for {len(tables['pred_ids']):,} predictions and "
        f"{len(tables['gt_ids']):,} GT trees.",
        flush=True)
    best_gt_indices = tables['iou'].argmax(axis=1)
    best_gt_ids = tables['gt_ids'][best_gt_indices]
    row_indices = np.arange(len(tables['pred_ids']))
    instance_rows = []
    for index in row_indices:
        gt_index = best_gt_indices[index]
        instance_rows.append({
            'instance_id': int(tables['pred_ids'][index]),
            'oracle_max_iou': float(quality[index]),
            'best_gt_id': int(best_gt_ids[index]),
            'precision_at_best_gt': float(
                tables['precision'][index, gt_index]),
            'recall_at_best_gt': float(
                tables['recall'][index, gt_index]),
            'num_points': int(tables['pred_sizes'][index]),
        })

    threshold_rows = [
        evaluate_threshold(
            tables, quality, threshold,
            float(settings['min_iou_for_match']),
            float(settings['min_precision_for_pred']))
        for threshold in settings['thresholds']]
    baseline = next(
        row for row in threshold_rows
        if np.isclose(row['threshold'], 0.0))
    artifact = load_evaluation(paths['evaluation'])
    artifact_counts = evaluation_artifact_counts(artifact)
    validate_baseline(baseline, artifact_counts)
    best = select_oracle(
        threshold_rows, float(settings['max_completeness_drop_pp']))
    f1_gain = best['f1_score'] - baseline['f1_score']
    completeness_drop = baseline['completeness'] - best['completeness']
    gate = {
        'minimum_f1_gain_pp': float(settings['min_f1_gain_pp']),
        'maximum_completeness_drop_pp': float(
            settings['max_completeness_drop_pp']),
        'f1_gain_pp': float(f1_gain),
        'completeness_drop_pp': float(completeness_drop),
        'f1_gain_passed': bool(
            f1_gain + 1e-12 >= float(settings['min_f1_gain_pp'])),
        'completeness_passed': bool(
            completeness_drop <=
            float(settings['max_completeness_drop_pp']) + 1e-12),
    }
    gate['passed'] = bool(
        gate['f1_gain_passed'] and gate['completeness_passed'])
    report = {
        'name': name,
        'method': {
            'quality': 'maximum pointwise IoU with any labeled GT tree',
            'threshold_comparison': 'oracle_max_iou >= threshold',
            'thresholds_fixed_before_run': settings['thresholds'],
            'min_iou_for_match': settings['min_iou_for_match'],
            'min_precision_for_pred': settings['min_precision_for_pred'],
            'coordinate_tolerance_m': settings['coordinate_tolerance_m'],
            'max_coordinate_difference_m': accumulator['max_coordinate_difference_m'],
        },
        'provenance': {
            'git_branch': git_value('branch', '--show-current'),
            'git_commit': git_value('rev-parse', 'HEAD'),
            **{key: str(value) for key, value in paths.items()},
        },
        'point_count': accumulator['point_count'],
        'num_gt_instances': int(len(tables['gt_ids'])),
        'max_coordinate_difference_m': accumulator['max_coordinate_difference_m'],
        'num_pred_instances': int(len(tables['pred_ids'])),
        'artifact_counts': artifact_counts,
        'baseline': baseline,
        'best_oracle': best,
        'gate': gate,
        'threshold_results': threshold_rows,
    }

    output_dir = Path(spec['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'oracle_thresholds.csv', threshold_rows)
    write_csv(output_dir / 'oracle_instance_quality.csv', instance_rows)
    save_curve(output_dir / 'oracle_f1_completeness.png', threshold_rows)
    (output_dir / 'oracle_results.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_default),
        encoding='utf-8')
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    return report


def settings_from_args(args):
    return {
        'thresholds': args.thresholds or DEFAULT_THRESHOLDS,
        'min_iou_for_match': args.min_iou_for_match,
        'min_precision_for_pred': args.min_precision_for_pred,
        'max_completeness_drop_pp': args.max_completeness_drop_pp,
        'min_f1_gain_pp': args.min_f1_gain_pp,
        'chunk_size': args.chunk_size,
        'coordinate_tolerance_m': args.coordinate_tolerance_m,
    }


def run_config(path):
    config = load_yaml(path)
    settings = {
        key: config[key] for key in [
            'thresholds', 'min_iou_for_match',
            'min_precision_for_pred', 'max_completeness_drop_pp',
            'min_f1_gain_pp', 'chunk_size', 'coordinate_tolerance_m']}
    reports = {}
    primary = []
    for spec in config['runs']:
        name = spec['name']
        reports[name] = run_oracle(name, spec, settings)
        if spec.get('primary_gate', False):
            primary.append(name)
    if not primary:
        raise ValueError('At least one run must set primary_gate: true.')
    aggregate = {
        'config': str(path),
        'primary_runs': primary,
        'runs': reports,
        'passed': bool(all(reports[name]['gate']['passed'] for name in primary)),
    }
    output_dir = Path(config.get(
        'summary_output_dir', 'logs/vertical_quality'))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'e1_oracle_summary.json').write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, default=json_default),
        encoding='utf-8')
    summary_lines = ['# E1 Oracle 汇总', '']
    for name, report in reports.items():
        summary_lines.append(
            f"- {name}: baseline F1={report['baseline']['f1_score']:.6f}%, "
            f"best F1={report['best_oracle']['f1_score']:.6f}%, "
            f"gate={report['gate']['passed']}")
    summary_lines.extend(['', f"- Primary Gate：**{aggregate['passed']}**", ''])
    summary = '\n'.join(summary_lines)
    (output_dir / 'e1_oracle_summary.md').write_text(
        summary, encoding='utf-8')
    print(summary)
    return aggregate


def main():
    args = parse_args()
    if args.config:
        run_config(args.config)
        return
    spec = {
        'predictions': args.predictions,
        'ground_truth': args.ground_truth,
        'evaluation': args.evaluation,
        'propagated_predictions': args.propagated_predictions,
        'output_dir': args.output_dir,
    }
    run_oracle('direct', spec, settings_from_args(args))


if __name__ == '__main__':
    main()
