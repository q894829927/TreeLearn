"""CPU-only audit of saved final masks; never load a model or change predictions."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml


ROOT = Path(__file__).resolve().parents[2]
METRICS = ('f1', 'completeness', 'commission', 'coverage', 'point_precision', 'point_recall')


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.zeros(numerator.shape, dtype=float),
                     where=denominator > 0)


def scan_arrays(coords, labels, predictions, chunk_points=1_000_000, max_pairs=20_000_000):
    """Two bounded scans: discover IDs, then accumulate sufficient statistics."""
    if chunk_points <= 0 or max_pairs <= 0:
        raise ValueError('chunk_points and max_pairs must be positive.')
    if labels.ndim != 1 or not len(labels) or coords.shape != (len(labels), 3):
        raise ValueError('Expected nonempty aligned coordinates and GT labels.')
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(coords.dtype, np.floating):
        raise TypeError('Expected integer GT labels and floating coordinates.')
    for values in predictions.values():
        if values.shape != labels.shape or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Predictions must be aligned integer labels.')
    gt_ids, pred_ids = set(), {name: set() for name in predictions}
    ignored = 0
    for start in range(0, len(labels), chunk_points):
        stop = min(start + chunk_points, len(labels))
        gt = labels[start:stop]
        gt_ids.update(np.unique(gt[gt > 0]).tolist())
        ignored += int(np.count_nonzero(gt < 0))
        for name, values in predictions.items():
            values = values[start:stop]
            if np.any(values < 0):
                raise ValueError('Completed final predictions must be nonnegative.')
            pred_ids[name].update(np.unique(values[values > 0]).tolist())
        if start == 0 or stop == len(labels) or (start // chunk_points + 1) % 10 == 0:
            print(f'SCAN_IDS points={stop}/{len(labels)}', flush=True)
    gt_ids = np.asarray(sorted(gt_ids), dtype=np.int64)
    if not len(gt_ids):
        raise ValueError('No labeled GT trees.')
    tables = {}
    for name, identifiers in pred_ids.items():
        identifiers = np.asarray(sorted(identifiers), dtype=np.int64)
        if len(identifiers) * len(gt_ids) > max_pairs:
            raise ValueError('Diagnostic contingency exceeds max_pairs.')
        tables[name] = dict(ids=identifiers, total=np.zeros(len(identifiers), dtype=np.int64),
                            known=np.zeros(len(identifiers), dtype=np.int64),
                            intersections=np.zeros((len(identifiers), len(gt_ids)), dtype=np.int64))
    counts = np.zeros(len(gt_ids), dtype=np.int64)
    low, high = np.full(len(gt_ids), np.inf), np.full(len(gt_ids), -np.inf)
    for start in range(0, len(labels), chunk_points):
        stop = min(start + chunk_points, len(labels))
        gt = labels[start:stop]
        tree = gt > 0
        positions = np.searchsorted(gt_ids, gt[tree])
        counts += np.bincount(positions, minlength=len(gt_ids))
        z = coords[start:stop, 2][tree]
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite coordinates for a labeled tree.')
        np.minimum.at(low, positions, z)
        np.maximum.at(high, positions, z)
        for name, values in predictions.items():
            table = tables[name]
            pred = values[start:stop]
            positive = pred > 0
            rows = np.searchsorted(table['ids'], pred[positive])
            table['total'] += np.bincount(rows, minlength=len(table['ids']))
            table['known'] += np.bincount(rows[gt[positive] >= 0], minlength=len(table['ids']))
            both = tree & positive
            encoded = (np.searchsorted(table['ids'], pred[both]) * len(gt_ids)
                       + np.searchsorted(gt_ids, gt[both]))
            table['intersections'] += np.bincount(
                encoded, minlength=len(table['ids']) * len(gt_ids)).reshape(table['intersections'].shape)
        if start == 0 or stop == len(labels) or (start // chunk_points + 1) % 10 == 0:
            print(f'SCAN_COUNTS points={stop}/{len(labels)}', flush=True)
    return dict(ids=gt_ids, counts=counts, z_span=high - low,
                ignored_points=ignored, num_points=len(labels)), tables


def analyze_table(geometry, table):
    """Mirror aligned_voxels_ignore_negative_gt_iou_gt_0.5 exactly."""
    ids, counts = geometry['ids'], geometry['counts']
    active = np.flatnonzero(table['known'] > 0)
    intersection = table['intersections'][active]
    pc = table['known'][active]
    iou = ratio(intersection, pc[:, None] + counts[None, :] - intersection)
    rows, columns = linear_sum_assignment(-iou)
    valid = iou[rows, columns] > .5
    matched_rows, matched_columns = active[rows[valid]], columns[valid]
    detected = np.zeros(len(ids), dtype=bool)
    detected[matched_columns] = True
    pred_to_gt = {int(row): int(ids[col]) for row, col in zip(matched_rows, matched_columns)}
    tree_fraction = ratio(table['intersections'].sum(axis=1), table['known'])
    fp = tree_fraction >= .5
    fp[matched_rows] = False
    tp, false_positive, fn = len(matched_columns), int(fp.sum()), int((~detected).sum())
    coverage, precision, recall = np.zeros((3, len(ids)))
    best_rows = np.full(len(ids), -1, dtype=np.int64)
    if len(active):
        best = iou.argmax(axis=0)
        coverage = iou[best, np.arange(len(ids))]
        precision = ratio(intersection[best, np.arange(len(ids))], pc[best])
        recall = ratio(intersection[best, np.arange(len(ids))], counts)
        best_rows = np.where(coverage > 0, active[best], -1)
    covered = ratio(table['intersections'].sum(axis=0), counts)
    recalls = ratio(table['intersections'], counts[None, :])
    max_recall = recalls.max(axis=0) if len(recalls) else np.zeros(len(ids))
    half_recall = recalls >= .5
    covered_gt_per_pred = half_recall.sum(axis=1)
    categories, tree_rows = [], []
    for col, identifier in enumerate(ids):
        row = int(best_rows[col])
        if detected[col]:
            category = 'detected'
        elif covered[col] == 0:
            category = 'no_assigned_tree_points'
        elif coverage[col] > .5:
            category = 'unmatched_despite_valid_iou'
        elif recall[col] >= .5 and precision[col] <= .5:
            category = 'high_recall_low_purity'
        elif covered[col] >= .5 and max_recall[col] < .5:
            category = 'spread_across_predictions'
        elif covered[col] < .5:
            category = 'mostly_unassigned'
        else:
            category = 'other_below_iou_threshold'
        categories.append(category)
        known = int(table['known'][row]) if row >= 0 else 0
        tree_mass = int(table['intersections'][row].sum()) if row >= 0 else 0
        correct = int(table['intersections'][row, col]) if row >= 0 else 0
        tree_rows.append(dict(gt_id=int(identifier), detected=bool(detected[col]),
                              category=category, best_pred_id=int(table['ids'][row]) if row >= 0 else 0,
                              best_iou=float(coverage[col]), best_precision=float(precision[col]),
                              best_recall=float(recall[col]), assigned_tree_point_fraction=float(covered[col]),
                              max_single_prediction_recall=float(max_recall[col]),
                              best_other_tree_fraction=(tree_mass - correct) / max(known, 1),
                              best_background_fraction=(known - tree_mass) / max(known, 1),
                              best_pred_gt_at_half_recall=int(covered_gt_per_pred[row]) if row >= 0 else 0))
    pred_rows, accounting = [], Counter()
    for row, identifier in enumerate(table['ids']):
        if row in pred_to_gt:
            category = 'true_positive'
        elif fp[row]:
            category = 'false_positive'
        elif table['known'][row] == 0:
            category = 'ignored_only'
        else:
            category = 'excluded_low_tree_fraction'
        accounting[category] += 1
        col = int(table['intersections'][row].argmax())
        common = int(table['intersections'][row, col])
        pred_rows.append(dict(pred_id=int(identifier), category=category,
                              matched_gt_id=pred_to_gt.get(row, 0), total_points=int(table['total'][row]),
                              known_points=int(table['known'][row]),
                              ignored_fraction=1 - int(table['known'][row]) / int(table['total'][row]),
                              labeled_tree_fraction=float(tree_fraction[row]),
                              dominant_gt_id=int(ids[col]) if common else 0,
                              dominant_gt_precision=common / max(int(table['known'][row]), 1),
                              gt_at_half_recall=int(covered_gt_per_pred[row])))
    metrics = dict(tp=tp, fp=false_positive, fn=fn, num_gt=len(ids), num_predictions=len(active),
                   ignored_points=geometry['ignored_points'], f1=2 * tp / max(2 * tp + false_positive + fn, 1),
                   completeness=tp / len(ids), commission=false_positive / max(tp + false_positive, 1),
                   coverage=float(coverage.mean()), point_precision=float(precision.mean()),
                   point_recall=float(recall.mean()))
    for key in ('true_positive', 'false_positive', 'ignored_only', 'excluded_low_tree_fraction'):
        accounting.setdefault(key, 0)
    return dict(metrics=metrics, final_instance_accounting=dict(accounting),
                final_instance_count=len(table['ids']), tree_categories=dict(Counter(categories)),
                predictions_covering_multiple_gt_at_half_recall=int((covered_gt_per_pred >= 2).sum()),
                trees=tree_rows, predictions=pred_rows)


def compare_models(geometry, models):
    rows = []
    for col, identifier in enumerate(geometry['ids']):
        row = dict(gt_id=int(identifier), gt_points=int(geometry['counts'][col]),
                   z_span_m=float(geometry['z_span'][col]))
        for name, result in models.items():
            row.update({f'{name}_{key}': value for key, value in result['trees'][col].items() if key != 'gt_id'})
        row['transition'] = ('both_detected' if row['baseline_detected'] and row['refined_detected'] else
                             'lost' if row['baseline_detected'] else
                             'recovered' if row['refined_detected'] else 'both_missed')
        rows.append(row)
    groups = {}
    for name, lo, hi in (('lt_5m', 0, 5), ('5_to_10m', 5, 10), ('10_to_20m', 10, 20),
                         ('20_to_30m', 20, 30), ('ge_30m', 30, float('inf'))):
        selected = [row for row in rows if lo <= row['z_span_m'] < hi]
        group = dict(num_gt=len(selected), lost=sum(row['transition'] == 'lost' for row in selected),
                     recovered=sum(row['transition'] == 'recovered' for row in selected))
        for model in models:
            group[model] = {metric: (float(np.mean([row[f'{model}_{field}'] for row in selected]))
                                     if selected else None) for metric, field in (
                ('completeness', 'detected'), ('coverage', 'best_iou'),
                ('point_precision', 'best_precision'), ('point_recall', 'best_recall'))}
        groups[name] = group
    return rows, dict(transitions=dict(Counter(row['transition'] for row in rows)), z_span_groups=groups)


def assert_metrics(expected, actual):
    for key in ('tp', 'fp', 'fn', 'num_gt', 'num_predictions', 'ignored_points') + METRICS:
        if not np.isclose(expected[key], actual[key], rtol=0, atol=1e-10):
            raise ValueError(f'Saved evaluation and current arrays disagree: {key}: '
                             f'{expected[key]} != {actual[key]}. Check prediction/cache provenance.')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['pred_id'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def audit_forest(cache, prediction_dir, output, checkpoint_step, chunk_points=1_000_000):
    cache, prediction_dir, output = Path(cache), Path(prediction_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('brief.json', 'summary.json'):
        (output / filename).unlink(missing_ok=True)
    statistics = json.loads((prediction_dir / 'prediction.json').read_text(encoding='utf-8'))
    if statistics.get('partial', True):
        raise ValueError('Require a completed full-forest prediction.')
    expected = json.loads((prediction_dir / 'metrics.json').read_text(encoding='utf-8'))
    paths = dict(coords=cache / 'coords.npy', labels=cache / 'labels.npy', baseline=cache / 'baseline.npy',
                 refined=prediction_dir / 'refined_labels.npy')
    stamps = {name: dict(path=str(path.resolve()), size=path.stat().st_size,
                        mtime_ns=path.stat().st_mtime_ns) for name, path in paths.items()}
    arrays = {name: np.load(path, mmap_mode='r', allow_pickle=False) for name, path in paths.items()}
    try:
        geometry, tables = scan_arrays(arrays['coords'], arrays['labels'],
                                       {name: arrays[name] for name in ('baseline', 'refined')}, chunk_points)
        models = {name: analyze_table(geometry, table) for name, table in tables.items()}
        for name, result in models.items():
            assert_metrics(expected[name], result['metrics'])
        if statistics['final_instances'] != models['refined']['final_instance_count']:
            raise ValueError('prediction.json final_instances does not match refined_labels.npy.')
        for name, path in paths.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != (stamps[name]['size'], stamps[name]['mtime_ns']):
                raise ValueError('An input changed during the audit; stop concurrent evaluation.')
        rows, comparison = compare_models(geometry, models)
        brief = dict(schema=1, posthoc=True, scope='final_allocated_masks_only', checkpoint_step=checkpoint_step,
                     plot=expected['plot'], num_points=geometry['num_points'],
                     models={name: {key: value for key, value in result.items()
                                    if key not in ('trees', 'predictions')} for name, result in models.items()},
                     **comparison)
        write_csv(output / 'trees.csv', rows)
        for name, result in models.items():
            write_csv(output / f'{name}_predictions.csv', result['predictions'])
        write_json(output / 'summary.json', dict(brief, inputs=stamps, pipeline=statistics, notes=[
            'GT is used only after prediction. No checkpoint/model/GPU is loaded; no labels are changed.',
            'Metrics ignore negative GT; match Hungarian IoU strictly > 0.5; FP needs labeled-tree fraction >= 0.5.',
            'z_span_m is max(z)-min(z), not terrain-normalized tree height. Groups have GT metrics, not group F1.',
            'Categories describe final masks; high_recall_low_purity may mix other trees and/or background.',
            'Ignored/excluded predictions are not automatically false positives or correct extra trees.',
            'Final masks cannot reveal pre-NMS quality, query coverage, or why a particular proposal was rejected.',
            'Input stamps and reproduced metrics check consistency, not independent proof of dataset provenance.']))
        write_json(output / 'brief.json', brief)
        print(json.dumps(brief, indent=2), flush=True)
        print(f'SAVED {output / "brief.json"}', flush=True)
        return brief
    finally:
        for array in arrays.values():
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/forest_abc/full.yaml')
    parser.add_argument('--split', choices=('external', 'validation'), default='external')
    parser.add_argument('--plot', action='append')
    parser.add_argument('--evaluation-dir')
    parser.add_argument('--output')
    parser.add_argument('--expect-step', type=int)
    parser.add_argument('--chunk-points', type=int, default=1_000_000)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    evaluation = Path(args.evaluation_dir or Path(config['work_dir']) /
                      ('external_evaluation' if args.split == 'external' else 'evaluation'))
    step = int(json.loads((evaluation / 'brief.json').read_text(encoding='utf-8'))['checkpoint_step'])
    if args.expect_step is not None and step != args.expect_step:
        raise ValueError(f'Expected step {args.expect_step}, found step {step}; do not audit overwritten predictions.')
    cache_root = Path(config['external_cache_root'] if args.split == 'external' else config['cache_root'])
    manifest = json.loads((cache_root / 'manifest.json').read_text(encoding='utf-8'))['forests']
    candidates = [row for row in manifest if args.split == 'external' or
                  (row['split'] == 'validation' and row['plot'] in config['splits']['validation'])]
    rows = [row for row in candidates if args.plot is None or row['plot'] in args.plot]
    if not rows or len({row['plot'] for row in rows}) != len(rows):
        raise ValueError('No requested forests or duplicate manifest plot names.')
    if args.plot and set(args.plot) != {row['plot'] for row in rows}:
        raise ValueError('Some requested plots are missing from the manifest.')
    output = Path(args.output or Path(config['work_dir']) / 'diagnostics' / f'{args.split}_step_{step:06d}')
    for row in rows:
        print(f'AUDIT {row["plot"]}: existing final masks, CPU only, step={step}', flush=True)
        audit_forest(cache_root / row['cache'], evaluation / row['plot'], output / row['plot'],
                     step, args.chunk_points)


if __name__ == '__main__':
    main()
