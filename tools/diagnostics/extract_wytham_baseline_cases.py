"""Read-only CPU audit and GT-selected case extraction from saved official predictions.

No torch, spconv, model, checkpoint, training configuration or GPU lock is used.
GT-selected cases and editing bounds are diagnostic oracles, not deployable gains.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import zipfile

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.forest_abc.diagnose import analyze_table, scan_arrays, write_csv, write_json

PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'


def edit_bounds(gt_points, pred_points, intersection):
    """Exact integer lower bounds for editing ONE candidate against ONE GT.

Adding a missing target point increases intersection while leaving union fixed.
Removing a foreign known point decreases union while retaining intersection.
Only ignore-filtered point counts are used. These operations require GT.
"""
    g, p, i = int(gt_points), int(pred_points), int(intersection)
    if min(g, p, i) < 0 or i > min(g, p):
        raise ValueError('Invalid contingency counts.')
    union = g + p - i
    if not union:
        return dict(min_target_additions=None, min_foreign_removals=None)
    if 2 * i > union:
        return dict(min_target_additions=0, min_foreign_removals=0)
    difference = g + p - 3 * i
    add = difference // 2 + 1
    remove = difference + 1
    return dict(min_target_additions=add if add <= g - i else None,
                min_foreign_removals=remove if remove <= p - i and 2 * i > g else None)


def prioritize(geometry, table, baseline):
    locked = {row['pred_id'] for row in baseline['predictions'] if row['category'] == 'true_positive'}
    locked_rows = np.array([identifier in locked for identifier in table['ids']])
    locked_mass = table['intersections'][locked_rows].sum(0)
    pred_lookup = {int(identifier): row for row, identifier in enumerate(table['ids'])}
    result = []
    for column, record in enumerate(baseline['trees']):
        row = dict(record, gt_points=int(geometry['counts'][column]),
                   z_span_m=float(geometry['z_span'][column]))
        g = row['gt_points']
        pred_row = pred_lookup.get(row['best_pred_id'])
        p = int(table['known'][pred_row]) if pred_row is not None else 0
        common = int(table['intersections'][pred_row, column]) if pred_row is not None else 0
        safe = g - int(locked_mass[column])
        unassigned = g - int(table['intersections'][:, column].sum())
        row.update(best_pred_known_points=p, best_intersection_points=common,
            best_pred_is_locked_tp=row['best_pred_id'] in locked,
            target_points_in_locked_predictions=int(locked_mass[column]),
            target_points_outside_locked_predictions=safe,
            max_clean_iou_without_touching_locked_predictions=safe / g,
            clean_gt_oracle_possible_without_touching_locked_predictions=2 * safe > g,
            unassigned_target_points=unassigned,
            max_clean_iou_add_only_original_zero=unassigned / g,
            clean_gt_oracle_add_only_original_zero=2 * unassigned > g,
            **edit_bounds(g, p, common))
        additions = row['min_target_additions']
        safe_outside_best = safe - (common if not row['best_pred_is_locked_tp'] else 0)
        row['addition_oracle_possible_to_existing_unlocked_candidate'] = bool(
            not row['best_pred_is_locked_tp'] and pred_row is not None
            and additions is not None and additions <= safe_outside_best)
        row['min_target_addition_fraction'] = additions / g if additions is not None else None
        row['min_foreign_removal_fraction'] = (row['min_foreign_removals'] / p
            if p and row['min_foreign_removals'] is not None else None)
        if row['detected']:
            group = 'detected_control'
        elif row['best_iou'] >= .4:
            group = 'near_threshold'
        elif row['category'] == 'high_recall_low_purity':
            group = 'impure_mask'
        elif row['category'] in ('mostly_unassigned', 'no_assigned_tree_points'):
            group = 'foreground_missing'
        else:
            group = 'fragmented_or_other'
        row['diagnostic_group'] = group
        result.append(row)
    return result


def supplemental_signals(values, geometry, rows, chunk):
    """Summarize saved probabilities/votes without opening backbone features."""
    labels, baseline = values['labels'], values['baseline']
    identifiers, counts = geometry['ids'], geometry['counts']
    thresholds = (.1, .2, .3, .4)
    supported = {f'{threshold:.1f}': np.zeros(len(identifiers), dtype=np.int64) for threshold in thresholds}
    exposure = {key: dict(tree_gt_points=0, known_background_points=0, ignored_points=0) for key in supported}
    probability_sum = np.zeros(len(identifiers))
    vote_sum, vote_square = np.zeros((len(identifiers), 2)), np.zeros(len(identifiers))
    disagreements = 0
    for start in range(0, len(labels), chunk):
        stop = min(start + chunk, len(labels))
        gt, pred = labels[start:stop], baseline[start:stop]
        tree = gt > 0
        positions = np.searchsorted(identifiers, gt[tree])
        if 'tree_probability' in values:
            probability = values['tree_probability'][start:stop]
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError('Invalid saved tree probabilities.')
            disagreements += int(np.count_nonzero((probability >= .5) != (pred > 0)))
            probability_sum += np.bincount(positions, weights=probability[tree], minlength=len(identifiers))
            for threshold in thresholds:
                key = f'{threshold:.1f}'
                candidate = (pred == 0) & (probability >= threshold)
                supported[key] += np.bincount(positions[candidate[tree]], minlength=len(identifiers))
                exposure[key]['tree_gt_points'] += int(np.count_nonzero(candidate & tree))
                exposure[key]['known_background_points'] += int(np.count_nonzero(candidate & (gt == 0)))
                exposure[key]['ignored_points'] += int(np.count_nonzero(candidate & (gt < 0)))
        if 'votes_xy' in values:
            votes = np.asarray(values['votes_xy'][start:stop][tree], dtype=np.float64)
            if not np.isfinite(votes).all():
                raise ValueError('Nonfinite saved XY votes.')
            for axis in range(2):
                vote_sum[:, axis] += np.bincount(positions, weights=votes[:, axis], minlength=len(identifiers))
            vote_square += np.bincount(positions, weights=(votes * votes).sum(1), minlength=len(identifiers))
        if start == 0 or stop == len(labels) or (start // chunk + 1) % 20 == 0:
            print(f'SAVED_SIGNALS points={stop}/{len(labels)}', flush=True)
    for column, row in enumerate(rows):
        row['saved_tree_probability_mean'] = float(probability_sum[column] / counts[column]) if 'tree_probability' in values else None
        row['saved_vote_xy_rms_spread_m'] = float(np.sqrt(max(0., vote_square[column] / counts[column]
            - np.square(vote_sum[column] / counts[column]).sum()))) if 'votes_xy' in values else None
        for key, count in supported.items():
            row['max_clean_iou_add_only_p_ge_' + key.replace('.', '_')] = (
                float(count[column] / counts[column]) if 'tree_probability' in values else None)
    return dict(saved_tree_probability_available='tree_probability' in values,
        saved_xy_votes_available='votes_xy' in values,
        saved_probability_threshold_disagreements=disagreements if 'tree_probability' in values else None,
        additional_candidate_exposure=exposure if 'tree_probability' in values else None,
        note='Probability bands are posthoc diagnostics, not chosen deployment thresholds. Ignored GT is not treated as background.')


def select_cases(rows, per_group):
    chosen = []
    for name in ('near_threshold', 'impure_mask', 'foreground_missing', 'fragmented_or_other'):
        pool = [row for row in rows if row['diagnostic_group'] == name]
        # Prefer cases that don't require changing any correctly detected mask;
        # then prioritize smaller relative additions and higher existing IoU.
        pool.sort(key=lambda row: (not row['clean_gt_oracle_possible_without_touching_locked_predictions'],
            -(row.get('max_clean_iou_add_only_p_ge_0_3') or 0.) if name == 'foreground_missing' else 0.,
            row['best_pred_is_locked_tp'],
            row['min_target_addition_fraction'] if row['min_target_addition_fraction'] is not None else 10.,
            -row['best_iou'], row['gt_id']))
        chosen.extend(dict(row, selection_group=name) for row in pool[:per_group])
    controls = [row for row in rows if row['detected']]
    # Successful controls span height scales, not only the easiest tall trees.
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, float('inf'))):
        pool = [row for row in controls if lo <= row['z_span_m'] < hi]
        pool.sort(key=lambda row: (abs(row['best_iou'] - .7), row['gt_id']))
        chosen.extend(dict(row, selection_group='detected_control') for row in pool[:1])
    return chosen


def stamp(path):
    value = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def selected_boxes(coords, labels, selected, chunk):
    wanted = np.array([row['gt_id'] for row in selected])
    boxes = {int(identifier): [np.full(3, np.inf), np.full(3, -np.inf)] for identifier in wanted}
    for start in range(0, len(labels), chunk):
        gt, xyz = labels[start:start + chunk], coords[start:start + chunk]
        for identifier in np.unique(gt[np.isin(gt, wanted)]):
            points = xyz[gt == identifier]
            boxes[int(identifier)][0] = np.minimum(boxes[int(identifier)][0], points.min(0))
            boxes[int(identifier)][1] = np.maximum(boxes[int(identifier)][1], points.max(0))
    return boxes


def bounded_append(state, values, budget, rng):
    if not len(values['source_indices']):
        return
    priority = rng.random(len(values['source_indices']))
    if state:
        values = {key: np.concatenate((state[key], value)) for key, value in values.items()}
        priority = np.concatenate((state['priority'], priority))
    keep = np.argpartition(priority, budget - 1)[:budget] if len(priority) > budget else np.arange(len(priority))
    state.update({key: value[keep] for key, value in values.items()}, priority=priority[keep])


def export_cases(values, selected, output, chunk, padding,
                 target_budget, context_budget, seed):
    coords, labels, baseline = (values[key] for key in ('coords', 'labels', 'baseline'))
    boxes = selected_boxes(coords, labels, selected, chunk)
    states = {row['gt_id']: dict(target={}, context={}, target_seen=0, context_seen=0,
              rng=np.random.default_rng([seed, row['gt_id']])) for row in selected}
    print(f'EXTRACT {len(selected)} cases; full target bounding boxes + {padding} m XY context.', flush=True)
    for start in range(0, len(labels), chunk):
        end = min(start + chunk, len(labels))
        xyz, gt, pred = coords[start:end], labels[start:end], baseline[start:end]
        for row in selected:
            identifier, state = row['gt_id'], states[row['gt_id']]
            low, high = boxes[identifier]
            inside = np.all((xyz[:, :2] >= low[:2] - padding) & (xyz[:, :2] <= high[:2] + padding), axis=1)
            for name, mask, budget in (('target', gt == identifier, target_budget),
                                       ('context', inside & (gt != identifier), context_budget)):
                local = np.flatnonzero(mask)
                state[name + '_seen'] += len(local)
                selected_rows = local.astype(np.int64) + start
                payload = dict(coords=np.asarray(xyz[local], dtype=np.float32),
                    labels=np.asarray(gt[local], dtype=np.int32), baseline=np.asarray(pred[local], dtype=np.int32),
                    source_indices=selected_rows)
                payload.update({key: np.asarray(values[key][selected_rows], dtype=np.float32)
                                for key in ('tree_probability', 'votes_xy') if key in values})
                bounded_append(state[name], payload, budget, state['rng'])
        if start == 0 or end == len(labels) or (start // chunk + 1) % 20 == 0:
            print(f'EXTRACT points={end}/{len(labels)}', flush=True)
    cases = output / 'cases'
    cases.mkdir()
    for row in selected:
        identifier, state = row['gt_id'], states[row['gt_id']]
        arrays = [state[name] for name in ('target', 'context') if state[name]]
        exported = {key: np.concatenate([part[key] for part in arrays])
                    for key in arrays[0] if key != 'priority'}
        filename = f'{row["selection_group"]}_gt_{identifier:04d}'
        np.savez_compressed(cases / (filename + '.npz'), **exported)
        write_json(cases / (filename + '.json'), dict(row, point_file=filename + '.npz',
            bbox_min=boxes[identifier][0].tolist(), bbox_max=boxes[identifier][1].tolist(),
            target_points_in_full_cache=state['target_seen'], context_points_in_full_crop=state['context_seen'],
            exported_target_points=len(state['target'].get('labels', [])),
            exported_context_points=len(state['context'].get('labels', [])),
            note='GT-selected display sample. Global metrics use the full forest; never recompute them on this sampled crop.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', default='data/candidate_mask_mlp_a1_wytham_s42')
    parser.add_argument('--plot', default='wytham_vox0.1')
    parser.add_argument('--output', help='New output directory only; never overwrite prior artifacts.')
    parser.add_argument('--chunk-points', type=int, default=500000)
    parser.add_argument('--per-group', type=int, default=3)
    parser.add_argument('--target-budget', type=int, default=12000)
    parser.add_argument('--context-budget', type=int, default=38000)
    parser.add_argument('--padding', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-counts', nargs=3, type=int, default=[596, 222, 281], metavar=('TP', 'FP', 'FN'))
    args = parser.parse_args()
    if min(args.chunk_points, args.per_group, args.target_budget, args.context_budget) < 1 or args.padding < 0:
        raise ValueError('Invalid scan or sample budget.')
    if hasattr(os, 'nice'):
        os.nice(10)
    os.chdir(ROOT)
    root = Path(args.cache_root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    found = [row for row in manifest['forests'] if row['plot'] == args.plot]
    if len(found) != 1:
        raise ValueError('Require exactly one matching cache forest.')
    cache = root / found[0]['cache']
    paths = {name: cache / f'{name}.npy' for name in ('coords', 'labels', 'baseline')}
    paths.update({name: cache / f'{name}.npy' for name in ('tree_probability', 'votes_xy')
                  if (cache / f'{name}.npy').is_file()})
    inputs = {name: stamp(path) for name, path in paths.items()}
    values = {name: np.load(path, mmap_mode='r', allow_pickle=False) for name, path in paths.items()}
    try:
        if 'tree_probability' in values and values['tree_probability'].shape != values['labels'].shape:
            raise ValueError('Saved probabilities do not align with the GT cache.')
        if 'votes_xy' in values and values['votes_xy'].shape != (len(values['labels']), 2):
            raise ValueError('Saved votes do not align with the GT cache.')
        geometry, tables = scan_arrays(values['coords'], values['labels'], {'baseline': values['baseline']}, args.chunk_points)
        baseline = analyze_table(geometry, tables['baseline'])
        metrics = baseline['metrics']
        if [metrics[key] for key in ('tp', 'fp', 'fn')] != args.expected_counts:
            raise ValueError(f'Baseline differs from the intended 596/222/281 protocol: {metrics}. Do not mix old evaluations.')
        rows = prioritize(geometry, tables['baseline'], baseline)
        signals = supplemental_signals(values, geometry, rows, args.chunk_points)
        selected = select_cases(rows, args.per_group)
        output = Path(args.output) if args.output else Path('logs/wytham_baseline_repair_cases') / datetime.now().strftime('%Y%m%dT%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        write_csv(output / 'all_gt_trees.csv', rows)
        write_csv(output / 'selected_cases.csv', selected)
        write_csv(output / 'baseline_predictions.csv', baseline['predictions'])
        export_cases(values, selected, output,
            args.chunk_points, args.padding, args.target_budget, args.context_budget, args.seed)
        missed = [row for row in rows if not row['detected']]
        summary = dict(schema=1, posthoc=True, gt_selected=True, protocol=PROTOCOL, plot=args.plot,
            baseline_metrics=metrics, tree_categories=baseline['tree_categories'],
            missed_with_iou_040_to_050=sum(row['best_iou'] >= .4 for row in missed),
            missed_with_iou_045_to_050=sum(row['best_iou'] >= .45 for row in missed),
            missed_lt_10m_z_span=sum(row['z_span_m'] < 10 for row in missed),
            missed_clean_gt_oracle_without_touching_locked_predictions=sum(
                row['clean_gt_oracle_possible_without_touching_locked_predictions'] for row in missed),
            missed_clean_gt_oracle_add_only_original_zero=sum(
                row['clean_gt_oracle_add_only_original_zero'] for row in missed),
            saved_signals=signals,
            missed_addition_oracle_to_existing_unlocked_candidate=sum(
                row['addition_oracle_possible_to_existing_unlocked_candidate'] for row in missed),
            selected_count=len(selected), selected_ids=[row['gt_id'] for row in selected], inputs=inputs,
            sampling=dict(seed=args.seed, chunk_points=args.chunk_points, per_group=args.per_group,
                padding_m=args.padding, target_budget=args.target_budget, context_budget=args.context_budget),
            notes=['CPU only, one BLAS thread, lower scheduling priority; no GPU lock or model execution.',
                   'Ranking/edit budgets use GT. They identify diagnostic opportunities, not an implementable repair rule or promised F1.',
                   'A locked prediction is a complete original TP mask. This is a conservative diagnostic scope; deployment cannot identify true TPs using GT.',
                   'No changes to any baseline, ongoing training, checkpoint, grouping threshold or experiment request.',
                   'z_span is not terrain-normalized tree height. Raw LAZ GT IDs require explicit alignment before joining this filtered cache.'])
        for name, path in paths.items():
            if stamp(path) != inputs[name]:
                raise ValueError('An input changed during the audit; discard this incomplete package.')
        write_json(output / 'summary.json', summary)
        archive = output / 'casepack.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path != archive:
                    bundle.write(path, arcname=path.relative_to(output))
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f'SAVED {output / "summary.json"}\nCASEPACK {archive}', flush=True)
    finally:
        for array in values.values():
            array._mmap.close()


if __name__ == '__main__':
    main()
