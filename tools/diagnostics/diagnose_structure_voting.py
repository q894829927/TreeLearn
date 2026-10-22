"""CPU-only audit of completed structure-voting predictions, never a model run."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from tools.forest_abc.diagnose import (
    analyze_table, assert_metrics, compare_models, scan_arrays, write_csv, write_json)

PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'
WATCH = {
    'same_tree_fragment': (259, 375, 419, 629, 720, 801),
    'foreground_missing': (152, 347, 507),
    'impure_or_borderline': (129, 190, 543),
    'detected_control': (414, 374, 48, 611),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def mask_changes(baseline, refined, tables, models, chunk_points):
    """Compare full point sets, including unknown GT; ignore arbitrary ID renaming."""
    old_ids, new_ids = tables['baseline']['ids'], tables['refined']['ids']
    if len(old_ids) * len(new_ids) > 20_000_000:
        raise ValueError('Prediction overlap table exceeds the diagnostic memory limit.')
    overlap = np.zeros((len(old_ids), len(new_ids)), dtype=np.int64)
    changed, foreground = 0, 0
    for start in range(0, len(baseline), chunk_points):
        a, b = baseline[start:start + chunk_points], refined[start:start + chunk_points]
        changed += int(np.count_nonzero((a > 0) != (b > 0)))
        foreground += int(np.count_nonzero(a > 0))
        positive = (a > 0) & (b > 0)
        encoded = np.searchsorted(old_ids, a[positive]) * len(new_ids) + np.searchsorted(new_ids, b[positive])
        overlap += np.bincount(encoded, minlength=overlap.size).reshape(overlap.shape)
        if start == 0 or start + chunk_points >= len(baseline) or (start // chunk_points + 1) % 10 == 0:
            print(f'COMPARE_MASKS points={min(start + chunk_points, len(baseline))}/{len(baseline)}', flush=True)
    if changed:
        raise ValueError(f'Foreground changed at {changed} points; this is not the frozen-foreground experiment.')
    records = []
    original_tp = {row['pred_id']: row for row in models['baseline']['predictions']
                   if row['category'] == 'true_positive'}
    for i, identifier in enumerate(old_ids):
        if int(identifier) not in original_tp:
            continue
        counts = overlap[i]
        j = int(counts.argmax())
        common, old_total, new_total = int(counts[j]), int(tables['baseline']['total'][i]), int(tables['refined']['total'][j])
        records.append(dict(original_pred_id=int(identifier),
            originally_matched_gt_id=original_tp[int(identifier)]['matched_gt_id'],
            original_total_points=old_total, num_refined_masks_overlapping=int(np.count_nonzero(counts)),
            best_overlap_refined_pred_id=int(new_ids[j]), best_overlap_points=common,
            original_fraction_in_best_overlap=common / old_total,
            other_points_in_best_overlap=new_total - common,
            exact_full_mask_preserved=bool(common == old_total == new_total)))
    identity = dict(changed_foreground_points=changed, official_foreground_points=foreground)
    return identity, records


def audit_saved(cache, directory, output, expect_step=3000, chunk_points=500_000, watch=None):
    cache, directory, output = Path(cache), Path(directory), Path(output)
    if chunk_points < 1:
        raise ValueError('chunk_points must be positive.')
    if output.exists():
        raise FileExistsError(f'Use a new diagnostic output directory: {output}')
    paths = {name: cache / f'{name}.npy' for name in ('coords', 'labels', 'baseline')}
    paths.update(refined=directory / 'predictions.npy', metrics=directory / 'metrics.json',
        prediction=directory / 'prediction.json', request=directory.parent / 'request.json',
        brief=directory.parent / 'brief.json', identity=directory / 'foreground_identity.json')
    stamps = {name: dict(path=str(path.resolve()), size=path.stat().st_size,
                        mtime_ns=path.stat().st_mtime_ns) for name, path in paths.items()}
    expected, statistics, request, saved_brief, saved_identity = (
        read_json(paths[name]) for name in ('metrics', 'prediction', 'request', 'brief', 'identity'))
    plot = expected['plot']
    if (request['step'] != expect_step or saved_brief['checkpoint_step'] != expect_step
            or plot not in request['plots'] or saved_brief['protocol'] != PROTOCOL):
        raise ValueError('Evaluation request/step/plot/protocol differs from the requested experiment.')
    if statistics.get('partial', False) or saved_identity['changed_foreground_points'] != 0:
        raise ValueError('Require complete predictions with unchanged official foreground.')
    arrays = {}
    try:
        for name in ('coords', 'labels', 'baseline', 'refined'):
            arrays[name] = np.load(paths[name], mmap_mode='r', allow_pickle=False)
        if statistics['num_points'] != len(arrays['labels']):
            raise ValueError('Predictions do not cover the full aligned cache.')
        geometry, tables = scan_arrays(arrays['coords'], arrays['labels'],
            {name: arrays[name] for name in ('baseline', 'refined')}, chunk_points)
        models = {name: analyze_table(geometry, table) for name, table in tables.items()}
        for name, result in models.items():
            assert_metrics(expected[name], result['metrics'])
        if statistics['final_instances'] != models['refined']['final_instance_count']:
            raise ValueError('Final instance count does not match saved predictions.')
        identity, preserved = mask_changes(arrays['baseline'], arrays['refined'], tables, models, chunk_points)
        if identity != saved_identity or ('foreground_identity' in expected and identity != expected['foreground_identity']):
            raise ValueError('Recomputed foreground identity differs from the saved result.')
        rows, comparison = compare_models(geometry, models)
        transitions = {key: comparison['transitions'].get(key, 0)
                       for key in ('both_detected', 'both_missed', 'lost', 'recovered')}
        if transitions['recovered'] - transitions['lost'] != models['refined']['metrics']['tp'] - models['baseline']['metrics']['tp']:
            raise ValueError('Per-tree transitions do not explain the net TP change.')
        selected, absent = [], []
        lookup = {row['gt_id']: row for row in rows}
        watch = WATCH if watch is None and plot == 'wytham_vox0.1' else (watch or {})
        for group, ids in watch.items():
            for identifier in ids:
                if identifier in lookup:
                    selected.append(dict(watch_group=group, **lookup[identifier]))
                else:
                    absent.append(identifier)
        model_briefs = {name: {key: value for key, value in result.items()
                               if key not in ('trees', 'predictions')} for name, result in models.items()}
        report = dict(schema=1, posthoc=True, scope='saved_final_masks_only', protocol=PROTOCOL,
            plot=plot, checkpoint_step=expect_step, num_points=geometry['num_points'],
            models=model_briefs, foreground_identity=identity, transitions=transitions,
            lost_categories=dict(Counter(row['refined_category'] for row in rows if row['transition'] == 'lost')),
            recovered_baseline_categories=dict(Counter(row['baseline_category'] for row in rows if row['transition'] == 'recovered')),
            z_span_groups=comparison['z_span_groups'], watchlist=selected, missing_watch_gt_ids=absent,
            original_tp_full_mask_accounting=dict(total=len(preserved),
                exact_masks_preserved=sum(row['exact_full_mask_preserved'] for row in preserved),
                masks_changed=sum(not row['exact_full_mask_preserved'] for row in preserved)))
        for name, path in paths.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != (stamps[name]['size'], stamps[name]['mtime_ns']):
                raise ValueError('An input changed during the audit; wait for evaluation to finish.')
        output.mkdir(parents=True, exist_ok=False)
        write_csv(output / 'all_gt_trees.csv', rows)
        write_csv(output / 'watchlist.csv', selected)
        write_csv(output / 'original_tp_full_masks.csv', preserved)
        for name, result in models.items():
            write_csv(output / f'{name}_predictions.csv', result['predictions'])
        write_json(output / 'summary.json', dict(report, inputs=stamps, notes=[
            'No model/checkpoint/features/GPU loaded. Saved arrays and metrics only; no predictions are modified.',
            'Watch IDs are GT-selected posthoc diagnostics, never inference inputs or independent validation.',
            'Raw instance ID numbers are not compared across models. Full-mask equality uses point-set overlap.',
            'Point-set changes are not the same as losing a detection; see lost/recovered separately.',
            'Negative GT is ignored in metrics, but retained when checking equality of complete original masks.',
            'Reproduced metrics and file stamps check consistency; point order was checked by the evaluation pipeline.',
            'Final masks cannot isolate seed selection, HDBSCAN or remaining-point assignment as the cause.']))
        write_json(output / 'brief.json', report)
        print('TRANSITIONS ' + json.dumps(transitions), flush=True)
        print('ORIGINAL_TP_MASKS ' + json.dumps(report['original_tp_full_mask_accounting']), flush=True)
        for row in selected:
            print(f'WATCH {row["gt_id"]}: {row["transition"]}; '
                  f'IoU {row["baseline_best_iou"]:.6f} -> {row["refined_best_iou"]:.6f}', flush=True)
        print(f'SAVED {output / "brief.json"}', flush=True)
        return report
    finally:
        for array in arrays.values():
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/structure_voting/full.yaml')
    parser.add_argument('--plot', default='wytham_vox0.1')
    parser.add_argument('--expect-step', type=int, default=3000)
    parser.add_argument('--chunk-points', type=int, default=500_000)
    parser.add_argument('--output', help='New directory only.')
    args = parser.parse_args()
    os.chdir(ROOT)
    if hasattr(os, 'nice'):
        os.nice(10)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    cache_root, work = Path(config['external_cache_root']), Path(config['work_dir'])
    matches = [row for row in read_json(cache_root / 'manifest.json')['forests'] if row['plot'] == args.plot]
    if len(matches) != 1:
        raise ValueError('Require one matching forest in the external cache manifest.')
    output = Path(args.output) if args.output else work / 'external_diagnostics' / f'step_{args.expect_step:06d}' / args.plot
    audit_saved(cache_root / matches[0]['cache'], work / 'external_evaluation' / args.plot,
                output, args.expect_step, args.chunk_points)


if __name__ == '__main__':
    main()
