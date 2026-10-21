"""CPU-only comparison of saved official, control and combined final masks."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from tools.forest_abc.diagnose import (
    analyze_table, assert_metrics, compare_models, scan_arrays, write_csv, write_json)

PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'
ARMS = ('control', 'combined')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def point_assignment_changes(labels, predictions, chunk_points):
    """Compare assigned/unassigned GT points, never arbitrary instance ID values."""
    if chunk_points <= 0:
        raise ValueError('chunk_points must be positive.')
    pairs = list(combinations(predictions, 2))
    result = {f'{left}_to_{right}': dict(both_assigned=0, became_unassigned=0,
        became_assigned=0, both_unassigned=0) for left, right in pairs}
    for start in range(0, len(labels), chunk_points):
        stop = min(start + chunk_points, len(labels))
        tree = labels[start:stop] > 0
        assigned = {name: values[start:stop][tree] > 0 for name, values in predictions.items()}
        for left, right in pairs:
            first, second = assigned[left], assigned[right]
            counts = result[f'{left}_to_{right}']
            counts['both_assigned'] += int(np.count_nonzero(first & second))
            counts['became_unassigned'] += int(np.count_nonzero(first & ~second))
            counts['became_assigned'] += int(np.count_nonzero(~first & second))
            counts['both_unassigned'] += int(np.count_nonzero(~first & ~second))
    return result


def audit_saved(cache, directories, output, steps, chunk_points=1_000_000):
    cache, output = Path(cache), Path(output)
    if set(directories) != set(ARMS) or set(steps) != set(ARMS):
        raise ValueError('Require both paired arms and their selected checkpoint steps.')
    directories = {name: Path(directory) for name, directory in directories.items()}
    paths = {name: cache / f'{name}.npy' for name in ('coords', 'labels', 'baseline')}
    statistics, expected = {}, {}
    plot = None
    for arm, directory in directories.items():
        statistics[arm] = read_json(directory / 'prediction.json')
        expected[arm] = read_json(directory / 'metrics.json')
        stamp = read_json(directory.parent / 'request.json')
        if (stamp['arm'] != arm or stamp['step'] != steps[arm]
                or expected[arm]['plot'] not in stamp['plots']):
            raise ValueError('Prediction request differs from the selected arm/step/plot.')
        if statistics[arm].get('partial', False):
            raise ValueError('Require completed full-forest predictions.')
        if plot is not None and plot != expected[arm]['plot']:
            raise ValueError('The two arms refer to different forests.')
        plot = expected[arm]['plot']
        paths[arm] = directory / 'predictions.npy'
        paths[f'{arm}_metrics'] = directory / 'metrics.json'
        paths[f'{arm}_prediction'] = directory / 'prediction.json'
        paths[f'{arm}_request'] = directory.parent / 'request.json'
    stamps = {name: dict(path=str(path.resolve()), size=path.stat().st_size,
                        mtime_ns=path.stat().st_mtime_ns) for name, path in paths.items()}
    arrays = {}
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('brief.json', 'summary.json'):
        (output / filename).unlink(missing_ok=True)
    try:
        for name in ('coords', 'labels', 'baseline') + ARMS:
            arrays[name] = np.load(paths[name], mmap_mode='r', allow_pickle=False)
        for arm in ARMS:
            if statistics[arm]['num_points'] != len(arrays['labels']):
                raise ValueError('Prediction point count differs from the original aligned cache.')
        predictions = {name: arrays[name] for name in ('baseline',) + ARMS}
        geometry, tables = scan_arrays(arrays['coords'], arrays['labels'], predictions, chunk_points)
        models = {name: analyze_table(geometry, table) for name, table in tables.items()}
        for arm in ARMS:
            assert_metrics(expected[arm]['baseline'], models['baseline']['metrics'])
            assert_metrics(expected[arm]['refined'], models[arm]['metrics'])
            if statistics[arm]['final_instances'] != models[arm]['final_instance_count']:
                raise ValueError('Saved final instance count does not match prediction IDs.')
        assignment = point_assignment_changes(arrays['labels'], predictions, chunk_points)
        pairs, compact_pairs = {}, {}
        for left, right in combinations(models, 2):
            name = f'{left}_to_{right}'
            rows, comparison = compare_models(geometry, dict(baseline=models[left], refined=models[right]))
            for row in rows:
                row['source_model'], row['target_model'] = left, right
            reasons = dict(Counter(row['refined_category'] for row in rows if row['transition'] == 'lost'))
            detail = dict(source=left, target=right, **comparison,
                          lost_target_categories=reasons, point_assignment=assignment[name])
            pairs[name] = detail
            compact_pairs[name] = dict(source=left, target=right,
                transitions=comparison['transitions'], lost_target_categories=reasons,
                point_assignment=assignment[name], z_span_groups={group: {
                    key: value[key] for key in ('num_gt', 'lost', 'recovered')}
                    for group, value in comparison['z_span_groups'].items()})
            write_csv(output / f'{name}_trees.csv', rows)
        model_briefs = {}
        for name, result in models.items():
            values = np.array([row['best_iou'] for row in result['trees']])
            assigned = int(tables[name]['intersections'].sum())
            model_briefs[name] = dict(
                **{key: value for key, value in result.items() if key not in ('trees', 'predictions')},
                gt_point_assignment_fraction=assigned / int(geometry['counts'].sum()),
                best_iou_boundary_counts=dict(exactly_half=int(np.count_nonzero(values == .5)),
                    from_045_through_05=int(np.count_nonzero((values >= .45) & (values <= .5))),
                    above_05_through_055=int(np.count_nonzero((values > .5) & (values <= .55)))))
            write_csv(output / f'{name}_predictions.csv', result['predictions'])
        for name, path in paths.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != (stamps[name]['size'], stamps[name]['mtime_ns']):
                raise ValueError('An input changed during the audit; rerun after evaluation completes.')
        brief = dict(schema=1, posthoc=True, scope='saved_final_masks_only', protocol=PROTOCOL,
                     plot=plot, checkpoint_steps=steps, num_points=geometry['num_points'],
                     models=model_briefs, pairwise=compact_pairs)
        write_json(output / 'summary.json', dict(brief, pairwise=pairs, inputs=stamps, notes=[
            'CPU only: no model, checkpoint, frozen features or GPU inference is loaded.',
            'All three metrics are reproduced against the saved per-forest evaluation.',
            'Equal TP counts do not imply identical detected GT IDs; lost/recovered are computed by GT ID.',
            'Assignment changes use only GT tree points. Instance ID numbers are not compared across models.',
            'Unassigned final points do not identify a semantic-head versus clustering/assignment cause.',
            'High-recall/low-purity masks may contain other trees and/or background, not necessarily merges.',
            'z_span is max(z)-min(z), not terrain-normalized height; no group F1 is invented.',
            'Best-IoU boundary counts are diagnostics and do not change the Hungarian matching protocol.',
            'Input stamps and matching metrics check consistency, not independent proof of provenance.']))
        write_json(output / 'brief.json', brief)
        for name, result in model_briefs.items():
            print(name + ': ' + json.dumps(result['metrics']), flush=True)
        for name, result in compact_pairs.items():
            print(name + ': ' + json.dumps(result['transitions']), flush=True)
        print(f'SAVED {output / "brief.json"}', flush=True)
        return brief
    finally:
        for array in arrays.values():
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/treelearn_residual_instance/full.yaml')
    parser.add_argument('--plot', default='wytham_vox0.1')
    parser.add_argument('--chunk-points', type=int, default=1_000_000)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    root, work = Path(config['external_cache_root']), Path(config['work_dir'])
    matches = [row for row in read_json(root / 'manifest.json')['forests'] if row['plot'] == args.plot]
    if len(matches) != 1:
        raise ValueError('Require exactly one matching external-cache forest.')
    comparison = read_json(work / 'external_comparison.json')
    if comparison['protocol'] != PROTOCOL:
        raise ValueError('External comparison uses a different evaluation protocol.')
    steps = {arm: comparison['arms'][arm]['checkpoint_step'] for arm in ARMS}
    directories = {arm: work / arm / 'external_evaluation' / args.plot for arm in ARMS}
    for arm in ARMS:
        brief = read_json(directories[arm].parent / 'brief.json')
        if brief['checkpoint_step'] != steps[arm] or brief['protocol'] != PROTOCOL or brief['arm'] != arm:
            raise ValueError('External comparison and arm evaluation differ.')
    audit_saved(root / matches[0]['cache'], directories,
                work / 'external_diagnostics' / args.plot, steps, args.chunk_points)


if __name__ == '__main__':
    main()
