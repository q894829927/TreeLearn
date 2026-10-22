"""Export complete original masks for posthoc local split feasibility inspection.

Read-only CPU tool: no torch, model, checkpoint tensors, or prediction writes.
GT selects diagnostic cases/controls and is exported separately from inputs.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.forest_abc.diagnose import analyze_table
from tree_learn.instance_relation import data as relation_data
from tree_learn.instance_relation.supervision import load_supervision, merge_table
from tree_learn.mask_refinement.data import file_digest, write_json


DEFAULT_MIXED_IDS = (1767, 250, 262, 1427)
EXPECTED_DIAGNOSTIC_GT = {1767: (38, 292), 250: (176, 424), 262: (622, 129), 1427: (49, 783)}
INPUT_NAMES = ('coords', 'baseline', 'votes_xy', 'tree_probability', 'features')


def select_cases(features, targets, mixed_ids, max_core_points):
    """GT-selected mixed masks and size/height/location-matched TP controls."""
    identifiers = targets['ids']
    if not np.array_equal(identifiers, features['ids']):
        raise ValueError('Original instance IDs differ between geometry and exact counts.')
    geometry, table = merge_table(targets, [])
    baseline = analyze_table(geometry, table)
    records = {r['pred_id']: r for r in baseline['predictions']}
    low, high = features['low'].astype(float), features['high'].astype(float)
    centers, span = (low + high) / 2, np.maximum(high[:, 2] - low[:, 2], .1)
    mixed_ids = list(mixed_ids)
    if not mixed_ids or len(set(mixed_ids)) != len(mixed_ids):
        raise ValueError('Provide distinct positive original instance IDs.')
    selected, used = [], set(mixed_ids)
    for identifier in mixed_ids:
        at = int(np.searchsorted(identifiers, identifier))
        if at >= len(identifiers) or identifiers[at] != identifier:
            raise ValueError(f'Original instance {identifier} is absent.')
        if records[identifier]['category'] == 'true_positive':
            raise ValueError(f'Diagnostic mixed case {identifier} is an original TP; refuse this initial case plan.')
        if int(targets['total'][at]) > max_core_points:
            raise ValueError(f'Complete core {identifier} exceeds max_core_points; do not silently subsample it.')
        for gt in EXPECTED_DIAGNOSTIC_GT.get(identifier, ()):
            col = int(np.searchsorted(targets['gt_ids'], gt))
            if (col >= len(targets['gt_ids']) or targets['gt_ids'][col] != gt
                    or targets['intersections'][at, col] / targets['gt_counts'][col] <= .9
                    or next(r['detected'] for r in baseline['trees'] if r['gt_id'] == gt)):
                raise ValueError(f'Diagnostic identity/recall changed for original {identifier}, GT{gt}.')
        selected.append(dict(original_id=identifier, node=at, role='mixed_case',
            total_points=int(targets['total'][at]), matched_gt_id=0))
    candidates = [i for i, identifier in enumerate(identifiers)
        if records[int(identifier)]['category'] == 'true_positive'
        and records[int(identifier)]['dominant_gt_precision'] >= .8
        and targets['known'][i] / max(targets['total'][i], 1) >= .5
        and targets['total'][i] <= max_core_points]
    for case in list(selected):
        at = case['node']
        eligible = [i for i in candidates if int(identifiers[i]) not in used]
        if not eligible:
            raise ValueError('Not enough distinct correctly detected controls; no silent fallback.')
        def distance(i):
            return (abs(np.log(targets['total'][i] / targets['total'][at]))
                + abs(np.log(span[i] / span[at]))
                + .1 * np.linalg.norm(centers[i, :2] - centers[at, :2]), int(identifiers[i]))
        chosen = min(eligible, key=distance)
        identifier = int(identifiers[chosen])
        used.add(identifier)
        selected.append(dict(original_id=identifier, node=chosen, role='detected_control',
            control_for_original_id=case['original_id'], total_points=int(targets['total'][chosen]),
            matched_gt_id=records[identifier]['matched_gt_id'], matching_distance=float(distance(chosen)[0])))
    return selected, baseline['metrics']


def collect_cores(baseline, cases, chunk_points, max_core_points):
    """Complete original-mask indices; labels are not an argument."""
    identifiers = np.array([r['original_id'] for r in cases], dtype=np.int64)
    states = {int(i): [] for i in identifiers}
    seen = {int(i): 0 for i in identifiers}
    for start in range(0, len(baseline), chunk_points):
        stop = min(start + chunk_points, len(baseline))
        block = baseline[start:stop]
        if np.any(block < 0):
            raise ValueError('Official baseline must have completed nonnegative instance labels.')
        for identifier in identifiers:
            local = np.flatnonzero(block == identifier).astype(np.int64) + start
            seen[int(identifier)] += len(local)
            if seen[int(identifier)] > max_core_points:
                raise ValueError(f'Complete core {identifier} exceeds the export limit; no core was truncated.')
            states[int(identifier)].append(local)
        if start == 0 or stop == len(baseline) or (stop // chunk_points) % 10 == 0:
            print(f'INDEX_CORE points={stop}/{len(baseline)}', flush=True)
    result = {}
    for case in cases:
        identifier = case['original_id']
        indices = np.concatenate(states[identifier])
        if len(indices) != case['total_points']:
            raise ValueError(f'Original instance {identifier} point count changed: {len(indices)} != {case["total_points"]}.')
        result[identifier] = indices
    return result


def retain_lowest(current, incoming, budget, seed):
    """Chunk-independent, deterministic context reservoir using index hashes."""
    if not len(incoming):
        return current
    indices = np.concatenate((current, incoming))
    if len(indices) <= budget:
        return indices
    priority = relation_data.priorities(indices, seed)
    # Hash ties are resolved by source index, independent of chunk boundaries.
    order = np.lexsort((indices, priority))[:budget]
    return indices[order]


def collect_context(coords, baseline, cases, cores, chunk_points, budget, strata, padding_xy, padding_z, seed):
    """Prediction-defined boxes, height-stratified context; never read GT."""
    if budget < strata or strata < 1 or budget % strata:
        raise ValueError('Context budget must be a positive multiple of height strata.')
    states = {}
    for case in cases:
        identifier = case['original_id']
        points = np.asarray(coords[cores[identifier]], dtype=np.float32)
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite coordinates in a complete core.')
        low, high = points.min(0).astype(float), points.max(0).astype(float)
        padding = np.array([padding_xy, padding_xy, padding_z])
        states[identifier] = dict(low=low - padding, high=high + padding,
            indices=[np.empty(0, dtype=np.int64) for _ in range(strata)], seen=np.zeros(strata, dtype=np.int64))
    for start in range(0, len(baseline), chunk_points):
        stop = min(start + chunk_points, len(baseline))
        xyz, pred = coords[start:stop], baseline[start:stop]
        for identifier, state in states.items():
            mask = np.all((xyz >= state['low']) & (xyz <= state['high']), axis=1) & (pred != identifier)
            local = np.flatnonzero(mask)
            if not len(local):
                continue
            z = xyz[local, 2]
            level = np.minimum(((z - state['low'][2]) / max(state['high'][2] - state['low'][2], .1) * strata).astype(int), strata - 1)
            for h in range(strata):
                incoming = local[level == h].astype(np.int64) + start
                state['seen'][h] += len(incoming)
                state['indices'][h] = retain_lowest(state['indices'][h], incoming, budget // strata, seed)
        if start == 0 or stop == len(baseline) or (stop // chunk_points) % 10 == 0:
            print(f'INDEX_CONTEXT points={stop}/{len(baseline)}', flush=True)
    return states


def audit_core(labels, core_mask, target, node):
    """Check every exported core against exact full-forest contingency counts."""
    gt = labels[core_mask]
    if len(gt) != int(target['total'][node]) or np.count_nonzero(gt >= 0) != target['known'][node]:
        raise ValueError('Complete core population or known count differs from saved official counts.')
    positive = gt[gt > 0]
    columns = np.searchsorted(target['gt_ids'], positive)
    if np.any(columns >= len(target['gt_ids'])) or not np.array_equal(target['gt_ids'][columns], positive):
        raise ValueError('Unexpected positive GT ID in the official core.')
    common = np.bincount(columns, minlength=len(target['gt_ids']))
    if not np.array_equal(common, target['intersections'][node]):
        raise ValueError('Core GT identities differ from the saved full-forest contingency.')
    gt_rows = []
    for col in np.flatnonzero(common):
        gt_rows.append(dict(gt_id=int(target['gt_ids'][col]), core_points=int(common[col]),
            gt_points=int(target['gt_counts'][col]), recall=float(common[col] / target['gt_counts'][col]),
            precision=float(common[col] / max(target['known'][node], 1))))
    return dict(known_points=int((gt >= 0).sum()), ignored_points=int((gt < 0).sum()),
        background_points=int((gt == 0).sum()), gt_components=sorted(gt_rows, key=lambda x: (-x['core_points'], x['gt_id'])))


def write_case(directory, case, inputs, labels, indices, core, context, targets):
    identifier = case['original_id']
    directory = Path(directory) / f'original_{identifier}'
    directory.mkdir()
    payload = {key: np.asarray(values[indices]) for key, values in inputs.items()}
    if any(not np.isfinite(value).all() for value in payload.values()):
        raise ValueError('Nonfinite selected official input values.')
    if np.any((payload['tree_probability'] < 0) | (payload['tree_probability'] > 1)):
        raise ValueError('Saved probabilities are outside [0, 1].')
    if len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError('Source indices must be strictly ordered, unique and aligned.')
    if core.dtype != np.bool_ or not np.array_equal(core, payload['baseline'] == identifier):
        raise ValueError('Core mask must exactly identify the selected original instance.')
    payload.update(source_indices=indices, is_core=core)
    np.savez_compressed(directory / 'inputs.npz', **payload)
    # Read pointwise GT only AFTER prediction-only input points were fixed/saved.
    gt = np.asarray(labels[indices], dtype=np.int64)
    audit = audit_core(gt, core, targets, case['node'])
    np.savez_compressed(directory / 'audit.npz', source_indices=indices, labels=gt)
    report = dict(schema=1, posthoc=True, eligible_for_advancement=False, **case,
        complete_core=True, core_points=int(core.sum()), context_points=int((~core).sum()),
        context_population_by_stratum=context['seen'].tolist(), context_sampled_by_stratum=[len(i) for i in context['indices']],
        context_complete=bool(sum(map(len, context['indices'])) == int(context['seen'].sum())),
        box_low=context['low'].tolist(), box_high=context['high'].tolist(),
        input_sha256=file_digest(directory / 'inputs.npz'), audit_sha256=file_digest(directory / 'audit.npz'), **audit)
    write_json(directory / 'case.json', report)
    return report


def check_output(output, protected):
    output = Path(output).resolve()
    for value in protected:
        root = Path(value).resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError(f'Use a separate output directory, outside protected input: {root}')


def export(config, row, output_root, chunk_points=500_000, context_points=8192,
           strata=4, padding_xy=2., padding_z=2., max_core_points=200_000, mixed_ids=DEFAULT_MIXED_IDS):
    if chunk_points <= 0 or max_core_points <= 0 or min(padding_xy, padding_z) < 0:
        raise ValueError('Invalid export limits or padding.')
    if strata < 1 or context_points < strata or context_points % strata:
        raise ValueError('Context points must be a positive multiple of height strata.')
    cache, prepared = Path(row['cache']), Path(config['base_prepared_root']) / 'external' / row['plot']
    check_output(output_root, [cache, prepared, config['external_cache_root'], config['base_prepared_root']])
    feature_marker = json.loads((prepared / 'features.json').read_text(encoding='utf-8'))
    if feature_marker['request'] != relation_data.feature_request(cache, dict(config['geometry'], seed=config['seed'])):
        raise ValueError('Official inputs no longer match the existing relation geometry.')
    features = relation_data.load_features(prepared)
    targets = load_supervision(prepared, config['supervision'])
    cases, baseline = select_cases(features, targets, mixed_ids, max_core_points)
    if row['plot'] == 'wytham_vox0.1' and [baseline[k] for k in ('tp', 'fp', 'fn')] != config['expected_wytham_baseline_counts']:
        raise ValueError('Official Wytham baseline counts changed.')
    paths = {name: cache / (name + '.npy') for name in (*INPUT_NAMES, 'labels')}
    before = {name: relation_data.stamp(path) for name, path in paths.items()}
    guarded = [cache / 'cache.json', prepared / 'features.json', prepared / 'supervision.json']
    hashes = {str(path): file_digest(path) for path in guarded}
    values = {}
    try:
        for name, path in paths.items():
            values[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        n = len(values['baseline'])
        shapes = dict(coords=(n, 3), baseline=(n,), votes_xy=(n, 2), tree_probability=(n,), features=(n, 32), labels=(n,))
        if n != int(targets['num_points']) or any(values[name].shape != shape for name, shape in shapes.items()):
            raise ValueError('Expected aligned official cache inputs and a matching full-point count.')
        if any(not np.issubdtype(values[name].dtype, np.integer) for name in ('baseline', 'labels')):
            raise TypeError('Baseline and GT must have integer labels.')
        print('CASE_PLAN ' + json.dumps(cases, ensure_ascii=False), flush=True)
        cores = collect_cores(values['baseline'], cases, chunk_points, max_core_points)
        contexts = collect_context(values['coords'], values['baseline'], cases, cores, chunk_points,
            context_points, strata, padding_xy, padding_z, config['seed'])
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='export_' + datetime.now().strftime('%Y%m%dT%H%M%S') + '_', dir=output_root))
        reports = []
        for case in cases:
            identifier = case['original_id']
            context = contexts[identifier]
            indices = np.sort(np.concatenate((cores[identifier], *context['indices'])))
            core = values['baseline'][indices] == identifier
            report = write_case(output, case, {name: values[name] for name in INPUT_NAMES}, values['labels'],
                indices, core, context, targets)
            reports.append(report)
            print(f'SAVE original={identifier} role={case["role"]} complete_core={report["core_points"]} context={report["context_points"]}', flush=True)
        if any(relation_data.stamp(path) != before[name] for name, path in paths.items()) or any(file_digest(p) != hashes[str(p)] for p in guarded):
            raise ValueError('Official cache or compact supervision changed during export; no completed bundle written.')
        np.savez_compressed(output / 'reference_counts.npz', **{key: targets[key] for key in (
            'ids', 'gt_ids', 'gt_counts', 'total', 'known', 'intersections', 'original_tp', 'num_points', 'ignored_points')})
        normalization = Path(config['work_dir']) / 'case_analysis' / 'normalization.json'
        if normalization.exists():
            write_json(output / 'normalization.json', json.loads(normalization.read_text(encoding='utf-8')))
        summary = dict(schema=1, posthoc=True, eligible_for_advancement=False, plot=row['plot'], baseline=baseline,
            cases=reports, inputs=before, guarded_sha256=hashes, export_code_sha256=file_digest(__file__),
            settings=dict(chunk_points=chunk_points, context_points=context_points, strata=strata,
                padding_xy=padding_xy, padding_z=padding_z, max_core_points=max_core_points, seed=config['seed']),
            gt_selected_case_plan=True, model_executed=False, predictions_modified=False,
            control_selection='Unique original TPs, purity>=.8, known fraction>=.5; minimize abs(log size ratio)+abs(log Z-span ratio)+0.1*XY center distance.',
            notes=['All points of each original core are retained; no core subsampling.',
                'Context is prediction-box-defined and hash-sampled by height; it is not full-resolution context.',
                'inputs.npz contains only saved observable inputs and original instance IDs.',
                'audit.npz and reference_counts.npz contain GT for posthoc analysis only; GT<0 remains unknown.',
                'Case and control selection uses GT; this is not a deployable candidate selector or an effectiveness evaluation.',
                'No local-graph inference, training, proposed split or threshold tuning was performed.'])
        write_json(output / 'brief.json', summary)
        (output / 'README.txt').write_text(
            'POSTHOC DIAGNOSTIC CASES, NOT MODEL RESULTS\n'
            'Each original_<ID>/inputs.npz: complete original core + sampled local context.\n'
            'Fields: coords, votes_xy, tree_probability, features (32), baseline, source_indices, is_core.\n'
            'Each audit.npz: aligned source_indices and GT labels; negative labels are unknown, 0 is known background.\n'
            'Case selection and TP controls are GT-selected. Do not use audit labels as deployment inputs.\n'
            'Core includes all original points, including unknown/background; context is height-stratified, sampled and bounded.\n'
            'Coordinates are original cached coordinates in metres, without terrain normalization.\n'
            'No model executed, no predictions modified, no whole-forest data arrays copied.\n', encoding='utf-8')
        archive = output / 'split_casepack.zip'
        with zipfile.ZipFile(archive.with_suffix('.zip.tmp'), 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.suffix not in ('.zip', '.tmp'):
                    bundle.write(path, path.relative_to(output).as_posix())
        archive.with_suffix('.zip.tmp').replace(archive)
        print(f'UPLOAD={archive.resolve()} bytes={archive.stat().st_size}', flush=True)
        return archive
    finally:
        for value in values.values():
            value._mmap.close()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/instance_relation_v3/full.yaml')
    parser.add_argument('--plot', default='wytham_vox0.1')
    parser.add_argument('--mixed-ids', type=int, nargs='+', default=list(DEFAULT_MIXED_IDS))
    parser.add_argument('--output-root')
    parser.add_argument('--context-points', type=int, default=8192)
    parser.add_argument('--max-core-points', type=int, default=200_000)
    parser.add_argument('--chunk-points', type=int, default=500_000)
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    matches = [row for row in relation_data.entries(config, external=True) if row['plot'] == args.plot]
    if len(matches) != 1:
        raise ValueError('Select exactly one existing external official cache.')
    output = args.output_root or str(Path(config['work_dir']) / 'split_case_analysis')
    export(config, matches[0], output, chunk_points=args.chunk_points, context_points=args.context_points,
        max_core_points=args.max_core_points, mixed_ids=args.mixed_ids)


if __name__ == '__main__':
    main()
