"""GT is opened only after the deployable features and candidate list are fixed."""

import json
from pathlib import Path

import numpy as np

from tools.forest_abc.diagnose import analyze_table, scan_arrays
from tree_learn.mask_refinement.data import file_digest, write_json
from .data import stamp


def make_targets(geometry, table, pairs, settings):
    counts, known, common = geometry['counts'], table['known'], table['intersections']
    union = known[:, None] + counts[None] - common
    iou = np.divide(common, union, out=np.zeros_like(union, dtype=float), where=union > 0)
    node_quality = iou.max(1)
    dominant = common.argmax(1)
    purity = common[np.arange(len(common)), dominant] / np.maximum(known, 1)
    coverage = known / np.maximum(table['total'], 1)
    node_valid = coverage >= settings['min_known_fraction']
    baseline = analyze_table(geometry, table)
    tp = np.array([r['category'] == 'true_positive' for r in baseline['predictions']])
    left, right = pairs.T
    # Keep exact-count threshold decisions in float64; cast only the training targets.
    quality = np.empty((len(pairs), 3), dtype=np.float64)
    target_gt = np.empty(len(pairs), dtype=np.int64)
    for start in range(0, len(pairs), 2048):
        stop = min(start + 2048, len(pairs))
        aa, bb = left[start:stop], right[start:stop]
        joined = common[aa] + common[bb]
        mass = known[aa] + known[bb]
        denom = mass[:, None] + counts[None] - joined
        scores = np.divide(joined, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
        best = scores.argmax(1)
        intersection = joined[np.arange(len(best)), best]
        quality[start:stop] = np.column_stack((intersection / np.maximum(mass, 1), intersection / counts[best],
                                               scores[np.arange(len(best)), best]))
        target_gt[start:stop] = geometry['ids'][best]
    same = ((dominant[left] == dominant[right]) & (purity[left] >= settings['same_tree_purity'])
            & (purity[right] >= settings['same_tree_purity']) & (known[left] > 0) & (known[right] > 0))
    merge = (same & (quality[:, 2] > .5) & (quality[:, 0] >= settings['min_union_precision'])
             & (quality[:, 2] > np.maximum(node_quality[left], node_quality[right]) + settings['min_iou_gain'])
             & ~(tp[left] & tp[right]))
    return dict(node_quality=node_quality.astype(np.float32), node_valid=node_valid,
        pair_same=same.astype(np.float32), pair_merge=merge.astype(np.float32),
        pair_gain=(quality[:, 2] - np.maximum(node_quality[left], node_quality[right])).astype(np.float32),
        pair_quality=quality.astype(np.float32), pair_target_gt=target_gt, pair_valid=node_valid[left] & node_valid[right],
        original_tp=tp), baseline


def prepare_supervision(cache, output, features, settings, chunk=500_000):
    cache, output = Path(cache), Path(output)
    request = dict(features_sha256=file_digest(output / 'features.npz'), settings=settings,
                   labels=stamp(cache / 'labels.npy'), code=file_digest(__file__))
    marker = output / 'supervision.json'
    if marker.exists():
        if json.loads(marker.read_text())['request'] != request:
            raise ValueError(f'Stale relation supervision: {output}')
        return load_supervision(output, settings)
    arrays = {name: np.load(cache / f'{name}.npy', mmap_mode='r', allow_pickle=False)
              for name in ('coords', 'labels', 'baseline')}
    try:
        geometry, tables = scan_arrays(arrays['coords'], arrays['labels'],
                                       {'baseline': arrays['baseline']}, chunk)
        table = tables['baseline']
        if not np.array_equal(features['ids'], table['ids']):
            raise ValueError('Supervision IDs differ from prediction-only features.')
        if not np.array_equal(features['population'].sum(1), table['total']):
            raise ValueError('Official instance sizes changed between feature and supervision scans.')
        targets, baseline = make_targets(geometry, table, features['pairs'], settings)
        if stamp(cache / 'labels.npy') != request['labels']:
            raise ValueError('Labels changed during supervision extraction.')
        np.savez(output / 'supervision.npz', **targets, gt_ids=geometry['ids'], gt_counts=geometry['counts'],
                 z_span=geometry['z_span'], ignored_points=geometry['ignored_points'],
                 num_points=geometry['num_points'], ids=table['ids'], total=table['total'],
                 known=table['known'], intersections=table['intersections'])
        write_json(marker, dict(request=request, baseline=baseline['metrics'],
            num_valid_pairs=int(targets['pair_valid'].sum()),
            num_merge_positive=int(np.count_nonzero(targets['pair_valid'] & (targets['pair_merge'] > 0))),
            num_valid_nodes=int(targets['node_valid'].sum()),
            supervision_sha256=file_digest(output / 'supervision.npz')))
        return targets_and_table(output)
    finally:
        for array in arrays.values():
            array._mmap.close()


def targets_and_table(output):
    with np.load(Path(output) / 'supervision.npz', allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def load_supervision(output, settings=None):
    output = Path(output)
    meta = json.loads((output / 'supervision.json').read_text())
    if stamp(meta['request']['labels']['path']) != meta['request']['labels']:
        raise ValueError(f'Training/evaluation GT changed since preparation: {output}')
    if (meta['request']['code'] != file_digest(__file__)
            or meta['request']['features_sha256'] != file_digest(output / 'features.npz')
            or (settings is not None and meta['request']['settings'] != settings)):
        raise ValueError(f'Relation supervision belongs to other features/settings/code: {output}')
    if file_digest(output / 'supervision.npz') != meta['supervision_sha256']:
        raise ValueError(f'Changed relation supervision: {output}')
    return targets_and_table(output)


def merge_table(supervision, accepted):
    """Exact union counts for disjoint original masks; no full-forest rescan."""
    ids = supervision['ids']
    mapping = np.arange(len(ids))
    used = set()
    for a, b in accepted:
        if a == b or min(a, b) < 0 or max(a, b) >= len(ids) or a in used or b in used:
            raise ValueError('Merge actions must be disjoint original instance pairs.')
        used.update((int(a), int(b)))
        mapping[max(a, b)] = min(a, b)
    representatives, inverse = np.unique(mapping, return_inverse=True)
    table = dict(ids=ids[representatives])
    for name in ('total', 'known', 'intersections'):
        values = supervision[name]
        merged = np.zeros((len(representatives),) + values.shape[1:], dtype=values.dtype)
        np.add.at(merged, inverse, values)
        table[name] = merged
    geometry = dict(ids=supervision['gt_ids'], counts=supervision['gt_counts'], z_span=supervision['z_span'],
                    ignored_points=int(supervision['ignored_points']), num_points=int(supervision['num_points']))
    return geometry, table
