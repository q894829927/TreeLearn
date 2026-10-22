"""Separate GT supervision: exact counterfactual Hungarian changes for real pairs."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from tree_learn.instance_relation.data import stamp
from tree_learn.instance_relation.supervision import load_supervision, merge_table
from tree_learn.mask_refinement.data import file_digest, write_json
from tools.forest_abc.diagnose import analyze_table


def action_targets(common, known, counts, pairs):
    """Evaluate every provided union with the same maximum-total-IoU matcher."""
    denominator = known[:, None] + counts[None] - common
    iou = np.divide(common, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    active = np.flatnonzero(known > 0)
    compact = iou[active]
    positions = np.full(len(known), -1, dtype=np.int64)
    positions[active] = np.arange(len(active))
    counted = common.sum(1) / np.maximum(known, 1) >= .5

    def matching(matrix, eligible):
        r, c = linear_sum_assignment(-matrix)
        detected = c[matrix[r, c] > .5]
        return set(detected.tolist()), int(eligible - len(detected))

    baseline, baseline_fp = matching(compact, int(counted.sum()))
    outputs = dict(delta_tp=[], delta_fp=[], lost_count=[], recovered_count=[], quality=[])
    eligible_before = int(counted.sum())
    for number, (a, b) in enumerate(pairs):
        if a < 0 or a >= b or b >= len(known):
            raise ValueError('Counterfactual targets require sorted distinct original instance pairs.')
        joined = common[a] + common[b]
        mass = known[a] + known[b]
        denominator = mass + counts - joined
        union_iou = np.divide(joined, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
        if positions[a] >= 0 and positions[b] >= 0:
            matrix = np.delete(compact, positions[b], axis=0)
            matrix[positions[a]] = union_iou
        else:
            retained = known > 0
            retained[a], retained[b] = mass > 0, False
            after = np.flatnonzero(retained)
            matrix = iou[after].copy()
            if mass > 0:
                matrix[np.searchsorted(after, a)] = union_iou
        eligible = eligible_before - int(counted[a]) - int(counted[b]) + int(joined.sum() / max(mass, 1) >= .5)
        detected, fp = matching(matrix, eligible)
        lost, recovered = len(baseline - detected), len(detected - baseline)
        best = int(union_iou.argmax())
        outputs['delta_tp'].append(len(detected) - len(baseline))
        outputs['delta_fp'].append(fp - baseline_fp)
        outputs['lost_count'].append(lost)
        outputs['recovered_count'].append(recovered)
        outputs['quality'].append([joined[best] / max(mass, 1), joined[best] / counts[best], union_iou[best]])
        if number == 0 or (number + 1) % 250 == 0 or number + 1 == len(pairs):
            print(f'COUNTERFACTUAL_TARGETS pairs={number + 1}/{len(pairs)}', flush=True)
    return {key: np.asarray(value, dtype=np.float32 if key == 'quality' else np.int64).reshape(
        (len(pairs), 3) if key == 'quality' else (len(pairs),)) for key, value in outputs.items()}


def utility(recovered, lost, delta_fp):
    return np.clip((np.asarray(recovered) - 2 * np.asarray(lost) - .25 * np.asarray(delta_fp)) / 2, -1, 1)


def load(row, config):
    output = Path(row['prepared'])
    meta = json.loads((output / 'targets.json').read_text(encoding='utf-8'))
    if (meta['request'] != request(row, config)
            or meta['sha256'] != file_digest(output / 'targets.npz')):
        raise ValueError(f'Stale counterfactual supervision: {output}')
    with np.load(output / 'targets.npz', allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def request(row, config):
    return dict(code=file_digest(__file__), settings=config['supervision'],
        points_sha256=file_digest(Path(row['prepared']) / 'points.json'),
        base_targets_sha256=file_digest(Path(row['base_prepared']) / 'supervision.json'),
        labels=stamp(Path(row['cache']) / 'labels.npy'))


def prepare(row, values, config):
    output = Path(row['prepared'])
    wanted = request(row, config)
    if (output / 'targets.json').exists():
        return load(row, config)
    original = load_supervision(row['base_prepared'], config['supervision'])
    if not np.array_equal(values['ids'], original['ids']):
        raise ValueError('Original supervision and point samples have different instances.')
    valid = original['pair_valid']
    locations = np.flatnonzero(valid)
    exact = action_targets(original['intersections'], original['known'], original['gt_counts'], values['pairs'][valid])
    result = {key: original[key] for key in ('gt_ids', 'gt_counts', 'z_span', 'ignored_points', 'num_points',
        'ids', 'total', 'known', 'intersections', 'original_tp')}
    p = len(values['pairs'])
    for name in ('delta_tp', 'delta_fp', 'lost_count', 'recovered_count'):
        result[name] = np.zeros(p, dtype=np.int64)
        result[name][locations] = exact[name]
    result['quality'] = original['pair_quality'].copy()
    result['quality'][locations] = exact['quality']
    safe = valid & (original['pair_merge'] > 0) & (result['lost_count'] == 0)
    action = np.zeros(p, dtype=np.int64)
    action[safe] = 2
    action[safe & (result['recovered_count'] > 0)] = 1
    result.update(action=action, valid=valid.copy(),
        risk=((result['lost_count'] > 0) | (original['pair_same'] == 0)).astype(np.float32),
        utility=utility(result['recovered_count'], result['lost_count'], result['delta_fp']).astype(np.float32))
    labels = np.load(Path(row['cache']) / 'labels.npy', mmap_mode='r', allow_pickle=False)
    try:
        point_labels = np.full(values['valid'].shape, -1, dtype=np.int64)
        point_labels[values['valid']] = labels[values['indices'][values['valid']]]
        result['point_labels'] = point_labels
    finally:
        labels._mmap.close()
    if request(row, config) != wanted:
        raise ValueError('Supervision inputs changed during counterfactual replay.')
    np.savez(output / 'targets.npz', **result)
    geometry, table = merge_table(result, [])
    baseline = analyze_table(geometry, table)['metrics']
    write_json(output / 'targets.json', dict(request=wanted, sha256=file_digest(output / 'targets.npz'),
        baseline=baseline, num_valid_pairs=int(valid.sum()),
        action_counts={name: int(np.count_nonzero(valid & (action == i)))
                       for i, name in enumerate(('keep', 'recover', 'complete'))},
        harmful_pairs=int(np.count_nonzero(valid & (result['lost_count'] > 0)))))
    return result
