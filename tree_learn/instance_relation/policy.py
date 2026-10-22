"""Prediction-only, simultaneous keep/merge decisions; all other labels stay exact."""

from pathlib import Path

import numpy as np


def select_pairs(pairs, scores, node_quality, policy):
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    merge, same, quality = scores['merge'], scores['same'], scores['quality']
    a, b = pairs.T
    gain = scores['gain']
    valid = ((merge >= policy['merge_threshold']) & (same >= policy['same_threshold'])
             & (quality[:, 0] >= policy['min_union_precision']) & (quality[:, 1] >= policy['min_union_recall'])
             & (quality[:, 2] >= policy['min_union_iou']) & (gain >= policy['min_iou_gain'])
             & (node_quality[a] < policy['protect_threshold']) & (node_quality[b] < policy['protect_threshold']))
    order = np.flatnonzero(valid)
    confidence = merge * quality[:, 2]
    order = sorted(order.tolist(), key=lambda i: (-float(confidence[i]), int(a[i]), int(b[i])))
    best = {}
    for edge in order:
        best.setdefault(int(a[edge]), edge)
        best.setdefault(int(b[edge]), edge)
    chosen = [edge for edge in order if best[int(a[edge])] == edge == best[int(b[edge])]]
    accepted = pairs[chosen].copy()
    if len(np.unique(accepted)) != 2 * len(accepted):
        raise RuntimeError('Relation policy created repeated/transitive merge actions.')
    return accepted


def write_prediction(baseline_path, output_path, ids, accepted, chunk=500_000, replace_existing=False):
    """No GT is opened. Only the larger ID in each accepted pair is relabeled."""
    ids = np.asarray(ids, dtype=np.int64)
    mapping = ids.copy()
    used = set()
    for a, b in accepted:
        if a == b or min(a, b) < 0 or max(a, b) >= len(ids) or a in used or b in used:
            raise ValueError('Only disjoint pairs of original instances may be merged.')
        used.update((int(a), int(b)))
        mapping[max(a, b)] = ids[min(a, b)]
    output_path = Path(output_path)
    if output_path.resolve() == Path(baseline_path).resolve():
        raise ValueError('Never overwrite the official baseline.')
    if output_path.exists() and not replace_existing:
        raise FileExistsError(output_path)
    baseline = np.load(baseline_path, mmap_mode='r', allow_pickle=False)
    temporary = output_path.with_suffix('.partial.npy')
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=baseline.dtype, shape=baseline.shape)
    changed, selected_points, foreground = 0, 0, 0
    try:
        for start in range(0, len(baseline), chunk):
            stop = min(start + chunk, len(baseline))
            original = baseline[start:stop]
            positive = original > 0
            local = original.copy()
            positions = np.searchsorted(ids, original[positive])
            if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], original[positive]):
                raise ValueError('Original instance IDs differ from prepared descriptors.')
            local[positive] = mapping[positions]
            selected = np.isin(original, ids[sorted(used)])
            if not np.array_equal(local[~selected], original[~selected]):
                raise RuntimeError('An unselected instance or background point was edited.')
            if not np.array_equal(local > 0, positive):
                raise RuntimeError('Merge changed the official foreground.')
            changed += int(np.count_nonzero(local != original))
            selected_points += int(selected.sum())
            foreground += int(positive.sum())
            output[start:stop] = local
            if start == 0 or stop == len(baseline) or (stop // chunk) % 10 == 0:
                print(f'WRITE_MERGES points={stop}/{len(baseline)} changed={changed}', flush=True)
        output.flush()
        report = dict(num_points=len(baseline), accepted_pairs=len(accepted),
            selected_original_instances=len(used), selected_original_points=selected_points,
            relabeled_points=changed, unchanged_unselected_instances=len(ids) - len(used),
            changed_foreground_points=0, changed_unselected_points=0,
            official_foreground_points=foreground)
    finally:
        output._mmap.close()
        baseline._mmap.close()
    temporary.replace(output_path)
    return report
