"""Training-only fragments: identical outcome labels, mainly real-data training."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from tree_learn.instance_relation_v2.augmentation import resample, augment_real, prepare_bank
from .supervision import outcomes


def virtual_targets(point_labels, weights, counts):
    identifiers = np.asarray(sorted(counts), dtype=np.int64)
    sizes = np.asarray([counts[int(i)] for i in identifiers], dtype=float)
    common = np.asarray([[weights[node][point_labels[node] == i].sum(dtype=np.float64)
                          for i in identifiers] for node in range(2)])
    # Unknown annotation is not negative/background supervision.
    known = np.asarray([weights[node][point_labels[node] >= 0].sum(dtype=np.float64) for node in range(2)])
    iou = common / np.maximum(known[:, None] + sizes - common, 1e-8)
    r, c = linear_sum_assignment(-iou)
    before = set(c[iou[r, c] > .5].tolist())
    joined, total = common.sum(0), known.sum()
    joined_iou = joined / np.maximum(total + sizes - joined, 1e-8)
    best = int(joined_iou.argmax())
    after = {best} if joined_iou[best] > .5 else set()
    lost, recovered = len(before - after), len(after - before)
    category = int(outcomes(lost, recovered))
    return dict(action=category, risk=float(category == 2), utility=float(category == 1) - float(category == 2),
        quality=np.asarray([joined[best] / max(total, 1e-8), joined[best] / sizes[best], joined_iou[best]], dtype=np.float32))


def synthetic(bank, rng, capacity, settings, desired):
    result = None
    for _ in range(12):
        parent = int(rng.integers(len(bank['ids'])))
        neighbor = int(rng.choice(bank['neighbors'][parent]))
        counts = {int(bank['ids'][i]): int(bank['counts'][i]) for i in (parent, neighbor)}
        groups = []
        if desired == 2:
            # Nearby independently complete trees: merging should destroy a detection.
            for index in (parent, neighbor):
                mask = bank['valid'][index]
                groups.append((bank['points'][index, mask].copy(), bank['weights'][index, mask].copy(),
                               bank['labels'][index, mask].copy()))
        else:
            mask = bank['valid'][parent]
            points = bank['points'][parent, mask].copy()
            weights = bank['weights'][parent, mask].astype(np.float64).copy()
            labels = bank['labels'][parent, mask].copy()
            coordinate = points[:, 2] if rng.random() < .5 else points[:, :2] @ rng.normal(size=2)
            quantile = .5 if desired == 1 else float(rng.uniform(.7, .85))
            order = np.argsort(coordinate, kind='stable')
            cumulative = np.cumsum(weights[order])
            cut = min(max(int(np.searchsorted(cumulative, cumulative[-1] * quantile)) + 1, 1), len(points) - 1)
            split = np.zeros(len(points), dtype=bool)
            split[order[:cut]] = True
            weights *= float(rng.uniform(.82, .95))
            for part in (split, ~split):
                keep = np.flatnonzero(part)
                if len(keep) > 16 and rng.random() < .5:
                    axis = int(rng.integers(3))
                    ranked = keep[np.argsort(points[keep, axis], kind='stable')]
                    keep = ranked[:max(8, int(len(ranked) * rng.uniform(.85, 1.)))]
                groups.append((points[keep], weights[keep], labels[keep]))
            if rng.random() < .5:
                other = np.flatnonzero(bank['valid'][neighbor])
                for i, (p, w, g) in enumerate(groups):
                    chosen = rng.choice(other, size=min(8, len(other)), replace=False)
                    noise_mass = w.sum() * float(rng.uniform(.01, .08))
                    groups[i] = (np.r_[p, bank['points'][neighbor, chosen]],
                        np.r_[w, np.full(len(chosen), noise_mass / len(chosen))], np.r_[g, bank['labels'][neighbor, chosen]])
        samples = [resample(p, w, g, capacity, rng, density=float(rng.uniform(settings['density_min'], 1.)))
                   for p, w, g in groups]
        points, valid, weights, labels = [np.stack([sample[j] for sample in samples]) for j in range(4)]
        target = virtual_targets(labels, weights, counts)
        result = dict(points=points, valid=valid, weights=weights, point_labels=labels, **target)
        if target['action'] == desired:
            break
    return result
