"""Training-only GT point banks, spatial cuts, thinning and nearby-tree negatives."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from tree_learn.mask_refinement.data import file_digest, write_json
from .supervision import utility


def resample(points, weights, labels, capacity, rng, density=1.):
    """Thin observations without changing represented mass; retain height coverage."""
    if capacity < 4 or capacity % 4 or not len(points) or np.any(weights <= 0):
        raise ValueError('Require a nonempty weighted fragment and capacity divisible by four.')
    result = np.zeros((capacity, 38), dtype=np.float32)
    mass = np.zeros(capacity, dtype=np.float32)
    truth = np.full(capacity, -1, dtype=np.int64)
    valid = np.zeros(capacity, dtype=bool)
    z = points[:, 2]
    bands = np.minimum(((z - z.min()) / max(float(z.max() - z.min()), .001) * 4).astype(int), 3)
    for band in range(4):
        members = np.flatnonzero(bands == band)
        if not len(members):
            continue
        count = max(1, min(capacity // 4, int(np.ceil(len(members) * density))))
        chosen = rng.choice(members, size=count, replace=False)
        at = band * (capacity // 4) + np.arange(len(chosen))
        result[at], truth[at], valid[at] = points[chosen], labels[chosen], True
        selected = weights[chosen].astype(np.float64)
        mass[at] = selected * (weights[members].sum(dtype=np.float64) / selected.sum())
    return result, valid, mass, truth


def bank_request(row, config):
    return dict(code=file_digest(__file__), settings=config['augmentation'],
        targets=file_digest(Path(row['prepared']) / 'targets.json'),
        points=file_digest(Path(row['prepared']) / 'points.json'))


def prepare_bank(row, values, targets, config):
    if row['split'] != 'train':
        raise ValueError('Synthetic fragments may only be prepared from training forests.')
    output = Path(row['prepared'])
    wanted = bank_request(row, config)
    if (output / 'bank.json').exists():
        meta = json.loads((output / 'bank.json').read_text(encoding='utf-8'))
        if meta['request'] != wanted or meta['sha256'] != file_digest(output / 'bank.npz'):
            raise ValueError('Stale training fragment bank.')
        with np.load(output / 'bank.npz', allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}
    settings = config['augmentation']
    rng = np.random.default_rng(config['seed'])
    points, weights, gt = values['points'][values['valid']], values['weights'][values['valid']], targets['point_labels'][values['valid']]
    represented = targets['intersections'].sum(0)
    records = []
    for col, identifier in enumerate(targets['gt_ids']):
        members = np.flatnonzero(gt == identifier)
        coverage = represented[col] / targets['gt_counts'][col]
        if len(members) < settings['min_bank_points'] or coverage < settings['min_foreground_coverage']:
            continue
        selected_weights = weights[members].astype(np.float64)
        selected_weights *= represented[col] / selected_weights.sum()
        sampled = resample(points[members], selected_weights, gt[members], settings['bank_points'], rng)
        records.append((int(identifier), int(targets['gt_counts'][col]), *sampled))
    if len(records) < 2:
        raise ValueError(f'{row["plot"]}: fewer than two eligible GT trees for training fragment generation.')
    bank = dict(ids=np.asarray([r[0] for r in records]), counts=np.asarray([r[1] for r in records]),
        points=np.stack([r[2] for r in records]), valid=np.stack([r[3] for r in records]),
        weights=np.stack([r[4] for r in records]), labels=np.stack([r[5] for r in records]))
    centers = np.asarray([p[m, :2].mean(0) for p, m in zip(bank['points'], bank['valid'])])
    _, neighbors = cKDTree(centers).query(centers, k=min(len(centers), 9), workers=1)
    bank['neighbors'] = np.asarray([[j for j in row_neighbors if j != i][:min(len(centers) - 1, 8)]
                                   for i, row_neighbors in enumerate(neighbors)], dtype=np.int64)
    np.savez(output / 'bank.npz', **bank)
    write_json(output / 'bank.json', dict(request=wanted, sha256=file_digest(output / 'bank.npz'),
        trees=len(records), split='train', quality_targets='weighted synthetic observations; not full-forest measurements'))
    return bank


def virtual_targets(point_labels, weights, counts):
    """Exact two-mask matching on weighted synthetic observations (not real forests)."""
    identifiers = np.asarray(sorted(counts), dtype=np.int64)
    sizes = np.asarray([counts[int(i)] for i in identifiers], dtype=float)
    common = np.asarray([[weights[node][point_labels[node] == i].sum(dtype=np.float64)
                          for i in identifiers] for node in range(2)])
    known = weights.sum(1, dtype=np.float64)
    iou = common / np.maximum(known[:, None] + sizes - common, 1e-8)
    r, c = linear_sum_assignment(-iou)
    before = set(c[iou[r, c] > .5].tolist())
    joined = common.sum(0)
    total = known.sum()
    joined_iou = joined / np.maximum(total + sizes - joined, 1e-8)
    best = int(joined_iou.argmax())
    after = {best} if joined_iou[best] > .5 else set()
    lost, recovered = len(before - after), len(after - before)
    fp_before = int(np.count_nonzero(common.sum(1) / np.maximum(known, 1e-8) >= .5)) - len(before)
    fp_after = int(joined.sum() / max(total, 1e-8) >= .5) - len(after)
    dominant = common.argmax(1)
    purity = common[np.arange(2), dominant] / np.maximum(known, 1e-8)
    same = dominant[0] == dominant[1] and min(purity) >= .6
    quality = np.asarray([joined[best] / max(total, 1e-8), joined[best] / sizes[best], joined_iou[best]], dtype=np.float32)
    eligible = same and not lost and joined_iou[best] > .5 and joined[best] / max(total, 1e-8) >= .65
    eligible = eligible and joined_iou[best] > iou.max() + .02
    action = (1 if recovered else 2) if eligible else 0
    return dict(action=action, risk=float(bool(lost) or not same),
        utility=float(utility(recovered, lost, fp_after - fp_before)), quality=quality)


def synthetic(bank, rng, capacity, settings, desired):
    """Generate recover/complete/keep examples; labels are recalculated after corruption."""
    result = None
    for _ in range(12):
        parent = int(rng.integers(len(bank['ids'])))
        neighbor = int(rng.choice(bank['neighbors'][parent]))
        counts = {int(bank['ids'][i]): int(bank['counts'][i]) for i in (parent, neighbor)}
        groups = []
        if desired == 0:
            # Two nearby independent trees: deliberately include intact-tree negatives.
            for index in (parent, neighbor):
                mask = bank['valid'][index]
                groups.append((bank['points'][index, mask].copy(), bank['weights'][index, mask].copy(),
                               bank['labels'][index, mask].copy()))
        else:
            mask = bank['valid'][parent]
            points = bank['points'][parent, mask].copy()
            weights = bank['weights'][parent, mask].astype(np.float64).copy()
            labels = bank['labels'][parent, mask].copy()
            # Occlusion removes mass. Density thinning below preserves that remaining mass.
            if rng.random() < .5:
                direction = rng.normal(size=2)
                coordinate = points[:, :2] @ direction
            else:
                coordinate = points[:, 2].copy()
            quantile = .5 if desired == 1 else float(rng.uniform(.7, .85))
            order = np.argsort(coordinate, kind='stable')
            cumulative = np.cumsum(weights[order])
            cut = int(np.searchsorted(cumulative, cumulative[-1] * quantile)) + 1
            cut = min(max(cut, 1), len(points) - 1)
            split = np.zeros(len(points), dtype=bool)
            split[order[:cut]] = True
            weights *= float(rng.uniform(.82, .95))
            for part in (split, ~split):
                keep = np.flatnonzero(part)
                # Remove a spatial portion, rather than arbitrary label-conditioned points.
                if len(keep) > 16 and rng.random() < .5:
                    axis = int(rng.integers(3))
                    ranked = keep[np.argsort(points[keep, axis], kind='stable')]
                    keep = ranked[:max(8, int(len(ranked) * rng.uniform(.85, 1.)))]
                groups.append((points[keep], weights[keep], labels[keep]))
            # Mix a small amount of real neighboring-tree contamination into positives.
            if rng.random() < .5:
                other = np.flatnonzero(bank['valid'][neighbor])
                for i, (p, w, g) in enumerate(groups):
                    chosen = rng.choice(other, size=min(8, len(other)), replace=False)
                    noise_mass = w.sum() * float(rng.uniform(.01, .08))
                    groups[i] = (np.r_[p, bank['points'][neighbor, chosen]],
                        np.r_[w, np.full(len(chosen), noise_mass / len(chosen))],
                        np.r_[g, bank['labels'][neighbor, chosen]])
        samples = [resample(p, w, g, capacity, rng, density=float(rng.uniform(settings['density_min'], 1.)))
                   for p, w, g in groups]
        points, valid, weights, labels = [np.stack([sample[j] for sample in samples]) for j in range(4)]
        target = virtual_targets(labels, weights, counts)
        result = dict(points=points, valid=valid, weights=weights, point_labels=labels, **target)
        if target['action'] == desired:
            break
    return result


def augment_real(points, valid, weights, rng, density_min):
    """Geometry-preserving density augmentation; no GT is consulted."""
    points, valid, weights = points.copy(), valid.copy(), weights.copy()
    for i in range(2):
        for band in np.array_split(np.arange(points.shape[1]), 4):
            original = band[valid[i, band]]
            if not len(original):
                continue
            keep = rng.choice(original, size=max(1, int(np.ceil(len(original) * rng.uniform(density_min, 1.)))), replace=False)
            mass = weights[i, original].sum(dtype=np.float64)
            valid[i, original] = False
            valid[i, keep] = True
            weights[i, original] = 0
            weights[i, keep] = mass / len(keep)
    points[~valid] = 0
    return points, valid, weights
