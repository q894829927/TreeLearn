"""GT-free descriptors and candidate pairs from existing official forest caches."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tree_learn.mask_refinement.data import file_digest, write_json

INPUTS = ('coords', 'baseline', 'votes_xy', 'tree_probability')
SCHEMA = 1


def stamp(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def entries(config, external=False):
    root = Path(config['external_cache_root'] if external else config['cache_root'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    result = []
    for record in manifest['forests']:
        split = 'external' if external else record['split']
        if not external and record['plot'] not in config['splits'].get(split, []):
            continue
        cache = root / record['cache']
        meta = json.loads((cache / 'cache.json').read_text(encoding='utf-8'))
        if meta['signature'] != record['signature']:
            raise ValueError(f'Cache signature differs from manifest: {cache}')
        if Path(meta['checkpoint']).name != Path(config['checkpoint']).name:
            raise ValueError(f'Cache does not identify the configured official checkpoint: {cache}')
        result.append(dict(plot=record['plot'], split=split, cache=str(cache),
            prepared=str(Path(config['prepared_root']) / split / record['plot'])))
    if not result or len({r['plot'] for r in result}) != len(result):
        raise ValueError('No requested forests or duplicate manifest plots.')
    if not external:
        for split in ('train', 'validation'):
            if sorted(r['plot'] for r in result if r['split'] == split) != sorted(config['splits'][split]):
                raise ValueError(f'Manifest does not contain the configured {split} forests.')
        if set(config['splits']['train']) & set(config['splits']['validation']):
            raise ValueError('Training and validation forests overlap.')
    return result


def feature_request(cache, settings):
    return dict(schema=SCHEMA, settings=settings, source_code=file_digest(__file__),
                inputs={name: stamp(Path(cache) / f'{name}.npy') for name in INPUTS},
                metadata=stamp(Path(cache) / 'cache.json'))


def priorities(indices, seed):
    """Counter hash: samples are independent of chunk size and instance numbering."""
    value = np.asarray(indices, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _progress(stage, stop, count, chunk):
    if stop == count or stop <= chunk or (stop // chunk) % 10 == 0:
        print(f'{stage} points={stop}/{count}', flush=True)


def _sample_descriptors(data, settings):
    k, strata = data['population'].shape
    descriptors, lower_centers, vote_centers = [], [], []
    for i in range(k):
        valid = data['sample_valid'][i]
        samples = data['samples'][i]
        xyz, votes = samples[..., :3], samples[..., 3:5]
        span = np.maximum(data['high'][i] - data['low'][i], .1)
        center = (data['high'][i] + data['low'][i]) / 2
        count = int(data['population'][i].sum())
        # Weight strata by their full population, not by equal sample budgets.
        weights = data['population'][i] / max(count, 1)
        vm = np.zeros(2)
        for h in range(strata):
            if valid[h].any():
                vm += weights[h] * votes[h][valid[h]].mean(0)
        first = int(np.flatnonzero(data['population'][i])[0])
        lower = xyz[first][valid[first]].mean(0)
        lower_centers.append(lower)
        vote_centers.append(vm)
        p_mean = data['probability_sum'][i] / count
        p_var = max(0., data['probability_square'][i] / count - p_mean ** 2)
        features = [np.log1p(count), *np.log1p(span), np.log1p(count / np.prod(span)),
                    p_mean, np.sqrt(p_var), *np.abs(lower[:2] - center[:2]) / span[:2],
                    np.linalg.norm(vm - lower[:2]) / settings['vote_scale_m']]
        for h in range(strata):
            mask = valid[h]
            if not mask.any():
                features.extend([0.] * 11)
                continue
            xx, vv, pp = xyz[h][mask], votes[h][mask], samples[h, mask, 5]
            xm, vmean = xx.mean(0), vv.mean(0)
            vote_rms = np.sqrt(np.square(vv - vmean).sum(1).mean())
            centered = xx - xm
            eigen = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(xx)), 0.)
            features.extend([weights[h], float(pp.mean()), float(pp.std()), np.log1p(vote_rms),
                             *((xm - center) / span), *((vmean - vm) / settings['vote_scale_m']),
                             eigen[-1] / max(float(eigen.sum()), 1e-8),
                             np.log1p(np.sqrt(eigen.sum()))])
        descriptors.append(features)
    return (np.asarray(descriptors, dtype=np.float32), np.asarray(lower_centers, dtype=np.float32),
            np.asarray(vote_centers, dtype=np.float32))


def candidate_pairs(data, settings):
    """Union of two spatial graphs; no GT-based filtering or candidate truncation."""
    proposals = []
    for key, radius in (('vote_centers', settings['vote_radius_m']),
                        ('lower_centers', settings['lower_radius_m'])):
        tree = cKDTree(data[key][:, :2])
        count = (tree.count_neighbors(tree, radius) - len(data[key])) // 2
        if count > settings['max_pairs']:
            raise ValueError(f'{count} pairs in {key} exceed max_pairs; refuse before allocating candidates.')
        proposals.extend(tree.query_pairs(radius, output_type='ndarray').tolist())
    pairs = np.asarray(sorted(set(tuple(pair) for pair in proposals)), dtype=np.int64).reshape(-1, 2)
    if len(pairs) > settings['max_pairs']:
        raise ValueError(f'{len(pairs)} pairs exceed max_pairs; inspect radii instead of truncating.')
    return pairs


def _distances(a, b):
    if not len(a) or not len(b):
        return [0., 0., 0., 0.]
    first = cKDTree(b).query(a, workers=1)[0]
    second = cKDTree(a).query(b, workers=1)[0]
    both = np.r_[first, second]
    # Symmetric contact quantiles and fractions at two physical scales.
    return [float(np.quantile(both, .1)), float(np.median(both)),
            float(np.mean(both <= .2)), float(np.mean(both <= .5))]


def pair_descriptors(data, pairs, settings):
    result = []
    for number, (i, j) in enumerate(pairs):
        li, lj, hi, hj = data['low'][i], data['low'][j], data['high'][i], data['high'][j]
        span_i, span_j = np.maximum(hi - li, .1), np.maximum(hj - lj, .1)
        gap = np.maximum(np.maximum(li, lj) - np.minimum(hi, hj), 0.)
        overlap = np.maximum(np.minimum(hi, hj) - np.maximum(li, lj), 0.)
        count_i, count_j = data['population'][i].sum(), data['population'][j].sum()
        features = [*np.log1p(gap), *np.minimum(span_i, span_j) / np.maximum(span_i, span_j),
                    overlap[2] / max(float(min(span_i[2], span_j[2])), .1),
                    min(count_i, count_j) / max(count_i, count_j),
                    np.linalg.norm(data['vote_centers'][i] - data['vote_centers'][j]) / settings['vote_scale_m'],
                    np.linalg.norm(data['lower_centers'][i, :2] - data['lower_centers'][j, :2]),
                    abs(float(data['lower_centers'][i, 2] - data['lower_centers'][j, 2]))]
        si = data['samples'][i][data['sample_valid'][i]]
        sj = data['samples'][j][data['sample_valid'][j]]
        features.extend(_distances(si[:, :3], sj[:, :3]))
        features.extend(_distances(si[:, 3:5], sj[:, 3:5]))
        # Shared physical height bands of the pair, not misaligned per-node bins.
        bottom, height = min(li[2], lj[2]), max(float(max(hi[2], hj[2]) - min(li[2], lj[2])), .001)
        for h in range(settings['height_strata']):
            lo = bottom + height * h / settings['height_strata']
            high = bottom + height * (h + 1) / settings['height_strata']
            if h + 1 == settings['height_strata']:
                high += 1e-5
            aa = si[(si[:, 2] >= lo) & (si[:, 2] < high)]
            bb = sj[(sj[:, 2] >= lo) & (sj[:, 2] < high)]
            features.extend([float(len(aa) > 0 and len(bb) > 0), float(min(len(aa), len(bb))) / max(len(aa), len(bb), 1)])
            features.extend(_distances(aa[:, :3], bb[:, :3]))
            features.extend(_distances(aa[:, 3:5], bb[:, 3:5]))
        result.append(features)
        if number == 0 or (number + 1) % 1000 == 0 or number + 1 == len(pairs):
            print(f'PAIR_FEATURES pairs={number + 1}/{len(pairs)}', flush=True)
    width = 19 + 10 * settings['height_strata']
    return np.asarray(result, dtype=np.float32).reshape(len(pairs), width)


def prepare_features(cache, output, settings, seed, chunk=500_000):
    cache, output = Path(cache), Path(output)
    request = feature_request(cache, dict(settings, seed=seed))
    marker = output / 'features.json'
    if marker.exists():
        if json.loads(marker.read_text())['request'] != request:
            raise ValueError(f'Stale relation features; use a new prepared_root: {output}')
        return load_features(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.load(cache / f'{name}.npy', mmap_mode='r', allow_pickle=False) for name in INPUTS}
    try:
        n = len(arrays['baseline'])
        metadata = json.loads((cache / 'cache.json').read_text(encoding='utf-8'))
        if metadata['num_points'] != n:
            raise ValueError('Official cache metadata point count differs from baseline.npy.')
        if (arrays['coords'].shape != (n, 3) or arrays['votes_xy'].shape != (n, 2)
                or arrays['tree_probability'].shape != (n,) or not n
                or not np.issubdtype(arrays['baseline'].dtype, np.integer)):
            raise ValueError('Require nonempty aligned official coordinates, instances, probabilities and votes.')
        found = set()
        for start in range(0, n, chunk):
            b = arrays['baseline'][start:start + chunk]
            if np.any(b < 0):
                raise ValueError('Official instances must be complete and nonnegative.')
            found.update(np.unique(b[b > 0]).tolist())
            _progress('INSTANCE_IDS', min(start + chunk, n), n, chunk)
        ids = np.asarray(sorted(found), dtype=np.int64)
        if not len(ids) or len(ids) > settings['max_instances']:
            raise ValueError('No official instances or instance memory guard exceeded.')
        k, strata, budget = len(ids), settings['height_strata'], settings['points_per_stratum']
        if k * strata * budget > 20_000_000:
            raise ValueError('Height sample slots exceed the memory guard; inspect sampling configuration.')
        low, high = np.full((k, 3), np.inf), np.full((k, 3), -np.inf)
        ps, pq = np.zeros(k), np.zeros(k)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            b, p = arrays['baseline'][start:stop], arrays['tree_probability'][start:stop]
            xyz, votes = arrays['coords'][start:stop], arrays['votes_xy'][start:stop]
            if (not all(np.isfinite(v).all() for v in (p, xyz, votes)) or np.any((p < 0) | (p > 1))
                    or not np.array_equal(p >= .5, b > 0)):
                raise ValueError('Input signals differ from the completed official 0.5 foreground cache.')
            positive = b > 0
            rows = np.searchsorted(ids, b[positive])
            for axis in range(3):
                np.minimum.at(low[:, axis], rows, xyz[positive, axis])
                np.maximum.at(high[:, axis], rows, xyz[positive, axis])
            ps += np.bincount(rows, weights=p[positive], minlength=k)
            pq += np.bincount(rows, weights=np.square(p[positive].astype(float)), minlength=k)
            _progress('INSTANCE_BOUNDS', stop, n, chunk)
        population = np.zeros((k, strata), dtype=np.int64)
        chosen = [np.empty(0, dtype=np.int64) for _ in range(k * strata)]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            b = arrays['baseline'][start:stop]
            local = np.flatnonzero(b > 0)
            rows = np.searchsorted(ids, b[local])
            z = arrays['coords'][start:stop, 2][local]
            h = np.minimum(((z - low[rows, 2]) / np.maximum(high[rows, 2] - low[rows, 2], .001) * strata).astype(int), strata - 1)
            encoded = rows * strata + h
            population += np.bincount(encoded, minlength=k * strata).reshape(k, strata)
            order = np.argsort(encoded, kind='stable')
            keys, starts, counts = np.unique(encoded[order], return_index=True, return_counts=True)
            for key, begin, count in zip(keys, starts, counts):
                indices = np.r_[chosen[int(key)], local[order[begin:begin + count]] + start]
                if len(indices) > budget:
                    keep = np.argpartition(priorities(indices, seed), budget - 1)[:budget]
                    indices = indices[keep]
                chosen[int(key)] = indices
            _progress('HEIGHT_SAMPLES', stop, n, chunk)
        samples = np.zeros((k, strata, budget, 6), dtype=np.float32)
        valid = np.zeros((k, strata, budget), dtype=bool)
        for cell, indices in enumerate(chosen):
            indices = np.sort(indices)
            i, h = divmod(cell, strata)
            samples[i, h, :len(indices)] = np.column_stack((arrays['coords'][indices],
                arrays['votes_xy'][indices], arrays['tree_probability'][indices]))
            valid[i, h, :len(indices)] = True
        data = dict(ids=ids, low=low.astype(np.float32), high=high.astype(np.float32),
                    probability_sum=ps, probability_square=pq, population=population,
                    samples=samples, sample_valid=valid)
        data['node_features'], data['lower_centers'], data['vote_centers'] = _sample_descriptors(data, settings)
        data['pairs'] = candidate_pairs(data, settings)
        data['pair_features'] = pair_descriptors(data, data['pairs'], settings)
        if not all(np.isfinite(data[key]).all() for key in ('node_features', 'pair_features')):
            raise ValueError('Nonfinite relation descriptors.')
        if feature_request(cache, dict(settings, seed=seed)) != request:
            raise ValueError('Official input cache changed during feature extraction.')
        np.savez(output / 'features.npz', **data)
        write_json(marker, dict(request=request, num_points=n, num_instances=k, num_pairs=len(data['pairs']),
            node_dim=data['node_features'].shape[1], pair_dim=data['pair_features'].shape[1],
            npz_sha256=file_digest(output / 'features.npz'), gt_used=False))
        return data
    finally:
        for array in arrays.values():
            array._mmap.close()


def load_features(output):
    output = Path(output)
    meta = json.loads((output / 'features.json').read_text())
    if file_digest(output / 'features.npz') != meta['npz_sha256']:
        raise ValueError(f'Incomplete or changed relation features: {output}')
    with np.load(output / 'features.npz', allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}
