"""Prediction-only point samples; preserve the V1 candidate graph exactly."""

import json
from pathlib import Path

import numpy as np

from tree_learn.instance_relation import data as base
from tree_learn.mask_refinement.data import file_digest, write_json


def rows(config, external=False):
    settings = dict(config, prepared_root=config['base_prepared_root'])
    result = base.entries(settings, external)
    for row in result:
        row['base_prepared'] = row['prepared']
        row['prepared'] = str(Path(config['prepared_root']) / row['split'] / row['plot'])
    return result


def request(row, config):
    meta = json.loads((Path(row['base_prepared']) / 'features.json').read_text(encoding='utf-8'))
    expected = base.feature_request(row['cache'], dict(config['geometry'], seed=config['seed']))
    if meta['request'] != expected:
        raise ValueError('Official cache inputs or V1 geometry/seed changed; refuse stale V2 samples.')
    return dict(schema=2, code=file_digest(__file__), seed=config['seed'], geometry=config['geometry'],
                base_features_sha256=file_digest(Path(row['base_prepared']) / 'features.json'),
                frozen_features=base.stamp(Path(row['cache']) / 'features.npy'))


def load(row, config):
    output = Path(row['prepared'])
    meta = json.loads((output / 'points.json').read_text(encoding='utf-8'))
    if meta['request'] != request(row, config) or meta['sha256'] != file_digest(output / 'points.npz'):
        raise ValueError(f'Stale V2 point samples: {output}')
    with np.load(output / 'points.npz', allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def prepare(row, config):
    output, old = Path(row['prepared']), Path(row['base_prepared'])
    wanted = request(row, config)
    if (output / 'points.json').exists():
        return load(row, config)
    meta = json.loads((old / 'features.json').read_text(encoding='utf-8'))
    expected = base.feature_request(row['cache'], dict(config['geometry'], seed=config['seed']))
    if meta['request'] != expected:
        raise ValueError('V2 requires the unchanged V1 geometry/seed and completed V1 prediction-only features.')
    original = base.load_features(old)
    cache = Path(row['cache'])
    arrays = {key: np.load(cache / f'{key}.npy', mmap_mode='r', allow_pickle=False)
              for key in ('coords', 'baseline', 'features')}
    try:
        n = len(arrays['baseline'])
        if arrays['features'].shape != (n, 32) or arrays['coords'].shape != (n, 3):
            raise ValueError('Expected aligned official 32D frozen features and coordinates.')
        ids, low, high = original['ids'], original['low'].astype(float), original['high'].astype(float)
        strata, budget = original['sample_valid'].shape[1:]
        chosen = [np.empty(0, dtype=np.int64) for _ in range(len(ids) * strata)]
        population = np.zeros((len(ids), strata), dtype=np.int64)
        chunk = config['chunk_points']
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            labels = arrays['baseline'][start:stop]
            local = np.flatnonzero(labels > 0)
            positions = np.searchsorted(ids, labels[local])
            if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], labels[local]):
                raise ValueError('Official instance IDs differ from V1 features.')
            z = arrays['coords'][start:stop, 2][local]
            height = np.minimum(((z - low[positions, 2]) /
                np.maximum(high[positions, 2] - low[positions, 2], .001) * strata).astype(int), strata - 1)
            cell = positions * strata + height
            population += np.bincount(cell, minlength=len(ids) * strata).reshape(len(ids), strata)
            order = np.argsort(cell, kind='stable')
            keys, starts, counts = np.unique(cell[order], return_index=True, return_counts=True)
            for key, begin, count in zip(keys, starts, counts):
                indices = np.r_[chosen[int(key)], local[order[begin:begin + count]] + start]
                if len(indices) > budget:
                    indices = indices[np.argpartition(base.priorities(indices, config['seed']), budget - 1)[:budget]]
                chosen[int(key)] = indices
            if start == 0 or stop == n or (stop // chunk) % 10 == 0:
                print(f'POINT_INDICES {row["plot"]}: {stop}/{n}', flush=True)
        if not np.array_equal(population, original['population']):
            raise ValueError('V1 full instance populations changed.')
        shape = original['sample_valid'].shape
        indices = np.full(shape, -1, dtype=np.int64)
        frozen = np.zeros(shape + (32,), dtype=np.float32)
        for cell, index in enumerate(chosen):
            index = np.sort(index)
            i, h = divmod(cell, strata)
            indices[i, h, :len(index)] = index
            frozen[i, h, :len(index)] = arrays['features'][index]
            if not np.array_equal(arrays['coords'][index], original['samples'][i, h, :len(index), :3]):
                raise ValueError('Reconstructed samples differ from V1; do not silently change the candidate graph.')
        valid = indices >= 0
        if not np.array_equal(valid, original['sample_valid']) or not np.isfinite(frozen).all():
            raise ValueError('Invalid frozen point samples.')
        count = valid.sum(2)
        weights = np.broadcast_to(population[..., None] / np.maximum(count[..., None], 1), shape).copy()
        weights[~valid] = 0
        points = np.concatenate((original['samples'], frozen), axis=-1).reshape(len(ids), -1, 38)
        result = dict(ids=ids, pairs=original['pairs'], points=points, valid=valid.reshape(len(ids), -1),
            weights=weights.reshape(len(ids), -1).astype(np.float32), indices=indices.reshape(len(ids), -1),
            population=population, proximity=original['pair_features'][:, 8] + original['pair_features'][:, 9] / 4.)
        if request(row, config) != wanted:
            raise ValueError('Official frozen feature inputs changed during preparation.')
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / 'points.npz', **result)
        write_json(output / 'points.json', dict(request=wanted, sha256=file_digest(output / 'points.npz'),
            num_nodes=len(ids), num_pairs=len(result['pairs']), samples_per_node=points.shape[1],
            gt_opened=False, candidates_exactly_v1=True))
        return result
    finally:
        for array in arrays.values():
            array._mmap.close()
