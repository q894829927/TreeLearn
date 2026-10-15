"""Aligned forest caches and prediction-only candidate geometry for stage A1."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = 1
INPUT_ARRAYS = ('coords', 'features', 'votes_xy', 'tree_probability', 'baseline')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def export_cache(directory, coords, features, offsets, semantic_logits,
                 baseline, labels, metadata):
    """Write one aligned array at a time, without another large compressed NPZ."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'cache.json'
    if marker.exists():
        raise FileExistsError(f'Completed cache already exists: {directory}')
    count = len(coords)
    if (features is None or features.shape != (count, 32) or
            coords.shape != (count, 3) or offsets.shape != (count, 3) or
            semantic_logits.shape != (count, 2) or
            len(baseline) != count or len(labels) != count or count == 0):
        raise ValueError('A1 requires aligned, nonempty arrays and 32D features.')
    if np.any(np.asarray(baseline) < 0):
        raise ValueError('Export requires completed baseline point assignment.')
    for name, values, dtype in (
            ('coords', coords, np.float32), ('features', features, np.float32),
            ('baseline', baseline, np.int64), ('labels', labels, np.int64)):
        values = np.asarray(values, dtype=dtype)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite {name} in cache.')
        np.save(directory / f'{name}.npy', values, allow_pickle=False)
    votes = np.asarray(coords[:, :2] + offsets[:, :2], dtype=np.float32)
    logits = np.asarray(semantic_logits, dtype=np.float32)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability = probability[:, 0] / probability.sum(axis=1)
    if not np.isfinite(votes).all() or not np.isfinite(probability).all():
        raise ValueError('Nonfinite votes or probabilities.')
    np.save(directory / 'votes_xy.npy', votes, allow_pickle=False)
    np.save(directory / 'tree_probability.npy', probability, allow_pickle=False)
    write_json(marker, dict(metadata, schema=SCHEMA, num_points=count,
                            feature_dim=32))


def load_cache(directory, with_labels=False):
    """Inference does not open labels.npy or supervision metadata."""
    directory = Path(directory)
    meta = json.loads((directory / 'cache.json').read_text(encoding='utf-8'))
    if meta['schema'] != SCHEMA:
        raise ValueError('Unsupported A1 cache schema.')
    names = INPUT_ARRAYS + (('labels',) if with_labels else ())
    arrays = {name: np.load(directory / f'{name}.npy', mmap_mode='r',
                            allow_pickle=False) for name in names}
    for name, array in arrays.items():
        if len(array) != meta['num_points']:
            raise ValueError(f'Cache row mismatch: {name}')
    return arrays, meta


def prepare_regions(directory, settings):
    """XY-expanded candidate boxes, full Z; no GT in region or query inputs."""
    directory = Path(directory)
    marker = directory / 'regions.json'
    signature = digest(settings)
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old['settings_digest'] != signature:
            raise ValueError('Region settings changed. Use a new output_root.')
        return old
    arrays, _ = load_cache(directory)
    coords, baseline = arrays['coords'], arrays['baseline']
    tree_indices = np.flatnonzero(baseline > 0)
    order = tree_indices[np.argsort(baseline[tree_indices], kind='stable')]
    ids, starts, counts = np.unique(baseline[order], return_index=True,
                                    return_counts=True)
    if not len(ids):
        raise ValueError('No baseline candidates.')
    spatial = cKDTree(coords[:, :2])
    regions_dir = directory / 'regions'
    regions_dir.mkdir(exist_ok=True)
    centers, scales, vote_centers, descriptors = [], [], [], []
    total_references = 0
    for candidate_id, start, count in zip(ids, starts, counts):
        members = order[start:start + count]
        xyz = coords[members]
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        center = (low + high) / 2
        scale = np.maximum(high - low, float(settings['minimum_scale_m']))
        half_xy = (high[:2] - low[:2]) / 2 + float(settings['margin_xy_m'])
        indices = np.asarray(spatial.query_ball_point(
            center[:2], np.nextafter(float(np.max(half_xy)), np.inf), p=np.inf),
            dtype=np.int64)
        keep = np.all(np.abs(coords[indices, :2] - center[:2]) <=
                      half_xy + 1e-5, axis=1)
        indices = np.sort(indices[keep])
        if len(indices) > int(settings['max_region_points']):
            raise RuntimeError(
                f'Candidate {candidate_id}: {len(indices)} region points exceed '
                'max_region_points. Inspect candidate extent; do not truncate.')
        # All original members must remain reachable, including numerical edges.
        if not np.isin(members, indices, assume_unique=True).all():
            raise RuntimeError('Candidate region lost original member points.')
        total_references += len(indices)
        np.save(regions_dir / f'{int(candidate_id)}.npy', indices,
                allow_pickle=False)
        centers.append(center)
        scales.append(scale)
        vote_centers.append(np.median(arrays['votes_xy'][members], axis=0))
        # Chunked pooling bounds memory even for a very large merged candidate.
        feature_sum = np.zeros(32, dtype=np.float64)
        feature_max = np.full(32, -np.inf, dtype=np.float32)
        for offset in range(0, len(members), 65536):
            feature = arrays['features'][members[offset:offset + 65536]]
            feature_sum += feature.sum(axis=0, dtype=np.float64)
            feature_max = np.maximum(feature_max, feature.max(axis=0))
        descriptors.append(np.concatenate([feature_sum / count, feature_max]))
    np.savez(directory / 'candidates.npz', ids=ids,
             centers=np.asarray(centers, dtype=np.float32),
             scales=np.asarray(scales, dtype=np.float32),
             vote_centers=np.asarray(vote_centers, dtype=np.float32),
             descriptors=np.asarray(descriptors, dtype=np.float32))
    result = dict(schema=SCHEMA, settings=settings, settings_digest=signature,
                  num_candidates=len(ids), num_region_references=total_references)
    write_json(marker, result)
    return result


def load_candidates(directory):
    with np.load(Path(directory) / 'candidates.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def point_inputs(arrays, candidates, row, indices):
    """40D deployable input, invariant to a shared coordinate translation."""
    candidate_id = candidates['ids'][row]
    relative = (arrays['coords'][indices] - candidates['centers'][row]) / candidates['scales'][row]
    relative_vote = (arrays['votes_xy'][indices] - candidates['vote_centers'][row]) / candidates['scales'][row, :2]
    height = relative[:, 2:3] + 0.5
    return np.concatenate([
        arrays['features'][indices], relative, relative_vote,
        arrays['tree_probability'][indices, None],
        (arrays['baseline'][indices] == candidate_id)[:, None], height,
    ], axis=1).astype(np.float32)


def stratified_sample(indices, labels, target_id, size, rng):
    """Positive/neighbor/background sampling with population weights.

    Inverse sampling weights keep BCE and Dice estimates tied to the complete
    labeled region rather than the artificial class balance in a minibatch.
    """
    local = labels[indices]
    pools = [indices[(local == target_id) & (local > 0)],
             indices[(local > 0) & (local != target_id)],
             indices[local == 0]]
    fractions = np.asarray([0.5, 0.25, 0.25])
    active = [i for i, pool in enumerate(pools) if len(pool)]
    if not active or size < len(active):
        raise ValueError('Insufficient labeled sample support.')
    allocation = np.ones(len(active), dtype=int)
    remaining = size - len(active)
    raw = remaining * fractions[active] / fractions[active].sum()
    allocation += np.floor(raw).astype(int)
    allocation[np.argmax(raw - np.floor(raw))] += size - allocation.sum()
    selected, weights = [], []
    for group, draws in zip(active, allocation):
        pool = pools[group]
        selected.append(rng.choice(pool, size=int(draws), replace=True))
        weights.append(np.full(draws, len(pool) / draws, dtype=np.float32))
    selected, weights = np.concatenate(selected), np.concatenate(weights)
    weights *= len(weights) / weights.sum()
    return selected, weights


class ForestLRU:
    """Memory maps only a bounded number of forests in each training worker."""
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, path):
        key = str(path)
        if key not in self.items:
            self.items[key] = (load_cache(path, with_labels=True)[0],
                               load_candidates(path))
        self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return self.items[key]
