"""Mmap forests, disk-backed XY indices, and bounded point-query blocks."""

import json
from pathlib import Path

import numpy as np

from tree_learn.mask_refinement.data import file_digest, write_json


INPUTS = ('coords', 'features', 'votes_xy', 'tree_probability')


def fingerprint(cache):
    cache = Path(cache).resolve()
    stat = (cache / 'coords.npy').stat()
    return dict(cache=str(cache), metadata=file_digest(cache / 'cache.json'),
                coords_size=stat.st_size, coords_mtime_ns=stat.st_mtime_ns)


def prepare_index(cache, destination, cell_size=8., with_labels=True):
    """Only one forest's sort buffers are resident; no overlapping tile copies."""
    cache, destination = Path(cache), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    request = dict(schema=1, source=fingerprint(cache), cell_size=float(cell_size))
    marker = destination / 'index.json'
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding='utf-8'))
        if metadata['request'] != request:
            raise ValueError(f'Spatial index changed; use another index_root: {destination}')
    else:
        xyz = np.load(cache / 'coords.npy', mmap_mode='r', allow_pickle=False)
        origin = np.floor(np.min(xyz[:, :2], axis=0) / cell_size) * cell_size
        cell = np.floor((xyz[:, :2] - origin) / cell_size).astype(np.int64)
        dimensions = cell.max(axis=0) + 1
        keys = cell[:, 0] * dimensions[1] + cell[:, 1]
        order = np.argsort(keys, kind='stable')
        unique, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
        temporary = destination / 'order.tmp.npy'
        np.save(temporary, order, allow_pickle=False)
        temporary.replace(destination / 'order.npy')
        np.savez(destination / 'cells.npz', keys=unique, starts=starts, counts=counts)
        metadata = dict(request=request, origin=origin.tolist(), dimensions=dimensions.tolist(),
                        num_points=len(xyz))
        write_json(marker, metadata)
        del xyz, cell, keys, order
    if with_labels:
        labels_path = cache / 'labels.npy'
        stat = labels_path.stat()
        label_stamp = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        gt_marker = destination / 'gt.json'
        if not gt_marker.exists():
            labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
            ids, counts = np.unique(labels[labels > 0], return_counts=True)
            np.savez(destination / 'gt_counts.npz', ids=ids, counts=counts)
            write_json(gt_marker, label_stamp)
        elif json.loads(gt_marker.read_text(encoding='utf-8')) != label_stamp:
            raise ValueError(f'Labels changed since index preparation: {cache}')
    return metadata


def balanced_subset(xyz, size, rng, cell_size=4.):
    """Keep spatial/height representatives before filling the remaining budget."""
    if len(xyz) <= size:
        return np.arange(len(xyz), dtype=np.int64)
    cells = np.floor(xyz / cell_size).astype(np.int64)
    cells -= cells.min(axis=0)
    dims = cells.max(axis=0) + 1
    keys = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    shuffled = rng.permutation(len(xyz))
    _, first = np.unique(keys[shuffled], return_index=True)
    mandatory = shuffled[first]
    if len(mandatory) >= size:
        return np.sort(rng.choice(mandatory, size, replace=False))
    available = np.ones(len(xyz), dtype=bool)
    available[mandatory] = False
    extra = rng.choice(np.flatnonzero(available), size - len(mandatory), replace=False)
    return np.sort(np.concatenate([mandatory, extra]))


def population_sample(indices, labels, size, rng):
    """Shared GT strata for mask supervision, with original population mass."""
    known = labels[indices] >= 0
    indices = indices[known]
    if not len(indices):
        raise ValueError('Block has no known labels.')
    ids, inverse, counts = np.unique(labels[indices], return_inverse=True, return_counts=True)
    if len(ids) > size:
        raise ValueError('More label strata than supervision points; increase train_points.')
    minimum = min(16, size // len(ids))
    allocation = np.full(len(ids), minimum, dtype=np.int64)
    raw = (size - allocation.sum()) * counts / counts.sum()
    allocation += np.floor(raw).astype(np.int64)
    remainder = int(size - allocation.sum())
    allocation[np.argsort(-(raw - np.floor(raw)), kind='stable')[:remainder]] += 1
    order = np.argsort(inverse, kind='stable')
    starts = np.concatenate([[0], np.cumsum(counts)])
    selected, weights = [], []
    for row, draws in enumerate(allocation):
        pool = indices[order[starts[row]:starts[row + 1]]]
        selected.append(rng.choice(pool, int(draws), replace=True))
        weights.append(np.full(int(draws), len(pool) / draws, dtype=np.float32))
    return np.concatenate(selected), np.concatenate(weights)


class Forest:
    """Prediction never opens GT, baseline, candidate or supervision arrays."""
    def __init__(self, cache, index, with_labels=False):
        self.cache, self.index = Path(cache), Path(index)
        self.metadata = json.loads((self.index / 'index.json').read_text(encoding='utf-8'))
        if self.metadata['request']['source'] != fingerprint(self.cache):
            raise ValueError(f'Stale spatial index: {self.index}')
        self.arrays = {name: np.load(self.cache / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                       for name in INPUTS}
        count = self.metadata['num_points']
        if any(len(values) != count for values in self.arrays.values()):
            raise ValueError('Cache arrays do not align.')
        if self.arrays['features'].shape != (count, 32):
            raise ValueError('Expected 32-dimensional TreeLearn features.')
        self.order = np.load(self.index / 'order.npy', mmap_mode='r', allow_pickle=False)
        with np.load(self.index / 'cells.npz', allow_pickle=False) as cells:
            self.keys, self.starts, self.counts = (cells[k].copy() for k in ('keys', 'starts', 'counts'))
        self.cell_size = self.metadata['request']['cell_size']
        self.origin = np.asarray(self.metadata['origin'])
        self.dimensions = np.asarray(self.metadata['dimensions'])
        self.centers = self.origin + (np.column_stack([
            self.keys // self.dimensions[1], self.keys % self.dimensions[1]]) + .5) * self.cell_size
        self.labels = None
        self.gt_ids = self.gt_counts = None
        if with_labels:
            self.labels = np.load(self.cache / 'labels.npy', mmap_mode='r', allow_pickle=False)
            stat = (self.cache / 'labels.npy').stat()
            if json.loads((self.index / 'gt.json').read_text()) != dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns):
                raise ValueError('Label counts are stale; run prepare with a new index_root.')
            with np.load(self.index / 'gt_counts.npz', allow_pickle=False) as gt:
                self.gt_ids, self.gt_counts = gt['ids'].copy(), gt['counts'].copy()

    def block_indices(self, center, radius):
        lo = np.maximum(0, np.floor((np.asarray(center) - radius - self.origin) / self.cell_size).astype(int))
        hi = np.minimum(self.dimensions - 1,
                        np.floor((np.asarray(center) + radius - self.origin) / self.cell_size).astype(int))
        pieces = []
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            search = ix * self.dimensions[1] + np.arange(lo[1], hi[1] + 1)
            rows = np.searchsorted(self.keys, search)
            valid = rows < len(self.keys)
            rows, search = rows[valid], search[valid]
            for row in rows[self.keys[rows] == search]:
                pieces.append(self.order[self.starts[row]:self.starts[row] + self.counts[row]])
        if not pieces:
            return np.empty(0, dtype=np.int64)
        indices = np.concatenate(pieces)
        xy = self.arrays['coords'][indices, :2]
        return indices[np.square(xy - center).sum(axis=1) <= radius * radius]

    def inputs(self, indices, center3, radius):
        return np.concatenate([
            self.arrays['features'][indices],
            (self.arrays['coords'][indices] - center3) / radius,
            (self.arrays['votes_xy'][indices] - center3[:2]) / radius,
            self.arrays['tree_probability'][indices, None]], axis=1).astype(np.float32)

    def make_block(self, center, config, seed, training=False):
        rng = np.random.default_rng(seed)
        radius = config['radius']
        indices = self.block_indices(center, radius)
        if not len(indices):
            raise ValueError('Empty spatial block.')
        xyz = self.arrays['coords'][indices]
        center3 = np.array([*center, np.median(xyz[:, 2])], dtype=np.float32)
        # A representative is always a real fine point, so its anchor has a
        # well-defined GT identity. Prefer tree probability within each cell.
        cells = np.floor(xyz / config['token_voxel']).astype(np.int64)
        cells -= cells.min(axis=0)
        dims = cells.max(axis=0) + 1
        keys = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
        order = np.lexsort((indices, -self.arrays['tree_probability'][indices], keys))
        _, first = np.unique(keys[order], return_index=True)
        representatives = indices[order[first]]
        choice = balanced_subset(self.arrays['coords'][representatives], config['max_tokens'], rng)
        tokens = representatives[choice]
        positions = self.arrays['coords'][tokens]
        core = np.square(positions[:, :2] - center).sum(axis=1) <= config['core_radius'] ** 2
        if not core.any():
            # Rare sparse-edge block: do not force a query outside the core.
            raise ValueError('No query anchors in the central region.')
        result = dict(indices=indices, center=center3, token_indices=tokens,
                      tokens=self.inputs(tokens, center3, radius), core=core)
        if training:
            if self.labels is None:
                raise ValueError('Training requires labels.')
            sampled, weights = population_sample(indices, self.labels, config['train_points'], rng)
            result.update(points=self.inputs(sampled, center3, radius),
                          point_labels=self.labels[sampled].astype(np.int64), weights=weights,
                          token_labels=self.labels[tokens].astype(np.int64))
        return result

    def target_sizes(self, targets):
        result = np.zeros(len(targets), dtype=np.float32)
        positive = targets > 0
        if positive.any():
            rows = np.searchsorted(self.gt_ids, targets[positive])
            result[positive] = self.gt_counts[rows]
        return result
