"""Raw coordinate blocks; cached teacher features/predictions are never inputs."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tree_learn.query_segmentation.data import (
    Forest, fingerprint, prepare_index, balanced_subset)


def entries(config, external=False):
    root = Path(config['external_cache_root'] if external else config['cache_root'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for record in manifest['forests']:
        split = 'external' if external else record['split']
        if not external and record['plot'] not in config['splits'].get(split, []):
            continue
        cache = root / record['cache']
        for filename in ('coords.npy', 'cache.json'):
            if not (cache / filename).is_file():
                raise FileNotFoundError(cache / filename)
        rows.append(dict(plot=record['plot'], split=split, cache=str(cache),
                         index=str(Path(config['index_root']) / split / record['plot'])))
    if not rows:
        raise ValueError(f'No coordinate forests found in {root}.')
    if not external:
        actual = {s: sorted(r['plot'] for r in rows if r['split'] == s)
                  for s in ('train', 'validation')}
        if actual != {s: sorted(v) for s, v in config['splits'].items()}:
            raise ValueError('Coordinate-cache manifest does not contain the configured splits.')
    return rows


def neighborhood(xyz, count):
    """Explicit self at column zero, including equal-coordinate ties."""
    count = min(int(count), len(xyz) - 1)
    if count < 1:
        raise ValueError('At least two input voxels are required.')
    _, neighbors = cKDTree(xyz).query(xyz, k=count + 1, workers=1)
    result = np.empty((len(xyz), count + 1), dtype=np.int64)
    result[:, 0] = np.arange(len(xyz))
    # Distinct quantized voxel representatives have distinct coordinates, so
    # self is first. The explicit check also catches corrupted/duplicate input.
    if np.array_equal(neighbors[:, 0], result[:, 0]):
        result[:, 1:] = neighbors[:, 1:]
    else:
        for row in range(len(xyz)):
            result[row, 1:] = neighbors[row][neighbors[row] != row][:count]
    return result


class RawForest:
    block_indices = Forest.block_indices

    def __init__(self, cache, index, with_labels=False):
        self.cache, self.index = Path(cache), Path(index)
        self.metadata = json.loads((self.index / 'index.json').read_text(encoding='utf-8'))
        if self.metadata['request']['source'] != fingerprint(self.cache):
            raise ValueError(f'Stale coordinate index: {index}')
        self.arrays = {'coords': np.load(self.cache / 'coords.npy', mmap_mode='r', allow_pickle=False)}
        self.order = np.load(self.index / 'order.npy', mmap_mode='r', allow_pickle=False)
        with np.load(self.index / 'cells.npz', allow_pickle=False) as cells:
            self.keys, self.starts, self.counts = [cells[k].copy() for k in ('keys', 'starts', 'counts')]
        self.cell_size = self.metadata['request']['cell_size']
        self.origin = np.asarray(self.metadata['origin'])
        self.dimensions = np.asarray(self.metadata['dimensions'])
        self.centers = self.origin + (np.column_stack([
            self.keys // self.dimensions[1], self.keys % self.dimensions[1]]) + .5) * self.cell_size
        self.labels = None
        if with_labels:
            self.labels = np.load(self.cache / 'labels.npy', mmap_mode='r', allow_pickle=False)
            stamp = (self.cache / 'labels.npy').stat()
            expected = dict(size=stamp.st_size, mtime_ns=stamp.st_mtime_ns)
            if json.loads((self.index / 'gt.json').read_text()) != expected:
                raise ValueError('Training labels changed; prepare a fresh index_root.')
            with np.load(self.index / 'gt_counts.npz', allow_pickle=False) as gt:
                self.gt_ids, self.gt_counts = gt['ids'].copy(), gt['counts'].copy()

    def make_block(self, center, config, seed, training=False):
        rng = np.random.default_rng(seed)
        indices = self.block_indices(center, config['radius'])
        if len(indices) < config['minimum_points']:
            return None
        xyz = self.arrays['coords'][indices].astype(np.float32)
        center3 = np.array([*center, np.median(xyz[:, 2])], dtype=np.float32)
        local = xyz - center3
        if training:
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]], dtype=np.float32)
            local[:, :2] = local[:, :2] @ rotation
            if rng.random() < .5:
                local[:, 0] *= -1
            local *= rng.uniform(.9, 1.1)
        grid = np.floor(local / config['voxel_size']).astype(np.int32)
        grid -= grid.min(0)
        _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        total_voxels = len(first)
        if total_voxels < config['minimum_points']:
            return None
        limit = config['max_train_voxels'] if training else config['max_inference_voxels']
        selected = (np.sort(rng.choice(total_voxels, limit, replace=False))
                    if training and total_voxels > limit else
                    balanced_subset(local[first], limit, rng, cell_size=2.))
        representatives = first[selected]
        inputs = local[representatives].copy()
        result = dict(indices=indices, center=center3, xyz=inputs,
                      source_xyz=xyz[representatives].copy(), grid=grid[representatives].copy(),
                      neighbors=neighborhood(inputs, config['neighbors']),
                      core=np.square(xyz[representatives, :2] - center).sum(-1)
                      <= config['core_radius'] ** 2,
                      raw_points=len(indices), available_voxels=total_voxels)
        if training:
            if self.labels is None:
                raise ValueError('Training requires labels.')
            raw_labels = self.labels[indices].astype(np.int64)
            known = raw_labels >= 0
            # Majority of known labels in each voxel; unknown points carry no
            # negative supervision. Coordinates/representatives never use GT.
            pairs, counts = np.unique(np.column_stack((inverse[known], raw_labels[known])),
                                      axis=0, return_counts=True)
            voxel_labels = np.full(total_voxels, -1, dtype=np.int64)
            if len(pairs):
                order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))
                _, best = np.unique(pairs[order, 0], return_index=True)
                voxel_labels[pairs[order[best], 0]] = pairs[order[best], 1]
            known_counts = np.bincount(inverse[known], minlength=total_voxels)
            labels = voxel_labels[selected]
            sizes = np.zeros(len(labels), dtype=np.float32)
            positive = labels > 0
            sizes[positive] = self.gt_counts[np.searchsorted(self.gt_ids, labels[positive])]
            # Uniform voxel subsampling has known inclusion probability.
            weights = known_counts[selected].astype(np.float32) * total_voxels / len(selected)
            result.update(labels=labels, weights=weights, whole_sizes=sizes)
        return result


def prepare_rows(config, rows):
    for row in rows:
        print(f'INDEX {row["split"]}/{row["plot"]}: coordinates only', flush=True)
        prepare_index(row['cache'], row['index'], config['blocks']['stride'],
                      with_labels=row['split'] == 'train')
