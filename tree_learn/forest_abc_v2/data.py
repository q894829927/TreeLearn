"""Coordinate-only prediction; full-tree center targets are training-only."""

import json
from pathlib import Path

import numpy as np

from tree_learn.forest_abc.data import RawForest, neighborhood
from tree_learn.mask_refinement.data import write_json


def center_request(row):
    cache = Path(row['cache'])
    return {name: dict(size=(cache / f'{name}.npy').stat().st_size,
                       mtime_ns=(cache / f'{name}.npy').stat().st_mtime_ns)
            for name in ('coords', 'labels')}


def prepare_centers(row):
    """Accumulate full-tree XY centroids in bounded chunks, never on val/test."""
    if row['split'] != 'train':
        raise ValueError('Center supervision is only prepared for training forests.')
    folder = Path(row['index'])
    marker, target = folder / 'v2_centers.json', folder / 'v2_centers.npz'
    request = center_request(row)
    if marker.exists():
        if json.loads(marker.read_text()) != request or not target.is_file():
            raise ValueError(f'Stale training centers: {folder}')
        return
    with np.load(folder / 'gt_counts.npz', allow_pickle=False) as saved:
        ids, expected = saved['ids'].copy(), saved['counts'].copy()
    coords = np.load(Path(row['cache']) / 'coords.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(Path(row['cache']) / 'labels.npy', mmap_mode='r', allow_pickle=False)
    counts = np.zeros(len(ids), dtype=np.int64)
    sums = np.zeros((len(ids), 2), dtype=np.float64)
    print(f'CENTER_SCAN_BEGIN {row["plot"]}: points={len(labels)}', flush=True)
    try:
        for start in range(0, len(labels), 500_000):
            local = labels[start:start + 500_000]
            positive = local > 0
            columns = np.searchsorted(ids, local[positive])
            if np.any(columns >= len(ids)) or not np.array_equal(ids[columns], local[positive]):
                raise ValueError('Training GT IDs changed since index preparation.')
            points = coords[start:start + len(local), :2][positive]
            counts += np.bincount(columns, minlength=len(ids))
            for axis in range(2):
                sums[:, axis] += np.bincount(columns, weights=points[:, axis], minlength=len(ids))
            if (start // 500_000 + 1) % 10 == 0 or start + len(local) == len(labels):
                print(f'CENTER_SCAN {row["plot"]}: {start + len(local)}/{len(labels)}', flush=True)
        if not np.array_equal(counts, expected) or request != center_request(row):
            raise ValueError('Training center inputs/counts changed during preparation.')
        centers = (sums / counts[:, None]).astype(np.float32)
        if not np.isfinite(centers).all():
            raise ValueError('Non-finite training centers.')
        temporary = folder / 'v2_centers.tmp.npz'
        np.savez(temporary, ids=ids, centers=centers)
        temporary.replace(target)
        write_json(marker, request)
    finally:
        coords._mmap.close()
        labels._mmap.close()
    print(f'CENTERS {row["plot"]}: {len(ids)} full training trees', flush=True)


class CenterForest(RawForest):
    def __init__(self, cache, index, with_labels=False):
        super().__init__(cache, index, with_labels=with_labels)
        if with_labels:
            row = dict(cache=str(cache))
            if json.loads((self.index / 'v2_centers.json').read_text()) != center_request(row):
                raise ValueError('Stale center targets; run V2 prepare.')
            with np.load(self.index / 'v2_centers.npz', allow_pickle=False) as saved:
                if not np.array_equal(saved['ids'], self.gt_ids):
                    raise ValueError('Center IDs do not match training labels.')
                self.gt_centers = saved['centers'].copy()
            if self.gt_centers.shape != (len(self.gt_ids), 2) or not np.isfinite(self.gt_centers).all():
                raise ValueError('Invalid full-tree center targets.')

    def make_block(self, center, config, seed, training=False):
        if not training:
            return super().make_block(center, config, seed, training=False)
        if self.labels is None:
            raise ValueError('Training requires training-only center targets.')
        rng = np.random.default_rng(seed)
        indices = self.block_indices(center, config['radius'])
        if len(indices) < config['minimum_points']:
            return None
        xyz = self.arrays['coords'][indices].astype(np.float32)
        center3 = np.array([*center, np.median(xyz[:, 2])], dtype=np.float32)
        local = xyz - center3
        angle = rng.uniform(-np.pi, np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]], dtype=np.float32)
        if rng.random() < .5:
            rotation[:, 0] *= -1
        scale = np.float32(rng.uniform(config['scale_min'], config['scale_max']))
        local[:, :2] = local[:, :2] @ rotation
        local *= scale
        grid = np.floor(local / config['voxel_size']).astype(np.int32)
        grid -= grid.min(0)
        _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        total = len(first)
        if total < config['minimum_points']:
            return None
        density = rng.uniform(config['density_min'], 1.)
        limit = max(config['minimum_points'], min(config['max_train_voxels'], int(total * density)))
        selected = (np.sort(rng.choice(total, limit, replace=False)) if total > limit
                    else np.arange(total))
        representatives = first[selected]
        inputs = local[representatives].copy()
        raw_labels = self.labels[indices].astype(np.int64)
        known = raw_labels >= 0
        pairs, counts = np.unique(np.column_stack((inverse[known], raw_labels[known])),
                                  axis=0, return_counts=True)
        voxel_labels = np.full(total, -1, dtype=np.int64)
        if len(pairs):
            order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))
            _, first_label = np.unique(pairs[order, 0], return_index=True)
            voxel_labels[pairs[order[first_label], 0]] = pairs[order[first_label], 1]
        labels = voxel_labels[selected]
        positive = labels > 0
        columns = np.searchsorted(self.gt_ids, labels[positive])
        offsets = np.zeros((len(labels), 2), dtype=np.float32)
        target_centers = (self.gt_centers[columns] - center3[:2]) @ rotation * scale
        offsets[positive] = target_centers - inputs[positive, :2]
        sizes = np.zeros(len(labels), dtype=np.float32)
        sizes[positive] = self.gt_counts[columns]
        weights = np.bincount(inverse[known], minlength=total)[selected].astype(np.float32)
        weights *= np.float32(total / len(selected))
        return dict(indices=indices, center=center3, xyz=inputs,
                    source_xyz=xyz[representatives].copy(), grid=grid[representatives].copy(),
                    neighbors=neighborhood(inputs, config['neighbors']),
                    core=np.square(xyz[representatives, :2] - center).sum(-1) <= config['core_radius'] ** 2,
                    raw_points=len(indices), available_voxels=total, labels=labels,
                    weights=weights, whole_sizes=sizes, center_offsets=offsets,
                    augmentation_scale=float(scale), augmentation_density=float(density))
