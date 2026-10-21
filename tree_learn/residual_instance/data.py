"""Deterministic paired crops from existing aligned 0.1 m coordinate caches."""

from collections import OrderedDict
from pathlib import Path
import hashlib

import numpy as np
import torch

from tree_learn.forest_abc.data import RawForest


def legacy_base_offsets(xyz, labels):
    """Original TreeDataset legacy target, computed before point subsampling."""
    offsets = np.zeros_like(xyz, dtype=np.float32)
    valid = np.zeros(len(xyz), dtype=bool)
    positive = np.flatnonzero(labels > 0)
    order = positive[np.argsort(labels[positive], kind='stable')]
    starts = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(order)]
    for begin, end in zip(starts[:-1], starts[1:]):
        rows = order[begin:end]
        if not len(rows):
            continue
        points = xyz[rows]
        low = np.partition(points[:, 2], 10)[3] if len(points) > 11 else points[:, 2].min()
        base = points[points[:, 2] <= low + .5].mean(0)
        offsets[rows] = base - points
        valid[rows] = True
    return offsets, valid


class PairedCropSampler:
    """Crop k is a pure function of seed and step, independent of model RNG."""

    def __init__(self, rows, settings, seed):
        self.rows, self.settings, self.seed = rows, settings, seed
        self.forests = OrderedDict()

    def draw(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        settings = self.settings
        for _ in range(100):
            row = self.rows[int(rng.integers(len(self.rows)))]
            plot = row['plot']
            if plot not in self.forests:
                while len(self.forests) >= settings['max_open_forests']:
                    self.forests.popitem(last=False)
                self.forests[plot] = RawForest(row['cache'], row['index'], with_labels=True)
            forest = self.forests[plot]
            self.forests.move_to_end(plot)
            center = forest.centers[int(rng.integers(len(forest.centers)))].copy()
            center += rng.uniform(-4., 4., size=2)
            radius = settings['outer_half_extent']
            indices = forest.block_indices(center, radius * 1.6)
            if len(indices) < settings['minimum_points']:
                continue
            xyz = forest.arrays['coords'][indices].astype(np.float32)
            xyz[:, :2] -= center.astype(np.float32)
            # Mild original-style augmentation; no aggressive scale/density shift.
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]], dtype=np.float32)
            xyz[:, :2] = xyz[:, :2] @ rotation
            xyz[:, 0] *= rng.choice(np.array([-1., 1.], dtype=np.float32))
            xyz *= np.float32(rng.uniform(.95, 1.05))
            keep = np.max(np.abs(xyz[:, :2]), axis=1) <= radius
            xyz, indices = xyz[keep], indices[keep]
            labels = np.asarray(forest.labels[indices], dtype=np.int64)
            if len(xyz) < settings['minimum_points']:
                continue
            offsets, valid_offsets = legacy_base_offsets(xyz, labels)
            if len(xyz) > settings['max_points']:
                selected = np.sort(rng.choice(len(xyz), settings['max_points'], replace=False))
                xyz, indices, labels = xyz[selected], indices[selected], labels[selected]
                offsets, valid_offsets = offsets[selected], valid_offsets[selected]
            inner = np.max(np.abs(xyz[:, :2]), axis=1) <= settings['inner_half_extent']
            known = inner & (labels >= 0)
            off = inner & valid_offsets
            if np.count_nonzero(off) < 16 or not known.any():
                continue
            semantic = np.where(labels == 0, 1, 0).astype(np.int64)
            batch = dict(coords=torch.from_numpy(xyz.copy()),
                input_feats=torch.ones((len(xyz), 1), dtype=torch.float32),
                batch_ids=torch.zeros(len(xyz), dtype=torch.long), batch_size=1,
                instance_labels=torch.from_numpy(labels.copy()),
                semantic_labels=torch.from_numpy(semantic),
                offset_labels=torch.from_numpy(offsets.copy()),
                masks_inner=torch.from_numpy(inner), masks_sem=torch.from_numpy(known),
                masks_off=torch.from_numpy(off))
            plan = hashlib.sha256(plot.encode() + indices.tobytes() + xyz.tobytes()).hexdigest()
            return batch, dict(plot=plot, points=len(xyz), crop_sha256=plan)
        raise RuntimeError('100 crops lacked valid supervised tree points.')


def tile_batch(path):
    """Production inference inputs only; never read tile instance_label."""
    with np.load(Path(path), allow_pickle=False) as tile:
        xyz = np.asarray(tile['points'], dtype=np.float32)
        features = np.asarray(tile['feat'], dtype=np.float32)
    return dict(coords=torch.from_numpy(xyz), input_feats=torch.from_numpy(features),
                batch_ids=torch.zeros(len(xyz), dtype=torch.long), batch_size=1)
