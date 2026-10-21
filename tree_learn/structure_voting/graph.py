"""Coordinate-only, batch-isolated support graphs; no GT is an input."""

import numpy as np
from scipy.spatial import cKDTree


class LocalSupportGraph:
    def __init__(self, xyz, batches, foreground, voxel_size, neighbors, radius):
        self.xyz = np.asarray(xyz, dtype=np.float32)
        self.batches = np.asarray(batches, dtype=np.int64)
        self.neighbors, self.radius = int(neighbors), float(radius)
        self.supports = {}
        for batch_id in np.unique(self.batches):
            rows = np.flatnonzero((self.batches == batch_id) & foreground)
            if not len(rows):
                continue
            # One actual point per occupied cell, with its own feature/label.
            # No cap drops spatial regions. Query memory is bounded separately.
            points = self.xyz[rows]
            cells = np.floor((points - points.min(0)) / voxel_size).astype(np.int64)
            _, first = np.unique(cells, axis=0, return_index=True)
            rows = rows[first]
            self.supports[int(batch_id)] = (rows, cKDTree(self.xyz[rows]))

    def query(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        indices = np.broadcast_to(rows[:, None], (len(rows), self.neighbors)).copy()
        valid = np.zeros_like(indices, dtype=bool)
        for batch_id in np.unique(self.batches[rows]):
            if int(batch_id) not in self.supports:
                continue
            selected = np.flatnonzero(self.batches[rows] == batch_id)
            support, tree = self.supports[int(batch_id)]
            k = min(self.neighbors + 1, len(support))
            distance, neighbor = tree.query(self.xyz[rows[selected]], k=k, workers=1)
            distance, neighbor = distance.reshape(len(selected), k), neighbor.reshape(len(selected), k)
            global_rows = support[neighbor]
            keep = (distance <= self.radius) & (global_rows != rows[selected, None])
            order = np.argsort(np.where(keep, distance, np.inf), axis=1, kind='stable')
            take = min(self.neighbors, k)
            order = order[:, :take]
            indices[selected, :take] = np.take_along_axis(global_rows, order, axis=1)
            valid[selected, :take] = np.take_along_axis(keep, order, axis=1)
        return indices, valid
