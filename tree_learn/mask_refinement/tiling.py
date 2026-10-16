"""Stream the fixed A1 baseline's tiles without retaining overlapping chunks.

Grid rounding, boundaries, CUDA subtraction, tile order and output dtypes follow
SampleGenerator.tile_generate_and_save. This implementation deliberately supports
only the unrotated, unfiltered configuration used by the frozen A1 baseline.
"""

import json
from pathlib import Path

import numpy as np
import torch


def tile_generate_and_save_streaming(self, inner_edge, outer_edge, stride,
                                     compressed=False, plot_corners=None, logger=None):
    if plot_corners is not None or any(value is not None for value in (
            self.n_neigh_sor, self.multiplier_sor, self.rad, self.npoints_rad)):
        raise ValueError('Streaming A1 tiling requires no plot rotation or denoising.')
    logger.info('streaming tiles: retaining one overlapping chunk at a time')
    xmin = np.round(self.x_range[0] - 1.5 * outer_edge, 2)
    xmax = np.round(self.x_range[1] + 1.5 * outer_edge, 2)
    ymin = np.round(self.y_range[0] - 1.5 * outer_edge, 2)
    ymax = np.round(self.y_range[1] + 1.5 * outer_edge, 2)

    ncols = int(np.round((xmax - xmin - 2 * outer_edge) / inner_edge))
    inner_edge_x = np.round((xmax - xmin - 2 * outer_edge) / ncols, 5)
    ncols = int((ncols - 1) / stride + 1)
    nrows = int(np.round((ymax - ymin - 2 * outer_edge) / inner_edge))
    inner_edge_y = np.round((ymax - ymin - 2 * outer_edge) / nrows, 5)
    nrows = int((nrows - 1) / stride + 1)
    inner = np.empty((nrows * ncols, 4))
    for i in range(nrows):
        for j in range(ncols):
            inner[i * ncols + j] = np.array([
                xmin + outer_edge + stride * j * inner_edge_x,
                xmin + outer_edge + (stride * j + 1) * inner_edge_x,
                ymax - outer_edge - (stride * i + 1) * inner_edge_y,
                ymax - outer_edge - stride * i * inner_edge_y])
    inner = np.round(inner, 5)
    outer = inner + np.array([-outer_edge, outer_edge, -outer_edge, outer_edge]).reshape(1, 4)
    host_points = np.hstack((self.points, self.label.reshape(-1, 1)))
    host_points = np.hstack([host_points, self.feats])
    points = torch.from_numpy(host_points).cuda()
    del host_points
    outer = torch.from_numpy(outer).cuda()
    x_points, y_points = points[:, 0], points[:, 1]
    count = 0

    for index, (xmin_outer, xmax_outer, ymin_outer, ymax_outer) in enumerate(outer):
        outer_mask = ((x_points >= xmin_outer) & (x_points <= xmax_outer) &
                      (y_points >= ymin_outer) & (y_points <= ymax_outer))
        chunk = points[outer_mask].cpu().numpy()
        del outer_mask
        inner_mask = ((chunk[:, 0] >= inner[index][0]) & (chunk[:, 0] < inner[index][1]) &
                      (chunk[:, 1] > inner[index][2]) & (chunk[:, 1] <= inner[index][3]))
        if not inner_mask.any():
            del chunk, inner_mask
            continue

        # The original casts valid bounds to float32 before computing centers.
        bounds = inner[index].astype(np.float32)
        center_x = np.round((bounds[0] + bounds[1]) / 2, 6)
        center_y = np.round((bounds[2] + bounds[3]) / 2, 6)
        subtraction = np.concatenate([
            np.array([center_x, center_y, 0, 0]), np.zeros(self.feats.shape[1])]).reshape(1, -1)
        chunk = (torch.from_numpy(chunk).cuda() - torch.from_numpy(subtraction).cuda()).cpu().numpy()
        chunk = chunk.astype(np.float32)
        metadata = dict(plot_name=self.plot_name, n_neigh_sor=self.n_neigh_sor,
                        multiplier_sor=self.multiplier_sor, rad=self.rad,
                        npoints_rad=self.npoints_rad, inner_edge=inner_edge, outer_edge=outer_edge)
        data = dict(points=chunk[:, :3], feat=chunk[:, 4:],
                    instance_label=chunk[:, 3].astype(np.int32), center=np.array([center_x, center_y, 0]))
        path = Path(self.save_dir_data) / f'{self.plot_name}_{count}.npz'
        save = np.savez_compressed if compressed else np.savez
        save(path, **data)
        with (Path(self.save_dir_meta_data) / f'{self.plot_name}_{count}.json').open('w') as stream:
            json.dump(metadata, stream)
        count += 1
        # Drop all views before selecting the next chunk; do not accumulate data.
        del data, chunk, inner_mask
        if count == 1 or count % 25 == 0:
            logger.info('streaming tiles: saved %d, visited %d/%d grid cells', count, index + 1, len(inner))
    logger.info('streaming tiles complete: saved %d tiles', count)
