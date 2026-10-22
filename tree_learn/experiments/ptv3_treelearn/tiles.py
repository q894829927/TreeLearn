"""Bounded-memory execution of main's exact default tile geometry and transforms."""
import json
import os

import numpy as np
import torch

from tree_learn.util.data_preparation import SampleGenerator, sor_filter, rad_filter


class StreamingSampleGenerator(SampleGenerator):
    def tile_generate_and_save(self, inner_edge, outer_edge, stride, compressed=False, plot_corners=None, logger=None):
        if plot_corners is not None:
            return super().tile_generate_and_save(inner_edge, outer_edge, stride, compressed, plot_corners, logger)
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
                inner[i*ncols+j] = [xmin + outer_edge + stride*j*inner_edge_x,
                    xmin + outer_edge + (stride*j+1)*inner_edge_x,
                    ymax - outer_edge - (stride*i+1)*inner_edge_y,
                    ymax - outer_edge - stride*i*inner_edge_y]
        inner = np.round(inner, 5)
        outer = inner + np.array([-outer_edge, outer_edge, -outer_edge, outer_edge]).reshape(1, 4)
        points = np.hstack([self.points, self.label.reshape(-1, 1), self.feats])
        points = torch.from_numpy(points).cuda()
        outer = torch.from_numpy(outer).cuda()
        count = 0
        logger.info("Streaming main tiles: inner=%s outer=%s stride=%s", inner_edge, outer_edge, stride)
        for row, (x0, x1, y0, y1) in zip(inner, outer):
            selected = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
            chunk = points[selected].cpu().numpy()
            inside = (chunk[:, 0] >= row[0]) & (chunk[:, 0] < row[1]) & (chunk[:, 1] > row[2]) & (chunk[:, 1] <= row[3])
            if not inside.any():
                continue
            # main casts the selected inner rectangle to float32 before computing
            # its center, then subtracts a float64 center on CUDA. Preserve both.
            row = row.astype(np.float32)
            cx = np.round((row[0] + row[1])/2, 6)
            cy = np.round((row[2] + row[3])/2, 6)
            center = np.concatenate([np.array([cx, cy, 0, 0]), np.zeros(self.feats.shape[1])]).reshape(1, -1)
            chunk = (torch.from_numpy(chunk).cuda() - torch.from_numpy(center).cuda()).cpu().numpy()
            if self.n_neigh_sor is not None and self.multiplier_sor is not None:
                chunk = chunk[sor_filter(chunk, self.n_neigh_sor, self.multiplier_sor)]
            if self.rad is not None and self.npoints_rad is not None:
                chunk = chunk[rad_filter(chunk, self.rad, self.npoints_rad)]
            chunk = chunk.astype(np.float32)
            data = dict(points=chunk[:, :3], feat=chunk[:, 4:], instance_label=chunk[:, 3].astype(np.int32),
                        center=np.array([cx, cy, 0]))
            metadata = dict(plot_name=self.plot_name, n_neigh_sor=self.n_neigh_sor, multiplier_sor=self.multiplier_sor,
                            rad=self.rad, npoints_rad=self.npoints_rad, inner_edge=inner_edge, outer_edge=outer_edge)
            stem = self.plot_name + "_" + str(count)
            writer = np.savez_compressed if compressed else np.savez
            writer(os.path.join(self.save_dir_data, stem + ".npz"), **data)
            with open(os.path.join(self.save_dir_meta_data, stem + ".json"), "w") as stream:
                json.dump(metadata, stream)
            count += 1
            if count % 100 == 0:
                logger.info("Saved %d identical-geometry tiles", count)
        logger.info("Saved %d tiles", count)
