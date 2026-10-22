"""Semantic/offset/seed diagnostics; never used for training or selection."""
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from tree_learn.dataset import TreeDataset
from tree_learn.util.data_preparation import load_data
from .provenance import sha256, write_json


def diagnose(pointwise_path, gt_path, translation, output):
    data = np.load(pointwise_path)
    coords = data["coords"].astype(np.float64) + np.array(translation)
    logits = data["semantic_prediction_logits"]
    offsets = data["offset_predictions"]
    verticality = data["input_feats"][:, -1]
    tree = torch.from_numpy(logits).float().softmax(-1)[:, 0].numpy() >= .5
    seeds = tree & (verticality > .6) & (np.abs(offsets[:, 2]) < 4)
    initial, final = data["instance_preds_after_initial_clustering"], data["instance_preds"]
    gt = load_data(str(gt_path))
    labels = gt[:, 3].astype(int)
    xyz = gt[:, :3]
    nearest = NearestNeighbors(n_neighbors=1, n_jobs=2).fit(coords.astype(np.float32))
    distances, indices = nearest.kneighbors(xyz.astype(np.float32))
    indices = indices[:, 0]
    # Main's base computation on the whole visible annotated GT forest, clearly
    # distinct from training's augmented per-crop anchors. No height relabeling.
    offset_gt, valid = TreeDataset.getOffset(None, xyz, labels, np.where(labels == 0, 1, 0))
    mask = (labels > 0) & valid
    error = (coords[indices] + offsets[indices]) - (xyz + offset_gt)
    xyz_error, xy_error = np.linalg.norm(error, axis=1), np.linalg.norm(error[:, :2], axis=1)
    rows = []
    for label in np.unique(labels[labels > 0]):
        selected = labels == label
        rows.append({"instance": int(label), "points": int(selected.sum()), "z_extent_m": float(np.ptp(xyz[selected, 2])),
                     "semantic_tree_recall": float(tree[indices[selected]].mean()),
                     "seed_fraction_at_gt_points": float(seeds[indices[selected]].mean()),
                     "offset_xyz_error_m": float(xyz_error[selected].mean()), "offset_xy_error_m": float(xy_error[selected].mean()),
                     "overlapping_final_instances": np.unique(final[indices[selected]][final[indices[selected]] > 0]).astype(int).tolist()})
    result = {"semantic_tree_recall": float(tree[indices[mask]].mean()) if mask.any() else None,
              "offset_xy_error_m": float(xy_error[mask].mean()) if mask.any() else None,
              "offset_xyz_error_m": float(xyz_error[mask].mean()) if mask.any() else None,
              "foreground_points": int(tree.sum()), "seed_points": int(seeds.sum()),
              "initial_instances": int(len(np.unique(initial[initial > 0]))),
              "final_instances": int(len(np.unique(final[final > 0]))),
              "unassigned_tree_points": int((final == -1).sum()),
              "gt_alignment_distance_quantiles_m": np.quantile(distances, [0, .5, .95, 1]).tolist(),
              "offset_reference": "main getOffset on whole visible GT; not per-crop training anchors",
              "pointwise_sha256": sha256(pointwise_path), "gt_sha256": sha256(gt_path),
              "per_instance": rows, "role": "evaluation diagnostics only"}
    write_json(output, result)
    return result
