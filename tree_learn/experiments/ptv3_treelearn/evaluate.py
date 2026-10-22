"""Main matching/commission rules plus machine-readable metrics and empty guards."""
from pathlib import Path
import time

import numpy as np
import torch

from tree_learn.util.data_preparation import load_data
from tree_learn.util.parser import get_config
from tree_learn.util.pipeline import make_labels_consecutive
from tree_learn.util.eval import get_detections, get_detection_failures, evaluate_instance_segmentation
from .pipeline import safe_propagate
from .provenance import sha256, write_json, git_state
from .train import code_hash


def evaluate_arrays(gt_coords, gt_raw, pred_coords, pred_raw, config):
    pred_raw = safe_propagate(pred_coords, pred_raw, gt_coords, 5)
    gt = np.full(len(gt_raw), -1, dtype=np.int64)
    pred = np.full(len(pred_raw), -1, dtype=np.int64)
    gt[gt_raw > 0], gt_mapping = make_labels_consecutive(gt_raw[gt_raw > 0], 0)
    pred[pred_raw > 0], pred_mapping = make_labels_consecutive(pred_raw[pred_raw > 0], 0)
    gt_mapping[-1] = pred_mapping[-1] = 0
    ng, npred = len(gt_mapping) - 1, len(pred_mapping) - 1
    matched_gt = np.empty(0, dtype=int)
    full = {}
    if ng and npred:
        matched_gt, matched_pred, iou, precision, recall = get_detections(gt, pred, config.thresholds.min_iou_for_match, -1)
        # main's diagnostic undersegmentation helper assumes >=2 GT trees.
        # Its commission filtering is independent of that helper; handle one GT
        # identically below and do not invoke the unsupported diagnostic branch.
        if ng > 1:
            failures = get_detection_failures(matched_gt, matched_pred, np.arange(ng), np.arange(npred),
                iou, precision, recall, config.thresholds.min_precision_for_pred, config.thresholds.min_recall_for_gt)
            unmatched_pred, corresponding = failures[1], failures[2]
            fp = int(np.isfinite(corresponding).sum())
        else:
            unmatched_pred = np.setdiff1d(np.arange(npred), matched_pred)
            fp = int((precision[unmatched_pred].sum(1) >= config.thresholds.min_precision_for_pred).sum())
        best = iou.argmax(0)
        # The normal path invokes the exact main segmentation evaluator.
        enough_partition_points = all((gt == label).sum() >= 5 for label in range(ng))
        partitions = config.partitions if enough_partition_points else None
        segmentation = evaluate_instance_segmentation(pred, gt, np.arange(ng), best, gt_coords,
            gt_mapping, pred_mapping, partitions.xy_partition if partitions else None,
            partitions.z_partition if partitions else None)
        full = {"matched_gts": matched_gt, "matched_preds": matched_pred, "iou_matrix": iou,
                "precision_matrix": precision, "recall_matrix": recall,
                "no_partition": segmentation[0], "xy_partition": segmentation[1], "z_partition": segmentation[2]}
        coverage = float(iou.max(0).mean())
        segmentation_precision = float(segmentation[0]["prec"].mean())
        segmentation_recall = float(segmentation[0]["rec"].mean())
    else:
        # With no annotated trees, main's commission test excludes predictions.
        fp = 0
        coverage = 0. if ng else None
        segmentation_precision = segmentation_recall = 0. if ng else None
    tp, fn = len(matched_gt), ng - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / ng if ng else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    bins = {}
    for low, high in ((0, 10), (10, 20), (20, float("inf"))):
        members = [label for label in range(ng) if low <= np.ptp(gt_coords[gt == label, 2]) < high]
        hits = len(set(members) & set(matched_gt.tolist()))
        bins[f"{low}_{high}"] = {"gt": len(members), "tp": hits, "fn": len(members)-hits,
                                     "recall": hits / len(members) if members else None}
    metrics = {"TP": tp, "FP": fp, "FN": fn, "detection_precision": precision,
               "metric_units": "fractions 0..1",
               "detection_recall": recall, "detection_f1": f1, "completeness": recall,
               "omission": 1-recall if recall is not None else None,
               "commission": 1-precision if tp+fp else None, "coverage": coverage,
               "segmentation_precision": segmentation_precision, "segmentation_recall": segmentation_recall,
               "z_extent_bins_m": bins, "gt_trees": ng, "predicted_trees": npred,
               "unknown_prediction_points": int((pred_raw == -1).sum()),
               "rules": "main strict IoU>0.5 Hungarian, unmatched summed precision>=0.5 commission, 5NN alignment",
               "scale_name": "Z Extent; not ground-normalized tree height"}
    return metrics, full, pred_raw


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    source_code_hash, source_git = code_hash(), git_state()
    config = get_config("configs/evaluation/evaluate.yaml")
    gt, pred = load_data(str(args.gt)), load_data(str(args.prediction))
    metrics, full, aligned = evaluate_arrays(gt[:, :3], gt[:, 3].astype(int), pred[:, :3], pred[:, 3].astype(int), config)
    torch.save(full, output / "evaluation_results.pt")
    np.savez_compressed(output / "predictions_on_gt.npz", points=gt[:, :3], labels=aligned)
    write_json(output / "metrics.json", metrics)
    write_json(output / "evaluation_manifest.json", {"prediction": str(Path(args.prediction).resolve()),
               "prediction_sha256": sha256(args.prediction), "gt": str(Path(args.gt).resolve()), "gt_sha256": sha256(args.gt),
               "config": dict(config), "git": source_git, "code_sha256": source_code_hash, "seconds": time.perf_counter()-began,
               "role": "external development evaluation; never training or checkpoint selection"})
