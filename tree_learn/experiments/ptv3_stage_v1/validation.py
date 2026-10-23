"""Fixed regional development evaluation; never represents a full-forest score."""
import argparse
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from munch import Munch
from torch.utils.data import DataLoader

from tree_learn.dataset import TreeDataset
from tree_learn.util.data_preparation import load_data
from tree_learn.util import pipeline as main_pipeline
from tree_learn.util.parser import get_config
from tree_learn.experiments.ptv3_treelearn.model import build_model
from tree_learn.experiments.ptv3_treelearn.train import load_config, code_hash
from tree_learn.experiments.ptv3_treelearn.provenance import (
    sha256, canonical_hash, write_json, seed_all, git_state, environment)
from tree_learn.experiments.ptv3_treelearn.pipeline import safe_instances, safe_assign
from tree_learn.experiments.ptv3_treelearn.evaluate import evaluate_arrays
from tree_learn.experiments.ptv3_treelearn.diagnostics import diagnose
from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model

CORE = (-24.0, 24.0, -24.0, 24.0)
DEFAULT_CACHE = "/home/szu-stu/projects/zrx/code/TreeLearn/data/ptv3_treelearn_baseline_20261023/e2_default_L1W"
DEFAULT_MANIFEST = "docs/experiments/ptv3_treelearn/evidence/e2_default_L1W_pipeline.json"


def inside(coords, box=CORE):
    x0, x1, y0, y1 = box
    return ((coords[:, 0] >= x0) & (coords[:, 0] <= x1) &
            (coords[:, 1] >= y0) & (coords[:, 1] <= y1))


def complete_tree_ids(coords, labels, box=CORE):
    positive = labels > 0
    ids, inverse = np.unique(labels[positive], return_inverse=True)
    low = np.full((len(ids), 2), np.inf)
    high = np.full((len(ids), 2), -np.inf)
    for axis in (0, 1):
        np.minimum.at(low[:, axis], inverse, coords[positive, axis])
        np.maximum.at(high[:, axis], inverse, coords[positive, axis])
    x0, x1, y0, y1 = box
    return ids[(low[:, 0] >= x0) & (high[:, 0] <= x1) &
               (low[:, 1] >= y0) & (high[:, 1] <= y1)]


def masked_gt(coords, labels, box=CORE):
    complete = complete_tree_ids(coords, labels, box)
    mask = inside(coords, box)
    target = labels[mask].copy()
    target[(target > 0) & ~np.isin(target, complete)] = -1
    return coords[mask], target, complete


def tile_contributes(center, translation, box=CORE):
    # Main rounds ensemble coordinates to 0.01m. Include the rounding margin.
    x, y = np.asarray(center)[:2] + np.asarray(translation)[:2]
    x0, x1, y0, y1 = box
    return x0-4.02 <= x <= x1+4.02 and y0-4.02 <= y <= y1+4.02


def prepare(args):
    root = Path(args.root).resolve()
    if (root / "protocol.json").exists():
        raise FileExistsError("Refusing to replace a frozen validation protocol")
    root.mkdir(parents=True, exist_ok=True)
    status = root / "stage_status.json"
    write_json(status, dict(status="preparing_validation", pid=os.getpid(), updated_unix=time.time()))
    old = json.loads(Path(DEFAULT_MANIFEST).read_text())
    forest = Path("data/source/pipeline/L1W/forest/L1W.laz").resolve()
    gt_path = Path("data/source/benchmark/L1W_voxelized01_for_eval.laz").resolve()
    if sha256(forest) != old["forest_sha256"]:
        raise ValueError("The cached full-context tiles came from a different L1W forest")
    native = get_config("configs/pipeline/pipeline.yaml")
    for key in ("voxel_size", "search_radius_features", "inner_edge", "outer_edge", "stride"):
        if native.sample_generation[key] != old["pipeline"]["sample_generation"][key]:
            raise ValueError("Changed cached tile geometry/features: " + key)
    translation = old["pointwise_coordinate_translation"]
    tile_dir = root / "tiles" / "npz"
    tile_dir.mkdir(parents=True, exist_ok=False)
    tiles = []
    for path in sorted((Path(DEFAULT_CACHE) / "tiles" / "npz").glob("*.npz")):
        with np.load(path) as data:
            center = data["center"].tolist()
        if tile_contributes(center, translation):
            with np.load(path) as data:
                labels = data["instance_label"]
                if not np.all(labels == -1):
                    raise ValueError("Inference tiles unexpectedly contain supervision")
                points = len(labels)
            (tile_dir / path.name).symlink_to(path)
            tiles.append(dict(path=str(path), sha256=sha256(path), points=points, center=center))
            if len(tiles) % 25 == 0:
                print(f"Verified {len(tiles)} full-context validation tiles", flush=True)
    gt = load_data(str(gt_path))
    coords, labels, complete = masked_gt(gt[:, :3], gt[:, 3].astype(np.int64))
    target = root / "gt_core.npz"
    np.savez(target, points=coords, labels=labels)
    grouping = dict(native.grouping)
    evaluation = dict(get_config("configs/evaluation/evaluate.yaml"))
    protocol = dict(version="ptv3-stage-l1w-v1", role="L1W regional development validation",
        authorized_change="User approved L1W for staged validation and continuation decisions on 2026-10-23",
        reserved_external_domain="Wytham; no stage selection or tuning",
        core_xy=list(CORE), gt_core=str(target), gt_core_sha256=sha256(target),
        gt_source=str(gt_path), gt_source_sha256=sha256(gt_path),
        forest=str(forest), forest_sha256=old["forest_sha256"],
        cache_pipeline_manifest_sha256=sha256(DEFAULT_MANIFEST), translation=translation,
        tiles=tiles, tile_root=str(tile_dir), tile_count=len(tiles),
        complete_tree_ids=complete.tolist(), complete_trees=len(complete), gt_points=len(coords),
        excluded_partial_tree_points=int(((gt[:, 3][inside(gt[:, :3])] > 0) & (labels == -1)).sum()),
        grouping=grouping, evaluation=evaluation, inference_seed=42,
        tile_context="native complete 35m inputs; native 8m inner masks; stride=0.5",
        boundary_policy="GT trees extending outside the fixed 48m core become unknown, never non-tree",
        clustering_scope="union of selected inner tiles, then evaluate the core; regional, not full-forest grouping",
        precision="main FP32 inference; upstream FlashAttention internal QKV precision unchanged",
        checkpoint_nodes=dict(completed_training=2900, proposed_next=5000,
                              warmup_end=12500, optional_later=25000),
        continuation_policy="pause for review after a bounded stage; no automatic 250000-update continuation",
        limitation="24 annotated complete trees in one fixed region; not full L1W or an architecture-fair comparison")
    write_json(root / "protocol.json", protocol)
    write_json(status, dict(status="ready_for_evaluation", pid=os.getpid(),
        protocol_sha256=sha256(root / "protocol.json"), updated_unix=time.time()))
    print(json.dumps({key: protocol[key] for key in ("tile_count", "gt_points", "complete_trees", "excluded_partial_tree_points")}), flush=True)


def evaluate(args):
    root = Path(args.root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("ptv3_stage_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "evaluation.log")):
        logger.addHandler(handler)
    def status(phase, **extra):
        write_json(root / "stage_status.json", dict(status=phase, pid=os.getpid(),
            evaluation_output=str(output), tag=args.tag, updated_unix=time.time(), **extra))
    status("evaluating", operation="loading_and_verifying")
    began = time.perf_counter()
    try:
        if sha256(protocol["gt_core"]) != protocol["gt_core_sha256"]:
            raise ValueError("Changed fixed validation GT")
        for row in protocol["tiles"]:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError("Changed validation tile: " + row["path"])
        cfg = load_config(args.config)
        seed_all(protocol["inference_seed"])
        if cfg["model"].get("backbone") == "ptv3":
            model = build_optimized_model(cfg, "block_nested")
        else:
            model = build_model(cfg)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["net"], strict=True)
        update = checkpoint.get("global_update")
        del checkpoint
        model = model.cuda().eval()
        # Keep evaluation RNG independent of constructor consumption.
        seed_all(protocol["inference_seed"])
        def omit_features(module, inputs, result):
            result["backbone_feats"] = result["backbone_feats"][:, :0]
            return result
        model.register_forward_hook(omit_features)
        dataset = TreeDataset(data_root=protocol["tile_root"], inner_square_edge_length=8,
                              training=False, logger=logger)
        dataset.data_paths = [str(Path(protocol["tile_root"]) / Path(row["path"]).name)
                              for row in protocol["tiles"]]
        loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False,
                            collate_fn=dataset.collate_fn, pin_memory=True,
                            generator=torch.Generator().manual_seed(1234609))
        completed = [0]
        def progress(module, inputs, result):
            completed[0] += 1
            status("evaluating", operation="pointwise", completed_tiles=completed[0], total_tiles=len(dataset))
        model.register_forward_hook(progress)
        torch.cuda.reset_peak_memory_stats()
        pointwise_began = time.perf_counter()
        result = main_pipeline.get_pointwise_preds(model, loader, Munch.fromDict(cfg["model"]), logger)
        logits, sem, offsets, off_gt, coords, inst, features, input_feats = result
        torch.cuda.synchronize()
        pointwise_seconds = time.perf_counter() - pointwise_began
        del model, result, loader
        status("evaluating", operation="ensemble")
        coords, logits, sem, offsets, off_gt, inst, features, input_feats = main_pipeline.ensemble(
            coords, logits, sem, offsets, off_gt, inst, features, input_feats)
        status("evaluating", operation="native_regional_grouping")
        grouping = Munch.fromDict(protocol["grouping"])
        initial = safe_instances(coords, offsets, logits, grouping, input_feats[:, -1])
        predictions = initial.copy()
        tree = predictions != 0
        predictions[tree] = safe_assign(coords[tree] + offsets[tree], predictions[tree])
        pointwise = output / "pointwise_results.npz"
        np.savez(pointwise, coords=coords, semantic_prediction_logits=logits, offset_predictions=offsets,
                 input_feats=input_feats, instance_preds=predictions,
                 instance_preds_after_initial_clustering=initial)
        status("evaluating", operation="matching_and_diagnostics")
        with np.load(protocol["gt_core"]) as gt:
            gt_xyz, gt_labels = gt["points"], gt["labels"]
        translation = np.asarray(protocol["translation"])
        metrics, details, aligned = evaluate_arrays(gt_xyz, gt_labels,
            coords.astype(np.float64) + translation, predictions, Munch.fromDict(protocol["evaluation"]))
        torch.save(details, output / "evaluation_results.pt")
        np.savez(output / "predictions_on_gt.npz", points=gt_xyz, labels=aligned)
        diagnostics = diagnose(pointwise, protocol["gt_core"], translation, output / "diagnostics.json")
        metrics.update(role=protocol["role"], checkpoint_update=update, tag=args.tag,
                       semantic_tree_recall=diagnostics["semantic_tree_recall"],
                       offset_xy_error_m=diagnostics["offset_xy_error_m"],
                       offset_xyz_error_m=diagnostics["offset_xyz_error_m"])
        write_json(output / "metrics.json", metrics)
        write_json(output / "evaluation_manifest.json", dict(
            status="complete", role=protocol["role"], tag=args.tag, checkpoint_update=update,
            protocol_sha256=sha256(root / "protocol.json"), checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_sha256=sha256(args.checkpoint), config_sha256=canonical_hash(cfg),
            code_sha256=code_hash(), git=git_state(), environment=environment(),
            pointwise_sha256=sha256(pointwise), gt_core_sha256=protocol["gt_core_sha256"],
            pointwise_seconds=pointwise_seconds, total_seconds=time.perf_counter()-began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            is_formal_final=False, full_forest=False))
        status("stage_evaluation_complete", metrics=str(output / "metrics.json"), checkpoint_update=update)
        print(json.dumps(metrics, indent=2), flush=True)
    except BaseException:
        (output / "exception.txt").write_text(traceback.format_exc())
        status("evaluation_failed", exception=str(output / "exception.txt"))
        raise


def initialize(args):
    root = Path(args.root).resolve()
    path = root / "checkpoints" / "update_0_reconstructed.pth"
    if path.exists():
        raise FileExistsError(path)
    cfg = load_config(args.config)
    seed_all(42)
    model = build_optimized_model(cfg, "block_nested")
    torch.save(dict(net=model.state_dict(), global_update=0, seed=42,
                    role="reconstructed untrained initialization; not a saved historical checkpoint",
                    config_sha256=canonical_hash(cfg)), path)
    write_json(path.with_suffix(".json"), dict(path=str(path), sha256=sha256(path),
        seed=42, global_update=0, config_sha256=canonical_hash(cfg), git=git_state(),
        note="Same CPU seed_all/build_optimized_model initialization path as the frozen training source"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--root", required=True)
    init.add_argument("--config", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--root", required=True)
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--tag", required=True)
    evaluate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "initialize": initialize(args)
    else: evaluate(args)
