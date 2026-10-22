"""Run main's entire forest pipeline with the selected backbone and edge guards."""
from pathlib import Path
import resource
import time

import numpy as np
import torch
from munch import Munch

from tree_learn.util import pipeline as original
from tree_learn.util.parser import get_config
from .model import build_model
from .train import load_config, code_hash
from .provenance import sha256, write_json, load_initialization, git_state, environment, seed_all, canonical_hash


def safe_instances(coords, offset, logits, grouping_cfg, verticality_feat, tree_class_in_dataset=0,
                   non_trees_label_in_grouping=0, not_assigned_label_in_grouping=-1, start_num_preds=1):
    tree = torch.from_numpy(logits).float().softmax(-1)[:, tree_class_in_dataset].numpy() >= grouping_cfg.tree_conf_thresh
    seeds = tree & (verticality_feat > grouping_cfg.tau_vert) & (np.abs(offset[:, 2]) < grouping_cfg.tau_off)
    if seeds.sum() < (grouping_cfg.tau_min if grouping_cfg.use_hdbscan else 2):
        return np.where(tree, not_assigned_label_in_grouping, non_trees_label_in_grouping).astype(np.int64)
    return original.get_instances(coords, offset, logits, grouping_cfg, verticality_feat,
                    tree_class_in_dataset, non_trees_label_in_grouping, not_assigned_label_in_grouping, start_num_preds)


def safe_assign(coords, predictions, remaining_points_idx=-1, n_neighbors=5):
    query = predictions == remaining_points_idx
    reference_count = int((~query).sum())
    if not query.any() or reference_count == 0:
        return predictions.copy()
    return original.assign_remaining_points_nearest_neighbor(coords, predictions, remaining_points_idx,
                                                            min(n_neighbors, reference_count))


def safe_propagate(source_coords, source_preds, target_coords, n_neighbors=5, n_jobs=1):
    if not len(target_coords):
        return np.empty(0, dtype=np.int64)
    if not len(source_coords):
        return np.full(len(target_coords), -1, dtype=np.int64)
    return original.propagate_preds(source_coords, source_preds, target_coords,
                                    min(n_neighbors, len(source_coords)), n_jobs)


def safe_save_data(data, save_format, save_name, save_folder, use_offset=True):
    # main's LAS treeID is uint32, so -1 would become a huge positive instance.
    # Preserve unknown through treeID=0/classification=0, as load_data expects.
    if save_format in ("las", "laz") and (not len(data) or np.any(data[:, 3] < 0)):
        import laspy
        if not len(data):
            header = laspy.LasHeader(version="1.2", point_format=3)
            las = laspy.LasData(header)
            las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint32))
        else:
            safe = data.copy(); safe[safe[:, 3] < 0, 3] = 0
            original.save_data(safe, save_format, save_name, save_folder, use_offset)
            las = laspy.read(Path(save_folder) / (save_name + "." + save_format))
            las.classification[data[:, 3] < 0] = 0
        las.write(Path(save_folder) / (save_name + "." + save_format))
        return
    original.save_data(data, save_format, save_name, save_folder, use_offset)


def run(args):
    from tools.pipeline import pipeline as entry
    config = load_config(args.config)
    seed_all(args.seed)
    source_code_hash, source_git = code_hash(), git_state()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pipeline_preflight.json", {"code_sha256": source_code_hash, "git": source_git,
               "config_sha256": canonical_hash(config), "checkpoint_sha256": sha256(args.checkpoint),
               "forest": str(Path(args.forest).resolve()), "seed": args.seed,
               "compact_pointwise": args.compact_pointwise, "stream_tiles": args.stream_tiles})
    source = Path(args.forest).resolve()
    # Main centers and writes a sibling NPZ. Stage a symlink in the new run
    # directory so these writes can never touch the original forest/caches.
    forest_dir = output / "forest"
    forest_dir.mkdir()
    staged = forest_dir / source.name
    if source.suffix == ".npz":
        # Main's .npz -> .npy rename is followed by save_npz, producing a
        # mismatched .npy.npz path. Stage a lossless NPY input to avoid it.
        staged = staged.with_suffix(".npy")
        np.save(staged, original.load_data(str(source)))
    else:
        staged.symlink_to(source)
    cfg = get_config("configs/pipeline/pipeline.yaml")
    cfg.forest_path = str(staged)
    cfg.pretrain = args.checkpoint
    cfg.model = Munch.fromDict(config["model"])
    cfg.save_cfg.results_dir = "results"
    # Record the exact translation performed by main, for pointwise diagnostics.
    coordinate_translation = []
    load_data_original = original.load_data
    def audited_load_data(path):
        data = load_data_original(path)
        if not coordinate_translation:
            coordinate_translation.extend(np.mean(data[:, :3].astype(np.float64), 0).tolist())
        return data
    entry.load_data = audited_load_data
    # The original entry has a config-copy bug referencing global args; passing
    # no config_path avoids it. The resolved manifest is saved independently.
    def model_factory(**kwargs):
        model = build_model(config)
        if args.compact_pointwise:
            def omit_saved_features(module, inputs, result):
                result["backbone_feats"] = result["backbone_feats"][:, :0]
                return result
            model.register_forward_hook(omit_saved_features)
        return model
    entry.TreeLearn = model_factory
    entry.load_checkpoint = lambda checkpoint, logger, model: load_initialization(
        model, checkpoint, output / "initialization.json", "official")
    entry.get_pointwise_preds = original.get_pointwise_preds
    entry.save_data = safe_save_data
    entry.get_instances = safe_instances
    def assign_and_guard(*pos, **kw):
        preds = safe_assign(*pos, **kw)
        if not np.any(preds > 0):
            cfg.save_cfg.save_treewise = False
        return preds
    entry.assign_remaining_points_nearest_neighbor = assign_and_guard
    entry.propagate_preds = safe_propagate
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    previous_generator = original.SampleGenerator
    if args.stream_tiles:
        from .tiles import StreamingSampleGenerator
        original.SampleGenerator = StreamingSampleGenerator
    try:
        entry.run_treelearn_pipeline(cfg)
    finally:
        original.SampleGenerator = previous_generator
    torch.cuda.synchronize()
    write_json(output / "pipeline_manifest.json", {"config": config, "pipeline": dict(cfg),
              "git": source_git, "code_sha256": source_code_hash, "config_sha256": canonical_hash(config),
              "environment": environment(), "seed": args.seed, "forest": str(source),
              "forest_sha256": sha256(source), "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": sha256(args.checkpoint), "seconds": time.perf_counter() - began,
              "pointwise_coordinate_translation": coordinate_translation,
              "saved_backbone_features": not args.compact_pointwise,
              "streamed_tiles": args.stream_tiles,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "peak_host_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "normal_path": "main ensemble/thresholds/HDBSCAN/3D shifted 5NN/hash propagation",
              "adaptations": ["isolated forest staging", "strict checkpoint loading",
                              "empty/insufficient seeds, clusters and KNN references guarded"]})
