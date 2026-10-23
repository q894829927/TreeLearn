"""Versioned formal integration; the reference training implementation is unchanged."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from tree_learn.experiments.ptv3_treelearn import train as reference
from tree_learn.experiments.ptv3_treelearn.provenance import environment, git_state, sha256, write_json
from . import build_optimized_model
from .profile import memory
from .run import config as reference_config
from .stability import is_cuda_oom

ALLOCATOR = "max_split_size_mb:128,garbage_collection_threshold:0.7"
CONFIG = "configs/experiments/ptv3_memory_v1/e1_full_formal.yaml"
TOTAL_UPDATES = 250000


def formal_config():
    cfg = reference_config()
    cfg["experiment"].update(
        variant="full-crop-micro1-accum2-block-nested-allocator-v1",
        memory_adapter="block_nested", allocator_config=ALLOCATOR,
        allocator_memory_fraction=1.0, checkpoint_updates=50)
    return cfg


def validate_config(cfg):
    if cfg != formal_config():
        raise ValueError("Formal v1 requires the frozen full-crop protocol and original sample plan")


def checkpoint_target(path):
    path = Path(path)
    # Bound storage: one rolling recovery file, plus the original epoch saves.
    return path.with_name("recovery.pth") if path.stem.startswith("update_") else path


def run(args):
    cfg = reference.load_config(args.config)
    validate_config(cfg)
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != ALLOCATOR:
        raise ValueError("Use the isolated CLI to configure the allocator before torch import")
    torch.cuda.set_per_process_memory_fraction(1.0)
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise ValueError("This integration was validated with the native allocator")
    output = Path(args.work_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    # Keep launch metadata beside the run until reference.run creates its empty
    # directory. Its original non-overwrite guard remains fully effective.
    control = output.with_name(output.name + "_control")
    control.mkdir(parents=True, exist_ok=bool(args.resume))
    manifest = json.loads(Path(cfg["experiment"]["crop_manifest"]).read_text())
    crop_paths = {row["path"] for row in manifest["crops"]}
    state = dict(status="verifying_data", phase="formal", seed=args.seed,
                 target_updates=TOTAL_UPDATES, completed_updates=0,
                 verified_crops=0, total_crops=len(crop_paths), pid=os.getpid(),
                 started_unix=time.time(), resumed_from=args.resume)
    write_json(control / "run_status.json", state)
    sources = [Path(__file__), Path(__file__).with_name("adapters.py"),
               Path(reference.__file__), Path("tools/experiments/ptv3_memory_train_v1.py")]
    launch_manifest = dict(
        authorization="User explicitly requested direct long training on 2026-10-23",
        phase="formal", seed=args.seed, target_updates=TOTAL_UPDATES,
        cwd=str(Path.cwd()), command=sys.argv, git=git_state(), environment=environment(),
        config_file_sha256=sha256(args.config), allocator_config=ALLOCATOR,
        allocator_memory_fraction=1.0, variant="block_nested", maximum_points=None,
        checkpoint_policy="rolling recovery every 50 updates; last every epoch; original epoch snapshots",
        training_loop="unchanged reference.run; scoped model factory and scalar telemetry",
        data_integrity="unchanged eager SHA256 verification of every planned crop before training",
        source_sha256={str(path): sha256(path) for path in sources})
    write_json(control / "memory_training_manifest.json", launch_manifest)
    originals = reference.build_model, reference.write_json, reference.save_checkpoint, reference.sha256
    model_ref, handles, voxels = [], [], []
    verified = set()
    last_saved = [None]
    skipped = [0]

    def status(**updates):
        state.update(updates, updated_unix=time.time())
        write_json(control / "run_status.json", state)
        if output.exists():
            write_json(output / "run_status.json", state)

    def checked_hash(path):
        value = originals[3](path)
        if str(path) in crop_paths:
            verified.add(str(path))
            if len(verified) % 100 == 0 or len(verified) == len(crop_paths):
                status(verified_crops=len(verified))
            if len(verified) % 1000 == 0:
                print(f"Data SHA256 verified {len(verified)}/{len(crop_paths)} crops", flush=True)
        return value

    def builder(options):
        status(status="building_model", verified_crops=len(verified))
        write_json(output / "memory_training_manifest.json", launch_manifest)
        model = build_optimized_model(options, "block_nested")
        model_ref.append(model)
        def count_voxels(module, inputs):
            voxels.append(int(inputs[0]["feat"].shape[0]))
        handles.append(model.backbone.register_forward_pre_hook(count_voxels))
        return model

    def writer(path, value):
        originals[1](path, value)
        if Path(path).name != "monitoring.json":
            return
        if len(voxels) != 2 or any(row["points_before"] != row["points_after"]
                                   for row in value["sampling_audit"]):
            raise RuntimeError("Full-input / microbatch invariant failed")
        counts = {name: int(module.num_batches_tracked) for name, module in model_ref[0].named_modules()
                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        if not counts or any(count != value["update"] * 2 for count in counts.values()):
            raise RuntimeError("BatchNorm was not updated exactly once per microbatch")
        skipped[0] += int(value["optimizer_step_skipped"])
        row = dict(update=value["update"], voxels=list(voxels), bn_counts=counts, memory=memory(),
                   optimizer_step_skipped=value["optimizer_step_skipped"])
        stats = torch.cuda.memory_stats()
        row["allocator_stats"] = {name: stats[name] for name in (
            "inactive_split_bytes.all.current", "num_alloc_retries", "num_ooms")}
        with (output / "memory_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        voxels.clear()
        status(status="training", completed_updates=value["update"], epoch=value["epoch"],
               loss=value["semantic_loss"] + value["offset_loss"], lr=value["lr"],
               optimizer_skips_this_process=skipped[0], memory=row["memory"])

    def saver(path, model, optimizer, scheduler, scaler, update, identity, record):
        destination = checkpoint_target(path)
        # An epoch-end last.pth already contains this identical update.
        if destination.name == "recovery.pth" and last_saved[0] == update:
            return
        originals[2](destination, model, optimizer, scheduler, scaler, update, identity, record)
        last_saved[0] = update
        write_json(output / "latest_checkpoint.json", dict(
            path=str(destination), global_update=update, sha256=sha256(destination),
            is_formal_final=destination.name == "final.pth", saved_unix=time.time()))

    reference.build_model, reference.write_json = builder, writer
    reference.save_checkpoint, reference.sha256 = saver, checked_hash
    try:
        reference.run(SimpleNamespace(config=args.config, seed=args.seed, work_dir=str(output),
            phase="formal", max_updates=None, resume=args.resume, smoke_crop=None))
        status(status="complete", completed_updates=TOTAL_UPDATES)
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / f"exception_{time.time_ns()}.txt").write_text(traceback.format_exc())
        status(status="cuda_oom" if is_cuda_oom(error) else "failed", error=str(error),
               partial_voxels=list(voxels), failure_memory=memory())
        if is_cuda_oom(error):
            write_json(output / "oom_allocator_stats.json", dict(torch.cuda.memory_stats()))
            write_json(output / "oom_memory_snapshot.json", torch.cuda.memory_snapshot())
        raise
    finally:
        reference.build_model, reference.write_json, reference.save_checkpoint, reference.sha256 = originals
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resume")
    run(parser.parse_args())
