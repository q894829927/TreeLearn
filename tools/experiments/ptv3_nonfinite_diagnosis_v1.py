"""Bounded diagnostic replay against an immutable PTv3 training snapshot.

No formal run files are written. The original checkpoint, frozen implementation,
data, precision, optimizer, schedule and augmentation plan are read-only inputs.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Diagnostic state; default: original recovery.pth")
    parser.add_argument("--target", type=int, default=4617)
    args = parser.parse_args()
    base, out = Path(args.run_root).resolve(), Path(args.output).resolve()
    frozen = base / "source"
    if out.exists():
        parser.error("Use a new diagnostic output directory")
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out / "diagnostic_source.py")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.7"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(frozen))
    sys.path.insert(0, str(frozen / "work_dirs/ptv3_treelearn/runtime/packages"))
    os.chdir(frozen)
    import numpy as np
    import torch
    from munch import Munch
    from torch.utils.data import DataLoader
    from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
    from tree_learn.experiments.ptv3_treelearn import train as reference
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset, PlanSampler, sample_digest
    from tree_learn.experiments.ptv3_treelearn.provenance import (
        canonical_hash, environment, restore_rng, rng_state, seed_all, sha256, state_equal)
    from tree_learn.util.train import build_optimizer, build_cosine_scheduler

    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def append(name, value):
        with (out / name).open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def stats(t):
        t = t.detach()
        if not (t.is_floating_point() or t.is_complex()) or not t.numel():
            return dict(shape=list(t.shape), dtype=str(t.dtype), numel=t.numel())
        finite = bool(torch.isfinite(t).all())
        result = dict(shape=list(t.shape), dtype=str(t.dtype), numel=t.numel(), finite=finite)
        if finite:
            result.update(min=float(t.min()), max=float(t.max()))
        else:
            result.update(nan=int(torch.isnan(t).sum()), posinf=int(torch.isposinf(t).sum()),
                          neginf=int(torch.isneginf(t).sum()))
        return result

    def features(value):
        if torch.is_tensor(value):
            return [("tensor", value)]
        if isinstance(value, dict) and "feat" in value:
            return [("point.feat", value["feat"])]
        if hasattr(value, "features") and torch.is_tensor(value.features):
            return [("sparse.features", value.features)]
        if isinstance(value, (tuple, list)):
            return [(str(i) + "." + k, t) for i, x in enumerate(value) for k, t in features(x)]
        return []

    def pack(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if hasattr(value, "features"):
            return dict(kind="sparse", features=value.features.detach().cpu(),
                        indices=value.indices.detach().cpu(), spatial_shape=list(value.spatial_shape),
                        batch_size=value.batch_size)
        if isinstance(value, dict) and "feat" in value:
            return {k: v.detach().cpu() for k, v in value.items()
                    if torch.is_tensor(v) and k not in ("pooling_parent",)}
        if isinstance(value, (tuple, list)):
            return [pack(x) for x in value]
        return str(type(value))

    original_run = base / "stage_validation_v1/training/to_5000"
    manifest = json.loads((original_run / "experiment_manifest.json").read_text())
    config = manifest["config"]
    resume = Path(args.resume).resolve() if args.resume else original_run / "recovery.pth"
    checkpoint = torch.load(resume, map_location="cpu")
    start = checkpoint["global_update"]
    if not 0 < start < args.target <= 4617:
        raise ValueError("This diagnostic may only replay the existing failure through attempt 4617")
    if checkpoint["identity"]["code_sha256"] != reference.code_hash():
        raise ValueError("Frozen source differs from the checkpoint")
    if checkpoint["identity"]["config_sha256"] != canonical_hash(config):
        raise ValueError("Checkpoint/config mismatch")
    crop_path = Path(config["experiment"]["crop_manifest"])
    plan_path = Path(config["experiment"]["sample_plan"].format(seed=42))
    if sha256(crop_path) != checkpoint["identity"]["data_sha256"]:
        raise ValueError("Crop manifest changed")
    if sha256(plan_path) != checkpoint["identity"]["plan_sha256"]:
        raise ValueError("Sample plan changed")
    crops = json.loads(crop_path.read_text())["crops"]
    paths = [x["path"] for x in crops]
    plan = np.load(plan_path)
    indices, augmentation = plan["indices"], plan["augmentation"]
    used = np.unique(indices.reshape(-1)[2*start:2*args.target])
    for i in used:
        if sha256(paths[i]) != crops[i]["sha256"]:
            raise ValueError("Changed diagnostic input: " + paths[i])
    print("Verified", len(used), "replay inputs", flush=True)
    seed_all(42)
    torch.use_deterministic_algorithms(config["experiment"].get("deterministic_algorithms", False))
    torch.cuda.set_per_process_memory_fraction(1.0)
    model = build_optimized_model(config, "block_nested").cuda()
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_cosine_scheduler(Munch.fromDict(config["scheduler"]), optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config["fp16"], init_scale=1024.)
    model.load_state_dict(checkpoint["net"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"])
    restored = rng_state()
    equality = dict(model=all(torch.equal(t.detach().cpu(), checkpoint["net"][n])
                              for n, t in model.state_dict().items()),
                    optimizer=state_equal(optimizer.state_dict(), checkpoint["optimizer"]),
                    scheduler=state_equal(scheduler.state_dict(), checkpoint["scheduler"]),
                    scaler=state_equal(scaler.state_dict(), checkpoint["scaler"]),
                    rng=state_equal(restored, checkpoint["rng"]),
                    sampler=checkpoint["sampler"]["next_sample"] == start*2)
    if not all(equality.values()):
        raise RuntimeError("Checkpoint restoration failed")
    write("manifest.json", dict(purpose="diagnosis only; no formal resume or added training budget",
          frozen_source=str(frozen), source_sha256=reference.code_hash(), resume=str(resume),
          resume_sha256=sha256(resume), start_update=start, target_attempt=args.target,
          restored_exact=equality, environment=environment(), config=config,
          verification_scope="All distinct crops consumed by this diagnostic replay, not the unused prefix",
          verified_crops=len(used), original_run=str(original_run), command=sys.argv,
          replay_limitation="CUDA voxelization/scatter/spconv/FlashAttention backward can be nondeterministic"))
    del checkpoint
    options = dict(config["dataset_train"])
    options.pop("data_root", None)
    dataset = PlannedDataset(paths, maximum_points=None, **options)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
        sampler=PlanSampler(indices.reshape(-1)[:args.target*2],
                            augmentation.reshape(-1)[:args.target*2], start*2),
        collate_fn=dataset.collate_fn, drop_last=True, pin_memory=True,
        generator=torch.Generator().manual_seed(42 + 1234567))
    iterator = iter(loader)
    model.train()
    originals = {x["update"]: x for x in map(json.loads, (original_run/"updates.jsonl").read_text().splitlines())}
    active = [False]
    first_bad = []
    handles = []
    sequence = [0]
    input_stats = {}

    def before(name):
        def hook(module, inputs):
            if active[0]:
                input_stats[name] = {k: stats(t) for k, t in features(inputs)}
        return hook

    def after(name):
        def hook(module, inputs, output):
            if not active[0]:
                return
            values = {k: stats(t) for k, t in features(output)}
            sequence[0] += 1
            row = dict(sequence=sequence[0], module=name, type=type(module).__name__,
                       grad_enabled=torch.is_grad_enabled(), inputs=input_stats.get(name), outputs=values)
            append("module_trace.jsonl", row)
            if not first_bad and any(x.get("finite") is False for x in values.values()):
                first_bad.append(row)
                write("first_nonfinite_module.json", row)
                torch.save(dict(module=name, module_type=str(type(module)), inputs=pack(inputs),
                    state={k:v.detach().cpu() for k,v in module.state_dict().items()},
                    repr=repr(module)), out/"first_bad_operator.pt")
                print("FIRST NONFINITE", name, type(module).__name__, flush=True)
        return hook

    for name, module in model.named_modules():
        if name:
            handles.extend([module.register_forward_pre_hook(before(name)),
                            module.register_forward_hook(after(name))])
    result = dict(status="running", start_update=start, target_attempt=args.target)
    write("status.json", result)
    try:
        for update in range(start, args.target):
            number = update+1
            began = time.perf_counter()
            batches = [next(iterator) for _ in range(2)]
            sem_count = sum(int(b["masks_sem"].sum()) for b in batches)
            off_count = sum(int(b["masks_off"].sum()) for b in batches)
            digests = [sample_digest(b) for b in batches]
            scheduler.step(update//250+1)
            optimizer.zero_grad(set_to_none=True)
            if number == args.target:
                reference.save_checkpoint(out/"state_before_target.pth", model, optimizer,
                    scheduler, scaler, update, manifest, None)
                # Replace the administrative identity with the original one.
                saved = torch.load(out/"state_before_target.pth", map_location="cpu")
                saved["identity"] = {k:manifest[k] for k in ["config_sha256", "data_sha256", "plan_sha256",
                    "seed", "code_sha256", "environment_sha256", "effective_batch_size", "updates_per_epoch",
                    "total_updates", "phase", "unverified_smoke"]}
                torch.save(saved, out/"state_before_target.pth")
                del saved
                active[0] = True
            values = dict(semantic_loss=0., offset_loss=0.)
            for micro, batch in enumerate(batches):
                if active[0]:
                    write("target_input.json", dict(attempt=number, microbatch=micro+1,
                        digest=digests[micro], tensors={k:stats(v) for k,v in batch.items() if torch.is_tensor(v)},
                        sampling_audit=batch["sampling_audit"]))
                with torch.cuda.amp.autocast(enabled=config["fp16"]):
                    _, losses = model(batch, return_loss=True)
                    weighted_sem = losses["semantic_loss"] * (int(batch["masks_sem"].sum()) / max(sem_count,1))
                    weighted_off = losses["offset_loss"] * (int(batch["masks_off"].sum()) / max(off_count,1))
                    loss = weighted_sem + weighted_off
                if active[0]:
                    write("target_loss.json", dict(attempt=number, microbatch=micro+1,
                        semantic_loss=stats(losses["semantic_loss"]), offset_loss=stats(losses["offset_loss"]),
                        weighted_semantic=stats(weighted_sem), weighted_offset=stats(weighted_off),
                        total=stats(loss), semantic_count=sem_count, offset_count=off_count,
                        spatial_support=model.last_spatial_support))
                if not torch.isfinite(loss):
                    result.update(status="nonfinite_reproduced", failed_attempt=number,
                                  failed_microbatch=micro+1, first_bad_module=first_bad[0] if first_bad else None)
                    raise FloatingPointError("Nonfinite loss in diagnostic replay")
                scaler.scale(loss).backward()
                values["semantic_loss"] += float(weighted_sem.detach())
                values["offset_loss"] += float(weighted_off.detach())
            scaler.unscale_(optimizer)
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().double().norm()
                    for p in model.parameters() if p.grad is not None]))
            if torch.isfinite(norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            row = dict(update=number, **values, gradient_norm=float(norm) if torch.isfinite(norm) else None,
                       amp_scale=scaler.get_scale(), optimizer_step_skipped=scaler.get_scale()<previous_scale,
                       sample_digests=digests, seconds=time.perf_counter()-began,
                       bn_counts=sorted(set(int(m.num_batches_tracked) for m in model.modules()
                                       if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))))
            if number in originals:
                original = originals[number]
                row["original_comparison"] = dict(input_digests_equal=digests==original["sample_digests"],
                    semantic_loss_abs_error=abs(values["semantic_loss"]-original["semantic_loss"]),
                    offset_loss_abs_error=abs(values["offset_loss"]-original["offset_loss"]),
                    original_semantic_loss=original["semantic_loss"], original_offset_loss=original["offset_loss"])
            append("replay_updates.jsonl", row)
            print(json.dumps(row, allow_nan=False), flush=True)
            result["last_completed_update"] = number
            write("status.json", result)
        result["status"] = "target_finite_in_replay"
    except FloatingPointError:
        (out/"exception.txt").write_text(traceback.format_exc())
    except BaseException:
        result["status"] = "diagnostic_error"
        (out/"exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), ended_unix=time.time())
        write("status.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
