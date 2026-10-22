"""Shared training management for E0/E1, deliberately separate from main."""
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from munch import Munch
from torch.utils.data import DataLoader

from tree_learn.util.train import build_optimizer, build_cosine_scheduler
from .data import PlannedDataset, PlanSampler, sample_digest
from .model import build_model
from .provenance import (sha256, canonical_hash, write_json, git_state, environment,
                         seed_all, rng_state, restore_rng, load_initialization, state_equal)


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def code_hash():
    paths = sorted([*Path("tree_learn").rglob("*.py"), *Path("tools/experiments").rglob("*.py")])
    return canonical_hash({str(p): sha256(p) for p in paths})


def rewind_update_log(output, restored_update):
    """Keep only updates represented by the checkpoint; archive discarded logs."""
    path = Path(output) / "updates.jsonl"
    if not path.exists():
        return
    kept, discarded = [], []
    for line in path.read_text().splitlines(keepends=True):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            discarded.append(line)
            continue
        (kept if row["update"] <= restored_update else discarded).append(line)
    if discarded:
        archive = Path(output) / f"updates_after_checkpoint_{restored_update}_{time.time_ns()}.jsonl"
        archive.write_text("".join(discarded))
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(kept))
        os.replace(temporary, path)


def save_checkpoint(path, model, optimizer, scheduler, scaler, update, identity, last_record):
    checkpoint = {"net": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                  "global_update": update, "epoch": (update - 1) // identity["updates_per_epoch"] + 1,
                  "rng": rng_state(), "sampler": {"next_sample": update * identity["effective_batch_size"]},
                  "identity": identity, "last_record": last_record}
    temporary = str(path) + ".tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def run(args):
    config = load_config(args.config)
    torch.use_deterministic_algorithms(config["experiment"].get("deterministic_algorithms", False))
    if config["experiment"]["group"].startswith("E2"):
        raise ValueError("E2 is evaluation only")
    seed_all(args.seed)
    output = Path(args.work_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run {output}")
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ptv3_treelearn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "training.log")):
        logger.addHandler(handler)
    phase_budget = {"smoke": 100, "pilot": 500, "formal": 250000}[args.phase]
    budget = args.max_updates if args.max_updates is not None else phase_budget
    if not 0 < budget <= phase_budget:
        raise ValueError(f"{args.phase} permits 1..{phase_budget} updates")
    physical = config["dataloader"]["train"]["batch_size"]
    accumulation = config["experiment"].get("accumulation_steps", 1)
    effective = physical * accumulation
    if effective != 2 or config["examples_per_epoch"] % effective:
        raise ValueError("Frozen protocol requires effective batch=2 and whole epochs")
    per_epoch = config["examples_per_epoch"] // effective
    total = config["epochs"] * per_epoch
    if budget > total:
        raise ValueError("Budget exceeds the frozen sample plan")
    if args.smoke_crop:
        if args.phase != "smoke":
            raise ValueError("Unverified legacy crops are allowed only for short smoke tests")
        paths = [str(Path(args.smoke_crop).resolve())]
        schedule_rng = np.random.RandomState(args.seed)
        indices = np.zeros((config["epochs"], config["examples_per_epoch"]), dtype=np.int64)
        augmentation = schedule_rng.randint(0, 2**32 - 1, size=indices.shape, dtype=np.uint32)
        data_hash = canonical_hash({paths[0]: sha256(paths[0])})
        plan_hash = canonical_hash({"smoke": data_hash, "seed": args.seed})
    else:
        manifest_path = config["experiment"]["crop_manifest"]
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest["status"] != "verified" or len(manifest["crops"]) != 25000:
            raise ValueError("Training requires a verified full crop manifest")
        paths = [row["path"] for row in manifest["crops"]]
        plan_path = config["experiment"]["sample_plan"].format(seed=args.seed)
        metadata = json.loads(Path(plan_path + ".json").read_text())
        data_hash, plan_hash = sha256(manifest_path), sha256(plan_path)
        if metadata["crop_manifest_sha256"] != data_hash or metadata["plan_sha256"] != plan_hash or metadata["seed"] != args.seed:
            raise ValueError("Data/plan provenance mismatch")
        plan = np.load(plan_path)
        indices, augmentation = plan["indices"], plan["augmentation"]
        if indices.shape != (config["epochs"], config["examples_per_epoch"]) or augmentation.shape != indices.shape:
            raise ValueError("Plan dimensions do not match the training budget")
        # Verify every input that can actually be trained up to this stop. Formal
        # full-budget runs cover the whole planned pool; short P2/P3 runs need
        # not cold-read hundreds of GB of future, unused samples.
        for i in np.unique(indices.reshape(-1)[:budget * effective]):
            if sha256(paths[i]) != manifest["crops"][i]["sha256"]:
                raise ValueError(f"Changed crop: {paths[i]}")
    identity = {"config_sha256": canonical_hash(config), "data_sha256": data_hash,
                "plan_sha256": plan_hash, "seed": args.seed, "code_sha256": code_hash(),
                "environment_sha256": canonical_hash(environment()),
                "effective_batch_size": effective, "updates_per_epoch": per_epoch,
                "total_updates": total, "phase": args.phase, "unverified_smoke": bool(args.smoke_crop)}
    model = build_model(config).cuda()
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_cosine_scheduler(Munch.fromDict(config["scheduler"]), optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config["fp16"], init_scale=config["experiment"].get("amp_init_scale", 1024.))
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint["identity"] != identity:
            raise ValueError("Resume rejected: code, data, configuration, seed, phase or sample plan changed")
        model.load_state_dict(checkpoint["net"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["global_update"]
        if checkpoint["sampler"]["next_sample"] != start * effective:
            raise ValueError("Invalid sampler cursor in checkpoint")
        restore_rng(checkpoint["rng"])
        rewind_update_log(output, start)
        model_equal = all(torch.equal(value.detach().cpu(), checkpoint["net"][name]) for name, value in model.state_dict().items())
        restored_rng = rng_state()
        restoration = {"model_bitwise_equal": model_equal,
                       "optimizer_bitwise_equal": state_equal(optimizer.state_dict(), checkpoint["optimizer"]),
                       "scheduler_equal": state_equal(scheduler.state_dict(), checkpoint["scheduler"]),
                       "scaler_equal": state_equal(scaler.state_dict(), checkpoint["scaler"]),
                       "python_rng_equal": state_equal(restored_rng["python"], checkpoint["rng"]["python"]),
                       "numpy_rng_equal": state_equal(restored_rng["numpy"], checkpoint["rng"]["numpy"]),
                       "torch_rng_equal": torch.equal(restored_rng["torch"], checkpoint["rng"]["torch"]),
                       "cuda_rng_equal": all(torch.equal(a, b) for a, b in zip(restored_rng["cuda"], checkpoint["rng"]["cuda"])),
                       "sampler_cursor": checkpoint["sampler"]["next_sample"],
                       "global_update": start, "checkpoint_sha256": sha256(args.resume)}
        write_json(output / "resume_restoration.json", restoration)
        if not all(restoration[k] for k in ("model_bitwise_equal", "optimizer_bitwise_equal", "scheduler_equal", "scaler_equal",
                                           "python_rng_equal", "numpy_rng_equal", "torch_rng_equal", "cuda_rng_equal")):
            raise RuntimeError("Restored checkpoint state differs before any forward pass")
    elif config.get("pretrain"):
        report = load_initialization(model, config["pretrain"], output / "initialization.json", "hais")
        logger.info("HAIS required-layer coverage %.4f; full initialization=%s", report["required_coverage"], report["is_complete_initialization"])
    if start >= budget:
        raise ValueError("Requested stop must be later than the restored update")
    common_protocol = {key: config[key] for key in ("dataset_train", "dataloader", "optimizer", "scheduler", "epochs", "examples_per_epoch", "fp16", "grad_norm_clip")}
    common_protocol["input"] = {key: config["model"][key] for key in ("use_feats", "use_coords", "dim_coord", "dim_feat", "voxel_size", "max_num_points_per_voxel")}
    common_protocol["voxel_engineering"] = {key: config["model"].get(key) for key in ("deterministic_voxels", "spconv_fp32_accum", "cover_input_spatial_shape")}
    common_protocol["resource"] = {key: config["experiment"].get(key) for key in ("variant", "maximum_points", "accumulation_steps", "amp_init_scale")}
    manifest = {**identity, "config": config, "git": git_state(), "fairness_sha256": canonical_hash(common_protocol),
                "initialization_report_sha256": sha256(output / "initialization.json") if (output / "initialization.json").exists() else None,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "checkpoint_selection": "fixed full-budget final checkpoint; no L1W/Wytham selection",
                "validation_policy": "held-out evaluation callbacks disabled; original frequency is retained only as configuration provenance",
                "voxel_policy": {key: config["model"].get(key, False) for key in ("deterministic_voxels", "cover_input_spatial_shape")},
                "training_adaptations": ["separate data RNG and frozen sample plan", "unscale AMP gradients before clip(max_norm=1)",
                     "common AMP initial scale 1024; GradScaler may reject overflow steps and this is logged",
                     "save scheduler/scaler/RNG/data cursor and hashes", "no evaluation-set validation/early stopping"],
                "batchnorm_caveat": "microbatch accumulation has different BatchNorm statistics" if accumulation > 1 else None}
    write_json(output / "experiment_manifest.json", manifest)
    write_json(output / "environment.json", environment())
    dataset_options = dict(config["dataset_train"])
    dataset_options.pop("data_root", None)
    dataset = PlannedDataset(paths, maximum_points=config["experiment"].get("maximum_points"), **dataset_options)
    # A separate generator prevents DataLoader worker creation from consuming model RNG.
    loader = DataLoader(dataset, batch_size=physical, num_workers=config["dataloader"]["train"]["num_workers"],
                        sampler=PlanSampler(indices, augmentation, start * effective),
                        collate_fn=dataset.collate_fn, drop_last=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed + 1234567))
    iterator = iter(loader)
    model.train()
    record = None
    torch.cuda.reset_peak_memory_stats()
    for update in range(start, budget):
        update_began = time.perf_counter()
        batches = [next(iterator) for _ in range(accumulation)]
        semantic_count = sum(int(b["masks_sem"].sum()) for b in batches)
        offset_count = sum(int(b["masks_off"].sum()) for b in batches)
        digests = [sample_digest(b) for b in batches]
        scheduler.step(update // per_epoch + 1)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        values = {"semantic_loss": 0., "offset_loss": 0.}
        grid_audits = []
        for batch in batches:
            with torch.cuda.amp.autocast(enabled=config["fp16"]):
                _, losses = model(batch, return_loss=True)
                grid_audits.append(getattr(model, "last_spatial_support", None))
                # Sum losses with their own valid-point counts over the effective batch.
                weighted_sem = losses["semantic_loss"] * (int(batch["masks_sem"].sum()) / max(semantic_count, 1))
                weighted_off = losses["offset_loss"] * (int(batch["masks_off"].sum()) / max(offset_count, 1))
                loss = weighted_sem + weighted_off
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            values["semantic_loss"] += float(weighted_sem.detach())
            values["offset_loss"] += float(weighted_off.detach())
        bias_before = model.semantic_linear[-1].bias.detach().clone()
        scaler.unscale_(optimizer)
        norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().double().norm()
            for p in model.parameters() if p.grad is not None]))
        finite_gradients = bool(torch.isfinite(norm))
        if finite_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        elif not scaler.is_enabled():
            raise FloatingPointError("Nonfinite unscaled gradients without GradScaler")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < previous_scale
        if not finite_gradients and not skipped:
            raise FloatingPointError("Nonfinite gradients were not rejected by GradScaler")
        torch.cuda.synchronize()
        record = {"update": update + 1, "epoch": update // per_epoch + 1, **values,
                  "lr": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm) if finite_gradients else None,
                  "seconds": time.perf_counter() - began, "amp_scale": scaler.get_scale(), "optimizer_step_skipped": skipped,
                  "data_and_digest_seconds": began - update_began,
                  "update_wall_seconds": time.perf_counter() - update_began,
                  "semantic_head_bias_delta_l2": float((model.semantic_linear[-1].bias.detach()-bias_before).norm()),
                  "semantic_points": semantic_count, "offset_points": offset_count, "sample_digests": digests,
                  "points": sum(len(b["coords"]) for b in batches),
                  "sampling_audit": [item for batch in batches for item in batch["sampling_audit"]],
                  "grid_support": grid_audits,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        with open(output / "updates.jsonl", "a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        write_json(output / "monitoring.json", record)
        logger.info("update=%d loss=%.5f lr=%.8f sec=%.3f points=%d peak_GiB=%.3f", update + 1,
                    values["semantic_loss"] + values["offset_loss"], record["lr"], record["seconds"], record["points"], record["peak_allocated_bytes"] / 2**30)
        if (update + 1) % per_epoch == 0 or update + 1 == budget:
            save_checkpoint(output / "last.pth", model, optimizer, scheduler, scaler, update + 1, identity, record)
        if (update + 1) % (config["save_frequency"] * per_epoch) == 0:
            save_checkpoint(output / f"epoch_{(update+1)//per_epoch}.pth", model, optimizer, scheduler, scaler, update + 1, identity, record)
        interval = config["experiment"].get("checkpoint_updates", 0)
        if interval and (update + 1) % interval == 0:
            save_checkpoint(output / f"update_{update+1}.pth", model, optimizer, scheduler, scaler, update + 1, identity, record)
    if budget == total:
        save_checkpoint(output / "final.pth", model, optimizer, scheduler, scaler, budget, identity, record)
    else:
        save_checkpoint(output / f"{args.phase}_final.pth", model, optimizer, scheduler, scaler, budget, identity, record)
    write_json(output / "checkpoint_provenance.json", {"last_sha256": sha256(output / "last.pth"),
               "final_sha256": sha256(output / "final.pth") if budget == total else None,
               "identity": identity, "global_update": budget, "is_formal_final": budget == total})
