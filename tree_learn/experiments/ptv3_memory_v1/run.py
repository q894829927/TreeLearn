"""Frozen-input paired runs and full raw crop tests, one process per run."""
import argparse
import contextlib
import gc
import json
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset, sample_digest
from tree_learn.experiments.ptv3_treelearn.model import build_model
from tree_learn.experiments.ptv3_treelearn.train import load_config, code_hash
from tree_learn.experiments.ptv3_treelearn.provenance import (seed_all, rng_state, restore_rng, sha256,
                                                            write_json, environment, git_state)
from .profile import MemoryAudit, memory
from . import resolved_config


def config():
    return load_config("configs/experiments/ptv3_treelearn/e1_ptv3_micro_checkpoint.yaml")


def batch_from_crop(path, augmentation_seed):
    options = dict(config()["dataset_train"])
    options.pop("data_root")
    dataset = PlannedDataset([path], maximum_points=None, **options)
    return dataset.collate_fn([dataset[(0, augmentation_seed)]])


def prepare(args):
    from tree_learn.model.tree_learn import voxelize
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    model = build_model(config())
    torch.save(model.state_dict(), folder/"initial_state.pth")
    batch = batch_from_crop(args.crop, args.augmentation_seed)
    xyz = batch["coords"].cuda(); feat = batch["input_feats"].cuda()
    ids = batch["batch_ids"].cuda()
    with torch.no_grad():
        features, indices, point_map, shape = voxelize(torch.hstack([xyz, feat]), ids, 1, .1, False, False, 3)
        grid = indices[:, 1:].int(); b = indices[:, 0].long()
        coord = (grid.float()+.5)*.1 + xyz.min(0).values
        frozen = dict(feat=features.cpu(), grid_coord=grid.cpu(), coord=coord.cpu(), batch=b.cpu(),
                      offset=torch.tensor([len(grid)]))
    torch.save(dict(batch=batch, backbone=frozen, point_map=point_map.cpu(), rng=rng_state()), folder/"input.pth")
    write_json(folder/"manifest.json", dict(crop=str(Path(args.crop).resolve()), crop_sha256=sha256(args.crop),
               points=len(xyz), voxels=len(grid), augmentation_seed=args.augmentation_seed, seed=args.seed,
               batch_digest=sample_digest(batch), input_sha256=sha256(folder/"input.pth"),
               initial_state_sha256=sha256(folder/"initial_state.pth"), code_sha256=code_hash(), git=git_state()))


def run(args):
    if args.save_tensors and args.accumulation != 1:
        raise ValueError("Paired tensor dumps require accumulation=1")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    cfg = resolved_config(config(), args.variant)
    model = build_model(cfg).cuda()
    if args.variant != "reference":
        from .adapters import apply_optimization
        apply_optimization(model, args.variant)
    fixture = Path(args.fixture) if args.fixture else None
    if fixture:
        data = torch.load(fixture/"input.pth", map_location="cpu")
        model.load_state_dict(torch.load(fixture/"initial_state.pth", map_location="cpu"), strict=True)
        batch = data["batch"]
        frozen = {k:v.cuda() for k,v in data["backbone"].items()}
        point_map = data["point_map"].cuda()
        def frozen_backbone(self, **kwargs):
            result = self.backbone(dict(frozen))
            return SimpleNamespace(features=result.feat), point_map
        model.forward_backbone = MethodType(frozen_backbone, model)
        initial_rng = data["rng"]
        input_identity = json.loads((fixture/"manifest.json").read_text())
    else:
        batch = batch_from_crop(args.crop, args.augmentation_seed)
        initial_rng = rng_state()
        input_identity = dict(crop=str(Path(args.crop).resolve()), crop_sha256=sha256(args.crop),
                              points=len(batch["coords"]), batch_digest=sample_digest(batch), augmentation_seed=args.augmentation_seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.98e-5, weight_decay=.001)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    restore_rng(initial_rng)
    provenance = dict(variant=args.variant, raw_input=fixture is None, input=input_identity,
                      code_sha256=code_hash(), git=git_state(), environment=environment(),
                      config=cfg, precision="same CUDA autocast FP16 and upstream Flash FP16 as baseline",
                      steps=args.steps, accumulation_steps=args.accumulation,
                      effective_batch_size=args.accumulation, physical_batch_size=1)
    sources = sorted(Path(__file__).parent.glob("*.py"))+[Path("tools/experiments/ptv3_memory_v1.py")]
    provenance["audit_sources"] = {str(path):sha256(path) for path in sources}
    provenance["command"] = sys.argv
    for path in sources:
        target = output/"sources"/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
    write_json(output/"manifest.json", provenance)
    audit = MemoryAudit(output, args.operators) if args.profile else None
    result = dict(status="running", completed_steps=0, initial=memory(), steps=[])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with audit.attach(model) if audit else contextlib.nullcontext():
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); tick = time.perf_counter()
                micros = []
                for micro in range(args.accumulation):
                    torch.cuda.synchronize(); micro_start = time.perf_counter()
                    if audit: audit.phase = "forward"
                    with torch.cuda.amp.autocast():
                        predictions = model(batch, return_loss=False)
                        loss, components = model.get_loss(model_output=predictions, **batch)
                    if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                    torch.cuda.synchronize(); forward_done = time.perf_counter()
                    forward_memory = memory()
                    if audit: audit.phase = "backward"
                    scaler.scale(loss/args.accumulation).backward()
                    torch.cuda.synchronize(); backward_done = time.perf_counter()
                    micros.append(dict(micro=micro+1, loss=float(loss.detach()),
                        forward_seconds=forward_done-micro_start, backward_seconds=backward_done-forward_done,
                        after_forward=forward_memory, after_backward=memory()))
                    if not args.save_tensors:
                        del predictions, loss, components
                scaler.unscale_(optimizer)
                gradients_finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                if not gradients_finite: raise FloatingPointError("nonfinite gradients")
                if args.save_tensors and step == 0:
                    torch.save(dict(predictions={k:v.detach().cpu() for k,v in predictions.items() if k!="backbone_feats"},
                        gradients={k:p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None},
                        buffers={k:b.detach().cpu().clone() for k,b in model.named_buffers()},
                        rng=rng_state(), loss=float(loss.detach())), output/"comparison.pth")
                torch.cuda.synchronize(); optimizer_start = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if audit: audit.phase = "optimizer"
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale()!=previous_scale: raise RuntimeError("optimizer step was skipped")
                torch.cuda.synchronize()
                optimizer_seconds = time.perf_counter()-optimizer_start
                forward_seconds = sum(m["forward_seconds"] for m in micros)
                backward_seconds = sum(m["backward_seconds"] for m in micros)
                record = dict(step=step+1, loss=sum(m["loss"] for m in micros)/len(micros),
                    gradients_finite=gradients_finite, forward_seconds=forward_seconds,
                    backward_seconds=backward_seconds, optimizer_seconds=optimizer_seconds,
                    compute_seconds=forward_seconds+backward_seconds+optimizer_seconds,
                    total_seconds=time.perf_counter()-tick, micros=micros, **memory())
                result["steps"].append(record); result["completed_steps"] += 1
                if args.save_tensors:
                    del predictions, loss, components
                gc.collect()
                record["after_release"] = memory()
                write_json(output/"progress.json", result)
        result["status"] = "pass"
    except BaseException as error:
        result.update(status="cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error", error=str(error))
        (output/"exception.txt").write_text(traceback.format_exc())
    finally:
        result.update(seconds=time.perf_counter()-started, final=memory(),
                      peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if audit:
            result["final"].update(audit.peaks)
            result.update(last_successful_operator=audit.last_operator, last_successful_module=audit.last_module,
                          point_created=audit.created_points, point_released=audit.released_points)
            audit.finish()
        write_json(output/"result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass": raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--crop")
    source.add_argument("--fixture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation-seed", type=int, default=1608637542)
    parser.add_argument("--steps", type=int, choices=(1,2,3), default=1)
    parser.add_argument("--accumulation", type=int, choices=(1,2), default=1)
    parser.add_argument("--variant", default="reference")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--operators", action="store_true")
    parser.add_argument("--save-tensors", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare" and not args.crop:
        parser.error("prepare requires --crop")
    if args.operators and not args.profile:
        parser.error("--operators requires --profile")
    if args.command == "prepare": prepare(args)
    else: run(args)
