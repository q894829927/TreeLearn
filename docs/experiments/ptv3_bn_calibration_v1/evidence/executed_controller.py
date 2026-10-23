"""Bounded training-domain BN calibration and unchanged frozen L1W evaluation.

Two inference-only artifacts are produced before either validation is started.
No optimizer, backward, training resume, label-based sample selection or tuning.
"""
import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

OUTPUT = None
POLICY = "training-population-fp32-sequential-bn-v1"


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, out = Path(args.run_root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    OUTPUT = out
    implementation = Path(__file__).resolve().parents[2] / "tree_learn/experiments/ptv3_bn_calibration_v1/__init__.py"
    shutil.copyfile(__file__, out / "executed_controller.py")
    shutil.copyfile(implementation, out / "executed_implementation.py")
    source, validation = base / "source_head_fp32_v1", base / "source_validation_v1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.7"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path[:0] = [str(source / "work_dirs/ptv3_treelearn/runtime/packages"), str(source)]
    os.chdir(source)
    import numpy as np
    import torch
    from tree_learn.experiments.ptv3_treelearn.provenance import sha256, write_json, seed_all, canonical_hash, environment
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset, sample_digest
    from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
    from tree_learn.experiments.ptv3_head_fp32_v1 import enable_head_fp32
    from tree_learn.model.tree_learn import voxelize
    spec = importlib.util.spec_from_file_location("calibration_impl", implementation)
    impl = importlib.util.module_from_spec(spec); spec.loader.exec_module(impl)
    began = time.perf_counter()

    def status(phase, **extra):
        record = dict(phase=phase, pid=os.getpid(), updated_unix=time.time(),
                      elapsed_seconds=time.perf_counter()-began, policy=POLICY, **extra)
        write_json(out / "status.json", record)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def frozen_sources():
        result = {}
        for name, expected in [("source", "649c1749ea654cb42add12db2eacff3da9532c09"),
                               ("source_validation_v1", "a32ec3cd6233f37e050ca03d0f8a1bf959324b91"),
                               ("source_head_fp32_v1", "e4ea04b3e46eb86bf7d094e6ff72a7b70c7fe50a")]:
            head = subprocess.check_output(["git", "-C", str(base / name), "rev-parse", "HEAD"], text=True).strip()
            clean = subprocess.check_output(["git", "-C", str(base / name), "status", "--porcelain"], text=True)
            if head != expected or clean:
                raise ValueError("Frozen source changed: " + name)
            result[name] = head
        return result

    stage = base / "stage_validation_v1"
    parent = stage / "checkpoints/update_5000_head_fp32_v1.pth"
    parent_sha = sha256(parent)
    if parent_sha != "c9cc21c42f26f2052e887cdc5b4a9a2ed0f314e22509804846c8cb0d771919fd":
        raise ValueError("Unexpected parent checkpoint")
    protocol_sha = sha256(stage / "protocol.json")
    if protocol_sha != "3f3d223ac9fe4247fd327bd8d7e10946e83efd51508340fd44273b4511d86b69":
        raise ValueError("Changed L1W protocol")
    status("preparing", frozen_sources=frozen_sources())
    manifest = json.loads((stage / "training/to_5000_head_fp32_v1/experiment_manifest.json").read_text())
    cfg = manifest["config"]
    crops_path = source / cfg["experiment"]["crop_manifest"]
    plan_path = source / cfg["experiment"]["sample_plan"].format(seed=42)
    crop_manifest = json.loads(crops_path.read_text())
    plan_meta = json.loads(Path(str(plan_path)+".json").read_text())
    if sha256(crops_path) != plan_meta["crop_manifest_sha256"] or sha256(plan_path) != manifest["plan_sha256"]:
        raise ValueError("Changed training manifest/plan")
    forests = sorted(crop_manifest["counts_by_forest"])
    if len(forests) != 18 or any("l1w" in x.lower() or "wytham" in x.lower() for x in forests):
        raise ValueError("Unexpected training domains")
    plan = np.load(plan_path)
    buckets = {name: [] for name in forests}; used = set()
    for position, (index, seed) in enumerate(zip(plan["indices"].ravel()[:10000], plan["augmentation"].ravel()[:10000])):
        row = crop_manifest["crops"][int(index)]; forest = row["source_forest"]
        if len(buckets[forest]) < 4 and int(index) not in used:
            buckets[forest].append(dict(plan_position=position, crop_manifest_index=int(index),
                                       augmentation_seed=int(seed), **row)); used.add(int(index))
    if any(len(rows) != 4 for rows in buckets.values()):
        raise ValueError("Insufficient previously consumed crops")
    selected = [buckets[forest][round_index] for round_index in range(4) for forest in forests]
    groups = [["backbone.embedding.stem.norm"]]
    groups += [[f"backbone.enc.enc{i}.down.norm.0"] for i in range(1, 5)]
    groups += [[f"backbone.dec.dec{i}.up.proj.1", f"backbone.dec.dec{i}.up.proj_skip.1"] for i in range(3, -1, -1)]
    heads = ["semantic_linear.1", "offset_linear.1"]
    groups += [heads]
    protocol = dict(policy=POLICY, parent=str(parent), parent_sha256=parent_sha,
                    source_commits=frozen_sources(), config_sha256=canonical_hash(cfg),
                    controller_sha256=sha256(out / "executed_controller.py"),
                    implementation_sha256=sha256(out / "executed_implementation.py"),
                    sample_plan_sha256=sha256(plan_path), crop_manifest_sha256=sha256(crops_path),
                    calibration_samples=selected, samples=72, crops_per_training_forest=4,
                    already_consumed_sample_prefix=10000, validation_protocol_sha256=protocol_sha,
                    precision="global FP32, native sparse eval kernel; FlashAttention native internal QKV precision",
                    statistic="point-weighted pooled FP64 central moments; unbiased aggregate running variance",
                    calibration_order=groups, variants=["head_only", "all_sequential"],
                    prefixes=[18,36,72], stability_flag_threshold=dict(normalized_mean_shift=.1, relative_std_change=.1),
                    stability_policy="Measure 36->72; flag failures without automatic sample/budget expansion or result-based tuning",
                    selection="First four unique consumed crops per training forest, round-robin; no labels or validation outcomes",
                    labels_used=False, automatic_training=False, optimizer_steps=0,
                    num_batches_tracked="Preserve original 10000; population statistics provenance is separate",
                    resumable_training_checkpoint=False)
    write_json(out / "protocol.json", protocol)
    options = dict(cfg["dataset_train"]); options.pop("data_root", None)
    dataset = PlannedDataset([row["path"] for row in selected], maximum_points=None, **options)
    seed_all(42); torch.cuda.set_per_process_memory_fraction(1.)
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(out / "environment.json", environment())
    checkpoint = torch.load(parent, map_location="cpu")
    if checkpoint["global_update"] != 5000:
        raise ValueError("Unexpected update")
    original = checkpoint["net"]; del checkpoint
    model = enable_head_fp32(build_optimized_model(cfg, "block_nested")).cuda()
    model.load_state_dict(original, strict=True); model.eval()
    bn = {name: module for name, module in model.named_modules()
          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
    if set(bn) != {name for group in groups for name in group}:
        raise ValueError("Unexpected BN topology")
    active = {}

    def fixed_backbone(self, **kwargs):
        data = active["backbone"]
        point = self.backbone(dict(data))
        if not torch.equal(point.grid_coord, data["grid_coord"]) or not torch.equal(point.batch, data["batch"]):
            raise ValueError("Changed voxel output order")
        return SimpleNamespace(features=point.feat), active["point_map"]

    model.forward_backbone = MethodType(fixed_backbone, model)
    fixtures = []
    cache = out / "fixtures"; cache.mkdir()
    for i, row in enumerate(selected):
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("Changed calibration crop")
        batch = dataset.collate_fn([dataset[(i, row["augmentation_seed"])]])
        audit = batch["sampling_audit"][0]
        if audit["points_before"] != row["points"] or audit["points_after"] != row["points"] or audit["instances_after"] != audit["instances_before"]:
            raise ValueError("Incomplete calibration input")
        with torch.no_grad():
            xyz, features, ids = [batch[key].cuda() for key in ("coords", "input_feats", "batch_ids")]
            vf, vi, pm, shape = voxelize(torch.hstack([xyz, features]), ids, 1, .1, False, False, 3)
            grid, batches = vi[:,1:].int(), vi[:,0].long()
            if len(pm) != row["points"] or (pm < 0).any():
                raise ValueError("Incomplete point map")
            data = dict(feat=vf, grid_coord=grid, coord=(grid.float()+.5)*.1+xyz.min(0).values,
                        batch=batches, offset=torch.tensor([len(grid)], device="cuda"))
            fixture = dict(backbone={key:value.cpu() for key,value in data.items()}, point_map=pm.cpu())
        path = cache / f"{i:03d}.pth"; torch.save(fixture, path); fixtures.append(fixture)
        row.update(input_digest=sample_digest(batch), fixture=str(path), fixture_sha256=sha256(path),
                   voxels=len(grid), input_audit=audit)
        del batch, xyz, features, ids, vf, vi, pm, grid, batches, data
        if i == 0:
            trace = []
            handles = [module.register_forward_pre_hook(lambda module, inputs, name=name: trace.append(name))
                       for name,module in bn.items()]
            active.update(backbone={key:value.cuda() for key,value in fixture["backbone"].items()},
                          point_map=fixture["point_map"].cuda())
            seed_all(42 + row["plan_position"])
            with torch.no_grad():
                prediction = model({}, return_loss=False)
            impl.validate_trace(groups, trace)
            if not all(torch.isfinite(prediction[key]).all() for key in ("offset_predictions", "semantic_prediction_logits")):
                raise FloatingPointError("Nonfinite GPU gate")
            preservation = impl.verify_state(original, model.state_dict(), [])
            write_json(out / "gpu_gate.json", dict(full_input_points=row["points"], trace=trace,
                       predictions_finite=True, preservation=preservation, optimizer_steps=0))
            for handle in handles: handle.remove()
            del prediction; active.clear()
        gc.collect()
        status("preparing_fixtures", completed=i+1, total=72)
    write_json(out / "calibration_inputs.json", selected)
    results = {}
    for variant, stages in [("head_only", [heads]), ("all_sequential", groups)]:
        folder = out / variant; folder.mkdir()
        model.load_state_dict(original, strict=True); model.eval()
        variant_began = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        stage_reports = []
        for group_index, names in enumerate(stages):
            moments = {name: impl.Moments() for name in names}
            prefixes = {}; trace = []; handles = []
            for name, module in bn.items():
                def hook(module, inputs, name=name):
                    trace.append(name)
                    if name in moments:
                        moments[name].add(inputs[0])
                handles.append(module.register_forward_pre_hook(hook))
            for i, fixture in enumerate(fixtures):
                active.update(backbone={key:value.cuda() for key,value in fixture["backbone"].items()},
                              point_map=fixture["point_map"].cuda())
                seed_all(42 + selected[i]["plan_position"])
                trace.clear()
                with torch.no_grad():
                    prediction = model({}, return_loss=False)
                if not all(torch.isfinite(prediction[key]).all() for key in ("offset_predictions", "semantic_prediction_logits")):
                    raise FloatingPointError("Nonfinite calibration forward")
                impl.validate_trace(groups, trace)
                if any(module.training for module in model.modules()):
                    raise ValueError("Calibration must keep every module in eval")
                if i+1 in [18,36,72]:
                    prefixes[str(i+1)] = {name:value.snapshot() for name,value in moments.items()}
                del prediction; active.clear(); gc.collect()
                status("calibrating", variant=variant, group=group_index, bn=names, completed=i+1, total=72)
            for handle in handles: handle.remove()
            installed = {name: moment.install(bn[name]) for name,moment in moments.items()}
            stability = {}
            for name, final in installed.items():
                half = prefixes["36"][name]
                final_mean = torch.tensor(final["mean"], dtype=torch.float64)
                final_std = torch.tensor(final["running_variance_unbiased"], dtype=torch.float64).add(bn[name].eps).sqrt()
                half_std = torch.tensor(half["running_variance_unbiased"], dtype=torch.float64).add(bn[name].eps).sqrt()
                mean_shift = float(((final_mean-torch.tensor(half["mean"], dtype=torch.float64)).abs()/final_std).max())
                std_change = float(((half_std-final_std).abs()/final_std).max())
                stability[name] = dict(max_mean_shift_in_final_std=mean_shift, max_relative_std_change=std_change,
                                       within_predeclared_tolerance=mean_shift <= .1 and std_change <= .1)
            report = dict(group=group_index, names=names, prefixes=prefixes, installed=installed, stability=stability)
            stage_reports.append(report); write_json(folder / f"group_{group_index:02d}.json", report)
        state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        chosen = [name for group in stages for name in group]
        preservation = impl.verify_state(original, state, chosen)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Unexpected gradient")
        artifact = folder / "inference_only.pth"
        torch.save(dict(net=state, global_update=5000, is_formal_final=False,
                        inference_only=True, resumable_training_checkpoint=False,
                        numerical_policy="treelearn-both-heads-fp32-v1", bn_policy=POLICY,
                        bn_variant=variant, parent_checkpoint_sha256=parent_sha,
                        calibration_protocol_sha256=sha256(out / "protocol.json")), artifact)
        result = dict(variant=variant, checkpoint=str(artifact), checkpoint_sha256=sha256(artifact),
                      preservation=preservation, groups=len(stages), full_forwards=72*len(stages),
                      seconds=time.perf_counter()-variant_began,
                      population_stability_within_tolerance=all(s["within_predeclared_tolerance"] for r in stage_reports for s in r["stability"].values()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(folder / "calibration_result.json", result); results[variant] = result
        status("calibration_variant_complete", **result)
        del state
    del model, original, fixtures; active.clear(); gc.collect(); torch.cuda.empty_cache()
    if sha256(parent) != parent_sha or sha256(stage / "protocol.json") != protocol_sha:
        raise ValueError("Changed original checkpoint/protocol")
    write_json(out / "calibration_complete_before_validation.json", dict(results=results, completed_unix=time.time()))
    comparison = json.loads((stage / "comparison_at_5000_head_fp32_v1.json").read_text())
    commands = {}
    for variant in ["head_only", "all_sequential"]:
        validation_root = out / variant / "validation"; validation_root.mkdir()
        shutil.copyfile(stage / "protocol.json", validation_root / "protocol.json")
        evaluation = validation_root / "evaluation"
        command = [sys.executable, "-u", str(validation / "tools/experiments/ptv3_stage_v1.py"),
                   "evaluate", "--root", str(validation_root), "--config", "configs/experiments/ptv3_memory_v1/e1_full_formal.yaml",
                   "--checkpoint", results[variant]["checkpoint"], "--tag", "e1_u5000_bn_"+variant+"_v1", "--output", str(evaluation)]
        commands[variant] = dict(argv=command, cwd=str(validation)); write_json(out / "evaluation_commands.json", commands)
        with (out / variant / "evaluation_console.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=validation, stdout=stream, stderr=subprocess.STDOUT)
            status("evaluating", variant=variant, child_pid=process.pid, argv=command)
            exit_code = process.wait()
        if exit_code:
            raise RuntimeError(f"Frozen evaluation failed: {variant}, exit {exit_code}")
        em = json.loads((evaluation / "evaluation_manifest.json").read_text())
        baseline_em = json.loads((stage / "continuations/to_5000_head_fp32_v1/evaluation/evaluation_manifest.json").read_text())
        for key in ("config_sha256", "code_sha256", "protocol_sha256", "gt_core_sha256"):
            if em[key] != baseline_em[key]:
                raise ValueError("Changed evaluation identity: " + key)
        if em["checkpoint_update"] != 5000 or em["is_formal_final"]:
            raise ValueError("Wrong evaluation checkpoint metadata")
        if em["checkpoint_sha256"] != results[variant]["checkpoint_sha256"]:
            raise ValueError("Changed calibrated artifact")
        with np.load(evaluation / "pointwise_results.npz") as data:
            if not np.isfinite(data["offset_predictions"]).all() or not np.isfinite(data["semantic_prediction_logits"]).all():
                raise FloatingPointError("Nonfinite validation predictions")
        metrics = json.loads((evaluation / "metrics.json").read_text())
        metrics.update(bn_variant=variant, bn_policy=POLICY, total_seconds=em["total_seconds"],
                       pointwise_seconds=em["pointwise_seconds"], checkpoint_sha256=em["checkpoint_sha256"])
        comparison["results"].append(metrics)
        results[variant]["evaluation"] = dict(manifest=em, metrics=metrics)
        write_json(out / "comparison.json", comparison)
        status("evaluation_variant_complete", variant=variant, metrics=metrics)
    columns = ["tag", "checkpoint_update", "detection_f1", "detection_precision", "detection_recall", "coverage",
               "semantic_tree_recall", "offset_xy_error_m", "offset_xyz_error_m", "TP", "FP", "FN", "total_seconds"]
    with (out / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(comparison["results"])
    if sha256(parent) != parent_sha or sha256(stage / "protocol.json") != protocol_sha:
        raise ValueError("Original state changed")
    write_json(out / "result.json", dict(status="complete", policy=POLICY, variants=results,
               total_seconds=time.perf_counter()-began, parent_checkpoint_unchanged=True,
               original_protocol_unchanged=True, frozen_sources=frozen_sources(),
               automatic_training=False, optimizer_steps=0, no_validation_data_for_calibration=True))
    status("complete", comparison=str(out / "comparison.json"), automatic_training=False)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if OUTPUT is not None:
            (OUTPUT / "exception.txt").write_text(traceback.format_exc())
            (OUTPUT / "failure.json").write_text(json.dumps(dict(phase="failed", pid=os.getpid(), updated_unix=time.time()))+"\n")
        raise
