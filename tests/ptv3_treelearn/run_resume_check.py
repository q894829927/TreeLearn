"""Actual CUDA uninterrupted-vs-restart test (4 updates, not a mock)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tree_learn.experiments.ptv3_treelearn.provenance import write_json


def run():
    output = ROOT / os.environ.get("TREELEARN_RESUME_OUTPUT", f"work_dirs/ptv3_treelearn/resume_check_{time.time_ns()}")
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / os.environ.get("TREELEARN_RESUME_CONFIG", "configs/experiments/ptv3_treelearn/e1_ptv3_resource.yaml")).read_text())
    config["experiment"]["checkpoint_updates"] = 2
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    base = [sys.executable, "-u", "tools/experiments/ptv3_treelearn.py", "train",
            "--config", str(config_path),
            "--phase", "smoke", "--smoke-crop", "data/source/train/random_crops/npz/A1N_0.npz"]
    for name, count, resume in (("continuous", 4, None), ("resumed", 4, output / "continuous/update_2.pth")):
        command = base + ["--work-dir", str(output/name), "--max-updates", str(count)]
        if resume:
            command += ["--resume", str(resume)]
        with open(output / (name + f"_{count}.log"), "w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    full = torch.load(output / "continuous/last.pth", map_location="cpu")
    resumed = torch.load(output / "resumed/last.pth", map_location="cpu")
    differences = {}
    for name, tensor in full["net"].items():
        other = resumed["net"][name]
        if not torch.equal(tensor, other):
            differences[name] = float((tensor.double()-other.double()).abs().max())
    rows = [json.loads(line) for line in (output / "continuous/updates.jsonl").read_text().splitlines()]
    rows2 = [json.loads(line) for line in (output / "resumed/updates.jsonl").read_text().splitlines()]
    report = {"model_bitwise_equal": not differences, "model_max_abs_diff": max(differences.values(), default=0.),
              "acceptance": {"restored_state": "exact", "cuda_replay_absolute_tolerance": 2e-4, "bitwise_cuda_replay_required": False},
              "checkpoint_state_restoration": json.loads((output / "resumed/resume_restoration.json").read_text()),
              "different_model_tensors": differences, "sampler_equal": full["sampler"] == resumed["sampler"],
              "scheduler_equal": full["scheduler"] == resumed["scheduler"], "scaler_equal": full["scaler"] == resumed["scaler"],
              "torch_rng_equal": torch.equal(full["rng"]["torch"], resumed["rng"]["torch"]),
              "cuda_rng_equal": all(torch.equal(a,b) for a,b in zip(full["rng"]["cuda"], resumed["rng"]["cuda"])),
              "sample_digests_equal": [r["sample_digests"] for r in rows[2:]] == [r["sample_digests"] for r in rows2],
              "global_update": resumed["global_update"], "optimizer_state_bitwise_equal": True}
    a, b = full["optimizer"]["state"], resumed["optimizer"]["state"]
    for key in a:
        for field in a[key]:
            if isinstance(a[key][field], torch.Tensor):
                report["optimizer_state_bitwise_equal"] &= torch.equal(a[key][field], b[key][field])
            else:
                report["optimizer_state_bitwise_equal"] &= a[key][field] == b[key][field]
    # Exact state restoration and identical input plan are mandatory. CUDA kernel
    # floating arithmetic differences are exposed, never relabeled as exact.
    restoration = report["checkpoint_state_restoration"]
    exact = all(restoration[k] for k in ("model_bitwise_equal", "optimizer_bitwise_equal", "scheduler_equal", "scaler_equal",
                                        "python_rng_equal", "numpy_rng_equal", "torch_rng_equal", "cuda_rng_equal"))
    report["status"] = "pass" if exact and all(report[k] for k in ("sampler_equal", "scheduler_equal", "scaler_equal", "torch_rng_equal", "cuda_rng_equal", "sample_digests_equal")) and report["model_max_abs_diff"] < 2e-4 else "fail"
    write_json(output / "result.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise RuntimeError("CUDA restart equivalence failed")


if __name__ == "__main__":
    run()
