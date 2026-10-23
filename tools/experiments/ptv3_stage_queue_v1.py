"""Run exactly three fixed regional comparisons, then wait for a stage decision."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runner = Path(__file__).with_name("ptv3_stage_v1.py").resolve()
    jobs = [
        ("e1_u2900", "configs/experiments/ptv3_memory_v1/e1_full_formal.yaml", root/"checkpoints/update_2900.pth"),
        ("e1_initial", "configs/experiments/ptv3_memory_v1/e1_full_formal.yaml", root/"checkpoints/update_0_reconstructed.pth"),
        ("e2_official", "configs/experiments/ptv3_treelearn/e2_official.yaml",
         Path("data/experiments/ptv3_treelearn/weights/model_weights_20241213.pth").resolve()),
    ]
    if (root / "queue_status.json").exists():
        parser.error("This queue already has a recorded attempt; preserve its outputs")
    completed = []
    for tag, config, checkpoint in jobs:
        output = root / "evaluations" / tag
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, "-u", str(runner), "evaluate", "--root", str(root),
                   "--config", config, "--checkpoint", str(checkpoint), "--tag", tag,
                   "--output", str(output)]
        with (root / (tag + ".log")).open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            write(root/"queue_status.json", dict(status="evaluating", controller_pid=os.getpid(),
                child_pid=process.pid, tag=tag, completed=completed, command=command, updated_unix=time.time()))
            code = process.wait()
        if code:
            write(root/"queue_status.json", dict(status="evaluation_failed", tag=tag,
                returncode=code, completed=completed, updated_unix=time.time()))
            raise SystemExit(code)
        completed.append(tag)
        rows = []
        protocols = set()
        for name in completed:
            folder = root/"evaluations"/name
            metrics = json.loads((folder/"metrics.json").read_text())
            manifest = json.loads((folder/"evaluation_manifest.json").read_text())
            protocols.add(manifest["protocol_sha256"])
            rows.append({**metrics, "total_seconds": manifest["total_seconds"],
                         "pointwise_seconds": manifest["pointwise_seconds"],
                         "checkpoint_sha256": manifest["checkpoint_sha256"]})
        if len(protocols) != 1:
            raise ValueError("Results do not share the fixed validation protocol")
        write(root/"comparison.json", dict(role="regional development comparison", results=rows,
              protocol_sha256=next(iter(protocols)), full_forest=False, formal_final=False))
        columns = ["tag", "checkpoint_update", "TP", "FP", "FN", "detection_precision", "detection_recall",
                   "detection_f1", "coverage", "semantic_tree_recall", "offset_xy_error_m",
                   "offset_xyz_error_m", "pointwise_seconds", "total_seconds"]
        with (root/"comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps(rows[-1], indent=2), flush=True)
    write(root/"queue_status.json", dict(status="complete", completed=completed,
        controller_pid=os.getpid(), automatic_training_resume=False, updated_unix=time.time()))
    write(root/"stage_status.json", dict(status="awaiting_stage_decision", completed=completed,
        comparison=str(root/"comparison.json"), automatic_resume=False, updated_unix=time.time()))


if __name__ == "__main__": main()
