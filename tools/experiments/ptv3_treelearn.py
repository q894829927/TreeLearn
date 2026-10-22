#!/usr/bin/env python
"""Independent CLI. Run from the worktree root using the existing TreeLearn Python."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT))
# Optional, isolated FlashAttention wheel; never install into the user's Conda env.
RUNTIME = Path(os.environ.get("TREELEARN_RUNTIME_PACKAGES", str(ROOT / "work_dirs/ptv3_treelearn/runtime/packages")))
if RUNTIME.exists():
    sys.path.insert(0, str(RUNTIME))


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data")
    audit.add_argument("--source-root", default="data/source")
    audit.add_argument("--output", default="work_dirs/ptv3_treelearn/audit")
    generate = commands.add_parser("generate-crops")
    generate.add_argument("--source-root", default="data/source")
    generate.add_argument("--destination", default="data/experiments/ptv3_treelearn/v1")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--resume-generation", action="store_true")
    generate.add_argument("--generation-workers", type=int, choices=[1, 2, 3, 4], default=1)
    plan = commands.add_parser("plan")
    plan.add_argument("--crop-manifest", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--phase", choices=["smoke", "pilot", "formal"], default="smoke")
    train.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    train.add_argument("--work-dir", required=True)
    train.add_argument("--resume")
    train.add_argument("--max-updates", type=int)
    train.add_argument("--smoke-crop", help="Explicit unverified real crop for <=100 smoke updates only")
    infer = commands.add_parser("infer")
    infer.add_argument("--config", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--forest", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    infer.add_argument("--compact-pointwise", action="store_true", help="Omit unused saved backbone features to reduce host RAM; predictions/grouping unchanged")
    infer.add_argument("--stream-tiles", action="store_true", help="Save one tile at a time using exactly main's tile geometry and transforms")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--gt", required=True)
    evaluate.add_argument("--output", required=True)
    diagnostics = commands.add_parser("diagnose")
    diagnostics.add_argument("--pointwise", required=True)
    diagnostics.add_argument("--pipeline-manifest", required=True)
    diagnostics.add_argument("--gt", required=True)
    diagnostics.add_argument("--output", required=True)
    aggregation = commands.add_parser("aggregate")
    aggregation.add_argument("--index", required=True)
    aggregation.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "audit-data":
        from tree_learn.experiments.ptv3_treelearn.audit import audit_sources
        audit_sources(args.source_root, args.output)
    elif args.command == "generate-crops":
        from tree_learn.experiments.ptv3_treelearn.audit import generate_crops
        generate_crops(args.source_root, args.destination, args.seed, args.resume_generation, args.generation_workers)
    elif args.command == "plan":
        from tree_learn.experiments.ptv3_treelearn.data import create_plan
        create_plan(args.crop_manifest, args.output, args.seed)
    elif args.command == "train":
        from tree_learn.experiments.ptv3_treelearn.train import run
        import torch
        try:
            run(args)
        except torch.cuda.OutOfMemoryError as error:
            from tree_learn.experiments.ptv3_treelearn.provenance import write_json
            write_json(Path(args.work_dir) / "resource_failure.json", {
                "status": "cuda_oom", "error": str(error),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            raise
    elif args.command == "infer":
        from tree_learn.experiments.ptv3_treelearn.pipeline import run
        run(args)
    elif args.command == "evaluate":
        from tree_learn.experiments.ptv3_treelearn.evaluate import run
        run(args)
    elif args.command == "diagnose":
        import json
        from tree_learn.experiments.ptv3_treelearn.diagnostics import diagnose
        manifest = json.loads(Path(args.pipeline_manifest).read_text())
        diagnose(args.pointwise, args.gt, manifest["pointwise_coordinate_translation"], args.output)
    elif args.command == "aggregate":
        from tree_learn.experiments.ptv3_treelearn.aggregate import aggregate
        aggregate(args.index, args.output)


if __name__ == "__main__":
    main()
