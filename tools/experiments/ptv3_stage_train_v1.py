"""Administrative stop budget around the immutable formal trainer; no model edits."""
import argparse
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--frozen-source", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--stage-end", type=int, choices=(5000, 12500, 25000), required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    frozen = Path(args.frozen_source).resolve()
    resume = Path(args.resume).resolve()
    required = "max_split_size_mb:128,garbage_collection_threshold:0.7"
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", required) != required:
        parser.error("Conflicting inherited allocator")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = required
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(frozen))
    sys.path.insert(0, str(frozen / "work_dirs/ptv3_treelearn/runtime/packages"))
    os.chdir(frozen)
    import torch
    from types import SimpleNamespace
    from tree_learn.experiments.ptv3_memory_v1 import formal
    from tree_learn.experiments.ptv3_treelearn.provenance import sha256, write_json
    if not (root / "protocol.json").is_file():
        parser.error("Prepare and freeze the L1W development protocol first")
    checkpoint = torch.load(resume, map_location="cpu")
    start = checkpoint["global_update"]
    if not 0 < start < args.stage_end:
        parser.error("The stage must advance an existing trained checkpoint")
    if checkpoint["identity"]["code_sha256"] != formal.reference.code_hash():
        parser.error("Frozen training source does not match the checkpoint")
    del checkpoint
    output = root / "training" / f"to_{args.stage_end}"
    if output.exists():
        parser.error("Use a new explicit attempt directory; existing stages are preserved")
    write_json(root / "bounded_training_manifest.json", dict(
        administrative_source=str(Path(__file__).resolve()), administrative_sha256=sha256(__file__),
        frozen_source=str(frozen), protocol_sha256=sha256(root / "protocol.json"),
        start_update=start, stop_update=args.stage_end, original_model_budget=250000,
        resume=str(resume), resume_sha256=sha256(resume),
        change="Only baseline args.max_updates; configuration and all per-update computation remain unchanged"))
    write_json(root / "stage_status.json", dict(status="bounded_training", pid=os.getpid(),
        training_work_dir=str(output), start_update=start, stop_update=args.stage_end, updated_unix=time.time()))
    original_run = formal.reference.run
    def bounded(reference_args):
        reference_args.max_updates = args.stage_end
        return original_run(reference_args)
    formal.reference.run = bounded
    # This scalar only labels the wrapper status/manifest. The baseline computes
    # identity.total_updates from the unchanged 1000x500 configuration itself.
    formal.TOTAL_UPDATES = args.stage_end
    try:
        formal.run(SimpleNamespace(config=formal.CONFIG, seed=42, work_dir=str(output),
                                   resume=str(resume)))
        final = output / "formal_final.pth"
        saved = root / "checkpoints" / f"update_{args.stage_end}.pth"
        if saved.exists(): raise FileExistsError(saved)
        # Same filesystem: preserve the complete checkpoint without another copy.
        os.link(final, saved)
        write_json(root / "stage_status.json", dict(status="awaiting_stage_evaluation", pid=os.getpid(),
            checkpoint=str(saved), checkpoint_sha256=sha256(saved), checkpoint_update=args.stage_end,
            training_work_dir=str(output), updated_unix=time.time(), automatic_resume=False))
    except BaseException as error:
        write_json(root / "stage_status.json", dict(status="bounded_training_failed", error=str(error),
            training_work_dir=str(output), pid=os.getpid(), updated_unix=time.time()))
        raise
    finally:
        formal.reference.run = original_run


if __name__ == "__main__": main()
