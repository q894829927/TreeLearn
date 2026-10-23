"""Formal full-crop PTv3 training with the validated, process-local adapter."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work_dirs/ptv3_treelearn/runtime/packages"))

if __name__ == "__main__":
    # These settings must precede even an indirect import of torch.
    required = "max_split_size_mb:128,garbage_collection_threshold:0.7"
    inherited = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if inherited is not None and inherited != required:
        raise SystemExit("Conflicting PYTORCH_CUDA_ALLOC_CONF; start a fresh process")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = required
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from tree_learn.experiments.ptv3_memory_v1.formal import main
    main()
