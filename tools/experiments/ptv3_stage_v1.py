"""Fixed, inexpensive L1W stage checks; training and Wytham remain separate."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work_dirs/ptv3_treelearn/runtime/packages"))
if __name__ == "__main__":
    required = "max_split_size_mb:128,garbage_collection_threshold:0.7"
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", required) != required:
        raise SystemExit("Conflicting inherited allocator configuration")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = required
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.cuda.set_per_process_memory_fraction(1.0)
    from tree_learn.experiments.ptv3_stage_v1.validation import main
    main()
