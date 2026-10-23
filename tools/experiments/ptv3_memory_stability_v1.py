"""Twelve-update stress checks only; no pilot or formal training mode."""
import os
import sys
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"work_dirs/ptv3_treelearn/runtime/packages"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

if __name__=="__main__":
    early=argparse.ArgumentParser(add_help=False)
    settings={"default":None,"split128":"max_split_size_mb:128",
              "gc70":"garbage_collection_threshold:0.7",
              "split128_gc70":"max_split_size_mb:128,garbage_collection_threshold:0.7"}
    early.add_argument("--allocator",choices=tuple(settings),default="default")
    selected,_=early.parse_known_args()
    required=settings[selected.allocator]
    existing=os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if existing is not None and existing!=required:
        early.error("Conflicting inherited PYTORCH_CUDA_ALLOC_CONF; use a fresh subprocess environment")
    if required is not None:os.environ["PYTORCH_CUDA_ALLOC_CONF"]=required
    from tree_learn.experiments.ptv3_memory_v1.stability import main
    main()
