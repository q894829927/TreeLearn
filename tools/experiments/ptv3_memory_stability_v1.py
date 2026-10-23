"""Twelve-update stress checks only; no pilot or formal training mode."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"work_dirs/ptv3_treelearn/runtime/packages"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

if __name__=="__main__":
    from tree_learn.experiments.ptv3_memory_v1.stability import main
    main()
