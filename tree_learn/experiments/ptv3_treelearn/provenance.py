"""File, checkpoint and environment provenance for the isolated experiment."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def git_state():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "status": git("status", "--short")}


def environment():
    packages = {}
    for name in ("torch", "numpy", "spconv-cu118", "cumm-cu118", "scikit-learn", "timm", "flash-attn", "laspy", "torch-scatter",
                 "open3d", "jakteristics", "lazrs", "scipy", "pandas", "shapely", "alphashape", "geopandas", "munch", "addict"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip() if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "python_executable": __import__("sys").executable,
            "platform": platform.platform(), "driver": driver,
            "packages": packages, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "spconv_bwd_splitk": os.environ.get("SPCONV_BWD_SPLITK"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "note": "CUDA scatter/spconv kernels can remain nondeterministic; exact restart is measured, not assumed."}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return left == right


def load_initialization(model, path, report_path, kind="hais"):
    """Main's exact-name/exact-shape partial loading, with a strict coverage gate.

    HAIS semantic/instance heads and its 3-channel input convolution are expected
    exclusions. All U-Net and output-layer tensors must load. Official TreeLearn
    checkpoints must match every tensor. No implicit prefix removal/permutation.
    """
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint["net"]
    target = model.state_dict()
    loaded, mismatch, skipped = {}, {}, {}
    for name, tensor in source.items():
        if name not in target:
            skipped[name] = list(tensor.shape)
        elif tensor.shape != target[name].shape:
            mismatch[name] = {"source": list(tensor.shape), "target": list(target[name].shape)}
        else:
            loaded[name] = tensor
    missing = {name: list(t.shape) for name, t in target.items() if name not in loaded}
    allowed = ("input_conv.", "semantic_linear.", "offset_linear.") if kind == "hais" else ()
    unexpected_missing = [name for name in missing if not name.startswith(allowed)]
    expected = sum(t.numel() for n, t in target.items() if not n.startswith(allowed))
    matched = sum(t.numel() for n, t in loaded.items() if not n.startswith(allowed))
    report = {"path": str(Path(path).resolve()), "sha256": sha256(path), "kind": kind,
              "loaded": {n: list(t.shape) for n, t in loaded.items()}, "missing": missing,
              "shape_mismatch": mismatch, "skipped_source": skipped,
              "required_tensor_numel": expected, "matched_required_numel": matched,
              "required_coverage": matched / expected if expected else 0,
              "unexpected_missing": unexpected_missing,
              "is_complete_initialization": not missing and not skipped,
              "status": "pass" if not unexpected_missing and matched == expected and expected > 0
                        and (kind == "hais" or not skipped) else "fail"}
    write_json(report_path, report)
    if report["status"] != "pass":
        raise RuntimeError(f"Initialization coverage failed; see {report_path}")
    model.load_state_dict(loaded, strict=False)
    return report
