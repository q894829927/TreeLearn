"""Explicit migration from the intact v1 update-4600 state; stop at 5000."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(source/"work_dirs/ptv3_treelearn/runtime/packages"), str(source)]
    os.chdir(source)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.7"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import yaml
    from types import SimpleNamespace
    from tree_learn.experiments.ptv3_memory_v1 import formal
    from tree_learn.experiments.ptv3_head_fp32_v1 import POLICY, enable_head_fp32, variant_config
    from tree_learn.experiments.ptv3_treelearn.provenance import canonical_hash, sha256, state_equal, write_json
    base, attempt = Path(args.run_root).resolve(), Path(args.attempt).resolve()
    stage = base/"stage_validation_v1"
    output = stage/"training/to_5000_head_fp32_v1"
    if output.exists() or (attempt/"migration.json").exists():
        raise FileExistsError("This continuation already has training or migration output")
    attempt.mkdir(parents=True, exist_ok=True)
    gates = json.loads((attempt/"gate_approval.json").read_text())
    if not gates["all_required_gates_passed"]:
        raise ValueError("Full-network diagnostic gates must pass before migration")
    if gates["adapter_sha256"] != sha256(source/"tree_learn/experiments/ptv3_head_fp32_v1/__init__.py"):
        raise ValueError("Numerical adapter differs from the validated version")
    for gate in gates["required_results"]:
        if sha256(gate["path"]) != gate["sha256"]:
            raise ValueError("Gate evidence changed")
        result = json.loads(Path(gate["path"]).read_text())
        if not (result["status"] == "passed" and result["all_gradients_present_and_finite"]
                and result["bn_updated_once_per_microbatch"] and result["model_and_bn_finite"]
                and not result["optimizer_step_skipped"]):
            raise ValueError("Unsafe full-network gate")
    parent = stage/"training/to_5000/recovery.pth"
    meta = json.loads((parent.parent/"latest_checkpoint.json").read_text())
    if meta["global_update"] != 4600 or sha256(parent) != meta["sha256"]:
        raise ValueError("The intact original update-4600 checkpoint is required")
    checkpoint = torch.load(parent, map_location="cpu")
    if checkpoint["global_update"] != 4600 or checkpoint["sampler"]["next_sample"] != 9200:
        raise ValueError("Unexpected parent checkpoint/cursor")
    original_config = formal.formal_config()
    cfg = variant_config()
    old_identity = copy.deepcopy(checkpoint["identity"])
    old_paths = sorted([*(base/"source/tree_learn").rglob("*.py"),
                        *(base/"source/tools/experiments").rglob("*.py")])
    original_code = canonical_hash({str(path.relative_to(base/"source")):sha256(path) for path in old_paths})
    if old_identity["config_sha256"] != canonical_hash(original_config) or old_identity["code_sha256"] != original_code:
        raise ValueError("Parent code/config is not the frozen v1 reference")
    new_identity = dict(old_identity, config_sha256=canonical_hash(cfg), code_sha256=formal.reference.code_hash())
    changes = {k:{"from":old_identity[k],"to":new_identity[k]} for k in old_identity if old_identity[k]!=new_identity[k]}
    if set(changes) != {"config_sha256", "code_sha256"}:
        raise ValueError("Only explicit code and numerical-policy configuration migration is allowed")
    checkpoint["identity"] = new_identity
    migrated = attempt/"resume_update_4600_head_fp32_v1.pth"
    torch.save(checkpoint, migrated)
    loaded = torch.load(migrated, map_location="cpu")
    restored = {k:state_equal(loaded[k], checkpoint[k]) for k in
                ("net","optimizer","scheduler","scaler","rng","sampler","global_update","epoch")}
    parent_state = torch.load(parent, map_location="cpu")
    unchanged = {k:state_equal(loaded[k], parent_state[k]) for k in restored}
    if not all(restored.values()) or not all(unchanged.values()):
        raise RuntimeError("State changed during administrative migration")
    receipt = dict(policy=POLICY, parent=str(parent), parent_sha256=sha256(parent),
        migrated=str(migrated), migrated_sha256=sha256(migrated), identity_changes=changes,
        exact_parent_state_preservation=unchanged, global_update=4600, stop_update=5000,
        declaration="New numerical variant from update 4600; not an unchanged-precision continuation",
        source=str(source), gate_approval_sha256=sha256(attempt/"gate_approval.json"),
        implementation_sha256={str(path):sha256(path) for path in
            [Path(__file__).resolve(),source/"tree_learn/experiments/ptv3_head_fp32_v1/__init__.py"]})
    write_json(attempt/"migration.json", receipt)
    del checkpoint, loaded, parent_state
    config_path = attempt/"config.yaml"
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_json(stage/"stage_status.json",dict(status="bounded_training_head_fp32_v1", stop_update=5000,
        start_update=4600, attempt=str(attempt), training_work_dir=str(output), automatic_later_training=False))
    original_validate, original_builder = formal.validate_config, formal.build_optimized_model
    original_run, original_total = formal.reference.run, formal.TOTAL_UPDATES
    def validate(candidate):
        if candidate != cfg: raise ValueError("Head FP32 v1 requires its exact frozen configuration")
    def builder(candidate, variant):
        return enable_head_fp32(original_builder(candidate,variant))
    def bounded(reference_args):
        reference_args.max_updates = 5000
        return original_run(reference_args)
    formal.validate_config, formal.build_optimized_model = validate, builder
    formal.reference.run, formal.TOTAL_UPDATES = bounded, 5000
    try:
        formal.run(SimpleNamespace(config=str(config_path),seed=42,work_dir=str(output),resume=str(migrated)))
        final = output/"formal_final.pth"
        destination = stage/"checkpoints/update_5000_head_fp32_v1.pth"
        os.link(final,destination)
        provenance = json.loads((output/"checkpoint_provenance.json").read_text())
        provenance.update(numerical_policy=POLICY, parent_checkpoint_sha256=receipt["parent_sha256"],
                          migration_receipt=str(attempt/"migration.json"))
        write_json(output/"checkpoint_provenance.json",provenance)
    finally:
        formal.validate_config,formal.build_optimized_model = original_validate,original_builder
        formal.reference.run,formal.TOTAL_UPDATES = original_run,original_total


if __name__ == "__main__": main()
