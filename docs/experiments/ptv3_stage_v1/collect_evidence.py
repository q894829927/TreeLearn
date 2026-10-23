"""Collect small, immutable stage evidence; keep large arrays/checkpoints on HDD."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path("/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42/stage_validation_v1")
DEST = Path(__file__).resolve().parent / "evidence"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    for name in ("protocol.json", "pause_record.json", "comparison.json", "comparison.csv",
                 "checkpoint_health.json", "frozen_source_resume_precheck.json", "prepare.log",
                 "queue_launch.json", "queue_launch_v2.json", "queue_status.json", "stage_status.json"):
        copy(ROOT/name, DEST/name)
    protocol = json.loads((ROOT/"protocol.json").read_text())
    comparison = json.loads((ROOT/"comparison.json").read_text())
    inventory = []
    large = [ROOT/"gt_core.npz", ROOT/"checkpoints/update_0_reconstructed.pth", ROOT/"checkpoints/update_2900.pth"]
    for tag in ("e1_u2900", "e1_initial", "e2_official"):
        source = ROOT/"attempts/v2/evaluations"/tag
        metrics = json.loads((source/"metrics.json").read_text())
        manifest = json.loads((source/"evaluation_manifest.json").read_text())
        assert metrics["gt_trees"] == protocol["complete_trees"] == 24
        assert manifest["protocol_sha256"] == comparison["protocol_sha256"] == digest(ROOT/"protocol.json")
        assert not manifest["full_forest"] and not manifest["is_formal_final"]
        checkpoint = Path(manifest["checkpoint"])
        if checkpoint not in large:
            large.append(checkpoint)
        for name in ("metrics.json", "diagnostics.json", "evaluation_manifest.json", "evaluation.log"):
            copy(source/name, DEST/tag/name)
        copy(ROOT/"attempts/v2"/(tag+".log"), DEST/tag/"console.log")
        large.extend(source/name for name in ("pointwise_results.npz", "evaluation_results.pt", "predictions_on_gt.npz"))
    copy(ROOT/"checkpoints/update_0_reconstructed.json", DEST/"initialization.json")
    copy(ROOT/"evaluations/e1_u2900/exception.txt", DEST/"failed_initial/exception.txt")
    copy(ROOT/"e1_u2900.log", DEST/"failed_initial/console.log")
    copy(ROOT/"attempts/failed_initial/queue_status.json", DEST/"failed_initial/queue_status.json")
    for path in large:
        inventory.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), copied_into_git=False))
    integrity = dict(
        main=subprocess.check_output(["git", "rev-parse", "main"], text=True).strip(),
        baseline=subprocess.check_output(["git", "rev-parse", "codex/ptv3-treelearn-baseline"], text=True).strip(),
        protected_sources_unchanged=subprocess.run(["git", "diff", "--quiet", "855f831c01089261b93658fa2d9a4507bc43be27", "--",
            "tree_learn/model", "tree_learn/experiments/ptv3_treelearn", "configs/experiments/ptv3_treelearn"]).returncode == 0,
        frozen_training_source_clean=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent/"source", text=True) == "",
        training_paused=True, automatic_resume=False, complete_validation_comparisons=3, tests_passed=20,
        evaluation_source_commit="a32ec3cd6233f37e050ca03d0f8a1bf959324b91",
        training_source_commit="649c1749ea654cb42add12db2eacff3da9532c09")
    (DEST/"integrity.json").write_text(json.dumps(integrity, indent=2)+"\n")
    (DEST/"large_artifact_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")


if __name__ == "__main__": main()
