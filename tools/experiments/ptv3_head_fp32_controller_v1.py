"""One bounded migrated training stage, then the unchanged frozen L1W evaluator."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--run-root",required=True)
    p.add_argument("--attempt",required=True)
    args=p.parse_args()
    base,attempt=Path(args.run_root).resolve(),Path(args.attempt).resolve()
    source=Path(__file__).resolve().parents[2]
    stage=base/"stage_validation_v1"
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open("rb") as f:
            for part in iter(lambda:f.read(8388608),b""):h.update(part)
        return h.hexdigest()
    def write(path,value):
        tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");os.replace(tmp,path)
    if (attempt/"controller_status.json").exists():raise FileExistsError("Existing controller attempt")
    if not json.loads((attempt/"gate_approval.json").read_text())["all_required_gates_passed"]:
        raise ValueError("Required gates did not pass")
    python=sys.executable
    commands=dict(training=[python,"-u",str(source/"tools/experiments/ptv3_head_fp32_train_v1.py"),
                           "--run-root",str(base),"--attempt",str(attempt)],
        evaluation=[python,"-u",str(base/"source_validation_v1/tools/experiments/ptv3_stage_v1.py"),
                    "evaluate","--root",str(stage),"--config","configs/experiments/ptv3_memory_v1/e1_full_formal.yaml",
                    "--checkpoint",str(stage/"checkpoints/update_5000_head_fp32_v1.pth"),
                    "--tag","e1_u5000_head_fp32_v1","--output",str(attempt/"evaluation")],
        training_cwd=str(source),evaluation_cwd=str(base/"source_validation_v1"),stop_update=5000,
        automatic_later_training=False)
    write(attempt/"commands.json",commands)
    write(attempt/"launch.json",dict(controller_pid=os.getpid(),launched_unix=time.time(),start_update=4600,
        stop_update=5000,new_update_attempts=400,source=str(source),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip(),
        controller_sha256=digest(__file__),commands_sha256=digest(attempt/"commands.json"),
        numerical_policy="treelearn-both-heads-fp32-v1",automatic_l1w_evaluation=True,automatic_later_training=False))
    try:
        for phase in ["training","evaluation"]:
            with (attempt/(phase+".log")).open("x") as log:
                child=subprocess.Popen(commands[phase],cwd=commands[phase+"_cwd"],stdout=log,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                write(attempt/"controller_status.json",dict(status=phase,controller_pid=os.getpid(),
                    child_pid=child.pid,command=commands[phase],updated_unix=time.time()))
                code=child.wait()
            if code:raise RuntimeError(phase+" exited with code "+str(code))
            if phase=="training":
                prov=json.loads((stage/"training/to_5000_head_fp32_v1/checkpoint_provenance.json").read_text())
                assert prov["global_update"]==5000 and prov["is_formal_final"] is False
        folder=attempt/"evaluation"
        manifest=json.loads((folder/"evaluation_manifest.json").read_text())
        previous=json.loads((stage/"attempts/v2/evaluations/e1_u2900/evaluation_manifest.json").read_text())
        for key in ["code_sha256","config_sha256","protocol_sha256"]:assert manifest[key]==previous[key]
        assert manifest["protocol_sha256"]==digest(stage/"protocol.json")
        assert manifest["checkpoint_update"]==5000 and manifest["is_formal_final"] is False
        metrics=json.loads((folder/"metrics.json").read_text());assert metrics["gt_trees"]==24
        comparison=json.loads((stage/"comparison.json").read_text())
        comparison["results"].append(dict(**metrics,total_seconds=manifest["total_seconds"],
            pointwise_seconds=manifest["pointwise_seconds"],checkpoint_sha256=manifest["checkpoint_sha256"],
            training_numerical_policy="treelearn-both-heads-fp32-v1",precision_change_update=4600,
            migration_receipt_sha256=digest(attempt/"migration.json")))
        comparison["numerical_variant_note"]="E1 update 5000 switches both training heads to FP32 from the original update-4600 state; inference protocol is unchanged."
        target=stage/"comparison_at_5000_head_fp32_v1.json";write(target,comparison)
        columns=["tag","checkpoint_update","TP","FP","FN","detection_precision","detection_recall",
                 "detection_f1","coverage","semantic_tree_recall","offset_xy_error_m","offset_xyz_error_m",
                 "pointwise_seconds","total_seconds","training_numerical_policy"]
        with target.with_suffix(".csv").open("x",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore",lineterminator="\n")
            writer.writeheader();writer.writerows(comparison["results"])
        done=dict(status="complete",checkpoint_update=5000,comparison=str(target),
                  numerical_policy="treelearn-both-heads-fp32-v1",automatic_later_training=False,updated_unix=time.time())
        write(attempt/"controller_status.json",done)
        write(stage/"stage_status.json",dict(done,status="awaiting_stage_decision",automatic_resume=False))
        print(json.dumps(metrics,indent=2),flush=True)
    except BaseException as error:
        (attempt/"controller_exception.txt").write_text(traceback.format_exc())
        failure=dict(status="continuation_failed",error=str(error),controller_pid=os.getpid(),
                     attempt=str(attempt),updated_unix=time.time(),automatic_restart=False)
        write(attempt/"controller_status.json",failure);write(stage/"stage_status.json",failure)
        raise


if __name__=="__main__":main()
