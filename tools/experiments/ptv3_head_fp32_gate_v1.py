"""One independent full-input training update; never resume the formal run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--case",choices=["failure","maximum"],required=True)
    p.add_argument("--policy",choices=["legacy","head_fp32"],required=True)
    args=p.parse_args()
    source=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(source/"work_dirs/ptv3_treelearn/runtime/packages"),str(source)]
    os.chdir(source)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128,garbage_collection_threshold:0.7"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import numpy as np
    import torch
    from munch import Munch
    from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
    from tree_learn.experiments.ptv3_head_fp32_v1 import enable_head_fp32,POLICY
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset,sample_digest
    from tree_learn.experiments.ptv3_treelearn.provenance import seed_all,restore_rng,sha256,write_json
    from tree_learn.util.train import build_optimizer,build_cosine_scheduler
    base,out=Path(args.run_root).resolve(),Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,out/"gate_source.py")
    original=base/"stage_validation_v1/training/to_5000"
    cfg=json.loads((original/"experiment_manifest.json").read_text())["config"]
    checkpoint_path=(base/"diagnosis/nonfinite_v1/replay_01/state_before_target.pth"
                     if args.case=="failure" else original/"recovery.pth")
    state=torch.load(checkpoint_path,map_location="cpu")
    if args.case=="failure":
        rows=json.loads((base/"monitor/failure_5000_report.json").read_text())["inferred_input_plan_for_failed_attempt"]
        inputs=[(r["crop_manifest_row"]["path"],r["augmentation_seed"],r["crop_manifest_row"]["sha256"]) for r in rows]
    else:
        crops=json.loads(Path(cfg["experiment"]["crop_manifest"]).read_text())["crops"]
        largest=max(crops,key=lambda x:x["points"])
        if largest["points"]!=2677003:raise ValueError("Unexpected maximum crop")
        inputs=[(largest["path"],3414486137,largest["sha256"])]*2
    for path,seed,digest in inputs:
        if sha256(path)!=digest:raise ValueError("Changed gate input")
    options=dict(cfg["dataset_train"]);options.pop("data_root",None)
    batches=[]
    for path,seed,digest in inputs:
        ds=PlannedDataset([path],maximum_points=None,**options)
        batches.append(ds.collate_fn([ds[(0,seed)]]))
    seed_all(42);torch.cuda.set_per_process_memory_fraction(1.)
    model=build_optimized_model(cfg,"block_nested").cuda()
    if args.policy=="head_fp32":enable_head_fp32(model)
    model.load_state_dict(state["net"],strict=True)
    optimizer=build_optimizer(model,cfg["optimizer"])
    scheduler=build_cosine_scheduler(Munch.fromDict(cfg["scheduler"]),optimizer)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
    start=state["global_update"];restore_rng(state["rng"]);del state
    model.train();scheduler.step(start//250+1);optimizer.zero_grad(set_to_none=True)
    counts0={n:int(m.num_batches_tracked) for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
    result=dict(status="running",case=args.case,policy=args.policy,checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256(checkpoint_path),start_update=start,formal_resume=False,
        inputs=[dict(path=x[0],seed=x[1],sha256=x[2],digest=sample_digest(batch),audit=batch["sampling_audit"])
                for x,batch in zip(inputs,batches)],numerical_policy=POLICY if args.policy=="head_fp32" else "legacy-amp")
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    sem_count=sum(int(b["masks_sem"].sum()) for b in batches)
    off_count=sum(int(b["masks_off"].sum()) for b in batches)
    micro=[];voxels=[]
    handle=model.backbone.register_forward_pre_hook(lambda module,inputs:voxels.append(len(inputs[0]["feat"])))
    try:
        for i,batch in enumerate(batches):
            with torch.cuda.amp.autocast():
                _,losses=model(batch,return_loss=True)
                sem=losses["semantic_loss"]*(int(batch["masks_sem"].sum())/max(sem_count,1))
                off=losses["offset_loss"]*(int(batch["masks_off"].sum())/max(off_count,1));loss=sem+off
            row=dict(microbatch=i+1,finite=bool(torch.isfinite(loss)),
                     semantic_loss=float(sem) if torch.isfinite(sem) else None,
                     offset_loss=float(off) if torch.isfinite(off) else None)
            micro.append(row)
            if not row["finite"]:
                result.update(status="nonfinite_loss",failed_microbatch=i+1);break
            scaler.scale(loss).backward()
        else:
            scaler.unscale_(optimizer)
            grads={n:bool(p.grad is not None and torch.isfinite(p.grad).all()) for n,p in model.named_parameters() if p.requires_grad}
            norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().double().norm() for p in model.parameters() if p.grad is not None]))
            result.update(gradient_count=len(grads),all_gradients_present_and_finite=all(grads.values()),
                          bad_gradient_parameters=[n for n,v in grads.items() if not v],
                          gradient_norm=float(norm) if torch.isfinite(norm) else None)
            if not all(grads.values()):raise FloatingPointError("Nonfinite or missing full-network gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            bias=model.semantic_linear[-1].bias.detach().clone();previous=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            result.update(status="passed",optimizer_step_skipped=scaler.get_scale()<previous,
                          bias_delta_l2=float((model.semantic_linear[-1].bias-bias).norm()),amp_scale=scaler.get_scale())
            if result["optimizer_step_skipped"] or result["bias_delta_l2"]==0:raise RuntimeError("Optimizer did not update")
        counts1={n:int(m.num_batches_tracked) for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
        bn_delta={n:counts1[n]-counts0[n] for n in counts0}
        result.update(bn_deltas=bn_delta,bn_updated_once_per_microbatch=all(v==len(micro) for v in bn_delta.values()),
            model_and_bn_finite=all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()))
        if result["status"]=="passed" and not(result["model_and_bn_finite"] and result["bn_updated_once_per_microbatch"]):
            raise RuntimeError("Model/BN safety invariant failed")
    except BaseException as error:
        result.update(status="failed",error=str(error));(out/"exception.txt").write_text(traceback.format_exc())
    finally:
        handle.remove();torch.cuda.synchronize()
        result.update(microbatches=micro,voxels=voxels,elapsed_seconds=time.perf_counter()-began,
                      allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      allocator_stats={k:torch.cuda.memory_stats()[k] for k in ["num_ooms","num_alloc_retries"]})
        write_json(out/"result.json",result);print(json.dumps(result,allow_nan=False),flush=True)
    if args.policy=="head_fp32" and result["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":main()
