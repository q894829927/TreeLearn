"""Compare reference and memory-adapted forwards on one fixed full input.

Uses the diagnostic pre-failure state, one frozen voxelization and restored RNG.
Both variants train their BN exactly once, independently. No optimizer steps.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
from types import MethodType, SimpleNamespace


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    base, replay, out = [Path(x).resolve() for x in [args.run_root,args.replay,args.output]]
    if out.exists():
        p.error("Use a new output directory")
    out.mkdir(parents=True)
    shutil.copyfile(__file__,out/"diagnostic_source.py")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.7"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    source = base/"source"
    sys.path[:0] = [str(source/"work_dirs/ptv3_treelearn/runtime/packages"),str(source)]
    os.chdir(source)
    import torch
    from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset,sample_digest
    from tree_learn.experiments.ptv3_treelearn.provenance import seed_all,restore_rng,rng_state,state_equal,sha256
    from tree_learn.model.tree_learn import voxelize
    seed_all(42)
    torch.cuda.set_per_process_memory_fraction(1.)
    state = torch.load(replay/"state_before_target.pth",map_location="cpu")
    cfg = json.loads((replay/"manifest.json").read_text())["config"]
    target = json.loads((replay/"target_input.json").read_text())
    audit = target["sampling_audit"][0]
    options = dict(cfg["dataset_train"]);options.pop("data_root",None)
    ds = PlannedDataset([audit["path"]],maximum_points=None,**options)
    batch = ds.collate_fn([ds[(0,audit["augmentation_seed"])]])
    if sample_digest(batch)!=target["digest"]:
        raise ValueError("Target input changed")
    with torch.no_grad():
        xyz, feats, ids = [batch[k].cuda() for k in ["coords","input_feats","batch_ids"]]
        vf, vi, pm, shape = voxelize(torch.hstack([xyz,feats]),ids,1,.1,False,False,3)
        grid, b = vi[:,1:].int(),vi[:,0].long()
        frozen = dict(feat=vf,grid_coord=grid,coord=(grid.float()+.5)*.1+xyz.min(0).values,
                      batch=b,offset=torch.tensor([len(grid)],device="cuda"))
    torch.save(dict(backbone={k:v.cpu() for k,v in frozen.items()},point_map=pm.cpu()),out/"fixed_voxels.pth")
    def stats(t):
        t=t.detach();finite=bool(torch.isfinite(t).all())
        d=dict(shape=list(t.shape),dtype=str(t.dtype),finite=finite,
               nan=int(torch.isnan(t).sum()),inf=int(torch.isinf(t).sum()))
        if finite:d.update(min=float(t.min()),max=float(t.max()))
        return d
    cpu_features=[];rows=[];end_rng=[]
    for variant in ["reference","block_nested"]:
        model=build_optimized_model(cfg,variant).cuda();model.load_state_dict(state["net"],strict=True);model.train()
        def backbone(self,**kwargs):
            point=self.backbone(dict(frozen))
            return SimpleNamespace(features=point.feat),pm
        model.forward_backbone=MethodType(backbone,model)
        record={"variant":variant}
        def capture_input(module,inputs):
            record["head_input"]=stats(inputs[0]);cpu_features.append(inputs[0].detach().cpu())
        def capture_linear(module,inputs,output):record["semantic_first_linear"]=stats(output)
        hs=[model.semantic_linear[0].register_forward_pre_hook(capture_input),
            model.semantic_linear[0].register_forward_hook(capture_linear)]
        restore_rng(state["rng"]);torch.cuda.reset_peak_memory_stats()
        with torch.cuda.amp.autocast(enabled=cfg["fp16"]):
            loss,losses=model(batch,return_loss=True)
        record.update(semantic_loss=stats(losses["semantic_loss"]),offset_loss=stats(losses["offset_loss"]),
                      total_loss=stats(loss),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      bn_counter_values=sorted(set(int(m.num_batches_tracked) for m in model.modules()
                                                  if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))))
        rows.append(record);end_rng.append(rng_state());print(json.dumps(record),flush=True)
        for h in hs:h.remove()
        del loss,losses,model,hs
        gc.collect();torch.cuda.empty_cache()
    a,b=[x.reshape(-1) for x in cpu_features];max_error=0.;sq_error=sq_ref=0.
    for start in range(0,len(a),1048576):
        aa,bb=a[start:start+1048576].double(),b[start:start+1048576].double();diff=aa-bb
        max_error=max(max_error,float(diff.abs().max()));sq_error+=float(diff.square().sum());sq_ref+=float(aa.square().sum())
    result=dict(status="complete",purpose="Diagnostic forward A/B only, no formal model or optimizer changes",
                state_sha256=sha256(replay/"state_before_target.pth"),
                voxel_fixture_sha256=sha256(out/"fixed_voxels.pth"),point_count=len(pm),voxels=len(grid),
                variants=rows,head_input_max_abs_error=max_error,head_input_relative_l2=(sq_error/max(sq_ref,1e-300))**.5,
                post_forward_rng_equal=state_equal(end_rng[0],end_rng[1]),
                scope="Same diagnostic state, fixed voxels/map, complete crop, training mode, enabled grad, original AMP and restored RNG; no backward or optimizer step.")
    (out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
