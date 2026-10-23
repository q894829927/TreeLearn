"""Read-only BN/precision counterfactuals. Never calibrate or resume training.

Batch-stat modes affect only BN in disposable model instances; DropPath/dropout
remain in eval mode, running buffers stay unchanged, and no optimizer is used.
These are local diagnostics, not replacement L1W evaluation scores.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

_diagnostic_output = None


def main():
    global _diagnostic_output
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--run-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--case",choices=["saved","train","l1w_center","l1w_ne"],required=True)
    p.add_argument("--precision",choices=["fp32","amp"],default="fp32")
    p.add_argument("--sparse-kernel",choices=["evaluation","training"],default="evaluation")
    p.add_argument("--fixture",help="Reuse a diagnostic fixed_voxels.pth with the identical input digest")
    args=p.parse_args()
    base,out=Path(args.run_root).resolve(),Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    _diagnostic_output = out
    shutil.copyfile(__file__,out/"diagnostic_source.py")
    source=base/"source_head_fp32_v1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128,garbage_collection_threshold:0.7"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    sys.path[:0]=[str(source/"work_dirs/ptv3_treelearn/runtime/packages"),str(source)]
    os.chdir(source)
    import numpy as np
    import torch
    from tree_learn.experiments.ptv3_treelearn.provenance import (
        sha256,write_json,seed_all,rng_state,restore_rng,state_equal)
    from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
    from tree_learn.experiments.ptv3_head_fp32_v1 import enable_head_fp32
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset,sample_digest
    from tree_learn.dataset import TreeDataset
    from tree_learn.model.tree_learn import voxelize
    from spconv.pytorch.conv import SparseConvolutionBase
    stage=base/"stage_validation_v1"

    def quantiles(x):
        return np.quantile(x,[0,.01,.5,.99,1]).tolist() if len(x) and np.isfinite(x).all() else None

    if args.case=="saved":
        folders={"u2900":stage/"attempts/v2/evaluations/e1_u2900",
                 "u5000":stage/"continuations/to_5000_head_fp32_v1/evaluation"}
        rows={};old_coords=old_feats=None
        for name,folder in folders.items():
            with np.load(folder/"pointwise_results.npz") as data:
                off=data["offset_predictions"];logits=data["semantic_prediction_logits"]
                coords=data["coords"];feats=data["input_feats"]
            prob=torch.from_numpy(logits).softmax(-1).numpy()[:,0]
            fg=prob>=.5;vert=feats[:,-1]>.6;z=np.abs(off[:,2])<4
            rows[name]=dict(points=len(off),pointwise_sha256=sha256(folder/"pointwise_results.npz"),
                finite=bool(np.isfinite(off).all() and np.isfinite(logits).all()),
                offset_axis_quantiles=[quantiles(off[:,i]) for i in range(3)],
                tree_probability_quantiles=quantiles(prob),foreground=int(fg.sum()),
                verticality_pass=int(vert.sum()),offset_z_pass=int(z.sum()),
                foreground_verticality_pass=int((fg&vert).sum()),seed_points=int((fg&vert&z).sum()))
            if old_coords is None:old_coords,old_feats=coords,feats
            else:
                rows["same_coords_bitwise"]=bool(np.array_equal(coords,old_coords))
                rows["same_input_features_bitwise"]=bool(np.array_equal(feats,old_feats))
        rows["checkpoint_bn"]={}
        for name,path in [("u2900",stage/"checkpoints/update_2900.pth"),
                          ("u4600",stage/"training/to_5000/recovery.pth"),
                          ("u5000",stage/"checkpoints/update_5000_head_fp32_v1.pth")]:
            net=torch.load(path,map_location="cpu")["net"];stats={}
            for k,v in net.items():
                if k.endswith("running_mean"):
                    prefix=k[:-len("running_mean")];variance=net[prefix+"running_var"]
                    stats[prefix[:-1]]=dict(mean=v.tolist(),var=variance.tolist(),
                        gamma=net[prefix+"weight"].tolist(),beta=net[prefix+"bias"].tolist(),
                        count=int(net[prefix+"num_batches_tracked"]),finite=bool(torch.isfinite(v).all() and torch.isfinite(variance).all()))
            rows["checkpoint_bn"][name]=stats
            del net
        write_json(out/"result.json",rows)
        print(json.dumps({k:v for k,v in rows.items() if k!="checkpoint_bn"}),flush=True)
        return

    cfg=json.loads((stage/"training/to_5000_head_fp32_v1/experiment_manifest.json").read_text())["config"]
    cp=stage/"checkpoints/update_5000_head_fp32_v1.pth"
    checkpoint=torch.load(cp,map_location="cpu")
    protocol=json.loads((stage/"protocol.json").read_text())
    if args.case=="train":
        item=json.loads((base/"monitor/failure_5000_report.json").read_text())["inferred_input_plan_for_failed_attempt"][0]
        path=Path(item["crop_manifest_row"]["path"]);seed=item["augmentation_seed"]
        options=dict(cfg["dataset_train"]);options.pop("data_root",None)
        dataset=PlannedDataset([str(path)],maximum_points=None,**options)
        batch=dataset.collate_fn([dataset[(0,seed)]])
    else:
        xy=(0,0) if args.case=="l1w_center" else (12,12)
        tile=min(protocol["tiles"],key=lambda t:sum((t["center"][i]+protocol["translation"][i]-xy[i])**2 for i in [0,1]))
        path=Path(tile["path"]);seed=None
        dataset=TreeDataset(str(path.parent),8,False,logging.getLogger("bn_diagnosis"))
        dataset.data_paths=[str(path)];batch=dataset.collate_fn([dataset[0]])
    seed_all(42);torch.cuda.set_per_process_memory_fraction(1.)
    model=enable_head_fp32(build_optimized_model(cfg,"block_nested")).cuda()
    model.load_state_dict(checkpoint["net"],strict=True);model.eval();del checkpoint
    bn_modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
    bn_saved={n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in bn_modules.items()}
    bn_attributes={n:(m.training,m.track_running_stats) for n,m in bn_modules.items()}
    if args.fixture:
        fixture_path=Path(args.fixture).resolve()
        fixture_manifest=json.loads((fixture_path.parent/"manifest.json").read_text())
        if fixture_manifest["input_digest"]!=sample_digest(batch):raise ValueError("Fixture input differs")
        loaded=torch.load(fixture_path,map_location="cpu")
        fixture={k:v.cuda() for k,v in loaded["backbone"].items()};pm=loaded["point_map"].cuda()
        grid=fixture["grid_coord"]
        if len(pm)!=len(batch["coords"]):raise ValueError("Fixture point map differs")
        os.link(fixture_path,out/"fixed_voxels.pth")
        del loaded
    else:
        with torch.no_grad():
            xyz,feats,ids=[batch[k].cuda() for k in ["coords","input_feats","batch_ids"]]
            vf,vi,pm,shape=voxelize(torch.hstack([xyz,feats]),ids,1,.1,False,False,3)
            grid,b=vi[:,1:].int(),vi[:,0].long()
            fixture=dict(feat=vf,grid_coord=grid,coord=(grid.float()+.5)*.1+xyz.min(0).values,
                         batch=b,offset=torch.tensor([len(grid)],device="cuda"))
        torch.save(dict(backbone={k:v.cpu() for k,v in fixture.items()},point_map=pm.cpu()),out/"fixed_voxels.pth")
    def fixed_backbone(self,**kwargs):
        point=self.backbone(dict(fixture))
        return SimpleNamespace(features=point.feat),pm
    model.forward_backbone=MethodType(fixed_backbone,model)
    seed_all(42);fixed_rng=rng_state()
    write_json(out/"manifest.json",dict(case=args.case,checkpoint=str(cp),checkpoint_sha256=sha256(cp),
        input_path=str(path),input_sha256=sha256(path),augmentation_seed=seed,input_digest=sample_digest(batch),
        points=len(pm),voxels=len(grid),voxel_fixture_sha256=sha256(out/"fixed_voxels.pth"),
        diagnostic_source_sha256=sha256(out/"diagnostic_source.py"),
        protocol_sha256=sha256(stage/"protocol.json"),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(),
        frozen_source=str(source),precision=args.precision,sparse_kernel=args.sparse_kernel,
        fixture_reused_from=str(Path(args.fixture).resolve()) if args.fixture else None,
        input_policy="Complete original input; same frozen voxels/map and restored RNG for every mode",
        scope="Local diagnostic only; no optimizer, no BN running-stat writes, no official evaluation or checkpoint selection",
        bn_policy="Only selected BN modules use batch statistics with track_running_stats=False; dropout and DropPath remain off"))
    selected=batch["masks_inner"].numpy();features=batch["input_feats"].numpy()[:,-1]
    sem_mask=batch["masks_sem"].numpy();off_mask=batch["masks_off"].numpy();labels=batch["semantic_labels"].numpy();gt=batch["offset_labels"].numpy()
    bn_rows={};handles=[]
    def inspect_bn(name):
        def hook(module,inputs):
            with torch.no_grad():
                variance,mean=torch.var_mean(inputs[0].detach().float(),dim=0,unbiased=False)
                saved_mean=bn_saved[name]["running_mean"].to(mean.device)
                saved_var=bn_saved[name]["running_var"].to(mean.device)
                shift=(mean-saved_mean)/torch.sqrt(saved_var+module.eps)
                bn_rows[name]=dict(input_dtype=str(inputs[0].dtype),batch_mean=mean.cpu().tolist(),
                    batch_variance=variance.cpu().tolist(),saved_mean=bn_saved[name]["running_mean"].tolist(),
                    saved_variance=bn_saved[name]["running_var"].tolist(),
                    normalized_mean_shift=shift.cpu().tolist(),max_abs_normalized_mean_shift=float(shift.abs().max()),
                    batch_std_to_running_std_max=float(torch.sqrt((variance+module.eps)/(saved_var+module.eps)).max()),
                    uses_batch_statistics=module.training)
        return hook
    for name,module in bn_modules.items():handles.append(module.register_forward_pre_hook(inspect_bn(name)))
    rows=[];initial_post_rng=None
    # Separate processes avoid confusing spconv's dtype-specific inference caches.
    for precision in [args.precision]:
        for mode in ["eval","head_batch","backbone_batch","all_batch"]:
            model.eval()
            # Optional diagnostic path: spconv 2.3.8 lacks a usable AMP inference
            # kernel for this input. Keep all non-convolution modules in eval.
            if args.sparse_kernel=="training":
                for module in model.modules():
                    if isinstance(module,SparseConvolutionBase):module.training=True
            for name,module in bn_modules.items():
                use=(mode=="all_batch" or (mode=="head_batch" and not name.startswith("backbone."))
                     or (mode=="backbone_batch" and name.startswith("backbone.")))
                module.training=use;module.track_running_stats=not use
            # Check that changing BN mode did not activate stochastic layers.
            assert not any(m.training for n,m in model.named_modules()
                           if type(m).__name__ in ["DropPath","Dropout"])
            restore_rng(fixed_rng);bn_rows.clear();torch.cuda.synchronize();began=time.perf_counter()
            with torch.no_grad(),torch.cuda.amp.autocast(enabled=precision=="amp"):
                pred=model(batch,return_loss=False)
            torch.cuda.synchronize();seconds=time.perf_counter()-began
            offsets=pred["offset_predictions"].float().cpu().numpy();logits=pred["semantic_prediction_logits"].float().cpu()
            probability=logits.softmax(-1).numpy()[:,0];tree=probability>=.5
            initial_post_rng=initial_post_rng or rng_state()
            unchanged=all(all(torch.equal(v.detach().cpu(),bn_saved[n][k]) for k,v in m.state_dict().items()) for n,m in bn_modules.items())
            row=dict(precision=precision,sparse_kernel=args.sparse_kernel,bn_mode=mode,seconds=seconds,
                predictions_finite=bool(np.isfinite(offsets).all() and bool(torch.isfinite(logits).all())),
                offset_axis_quantiles_inner=[quantiles(offsets[selected,i]) for i in range(3)],
                foreground_inner=int((tree&selected).sum()),verticality_inner=int(((features>.6)&selected).sum()),
                offset_z_pass_inner=int(((np.abs(offsets[:,2])<4)&selected).sum()),
                seeds_inner=int((tree&(features>.6)&(np.abs(offsets[:,2])<4)&selected).sum()),
                semantic_tree_recall=float(tree[sem_mask&(labels==0)].mean()) if (sem_mask&(labels==0)).any() else None,
                offset_xy_error_m=float(np.linalg.norm(offsets[off_mask,:2]-gt[off_mask,:2],axis=1).mean()) if off_mask.any() else None,
                offset_xyz_error_m=float(np.linalg.norm(offsets[off_mask]-gt[off_mask],axis=1).mean()) if off_mask.any() else None,
                bn_buffers_bitwise_unchanged=unchanged,post_forward_rng_equal_to_first=state_equal(rng_state(),initial_post_rng),
                bn_diagnostics=dict(bn_rows))
            assert unchanged and row["post_forward_rng_equal_to_first"]
            rows.append(row)
            with (out/"modes.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
            np.savez(out/(precision+"_"+mode+"_inner.npz"),offsets=offsets[selected],logits=logits.numpy()[selected])
            print(json.dumps({k:v for k,v in row.items() if k!="bn_diagnostics"},allow_nan=False),flush=True)
            del pred,offsets,logits
    for h in handles:h.remove()
    for name,module in bn_modules.items():module.training,module.track_running_stats=bn_attributes[name]
    write_json(out/"result.json",dict(status="complete",case=args.case,rows=rows,
        original_checkpoint_unchanged=sha256(cp)==json.loads((out/"manifest.json").read_text())["checkpoint_sha256"]))


if __name__=="__main__":
    try:
        main()
    except BaseException:
        # Never write into an existing output directory rejected by mkdir.
        if _diagnostic_output is not None:
            (_diagnostic_output/"exception.txt").write_text(traceback.format_exc())
        raise
