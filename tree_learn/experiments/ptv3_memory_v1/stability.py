"""Bounded mixed-crop integration with the unchanged baseline training loop."""
import argparse
import gc
import json
import resource
import shutil
import sys
import time
import traceback
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tree_learn.experiments.ptv3_treelearn import train as reference_train
from tree_learn.experiments.ptv3_treelearn.provenance import sha256, write_json, environment, git_state
from tree_learn.experiments.ptv3_treelearn.vendor.model import Point
from . import build_optimized_model
from .profile import memory
from .run import config

UPDATES = 12
SEEDS = (42, 43, 44)
# p10, p50, p90, largest crop, largest crops in two other forests.
PAIRS = ((3,3), (0,1), (3,0), (2,4), (1,3), (5,2),
         (3,3), (1,0), (0,3), (4,2), (3,1), (2,5))


def selected_crops(manifest):
    if manifest.get("status") != "verified" or len(manifest["crops"]) != 25000:
        raise ValueError("Require the verified 25,000-crop corpus")
    ordered = sorted(enumerate(manifest["crops"]), key=lambda pair:(pair[1]["points"],pair[1]["path"]))
    selected = [ordered[round((len(ordered)-1)*q)] for q in (.1,.5,.9)]
    largest = ordered[-1]
    selected.append(largest)
    forests = {largest[1]["source_forest"]}
    for row in reversed(ordered):
        if row[1]["source_forest"] not in forests:
            selected.append(row); forests.add(row[1]["source_forest"])
        if len(selected) == 6:
            break
    if len(selected) != 6 or len({index for index,_ in selected}) != 6:
        raise ValueError("Expected six distinct real crops")
    return [dict(index=index, **row) for index,row in selected]


def schedule(crops, seed):
    if seed not in SEEDS:
        raise ValueError("Only the three audited seeds are supported")
    rng = np.random.RandomState(seed ^ 0x504D5631)
    augmentation = rng.randint(0,2**32-1,size=UPDATES*2,dtype=np.uint32)
    return [dict(update=i//2+1,micro=i%2+1,role=role,crop_index=crops[role]["index"],
                 points=crops[role]["points"],path=crops[role]["path"],augmentation_seed=int(augmentation[i]))
            for i,role in enumerate(role for pair in PAIRS for role in pair)]


def prepare(output):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    cfg = config(); manifest_path = Path(cfg["experiment"]["crop_manifest"])
    manifest = json.loads(manifest_path.read_text()); crops = selected_crops(manifest)
    for row in crops:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("Changed source crop: "+row["path"])
    (output/"plans").mkdir()
    plans = {}
    for seed in SEEDS:
        source = Path(cfg["experiment"]["sample_plan"].format(seed=seed))
        metadata = json.loads(Path(str(source)+".json").read_text())
        if metadata["seed"] != seed or metadata["crop_manifest_sha256"] != sha256(manifest_path) or metadata["plan_sha256"] != sha256(source):
            raise ValueError("Parent plan provenance mismatch")
        with np.load(source) as old:
            indices, augmentation = old["indices"].copy(), old["augmentation"].copy()
        if indices.shape != (1000,500) or augmentation.shape != indices.shape:
            raise ValueError("Unexpected original plan dimensions")
        rows = schedule(crops,seed)
        indices.flat[:UPDATES*2] = [row["crop_index"] for row in rows]
        augmentation.flat[:UPDATES*2] = [row["augmentation_seed"] for row in rows]
        path = output/"plans"/f"s{seed}.npz"
        np.savez(path,indices=indices,augmentation=augmentation)
        metadata.update(plan_sha256=sha256(path),parent_plan_sha256=sha256(source),audit_only=True,
                        modified_prefix_samples=UPDATES*2,maximum_executable_updates=UPDATES)
        write_json(str(path)+".json",metadata)
        plans[str(seed)] = dict(path=str(path),sha256=sha256(path),samples=rows)
    cfg["experiment"]["sample_plan"] = str(output/"plans"/"s{seed}.npz")
    cfg["experiment"]["memory_adapter"] = "block_nested"
    cfg["experiment"]["stability_audit_only"] = True
    write_json(output/"config.json",cfg)
    write_json(output/"plan.json",dict(status="prepared",updates_per_seed=UPDATES,seeds=SEEDS,
        variant="block_nested",physical_batch=1,accumulation=2,maximum_points=None,
        crop_manifest_sha256=sha256(manifest_path),crops=crops,plans=plans,
        policy="short stress schedule; first 24 samples only; no pilot/formal execution"))


class Probe:
    """Scalar-only observation; no synchronization or tensors retained by hooks."""
    def __init__(self):
        self.model = self.optimizer = None
        self.handles, self.active, self.voxels, self.records = [], [], [], []
        self.points = {}; self.created = self.released = 0
        self.max_allocated = self.max_reserved = 0
        self.last_module = None
        self.current_update = 1

    def init_point(self, original, point, *args, **kwargs):
        original(point,*args,**kwargs)
        key=id(point); self.created+=1
        def release(ref):
            self.points.pop(key,None); self.released+=1
        self.points[key]=weakref.ref(point,release)

    def attach(self, model):
        self.model=model
        for name,module in model.named_modules():
            def before(mod,args,label=name): self.active.append(label or "model")
            def after(mod,args,result,label=name):
                self.last_module=label or "model"; self.active.pop()
            self.handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        def voxel_count(module,args): self.voxels.append(int(args[0]["feat"].shape[0]))
        self.handles.append(model.backbone.register_forward_pre_hook(voxel_count))

    def bn_counts(self):
        return {name:int(module.num_batches_tracked) for name,module in self.model.named_modules()
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)}

    def begin(self):
        self.current_update=len(self.records)+1
        self.voxels=[]; self.before_bn=self.bn_counts()
        torch.cuda.reset_peak_memory_stats()

    def observe(self,record,output):
        gc.collect()
        current=memory()
        self.max_allocated=max(self.max_allocated,current["peak_allocated"])
        self.max_reserved=max(self.max_reserved,current["peak_reserved"])
        after=self.bn_counts()
        bn_once=all(after[name]-count==2 for name,count in self.before_bn.items())
        bn_finite=all(bool(torch.isfinite(buffer).all()) for name,buffer in self.model.named_buffers()
                      if "running_mean" in name or "running_var" in name)
        row=dict(update=record["update"],memory=current,voxels=list(self.voxels),
            point_created=self.created,point_released=self.released,point_alive=len(self.points),
            bn_counts=after,bn_once_per_micro=bn_once,bn_finite=bn_finite,
            loss=record["semantic_loss"]+record["offset_loss"],lr=record["lr"],
            seconds=record["seconds"],amp_scale=record["amp_scale"],
            optimizer_step_skipped=record["optimizer_step_skipped"],gradient_norm=record["gradient_norm"])
        self.records.append(row)
        with (output/"stability_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        if not bn_once or not bn_finite: raise RuntimeError("BatchNorm stability check failed")
        if len(self.voxels)!=2 or self.points: raise RuntimeError("Unexpected forward count or live Point objects")
        if record["optimizer_step_skipped"]: raise FloatingPointError("AMP rejected an update")
        for sample in record["sampling_audit"]:
            if sample["points_before"]!=sample["points_after"]:
                raise RuntimeError("Point reduction is forbidden")


def run(plan_dir,seed):
    root=Path(plan_dir).resolve(); plan=json.loads((root/"plan.json").read_text())
    if plan["updates_per_seed"]!=UPDATES or seed not in SEEDS:
        raise ValueError("This entry point is restricted to 12 updates per seed")
    cfg=reference_train.load_config(root/"config.json")
    expected=config()
    expected["experiment"].update(sample_plan=str(root/"plans"/"s{seed}.npz"),
                                  memory_adapter="block_nested",stability_audit_only=True)
    if cfg!=expected:
        raise ValueError("The stress run may only change the sample plan and adapter metadata")
    if cfg["experiment"].get("maximum_points") is not None or cfg["dataloader"]["train"]["batch_size"]!=1 or cfg["experiment"]["accumulation_steps"]!=2:
        raise ValueError("Require full input, physical batch=1, accumulation=2")
    if sha256(plan["plans"][str(seed)]["path"])!=plan["plans"][str(seed)]["sha256"]:
        raise ValueError("Changed stress schedule")
    output=root/"runs"/f"s{seed}"
    if output.exists(): raise FileExistsError(output)
    probe=Probe(); originals=(reference_train.build_model,reference_train.build_optimizer,reference_train.write_json,Point.__init__)
    def builder(config):
        model=build_optimized_model(config,"block_nested"); probe.attach(model)
        sources=[Path(__file__),Path(__file__).with_name("adapters.py"),Path(reference_train.__file__),Path(sys.argv[0])]
        provenance=dict(seed=seed,variant="block_nested",updates=UPDATES,phase="smoke",pilot=False,
            plan_sha256=sha256(root/"plan.json"),config_sha256=sha256(root/"config.json"),
            git=git_state(),environment=environment(),command=sys.argv,
            source_sha256={str(p):sha256(p) for p in sources},
            integration="unchanged reference_train.run with model factory substitution and scalar telemetry",
            optimizer="unchanged baseline optimizer/scheduler, per-valid-point accumulation, GradScaler and clipping",
            peak_scope="reset after zero_grad each update; run maximum aggregated separately")
        write_json(output/"stability_manifest.json",provenance)
        for index,path in enumerate(sources):
            target=output/"sources"/f"{index}_{path.name}";target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
        return model
    def optimizer_builder(model,options):
        optimizer=originals[1](model,options); probe.optimizer=optimizer
        zero_grad=optimizer.zero_grad
        def zero(*args,**kwargs):
            result=zero_grad(*args,**kwargs);probe.begin();return result
        optimizer.zero_grad=zero
        return optimizer
    def writer(path,value):
        originals[2](path,value)
        if Path(path).name=="monitoring.json": probe.observe(value,output)
    def point_init(point,*args,**kwargs): probe.init_point(originals[3],point,*args,**kwargs)
    reference_train.build_model=builder;reference_train.build_optimizer=optimizer_builder
    reference_train.write_json=writer;Point.__init__=point_init
    started=time.perf_counter();result=dict(status="running",seed=seed,maximum_updates=UPDATES)
    try:
        reference_train.run(SimpleNamespace(config=str(root/"config.json"),seed=seed,work_dir=str(output),
            phase="smoke",max_updates=UPDATES,resume=None,smoke_crop=None))
        if len(probe.records)!=UPDATES: raise RuntimeError("Missing optimizer update records")
        steps=[int(state["step"]) for state in probe.optimizer.state.values() if "step" in state]
        if not steps or min(steps)!=UPDATES or max(steps)!=UPDATES:
            raise RuntimeError("AdamW state did not advance through all updates")
        if not all(bool(torch.isfinite(p).all()) for p in probe.model.parameters()):
            raise FloatingPointError("Nonfinite parameters")
        result.update(status="pass",optimizer_state_steps_min=min(steps),optimizer_state_steps_max=max(steps))
    except BaseException as error:
        result.update(status="cuda_oom" if isinstance(error,torch.cuda.OutOfMemoryError) else "error",
                      error=str(error),failed_update=probe.current_update,active_modules=list(probe.active),
                      last_successful_module=probe.last_module,partial_voxels=list(probe.voxels),failure_memory=memory())
        output.mkdir(parents=True,exist_ok=True);(output/"exception.txt").write_text(traceback.format_exc())
    finally:
        reference_train.build_model,reference_train.build_optimizer,reference_train.write_json,Point.__init__=originals
        for handle in probe.handles:handle.remove()
        gc.collect()
        final_memory=memory()
        result.update(completed_updates=len(probe.records),seconds=time.perf_counter()-started,
            max_allocated=max(probe.max_allocated,final_memory["peak_allocated"]),
            max_reserved=max(probe.max_reserved,final_memory["peak_reserved"]),final_memory=final_memory,
            point_created=probe.created,point_released=probe.released,point_alive=len(probe.points),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_json(output/"stability_result.json",result)
        print(json.dumps(result,indent=2),flush=True)
    if result["status"]!="pass": raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--plan-dir",required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,default=42)
    args=parser.parse_args()
    prepare(args.plan_dir) if args.command=="prepare" else run(args.plan_dir,args.seed)
