"""Bounded D1 micro-fit / D2 scratch sanity; never a formal sample-plan resume.

Two full fixed crops form each accumulation pair. PTv3 arms inherit the same
complete u5000 net/Adam/scaler/RNG; only block LR ratio is different. Epoch LR
time follows a labelled local continuation clock, not a changed formal cursor.
"""
import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

OUTPUT = None


def main():
    global OUTPUT
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root',required=True); p.add_argument('--diagnosis',required=True)
    p.add_argument('--output',required=True); p.add_argument('--block-lr-ratio',type=float,default=1.)
    p.add_argument('--model',choices=['ptv3','unet'],default='ptv3')
    p.add_argument('--gradient-report',action='store_true')
    p.add_argument('--steps',type=int,default=200)
    args=p.parse_args()
    assert 0<args.steps<=200 and args.block_lr_ratio in [1.,.1]
    base,out=Path(args.run_root).resolve(),Path(args.output).resolve()
    diagnostic=Path(args.diagnosis).resolve()
    root=Path(__file__).resolve().parents[2]; helper=root/'tree_learn/experiments/ptv3_vote_v1/__init__.py'
    out.mkdir(parents=True,exist_ok=False); OUTPUT=out
    shutil.copyfile(__file__,out/'executed_source.py');shutil.copyfile(helper,out/'executed_metrics.py')
    source=base/'source_head_fp32_v1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:128,garbage_collection_threshold:0.7'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    sys.path[:0]=[str(source/'work_dirs/ptv3_treelearn/runtime/packages'),str(source)];os.chdir(source)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from munch import Munch
    from tree_learn.experiments.ptv3_treelearn.provenance import (
        sha256,write_json,seed_all,rng_state,restore_rng,state_equal,canonical_hash)
    from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
    from tree_learn.experiments.ptv3_treelearn.model import build_model
    from tree_learn.experiments.ptv3_head_fp32_v1 import enable_head_fp32
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset,sample_digest
    from tree_learn.util.train import build_optimizer,build_cosine_scheduler
    from spconv.pytorch.conv import SparseConvolutionBase
    spec=importlib.util.spec_from_file_location('vote_metrics_v1',helper)
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    stage=base/'stage_validation_v1';cp=stage/'checkpoints/update_5000_head_fp32_v1.pth'
    cfg=json.loads((stage/'training/to_5000_head_fp32_v1/experiment_manifest.json').read_text())['config']
    checkpoint=torch.load(cp,map_location='cpu');assert checkpoint['global_update']==5000
    frozen={str(path):sha256(path) for path in [cp,stage/'protocol.json',stage/'stage_status.json']}
    d0=json.loads((diagnostic/'manifest.json').read_text())
    selected=[r for r in d0['inputs'] if r['case_id'][:2] in ['00','02','01','03']]
    selected.sort(key=lambda r:(r['role']!='small',r['case_id']))
    for i,row in enumerate(selected):row['trial_role']='fit' if i<2 else 'holdout'
    write_json(out/'manifest.json',dict(version='ptv3-vote-microfit-v1',model=args.model,block_lr_ratio=args.block_lr_ratio,
        parent=str(cp) if args.model=='ptv3' else None,parent_sha256=frozen[str(cp)] if args.model=='ptv3' else None,
        maximum_trial_steps=args.steps,formal_update_unchanged=5000,formal_sampler_unchanged=True,inputs=selected,
        precision='Backbone AMP / both original heads FP32',loss='Frozen main 50*CE + 3D Euclidean; own valid-count accumulation weights',
        local_lr_clock='PTv3: epoch=(5000+trial_step-1)//250+1; scratch sanity: epoch=(trial_step-1)//250+1; not a formal sample cursor',
        gate='Both fitting inputs XY error <= 0.5*zero prediction AND vote scatter lower, finite states; local engineering gate only',
        gradient_report_budget=4 if args.gradient_report else 0,
        source_sha256=sha256(out/'executed_source.py'),metrics_sha256=sha256(out/'executed_metrics.py'),
        frozen_hashes=frozen,config=cfg,started_unix=time.time(),pid=os.getpid()))
    seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1.)
    if args.model=='ptv3':model=enable_head_fp32(build_optimized_model(cfg,'block_nested')).cuda()
    else:
        # The native U-Net remains unchanged; explicitly remove PTv3-only options.
        cfg=json.loads(json.dumps(cfg));cfg['model']['backbone']='unet'
        cfg['model'].pop('ptv3');cfg['model'].pop('activation_checkpointing',None)
        model=enable_head_fp32(build_model(cfg)).cuda()
    optimizer=build_optimizer(model,cfg['optimizer'])
    scheduler=build_cosine_scheduler(Munch.fromDict(cfg['scheduler']),optimizer)
    scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=1024.)
    if args.model=='ptv3':
        model.load_state_dict(checkpoint['net'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
    named=list(model.named_parameters());parameters=[v for _,v in named]
    if args.model=='ptv3':
        assert len(named)==459 and all(torch.equal(v.detach().cpu(),checkpoint['net'][n]) for n,v in model.state_dict().items())
        assert state_equal(optimizer.state_dict(),checkpoint['optimizer']) and state_equal(scaler.state_dict(),checkpoint['scaler'])
        mapping=[];old_group=optimizer.param_groups[0];original_ids=checkpoint['optimizer']['param_groups'][0]['params']
        assert len(original_ids)==len(named)==len(old_group['params'])
        assert all(a is b for a,b in zip(old_group['params'],parameters))
        nonblock=[];blocks=[]
        for (name,param),old_id in zip(named,original_ids):
            block=name.startswith('backbone.') and '.block' in name
            (blocks if block else nonblock).append(param)
            mapping.append(dict(name=name,parent_optimizer_id=old_id,shape=list(param.shape),group='block' if block else 'base'))
        assert len(set(map(id,nonblock+blocks)))==459 and set(map(id,nonblock+blocks))==set(map(id,parameters))
        options={k:v for k,v in old_group.items() if k!='params'}
        optimizer.param_groups[:]=[{**options,'params':nonblock},{**options,'params':blocks}]
        optimizer.param_groups[1]['lr']*=args.block_lr_ratio
        optimizer.param_groups[1]['initial_lr']*=args.block_lr_ratio
        # Retain the parent's single-group cosine state. It updates group 0;
        # our explicit versioned adapter scales that exact base value for group 1.
        # This also preserves the multiplier during scalar warmup and lr_min.
        rate_checks=[]
        for epoch in [0,1,20,21,49,50,51,250,999,1000]:
            rate=float(scheduler._get_lr(epoch)[0]); pair=[rate,rate*args.block_lr_ratio]
            assert math.isclose(pair[1]/pair[0],args.block_lr_ratio,rel_tol=1e-14,abs_tol=0.)
            rate_checks.append(dict(epoch=epoch,base=pair[0],block=pair[1]))
        state_exact=all(all(state_equal(optimizer.state[param][key],checkpoint['optimizer']['state'][old_id][key])
            for key in optimizer.state[param]) for (_,param),old_id in zip(named,original_ids))
        assert state_exact
        write_json(out/'parent_restoration.json',dict(net_exact=True,adam_per_parameter_exact=state_exact,scheduler_exact=True,
            scaler_exact=True,parent_rng_will_restore=True,formal_sampler_cursor=int(checkpoint['sampler']['next_sample']),
            exhaustive_tensor_count=len(mapping),base_count=len(nonblock),block_count=len(blocks),mapping=mapping,
            scheduler_ratio_checks=rate_checks,note='Only group metadata/LR policy changes; Adam moments and steps remain exact'))
    bns={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
    initial_bn={n:int(m.num_batches_tracked) for n,m in bns.items()}
    inputs=[]
    for row in selected:
        assert sha256(row['path'])==row['sha256'] and sha256(row['fixture'])==row['fixture_sha256']
        options=dict(cfg['dataset_train']);options.pop('data_root',None)
        ds=PlannedDataset([row['path']],maximum_points=None,**options)
        batch=ds.collate_fn([ds[(0,row['augmentation_seed'])]])
        assert sample_digest(batch)==row['input_digest']
        assert all(a['points_before']==a['points_after'] and a['instance_retention']==1 for a in batch['sampling_audit'])
        f=torch.load(row['fixture'],map_location='cpu')
        inputs.append(dict(row=row,batch=batch,fixture={k:v.cuda() for k,v in f['backbone'].items()},pm=f['point_map'].cuda()))
    active=None
    if args.model=='ptv3':
        def fixed_backbone(self,**kwargs):
            point=self.backbone(dict(active['fixture']))
            assert torch.equal(point.grid_coord,active['fixture']['grid_coord'])
            return SimpleNamespace(features=point.feat),active['pm']
        model.forward_backbone=MethodType(fixed_backbone,model)

    def module_group(name):
        return '.'.join(name.split('.')[:4]) if '.block' in name else '.'.join(name.split('.')[:2])

    if args.gradient_report:
        assert args.model=='ptv3'
        rows=[]
        for active in inputs[:2]:
            for kind in ['semantic','offset']:
                model.load_state_dict(checkpoint['net'],strict=True);model.train();model.zero_grad(set_to_none=True)
                restore_rng(checkpoint['rng']);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                b=active['batch']
                with torch.cuda.amp.autocast():
                    pred=model(b,return_loss=False)
                    sm=b['masks_sem'].cuda();om=b['masks_off'].cuda()
                    semantic=50*F.cross_entropy(pred['semantic_prediction_logits'][sm].float(),b['semantic_labels'].cuda()[sm])
                    offset=torch.linalg.vector_norm(pred['offset_predictions'][om].float()-b['offset_labels'].cuda()[om],dim=1).mean()
                    loss=semantic if kind=='semantic' else offset
                assert torch.isfinite(loss)
                # Match the parent's scale, then report unscaled parameter gradients.
                scale=float(checkpoint['scaler']['scale']);(loss*scale).backward()
                groups={}
                for name,param in named:
                    if param.grad is not None:
                        value=param.grad.detach().double()/scale
                        g=module_group(name);r=groups.setdefault(g,dict(squared_norm=0.,elements=0,nonfinite=0))
                        r['squared_norm']+=float(value.square().sum());r['elements']+=value.numel();r['nonfinite']+=int((~torch.isfinite(value)).sum())
                for r in groups.values():
                    r['norm']=r.pop('squared_norm')**.5;r['rms']=r['norm']/max(r['elements']**.5,1)
                offsets=pred['offset_predictions'].detach().float().cpu().numpy()
                rows.append(dict(case=active['row']['case_id'],kind=kind,loss=float(loss),groups=groups,
                    seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    output_gradient_analysis=metrics.vector_metrics(offsets,b['offset_labels'].numpy(),b['masks_off'].numpy()),
                    optimizer_updates=0,training_mode='Native AMP/backbone DropPath/BN in disposable state; reset net and parent RNG before each independent backward'))
                del pred,semantic,offset,loss;model.zero_grad(set_to_none=True);gc.collect()
        write_json(out/'gradient_report.json',dict(rows=rows,backwards=4,optimizer_updates=0))
        model.load_state_dict(checkpoint['net'],strict=True)
    if args.model=='ptv3':restore_rng(checkpoint['rng'])
    else:seed_all(42)
    initial_rng=rng_state();torch.save(initial_rng,out/'initial_rng.pth')
    snapshots=[]

    def local_evaluate(step):
        nonlocal active
        saved_rng=rng_state();bn_saved={n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in bns.items()}
        model.eval()
        for m in bns.values():m.training=True;m.track_running_stats=False
        # AMP spconv has no eval kernel here. Use native training sparse kernels
        # while only BN uses batch stats and all stochastic layers stay off.
        for m in model.modules():
            if isinstance(m,SparseConvolutionBase):m.training=True
        assert not any(m.training for m in model.modules() if type(m).__name__ in ['DropPath','Dropout'])
        results=[]
        with torch.no_grad(),torch.cuda.amp.autocast():
            for active in inputs:
                seed_all(42+active['row']['plan_position']);b=active['batch'];pred=model(b,return_loss=False)
                offsets=pred['offset_predictions'].float().cpu().numpy();logits=pred['semantic_prediction_logits'].float().cpu().numpy()
                assert np.isfinite(offsets).all() and np.isfinite(logits).all()
                xyz=b['coords'].numpy();gt=b['offset_labels'].numpy();mask=b['masks_off'].numpy();label=b['instance_labels'].numpy()
                seeds=(logits[:,0]>=logits[:,1])&(b['input_feats'].numpy()[:,-1]>.6)&(np.abs(offsets[:,2])<4)&b['masks_inner'].numpy()
                vm=metrics.vector_metrics(offsets,gt,mask);vs=metrics.votes(xyz,gt,offsets,label,mask,seeds)
                passed=vm['xy_improvement_over_zero']>=.5 and vs['mean_vote_scatter_rms_m']<vs['mean_zero_vote_scatter_rms_m']
                results.append(dict(case=active['row']['case_id'],role=active['row']['trial_role'],offset=vm,votes=vs,seeds=int(seeds.sum()),gate=passed))
                del pred;gc.collect()
        bn_unchanged=all(all(torch.equal(v.detach().cpu(),bn_saved[n][k]) for k,v in m.state_dict().items()) for n,m in bns.items())
        assert bn_unchanged
        for m in bns.values():m.track_running_stats=True
        restore_rng(saved_rng);model.train()
        snapshot=dict(trial_step=step,results=results,bn_state_unchanged_by_evaluation=bn_unchanged,
            both_fit_gate=all(r['gate'] for r in results if r['role']=='fit'),updated_unix=time.time())
        snapshots.append(snapshot);write_json(out/f'node_{step:03d}.json',snapshot)
        print(json.dumps(dict(node=step,both_fit_gate=snapshot['both_fit_gate'],
            xy_improvements={r['case']:r['offset']['xy_improvement_over_zero'] for r in results})),flush=True)
        return snapshot

    local_evaluate(0)
    start=time.perf_counter();skips=0;consecutive_skips=0;memory_gate=False
    for step in range(1,args.steps+1):
        epoch=((5000 if args.model=='ptv3' else 0)+step-1)//250+1
        scheduler.step(epoch)
        if args.model=='ptv3':optimizer.param_groups[1]['lr']=optimizer.param_groups[0]['lr']*args.block_lr_ratio
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
        scount=sum(int(x['batch']['masks_sem'].sum()) for x in inputs[:2]);ocount=sum(int(x['batch']['masks_off'].sum()) for x in inputs[:2])
        values={'semantic_loss':0.,'offset_loss':0.}
        for active in inputs[:2]:
            b=active['batch']
            with torch.cuda.amp.autocast():
                _,losses=model(b,return_loss=True)
                ls=losses['semantic_loss']*(int(b['masks_sem'].sum())/scount)
                lo=losses['offset_loss']*(int(b['masks_off'].sum())/ocount)
                loss=ls+lo
            assert torch.isfinite(loss), 'Nonfinite training loss'
            scaler.scale(loss).backward();values['semantic_loss']+=float(ls.detach());values['offset_loss']+=float(lo.detach())
            del loss,losses,ls,lo
        scaler.unscale_(optimizer)
        norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().double().norm() for p in parameters if p.grad is not None]))
        finite=bool(torch.isfinite(norm))
        if finite:torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        previous=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped=scaler.get_scale()<previous
        assert finite or skipped
        skips+=int(skipped);consecutive_skips=consecutive_skips+1 if skipped else 0
        bn_ok=all(int(m.num_batches_tracked)==initial_bn[n]+2*step and bool(torch.isfinite(m.running_mean).all()) and bool(torch.isfinite(m.running_var).all()) for n,m in bns.items())
        assert bn_ok and consecutive_skips<5, 'BN failure or five consecutive AMP rejected steps'
        memory_gate=True
        record=dict(trial_step=step,**values,base_lr=optimizer.param_groups[0]['lr'],block_lr=optimizer.param_groups[-1]['lr'],
            gradient_norm=float(norm) if finite else None,optimizer_step_skipped=skipped,amp_scale=scaler.get_scale(),
            bn_count_and_finiteness_ok=bn_ok,seconds=time.perf_counter()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),full_points=sum(len(x['batch']['coords']) for x in inputs[:2]),
            formal_update=5000,formal_cursor_unchanged=True)
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        write_json(out/'status.json',dict(status='running',trial_step=step,optimizer_skips=skips,updated_unix=time.time(),pid=os.getpid()))
        if step%10==0:print(json.dumps(record),flush=True)
        optimizer.zero_grad(set_to_none=True);gc.collect()
        if step in [50,100,200] or step==args.steps:
            local_evaluate(step)
            torch.save(dict(net=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                rng=rng_state(),trial_step=step,formal_update=5000,not_a_formal_continuation=True,
                fixed_inputs=[r['input_digest'] for r in selected[:2]],block_lr_ratio=args.block_lr_ratio,
                parent_sha256=frozen[str(cp)] if args.model=='ptv3' else None),out/f'trial_{step:03d}.pth')
    after={path:sha256(path) for path in frozen};assert frozen==after
    result=dict(status='complete',model=args.model,block_lr_ratio=args.block_lr_ratio,trial_attempts=args.steps,
        optimizer_steps=args.steps-skips,optimizer_skips=skips,backwards=args.steps*2+(4 if args.gradient_report else 0),
        seconds=time.perf_counter()-start,memory_gate_full_input_passed=memory_gate,formal_checkpoint_and_protocol_unchanged=True,
        final_fit_gate=snapshots[-1]['both_fit_gate'],snapshots=snapshots,completed_unix=time.time())
    write_json(out/'result.json',result);write_json(out/'status.json',dict(status='complete',trial_step=args.steps,updated_unix=time.time(),pid=os.getpid()))


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUTPUT is not None:
            (OUTPUT/'exception.txt').write_text(traceback.format_exc())
            (OUTPUT/'failed.json').write_text(json.dumps(dict(status='failed',pid=os.getpid(),time=time.time())))
        raise
