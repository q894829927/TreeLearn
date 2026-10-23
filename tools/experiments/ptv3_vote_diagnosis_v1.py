"""D0: eighteen full-input, zero-update forwards with immutable BN/RNG controls.

Runtime dependencies come exclusively from the frozen Head FP32 source. This
versioned diagnostic is loaded separately; it never patches vendor files.
"""
import argparse
import gc
import importlib.util
import json
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
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-raw-first', help='Reuse the completed raw first forward from a failed diagnostic attempt')
    args = parser.parse_args()
    base, out = Path(args.run_root).resolve(), Path(args.output).resolve()
    worktree = Path(__file__).resolve().parents[2]
    helper_path = worktree/'tree_learn/experiments/ptv3_vote_v1/__init__.py'
    out.mkdir(parents=True, exist_ok=False); OUTPUT = out
    shutil.copyfile(__file__, out/'executed_source.py')
    shutil.copyfile(helper_path, out/'executed_metrics.py')
    source = base/'source_head_fp32_v1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.7'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path[:0] = [str(source/'work_dirs/ptv3_treelearn/runtime/packages'), str(source)]
    os.chdir(source)
    import numpy as np
    import torch
    from munch import Munch
    from tree_learn.experiments.ptv3_treelearn.provenance import (
        sha256, write_json, seed_all, rng_state, restore_rng, state_equal)
    from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
    from tree_learn.experiments.ptv3_head_fp32_v1 import enable_head_fp32
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset, sample_digest
    from tree_learn.experiments.ptv3_treelearn.pipeline import safe_instances, safe_assign
    spec = importlib.util.spec_from_file_location('vote_metrics_v1', helper_path)
    metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    stage, cal = base/'stage_validation_v1', base/'diagnosis/bn_calibration_v1'
    cp = stage/'checkpoints/update_5000_head_fp32_v1.pth'
    calibrated = cal/'all_sequential/inference_only.pth'
    checkpoint = torch.load(cp, map_location='cpu')
    assert checkpoint['global_update'] == 5000
    nets = {'original': checkpoint['net'], 'calibrated': torch.load(calibrated, map_location='cpu')['net']}
    del checkpoint
    cfg = json.loads((stage/'training/to_5000_head_fp32_v1/experiment_manifest.json').read_text())['config']
    protocol = json.loads((stage/'protocol.json').read_text())
    all_inputs = json.loads((cal/'calibration_inputs.json').read_text())
    selected = []
    for region in ['A1N', 'G3N', 'LG3']:
        candidates = [r for r in all_inputs if r['source_forest'] == region]
        selected.extend([dict(min(candidates, key=lambda x:x['points']), role='small'),
                         dict(max(candidates, key=lambda x:x['points']), role='large')])
    for i, row in enumerate(selected): row['case_id'] = f'{i:02d}_{row["source_forest"]}_{row["role"]}'
    frozen_paths = [cp, calibrated, stage/'protocol.json', stage/'stage_status.json']
    before_hashes = {str(p):sha256(p) for p in frozen_paths}
    write_json(out/'manifest.json', dict(version='ptv3-vote-d0-v1', pid=os.getpid(), started_unix=time.time(),
        checkpoint=str(cp), checkpoint_sha256=before_hashes[str(cp)], calibrated=str(calibrated),
        inputs=selected, selection='Per A1N/G3N/LG3, smallest/largest by prior metadata, independent of outputs',
        modes=['raw_eval', 'calibrated_eval', 'batch_bn'], ordinary_full_forwards_budget=18,
        optimizer_updates=0, backwards=0, oracle_grouping_budget=6,
        input_policy='Full original crops and exact saved voxel fixtures; no input sampling',
        precision='Global FP32 native evaluation; backbone FlashAttention internal half unchanged',
        bn_policy='Batch branch: only BN.training=True, track_running_stats=False; every mode DropPath/dropout off',
        diagnostic_sampling='Only display points, 256 points per known inner label for feature statistics, and 32 attention patches; full forwards/metrics use all points',
        reused_raw_first=args.reuse_raw_first,
        frozen_hashes=before_hashes, tool_sha256=sha256(out/'executed_source.py'), metrics_sha256=sha256(out/'executed_metrics.py')))
    write_json(out/'status.json', dict(status='running', completed_forwards=0, pid=os.getpid(), updated_unix=time.time()))
    seed_all(42); torch.cuda.set_per_process_memory_fraction(1.)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = enable_head_fp32(build_optimized_model(cfg, 'block_nested')).cuda()
    bns = {n:m for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
    all_results = []; completed = 0; oracle_count = 0

    def scalar_feature(tensor):
        with torch.no_grad():
            x = tensor.detach().float()
            variance, mean = torch.var_mean(x, dim=0, unbiased=False)
            return dict(points=len(x), channels=x.shape[-1], rms=float(torch.sqrt((variance+mean.square()).mean())),
                        common_mean_rms=float(torch.sqrt(mean.square().mean())), centered_rms=float(torch.sqrt(variance.mean())),
                        abs_max=float(x.abs().max()), finite=bool(torch.isfinite(x).all()))

    for row in selected:
        case = out/row['case_id']; case.mkdir()
        assert sha256(row['path']) == row['sha256']
        assert sha256(row['fixture']) == row['fixture_sha256']
        options = dict(cfg['dataset_train']); options.pop('data_root', None)
        dataset = PlannedDataset([row['path']], maximum_points=None, **options)
        batch = dataset.collate_fn([dataset[(0,row['augmentation_seed'])]])
        assert sample_digest(batch) == row['input_digest']
        xyz, labels, gt, sem, inner, off_mask, sem_mask, verticality = [batch[k].numpy() for k in
            ['coords','instance_labels','offset_labels','semantic_labels','masks_inner','masks_off','masks_sem','input_feats']]
        verticality = verticality[:,-1]
        loaded = torch.load(row['fixture'], map_location='cpu')
        pm_cpu = loaded['point_map'].numpy().astype(np.int64)
        grid_cpu = loaded['backbone']['grid_coord'].numpy()
        assert len(pm_cpu) == row['points'] and len(grid_cpu) == row['voxels']
        assert pm_cpu.min() >= 0 and pm_cpu.max() < len(grid_cpu)
        cell_delta = xyz-(grid_cpu[pm_cpu]*.1+xyz.min(0))
        anchor_deviations = []
        for label in np.unique(labels[off_mask]):
            anchors = xyz[(labels==label)]+gt[(labels==label)]
            anchor_deviations.append(float(np.max(np.linalg.norm(anchors-anchors.mean(0,dtype=np.float64),axis=1))))
        audit = dict(input_digest=sample_digest(batch), sampling=batch['sampling_audit'], points=len(xyz), voxels=len(grid_cpu),
            full_input_retained=all(a['points_before']==a['points_after'] and a['instance_retention']==1. for a in batch['sampling_audit']),
            valid_offset_mask_subset=bool(np.all(~off_mask | (inner & sem_mask & (labels>0)))),
            point_map_valid=True, voxel_cell_delta_min=cell_delta.min(0).tolist(), voxel_cell_delta_max=cell_delta.max(0).tolist(),
            voxel_cell_outside_tolerance_points=int(np.any((cell_delta < -2e-5)|(cell_delta > .10002),axis=1).sum()),
            gt_anchor_max_deviation_m=max(anchor_deviations, default=0), gt_definition='frozen main anchor - augmented xyz, metres',
            inner_points=int(inner.sum()), offset_points=int(off_mask.sum()), semantic_points=int(sem_mask.sum()),
            target_nonfinite=int((~np.isfinite(gt[off_mask])).sum()))
        assert audit['full_input_retained'] and audit['valid_offset_mask_subset'] and audit['target_nonfinite']==0
        assert audit['gt_anchor_max_deviation_m'] < 1e-4 and audit['voxel_cell_outside_tolerance_points']==0
        write_json(case/'input_audit.json', audit)
        fixture = {k:v.cuda() for k,v in loaded['backbone'].items()}; pm = loaded['point_map'].cuda()
        del loaded, cell_delta
        def fixed_backbone(self, **kwargs):
            point = self.backbone(dict(fixture))
            # Decoder output is required to recover the exact initial voxel order.
            assert torch.equal(point.grid_coord, fixture['grid_coord'])
            return SimpleNamespace(features=point.feat), pm
        model.forward_backbone = MethodType(fixed_backbone, model)
        nclasses = int(labels.max())+1
        initial_hist = metrics.histograms(pm_cpu, labels, len(grid_cpu), nclasses)
        dominant, voxel_mix = metrics.histogram_report(initial_hist, len(grid_cpu))
        write_json(case/'voxel_mixing.json', voxel_mix)
        # Fixed output-only sample; no changes to input, losses, masks or heads.
        random = np.random.default_rng(42)
        feature_indices = np.concatenate([random.choice(ix, min(256,len(ix)), replace=False)
            for label in np.unique(labels[inner & (labels>=0)])
            if len(ix:=np.flatnonzero(inner & (labels==label)))])
        feature_indices_gpu = torch.from_numpy(feature_indices).cuda()
        seed_all(42+row['plan_position']); fixed_rng = rng_state(); post_rng = None
        mode_results = {}; display = {}
        for mode in ['raw_eval','calibrated_eval','batch_bn']:
            if args.reuse_raw_first and row['case_id'].startswith('00_') and mode=='raw_eval':
                old=Path(args.reuse_raw_first)
                manifest=json.loads((old/'manifest.json').read_text())
                assert manifest['checkpoint_sha256']==before_hashes[str(cp)] and manifest['inputs']==selected
                previous=old/row['case_id']
                for filename in ['raw_eval.json','raw_eval_inner.npz']:
                    shutil.copyfile(previous/filename,case/filename)
                result=json.loads((case/'raw_eval.json').read_text())
                assert result['bn_state_bitwise_unchanged'] and result['post_rng_equal']
                all_results.append(result);mode_results[mode]=result;completed+=1
                continue
            model.load_state_dict(nets['calibrated' if mode=='calibrated_eval' else 'original'], strict=True)
            model.eval()
            for m in bns.values():
                m.training = mode=='batch_bn'; m.track_running_stats = mode!='batch_bn'
            assert not any(m.training for m in model.modules() if type(m).__name__ in ['DropPath','Dropout'])
            bn_before = {n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in bns.items()}
            activations = {}; hist_by_id = {}; layer_inputs = {}; hooks = []; pooling = {}; patches = {}
            detailed = mode=='calibrated_eval'
            # addict/Point may copy nested dict identities while preserving the
            # immutable grid tensor. Track its storage, never mutable Point ids.
            def embedding_pre(module, inputs): hist_by_id[inputs[0].grid_coord.data_ptr()] = initial_hist
            if detailed: hooks.append(model.backbone.embedding.register_forward_pre_hook(embedding_pre))
            def pre_hook(name):
                def hook(module, inputs):
                    p = inputs[0]; x = p if torch.is_tensor(p) else p.feat
                    layer_inputs[name] = scalar_feature(x)
                return hook
            def post_hook(name):
                def hook(module, inputs, output):
                    p = output; x = p if torch.is_tensor(p) else p.feat
                    after = scalar_feature(x); before = layer_inputs.pop(name)
                    activations[name] = dict(before=before, after=after, rms_ratio=after['rms']/max(before['rms'],1e-20))
                    kind = type(module).__name__
                    if kind=='SerializedPooling':
                        parent_hist = hist_by_id[p.pooling_parent.grid_coord.data_ptr()]
                        inverse = p.pooling_inverse.detach().cpu().numpy()
                        gh, lh, wh = parent_hist
                        hist = metrics.histograms(inverse[gh],lh,len(p.feat),nclasses,wh)
                        hist_by_id[p.grid_coord.data_ptr()] = hist
                        _, pooling[name] = metrics.histogram_report(hist,len(p.feat))
                    if kind=='SerializedAttention' and name.endswith('block0.attn'):
                        order = p.serialized_order[module.order_index][p.pad].detach().cpu().numpy()
                        coord = p.coord.detach().cpu().numpy()
                        hist = hist_by_id[p.grid_coord.data_ptr()]; dom,_ = metrics.histogram_report(hist,len(p.feat))
                        spans=[]; purities=[]; count = len(order)//module.patch_size
                        for k in np.unique(np.linspace(0,max(count-1,0),min(count,32),dtype=int)):
                            ix = order[k*module.patch_size:(k+1)*module.patch_size]
                            spans.append(np.ptp(coord[ix],axis=0).tolist())
                            known = dom[ix]; known = known[known>=0]
                            purities.append(float(np.unique(known,return_counts=True)[1].max()/len(known)) if len(known) else None)
                        patches[name] = dict(patch_size=module.patch_size,total_patches=count,sampled_spans_xyz_m=spans,
                                             dominant_voxel_label_purity=purities, minority_count_policy='All labels retained separately in pooling report')
                return hook
            if detailed:
                for name,module in model.named_modules():
                    if type(module).__name__ in ['Block','SerializedPooling','SerializedUnpooling'] or name.endswith(('.cpe','.attn','.mlp')):
                        hooks.extend([module.register_forward_pre_hook(pre_hook(name)),module.register_forward_hook(post_hook(name))])
            restore_rng(fixed_rng); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.no_grad(): pred = model(batch,return_loss=False)
            torch.cuda.synchronize(); seconds = time.perf_counter()-start
            for handle in hooks: handle.remove()
            offsets = pred['offset_predictions'].float().cpu().numpy()
            logits = pred['semantic_prediction_logits'].float().cpu().numpy()
            sampled_feats = pred['backbone_feats'][feature_indices_gpu].float().cpu().numpy()
            assert np.isfinite(offsets).all() and np.isfinite(logits).all()
            post_rng = post_rng or rng_state()
            unchanged = all(all(torch.equal(v.detach().cpu(),bn_before[n][k]) for k,v in m.state_dict().items()) for n,m in bns.items())
            assert unchanged and state_equal(post_rng,rng_state())
            tree = logits[:,0]>=logits[:,1]
            seeds = tree & (verticality>.6) & (np.abs(offsets[:,2])<4) & inner
            gt_seeds = (labels>0) & off_mask & (verticality>.6) & (np.abs(gt[:,2])<4)
            masks = {'all_valid_tree':off_mask, 'gt_trunk_seed':gt_seeds, 'model_seed_known_tree':seeds & off_mask,
                     'crown_relative_anchor_z_ge_4':off_mask & (np.abs(gt[:,2])>=4)}
            vote_rows = {name:metrics.votes(xyz,gt,offsets,labels,mask,seeds) for name,mask in masks.items()}
            result = dict(case_id=row['case_id'],mode=mode,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),bn_state_bitwise_unchanged=unchanged,post_rng_equal=True,
                strata={name:metrics.vector_metrics(offsets,gt,mask) for name,mask in masks.items()}, votes=vote_rows,
                model_seeds=int(seeds.sum()),gt_trunk_seeds=int(gt_seeds.sum()),
                seed_semantic_purity=float((labels[seeds & sem_mask]>0).mean()) if (seeds & sem_mask).any() else None,
                seed_unknown_points=int((seeds & ~sem_mask).sum()),
                semantic_tree_recall=float(tree[sem_mask & (sem==0)].mean()),
                features=metrics.feature_summary(sampled_feats,labels[feature_indices]),
                activations=activations,pooling=pooling,patches=patches)
            write_json(case/(mode+'.json'),result); all_results.append(result); mode_results[mode]=result
            np.savez(case/(mode+'_inner.npz'),coords=xyz[inner],labels=labels[inner],semantic=sem[inner],target=gt[inner],
                     valid_off=off_mask[inner],valid_sem=sem_mask[inner],verticality=verticality[inner],offsets=offsets[inner],logits=logits[inner])
            ix = feature_indices[:10000]; display[mode] = xyz[ix,:2]+offsets[ix,:2]
            completed += 1
            print(json.dumps(dict(case=row['case_id'],mode=mode,forward=completed,seconds=seconds,
                xy=result['strata']['all_valid_tree']['xy_mean_m'],zero_xy=result['strata']['all_valid_tree']['zero_xy_mean_m'],
                seed_xy=result['strata']['gt_trunk_seed'].get('xy_mean_m'),seeds=result['model_seeds'],features=result['features'])),flush=True)
            write_json(out/'status.json',dict(status='running',completed_forwards=completed,case=row['case_id'],mode=mode,pid=os.getpid(),updated_unix=time.time()))
            del pred,offsets,logits,sampled_feats,hist_by_id; gc.collect()
        # Native local grouping plus three cached-array oracles on two inputs.
        # These are partial-crop diagnostic purity/coverage summaries, not formal L1W metrics.
        if row['case_id'].startswith(('00_','02_')):
            arrays = dict(np.load(case/'calibrated_eval_inner.npz'))
            cluster_results=[]
            for kind in ['model','semantic','xy','z']:
                l,o = (arrays['logits'],arrays['offsets']) if kind=='model' else metrics.oracle_arrays(
                    arrays['logits'],arrays['offsets'],arrays['semantic'],arrays['target'],arrays['valid_sem'],arrays['valid_off'],kind)
                start=time.perf_counter()
                groups=safe_instances(arrays['coords'],o,l,Munch(protocol['grouping']),arrays['verticality'])
                seed_clusters=groups.copy();groups=safe_assign(arrays['coords']+o,groups)
                details=[]
                for group in np.unique(groups[groups>0]):
                    known=(groups==group)&arrays['valid_sem']; labs=arrays['labels'][known]
                    ids,counts=np.unique(labs,return_counts=True)
                    details.append(dict(group=int(group),known_points=int(known.sum()),
                        dominant_gt=int(ids[counts.argmax()]) if len(ids) else None,
                        dominant_fraction=float(counts.max()/counts.sum()) if len(ids) else None,
                        positive_tree_ids=int((ids>0).sum())))
                cluster_results.append(dict(kind=kind,seconds=time.perf_counter()-start,groups=details,
                    cluster_count=len(details),seed_cluster_count=len(np.unique(seed_clusters[seed_clusters>0])),
                    diagnostic_only=True,unknown_gt_unchanged=True))
                if kind!='model':oracle_count+=1
            write_json(case/'oracle_grouping.json',cluster_results)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ix = feature_indices[:10000]; colors=np.unique(labels[ix],return_inverse=True)[1]
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        for ax,name,xy in zip(axes,['input','GT votes','calibrated votes','batch BN votes'],
            [xyz[ix,:2],xyz[ix,:2]+gt[ix,:2],display['calibrated_eval'],display['batch_bn']]):
            ax.scatter(xy[:,0],xy[:,1],c=colors,s=1,cmap='tab20');ax.set_title(name);ax.set_aspect('equal')
        fig.suptitle(row['case_id']+'; output display sample only; full input evaluated')
        fig.tight_layout();fig.savefig(case/'votes.png',dpi=150);plt.close(fig)
        del fixture,pm,model.forward_backbone,batch;gc.collect();torch.cuda.empty_cache()
    after_hashes={str(p):sha256(p) for p in frozen_paths}
    assert after_hashes==before_hashes and completed==18 and oracle_count==6
    write_json(out/'result.json',dict(status='complete',completed_forwards=completed,optimizer_updates=0,backwards=0,
        oracle_grouping_replays=oracle_count,frozen_hashes_unchanged=True,rows=all_results,completed_unix=time.time()))
    write_json(out/'status.json',dict(status='complete',completed_forwards=completed,pid=os.getpid(),updated_unix=time.time()))


if __name__=='__main__':
    try: main()
    except BaseException:
        if OUTPUT is not None:
            (OUTPUT/'exception.txt').write_text(traceback.format_exc())
            (OUTPUT/'failed.json').write_text(json.dumps(dict(status='failed',pid=os.getpid(),time=time.time())))
        raise
