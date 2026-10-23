"""CPU evidence index for D0/D1/D2; never loads or changes training code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();base=Path(args.run_root);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    d0=base/'diagnosis/vote_diagnosis_v1/attempt_v2'
    result=json.loads((d0/'result.json').read_text());rows=[]
    for r in result['rows']:
        v=r['strata']['all_valid_tree'];vote=r['votes']['all_valid_tree']
        rows.append(dict(case=r['case_id'],mode=r['mode'],xy_m=v['xy_mean_m'],xyz_m=v['xyz_mean_m'],zero_xy_m=v['zero_xy_mean_m'],
            mean_oracle_xy_m=v['mean_oracle_xy_mean_m'],xy_improvement=v['xy_improvement_over_zero'],
            gt_trunk_xy_m=r['strata']['gt_trunk_seed'].get('xy_mean_m'),model_seed_xy_m=r['strata']['model_seed_known_tree'].get('xy_mean_m'),
            seeds=r['model_seeds'],gt_trunk_seeds=r['gt_trunk_seeds'],trees_with_seed=vote['trees_with_seed'],
            vote_scatter_rms_m=vote['mean_vote_scatter_rms_m'],zero_scatter_rms_m=vote['mean_zero_vote_scatter_rms_m'],
            feature_effective_rank=r['features']['effective_covariance_rank'],feature_common_mean_rms=r['features']['common_mean_rms'],
            seconds=r['seconds'],peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes']))
    with (out/'d0_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'d0_comparison.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    for name in ['manifest.json','status.json']:
        shutil.copyfile(d0/name,out/('d0_'+name))
    trials=[]
    for name in ['lr_1','lr_01','unet_scratch']:
        folder=base/'diagnosis/vote_microfit_v1'/name
        if not (folder/'result.json').exists():continue
        x=json.loads((folder/'result.json').read_text());updates=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
        trial=dict(name=name,model=x['model'],block_lr_ratio=x['block_lr_ratio'],attempts=x['trial_attempts'],optimizer_steps=x['optimizer_steps'],
            optimizer_skips=x['optimizer_skips'],final_fit_gate=x['final_fit_gate'],seconds=x['seconds'],
            peak_training_allocated_bytes=max(u['peak_allocated_bytes'] for u in updates),
            peak_training_reserved_bytes=max(u['peak_reserved_bytes'] for u in updates),
            all_bn_finite_and_expected_count=all(u['bn_count_and_finiteness_ok'] for u in updates),
            all_unscaled_gradient_norms_finite=all(u['gradient_norm'] is not None for u in updates),
            formal_checkpoint_unchanged=x['formal_checkpoint_and_protocol_unchanged'],snapshots=x['snapshots'])
        trials.append(trial)
        shutil.copyfile(folder/'result.json',out/(name+'_result.json'))
        shutil.copyfile(folder/'updates.jsonl',out/(name+'_updates.jsonl'))
        for file in ['manifest.json','parent_restoration.json','gradient_report.json']:
            if (folder/file).exists():shutil.copyfile(folder/file,out/(name+'_'+file))
    (out/'trials.json').write_text(json.dumps(trials,indent=2,allow_nan=False)+'\n')
    nodes=[]
    for t in trials:
        for node in t['snapshots']:
            for r in node['results']:
                nodes.append(dict(trial=t['name'],trial_step=node['trial_step'],case=r['case'],role=r['role'],
                    xy_m=r['offset']['xy_mean_m'],zero_xy_m=r['offset']['zero_xy_mean_m'],xy_improvement=r['offset']['xy_improvement_over_zero'],
                    scatter_m=r['votes']['mean_vote_scatter_rms_m'],zero_scatter_m=r['votes']['mean_zero_vote_scatter_rms_m'],seeds=r['seeds'],gate=r['gate']))
    if nodes:
        with (out/'trial_nodes.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(nodes[0]),lineterminator='\n');w.writeheader();w.writerows(nodes)
    evidence_paths=[d0/'result.json',d0/'manifest.json']+[base/'diagnosis/vote_microfit_v1'/r['name']/'result.json' for r in trials]
    (out/'runtime_index.json').write_text(json.dumps(dict(runtime_base=str(base),
        files=[dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in evidence_paths]),indent=2)+'\n')


if __name__=='__main__':main()
