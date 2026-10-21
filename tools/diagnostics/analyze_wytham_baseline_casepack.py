"""Read the user-supplied saved-prediction casepack; CPU statistics and plots only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import io, csv, json, zipfile, hashlib, itertools
from collections import Counter

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--casepack', required=True, type=Path)
parser.add_argument('--summary', type=Path, help='Optional independently supplied summary to verify.')
parser.add_argument('--output', required=True, type=Path, help='New directory; existing results are never overwritten.')
parser.add_argument('--no-plots', action='store_true', help='Only NumPy is required in this mode.')
args = parser.parse_args()
import numpy as np
if not args.no_plots:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

SOURCE = args.casepack
OUT = args.output
OUT.mkdir(parents=True, exist_ok=False)

def convert(value):
    if value == '': return None
    if value in ('True', 'False'): return value == 'True'
    try: return int(value)
    except ValueError:
        try: return float(value)
        except ValueError: return value

def table(z, name):
    return [{k: convert(v) for k,v in row.items()} for row in
            csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig')))]

def quant(values):
    return [float(x) for x in np.quantile(values, [.1,.5,.9])] if len(values) else None

def rms(points):
    return float(np.sqrt(np.square(points - points.mean(0)).sum(1).mean())) if len(points) else None

def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(name, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)

with zipfile.ZipFile(SOURCE) as z:
    if sum(m.file_size for m in z.infolist()) > 64*1024*1024:
        raise ValueError('Unexpected expanded archive size.')
    if z.testzip() is not None: raise ValueError('Corrupt archive.')
    summary=json.loads(z.read('summary.json'))
    if args.summary is not None:
        external=json.loads(args.summary.read_text(encoding='utf-8-sig'))
        if external != summary: raise ValueError('Separate summary differs from archive.')
    trees=table(z,'all_gt_trees.csv'); selected=table(z,'selected_cases.csv')
    predictions=table(z,'baseline_predictions.csv')
    pred_by_id={r['pred_id']:r for r in predictions}
    all_by_id={r['gt_id']:r for r in trees}
    missed=[r for r in trees if not r['detected']]
    detected=[r for r in trees if r['detected']]
    cases={}
    for name in z.namelist():
        if not name.endswith('.npz'): continue
        meta=json.loads(z.read(name[:-4]+'.json'))
        with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as data:
            a={k:data[k].copy() for k in data.files}
        gid=meta['gt_id']; t=a['labels']==gid; n=len(a['labels'])
        assert a['coords'].shape==(n,3) and a['votes_xy'].shape==(n,2)
        assert all(np.isfinite(a[k]).all() for k in ('coords','tree_probability','votes_xy'))
        assert len(np.unique(a['source_indices']))==n
        assert t.sum()==meta['exported_target_points']
        assert np.array_equal(a['tree_probability']>=.5,a['baseline']>0)
        cases[gid]=(meta,a)

assert len(trees)==summary['baseline_metrics']['num_gt']==877
assert len(detected)==596 and len(missed)==281
assert Counter(r['category'] for r in trees)==summary['tree_categories']
assert Counter(r['category'] for r in predictions)['false_positive']==222
assert set(cases)==set(summary['selected_ids'])
for key, values in [('coverage', 'best_iou'), ('point_precision', 'best_precision'),
                    ('point_recall', 'best_recall')]:
    observed=float(np.mean([r[values] for r in trees]))
    if abs(observed-summary['baseline_metrics'][key])>1e-12:
        raise ValueError('Per-GT table does not reproduce '+key)
for gid,(meta,a) in cases.items():
    if len(a['labels'][a['labels']==gid])!=meta['gt_points']: continue
    ids,counts=np.unique(a['baseline'][a['labels']==gid],return_counts=True)
    for pid,nr in zip(ids,counts):
        if pid and pred_by_id[int(pid)]['dominant_gt_id']==gid:
            pr=pred_by_id[int(pid)]
            assert round(pr['dominant_gt_precision']*pr['known_points'])==int(nr)


bins=[]
for lo,hi in ((0,5),(5,10),(10,20),(20,30),(30,float('inf'))):
    rows=[r for r in trees if lo<=r['z_span_m']<hi]
    bins.append(dict(z_span_bin=f'{lo}-{hi}',num_gt=len(rows),detected=sum(r['detected'] for r in rows),
                     missed=sum(not r['detected'] for r in rows),
                     missed_categories=dict(Counter(r['category'] for r in rows if not r['detected']))))

cohorts=[]
for name, rows in [('all_detected',detected),('all_missed',missed)]+[
    (category,[r for r in missed if r['category']==category]) for category in sorted({r['category'] for r in missed})]:
    cohorts.append(dict(cohort=name,count=len(rows),
        median_gt_points=float(np.median([r['gt_points'] for r in rows])),
        median_z_span_m=float(np.median([r['z_span_m'] for r in rows])),
        median_mean_probability=float(np.median([r['saved_tree_probability_mean'] for r in rows])),
        median_foreground_fraction=float(np.median([r['assigned_tree_point_fraction'] for r in rows])),
        median_vote_rms_m=float(np.median([r['saved_vote_xy_rms_spread_m'] for r in rows])),
        add_only_zero_clean_gt_oracle=sum(r['clean_gt_oracle_add_only_original_zero'] for r in rows),
        clean_gt_oracle_locked_masks=sum(r['clean_gt_oracle_possible_without_touching_locked_predictions'] for r in rows)))

support_counts={}
for key in ('max_clean_iou_add_only_original_zero','max_clean_iou_add_only_p_ge_0_1',
            'max_clean_iou_add_only_p_ge_0_2','max_clean_iou_add_only_p_ge_0_3','max_clean_iou_add_only_p_ge_0_4'):
    ids=[r['gt_id'] for r in missed if r[key]>.5]
    support_counts[key]=dict(count=len(ids),gt_ids=ids)

records=[]; fragments=[]; pairs=[]
for gid,(meta,a) in sorted(cases.items()):
    t=a['labels']==gid
    xyz=a['coords'][t]; p=a['tree_probability'][t]; b=a['baseline'][t]; v=a['votes_xy'][t]
    complete=len(xyz)==meta['gt_points']
    row={k:meta[k] for k in ('gt_id','selection_group','gt_points','z_span_m','detected',
         'category','best_pred_id','best_iou','assigned_tree_point_fraction','saved_tree_probability_mean',
         'saved_vote_xy_rms_spread_m','min_target_additions','min_foreign_removals',
         'best_pred_is_locked_tp','max_clean_iou_add_only_original_zero','max_clean_iou_add_only_p_ge_0_3')}
    row.update(target_complete=complete,exported_target_points=len(xyz),
        sampled_probability_q10_q50_q90=quant(p),sampled_xy_span_m=(xyz[:,:2].max(0)-xyz[:,:2].min(0)).tolist(),
        sample_vote_rms_foreground_m=rms(v[b>0]),sample_vote_rms_original_zero_m=rms(v[b==0]))
    for lo,hi in ((0.,.25),(.25,.5),(.5,.75),(.75,1.)):
        rel=(xyz[:,2]-meta['bbox_min'][2])/max(meta['z_span_m'],1e-6)
        mask=(rel>=lo)&(rel<(hi if hi<1 else 1.00001))
        row[f'height_{lo}_{hi}_sample_count']=int(mask.sum())
        row[f'height_{lo}_{hi}_foreground_fraction']=float(np.mean(b[mask]>0)) if mask.any() else None
    records.append(row)
    ids,counts=np.unique(b,return_counts=True)
    order=np.argsort(-counts)
    top=[]
    for j in order:
        pid=int(ids[j]); nr=int(counts[j]); pr=pred_by_id.get(pid,{})
        frag=dict(gt_id=gid,pred_id=pid,sampled_target_points=nr,target_complete=complete,
            fraction_of_sampled_target=nr/len(xyz),prediction_category=pr.get('category','original_zero'),
            matched_gt_id=pr.get('matched_gt_id',0),known_points=pr.get('known_points',0),
            dominant_gt_id=pr.get('dominant_gt_id',0),dominant_gt_precision=pr.get('dominant_gt_precision',None))
        fragments.append(frag)
        if pid and nr/len(xyz)>.02: top.append(pid)
    for pa,pb in itertools.combinations(top,2):
        pra,prb=pred_by_id[pa],pred_by_id[pb]
        u=(b==pa)|(b==pb); common=int(u.sum())
        known=pra['known_points']+prb['known_points']
        va,vb=v[b==pa],v[b==pb]
        rows_all=(a['baseline']==pa)|(a['baseline']==pb)
        # The numerator is exact only when all target points were exported.
        pairs.append(dict(gt_id=gid,pred_a=pa,pred_b=pb,target_complete=complete,
            either_locked_tp=pra['category']=='true_positive' or prb['category']=='true_positive',
            both_false_positive=pra['category']==prb['category']=='false_positive',
            known_union_points=known,observed_target_intersection=common,
            exact_union_iou=common/(meta['gt_points']+known-common) if complete else None,
            observed_target_vote_median_gap_m=float(np.linalg.norm(np.median(va,0)-np.median(vb,0))),
            union_sampled_points=int(rows_all.sum())))

rankings={}
rankings['near_threshold_unlocked']=[dict(r) for r in sorted(
    [r for r in missed if r['best_iou']>=.4 and not r['best_pred_is_locked_tp']],
    key=lambda r:r['min_target_addition_fraction'] if r['min_target_addition_fraction'] is not None else 10)]
rankings['fragmented']=[dict(r) for r in missed if r['category']=='spread_across_predictions']

# Full-forest integer intersections can also be recovered for a prediction's
# dominant GT from its full-forest precision and known point count. These counts
# are independent of whether that GT's display sample was capped.
dominant_groups={}
for pr in predictions:
    gid=pr['dominant_gt_id']
    if pr['category']!='false_positive' or gid not in all_by_id or all_by_id[gid]['detected']:
        continue
    common=round(pr['dominant_gt_precision']*pr['known_points'])
    if abs(common-pr['dominant_gt_precision']*pr['known_points'])>1e-7:
        raise ValueError('Cannot reconstruct an exact dominant-GT integer intersection.')
    dominant_groups.setdefault(gid,[]).append((pr,common))
dominant_unions=[]
for gid,group in sorted(dominant_groups.items()):
    for (pa,ia),(pb,ib) in itertools.combinations(group,2):
        common=ia+ib; known=pa['known_points']+pb['known_points']; g=all_by_id[gid]['gt_points']
        iou=common/(g+known-common)
        if iou>.5:
            dominant_unions.append(dict(gt_id=gid,pred_a=pa['pred_id'],pred_b=pb['pred_id'],
                known_union_points=known,exact_target_intersection=common,exact_union_iou=iou,
                baseline_best_iou=all_by_id[gid]['best_iou'],category=all_by_id[gid]['category'],
                z_span_m=all_by_id[gid]['z_span_m'],both_original_fp=True))
write_csv('full_forest_same_dominant_gt_pair_oracles.csv',dominant_unions)

result=dict(schema=1,posthoc=True,gt_selected=True,protocol=summary['protocol'],
    source_zip_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    summary_json_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest() if args.summary else None,
    archive_summary_equals_uploaded_summary=True if args.summary else None,validated_cases=len(cases),
    baseline_metrics=summary['baseline_metrics'],height_bins=bins,cohorts=cohorts,
    clean_target_support_oracles=support_counts,
    missed_at_half_iou_exactly=sum(r['best_iou']==.5 for r in missed),
    additional_candidate_exposure=summary['saved_signals']['additional_candidate_exposure'],
    full_forest_same_dominant_gt_pair_oracles=dominant_unions,
    distinct_missed_gt_with_two_fp_same_dominant_union_above_half=len({r['gt_id'] for r in dominant_unions}),
    sampled_pair_union_oracles=[p for p in pairs if p['exact_union_iou'] is not None
        and p['exact_union_iou']>.5 and not p['either_locked_tp']],
    notes=['CPU analysis of saved arrays only. No model or synthetic test execution.',
           'GT selects cases and pair-union counterfactuals. These are not deployable decisions or new model scores.',
           'Context and some targets are sampled; sampled prevalence is not full-forest prevalence.',
           'Vote medians in pair table condition on GT target points and cannot be used as a GT-free merge score.',
           'Original correct mask preservation requires a rule independent of GT at deployment.'])
write_json('analysis_summary.json',result)
write_json('selected_case_statistics.json',records)
write_json('ranked_candidates.json',rankings)
write_csv('selected_case_statistics.csv',records)
write_csv('target_prediction_fragments.csv',fragments)
write_csv('pair_union_counterfactuals.csv',pairs)
write_csv('cohort_statistics.csv',cohorts)

# Display targets in their full GT bbox; uniform context is visual aid only.
palette=['#e68128','#2969aa','#ac4792','#20945c','#dfbf2c','#7857a3']
def gallery(ids, name):
    fig,axes=plt.subplots(len(ids),3,figsize=(13,3.15*len(ids)),squeeze=False)
    rng=np.random.default_rng(42)
    for ri,gid in enumerate(ids):
        meta,a=cases[gid]; t=a['labels']==gid
        xyz=a['coords'][t]; p=a['tree_probability'][t]; b=a['baseline'][t]; v=a['votes_xy'][t]
        origin=np.median(xyz[:,:2],axis=0)
        eigvec=np.linalg.eigh(np.cov(xyz[:,:2].T))[1][:,-1]
        if eigvec[0]<0:eigvec=-eigvec
        relz=xyz[:,2]-meta['bbox_min'][2]
        context=np.flatnonzero(~t)
        if len(context)>8000: context=rng.choice(context,8000,replace=False)
        context_xyz=a['coords'][context]; context_lab=a['labels'][context]
        cx=(context_xyz[:,:2]-origin)@eigvec
        cz=context_xyz[:,2]-meta['bbox_min'][2]
        ax=axes[ri,0]
        for mask,col in ((context_lab<0,'#d7d7d7'),(context_lab==0,'#bcaa91'),(context_lab>0,'#7b92a4')):
            ax.scatter(cx[mask],cz[mask],s=.6,c=col,alpha=.35,rasterized=True)
        ax.scatter((xyz[:,:2]-origin)@eigvec,relz,s=2.2,c='#d33b35',alpha=.75,rasterized=True)
        ax.set_ylim(-.5,meta['z_span_m']+1.)
        ax.set_xlabel('Horizontal principal axis (m)');ax.set_ylabel('Relative z (m)')
        ax.set_title(f'GT {gid}: {meta["selection_group"]}\nIoU={meta["best_iou"]:.3f}; z span={meta["z_span_m"]:.2f} m',fontsize=10)
        ax=axes[ri,1]
        sc=ax.scatter((xyz[:,:2]-origin)@eigvec,relz,s=3,c=p,cmap='viridis',vmin=0,vmax=1,rasterized=True)
        ax.set_xlabel('Horizontal principal axis (m)');ax.set_ylabel('Relative z (m)')
        ax.set_title(f'Official tree probability; foreground={meta["assigned_tree_point_fraction"]:.1%}',fontsize=10)
        fig.colorbar(sc,ax=ax,fraction=.035,pad=.02)
        ax=axes[ri,2]
        count=Counter(b.tolist()); top=[pid for pid,nr in count.most_common() if pid][:5]
        for j,pid in enumerate([0]+top):
            mask=b==pid
            if mask.any():
                label='original zero' if pid==0 else f'pred {pid} ({pred_by_id[pid]["category"].replace("true_positive","TP").replace("false_positive","FP")})'
                ax.scatter(v[mask,0]-origin[0],v[mask,1]-origin[1],s=3.2,
                           c='#777777' if pid==0 else palette[(j-1)%len(palette)],alpha=.55,label=label,rasterized=True)
        rest=~np.isin(b,[0]+top)
        if rest.any():ax.scatter(v[rest,0]-origin[0],v[rest,1]-origin[1],s=2,c='#c0c0c0',label='other predictions')
        ax.set_aspect('equal',adjustable='datalim')
        ax.set_xlabel('Official vote x minus target median x (m)');ax.set_ylabel('Official vote y minus target median y (m)')
        ax.set_title(f'Target XY votes, colored by original instance\nRMS={meta["saved_vote_xy_rms_spread_m"]:.3f} m',fontsize=10)
        ax.legend(fontsize=6,loc='best',markerscale=1.5)
        for ax in axes[ri]: ax.grid(alpha=.15)
    fig.suptitle('Saved official TreeLearn cases: GT selected diagnostic views; context is sampled',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(OUT/name,dpi=150);plt.close(fig)

if not args.no_plots:
    gallery([375,419,247,374],'fragmentation_and_control.png')
    gallery([152,507,347,414],'semantic_omissions_and_control.png')
    gallery([190,259,543,129],'contamination_and_borderline.png')
    gallery([157,370,48,611],'additional_impure_cases_and_controls.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=[b['z_span_bin'].replace('inf','+')+' m' for b in bins]
    xx=np.arange(len(bins))
    axes[0].bar(xx,[b['detected'] for b in bins],color='#368b70',label='Detected')
    axes[0].bar(xx,[b['missed'] for b in bins],bottom=[b['detected'] for b in bins],color='#c65945',label='Missed')
    axes[0].set_xticks(xx,labels);axes[0].set_ylabel('Number of GT trees');axes[0].legend()
    axes[0].set_title('Full forest: detection by vertical span')
    for name,col in [('detected','#368b70'),('missed','#c65945')]:
        rr=detected if name=='detected' else missed
        axes[1].scatter([r['assigned_tree_point_fraction'] for r in rr],
                        [r['saved_vote_xy_rms_spread_m'] for r in rr],s=10,c=col,alpha=.6,label=name)
    axes[1].set_xlabel('GT points retained as official foreground');axes[1].set_ylabel('Within-GT XY vote RMS (m)')
    axes[1].set_title('Full forest: distinct semantic and grouping failures');axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/'full_forest_error_structure.png',dpi=160);plt.close(fig)

print(json.dumps(dict(validated_cases=len(cases), baseline_metrics=summary['baseline_metrics'],
    height_bins=bins, distinct_pair_oracle_candidates=result['distinct_missed_gt_with_two_fp_same_dominant_union_above_half'],
    add_only_support_oracle_counts={k:v['count'] for k,v in support_counts.items()}),indent=2))
print('OUTPUT',OUT)
