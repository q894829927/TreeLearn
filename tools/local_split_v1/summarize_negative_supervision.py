"""Summarize the completed negative audit and write an unexecuted R2 proposal."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tree_learn.local_split_v1 import data
from tree_learn.mask_refinement.data import file_digest, write_json
from tools.local_split_v1.audit_negative_supervision import replay_sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default='docs/experiments/assets/local_split_negative_audit_20261022')
    args = parser.parse_args()
    root = Path(args.assets)
    if (root / 'analysis.json').exists():
        raise FileExistsError('Preserve the previous summary.')
    summary = data.read_json(root / 'summary.json')
    parents = data.read_json(root / 'parents.json')
    difficulty = data.read_json(root / 'edge_difficulty.json')
    groups = defaultdict(lambda: dict(count=0, v1_cut=0, r1_cut=0,
        v1_cut_r1_keep=0, v1_keep_r1_cut=0, v1_bce_sum=0., r1_bce_sum=0.))
    for row in difficulty:
        key = (row['dimension'], row['band'], row['edge_class'])
        group = groups[key]
        group['count'] += row['v1']['count']
        for model in ('v1', 'r1'):
            group[model + '_cut'] += row[model]['cut']
            group[model + '_bce_sum'] += row[model]['count'] * row[model]['mean_negative_bce']
        for name in ('v1_cut_r1_keep', 'v1_keep_r1_cut'):
            group[name] += row[name]
    rows = []
    for (dimension, band, category), values in sorted(groups.items()):
        rows.append(dict(dimension=dimension, band=band, edge_class=category, **values,
            v1_detection_rate=values['v1_cut'] / values['count'],
            r1_detection_rate=values['r1_cut'] / values['count']))
    write_json(root / 'difficulty_groups.json', rows)
    with (root / 'difficulty_groups.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    sidecars = data.read_json(root / 'subclass_manifest.json')['records']
    records = []
    for record in sidecars:
        if file_digest(record['sidecar']) != record['sidecar_sha256']:
            raise ValueError('Changed GT subtype sidecar.')
        with np.load(record['sidecar'], allow_pickle=False) as z:
            records.append(dict(record, kind=z['kind']))
    replay, visits = replay_sampler(records)
    if replay != summary['sampler']:
        raise ValueError('Replayed exposure statistics differ.')
    exposures = {}
    for code, name in enumerate(('same', 'other_tree', 'tree_background')):
        counts = np.concatenate([v[r['kind'] == code] for r, v in zip(records, visits)])
        exposures[name] = dict(bank_rows=len(counts), seen=int(np.count_nonzero(counts)),
            minimum=int(counts.min()), median=float(np.median(counts)), maximum=int(counts.max()),
            mean=float(counts.mean()))
    classes = {}
    for name in ('other_tree', 'tree_background'):
        selected = [r for r in rows if r['dimension'] == 'distance' and r['edge_class'] == name]
        classes[name] = {key: sum(r[key] for r in selected) for key in
            ('count', 'v1_cut', 'r1_cut', 'v1_cut_r1_keep', 'v1_keep_r1_cut', 'v1_bce_sum', 'r1_bce_sum')}
    categories = []
    for category in ('mixed', 'correct'):
        selected = [r for r in parents if r['category'] == category]
        categories.append(dict(category=category, strict_negative=sum(r['other_tree_full'] for r in selected),
            v1_cut=sum(r['v1_other_tree_bank_cut'] for r in selected),
            r1_cut=sum(r['r1_other_tree_bank_cut'] for r in selected)))
    result = dict(optimizer_steps=0, classes=classes, bank_row_exposure=exposures, categories=categories,
        positive_bank_cut={model: sum(r[f'{model}_same_bank_cut'] for r in parents) for model in ('v1', 'r1')},
        audit_summary_sha256=file_digest(root / 'summary.json'),
        sampler_note='500-step replay only; no optimizer, backward pass, prediction edit or Wytham access.')
    write_json(root / 'analysis.json', result)
    proposal = dict(schema=1, status='proposed_not_executed', is_training_config=False, trial='E1-R2',
        source_signature=summary['source_signature'], audit_summary_sha256=file_digest(root / 'summary.json'),
        starting_point='fresh seed-42 initialization; compare with V1 and R1, do not resume either checkpoint',
        proposed_additional_optimizer_steps=500, optimizer_steps_executed_this_round=0,
        sole_training_change='Replace the original V1 loss with symmetric mean/hard edge BCE; keep total class weight equal.',
        formula='0.25*(mean_BCE_pos+mean_top25pct_BCE_pos+mean_BCE_neg+mean_top25pct_BCE_neg)',
        hard_fraction=0.25, hard_count='max(1, ceil(class_count*0.25)) independently for each class',
        total_positive_weight=0.5, total_negative_weight=0.5, total_loss_coefficient=1.0,
        contrast='V1 gives mean positive/negative BCE weight 0.5/0.5; R1 adds 0.5 hard-positive BCE. R2 changes the loss formula only.',
        training_negatives='All original label-0 training edges, including strict other-tree and known tree-background; no new GT filtering.',
        subclass_sidecars='diagnostic only; R2 does not require or modify their supervision',
        fixed='identical 16 parents, bank arrays, normalization, sampler, 5265-parameter network, candidates, seeds, graph and policy',
        training=dict(seed=42, optimizer='AdamW', learning_rate=3e-4, weight_decay=1e-3, grad_clip=2, edges_per_step=8192),
        edge_threshold=0.5, gate=dict(max_same_false_cut_rate=.05, min_strict_other_tree_detection_rate=.8,
                                    min_recovered=1, max_lost=0),
        cut_threshold_choices=[None, .1, .2, .3, .4, .5],
        scope='Fitted E1 check only; success is not a generalization result.',
        stop='One setting and one seed only; no automatic E2, Wytham, threshold relaxation or budget increase.',
        hypothesis='Symmetric hard-edge emphasis may improve ordering without the unilateral positive weight increase. It may still fail both edge gates.')
    write_json(root / 'next_trial_spec.json', proposal)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for axis, dimension, labels in [(axes[0], 'distance', ['0-.1', '.1-.2', '.2-.3', '.3-.4', '.4-.6']),
                                    (axes[1], 'feature_rms', ['0-.1', '.1-.25', '.25-.5', '.5-1', '1-2', '>=2'])]:
        selected = sorted([r for r in rows if r['dimension'] == dimension and r['edge_class'] == 'other_tree'], key=lambda r: r['band'])
        for model in ('v1', 'r1'):
            axis.plot(range(len(selected)), [r[model + '_detection_rate'] for r in selected], marker='o', label=model.upper())
        axis.axhline(.8, color='red', linestyle='--', label='Locked minimum')
        axis.set_xticks(range(len(labels)), labels)
        axis.set(xlabel=dimension, ylabel='Strict other-tree detection', ylim=(0, 1.05))
        axis.legend()
    fig.suptitle('All 17,702 strict negative edges were sampled | fixed score cutoff 0.5')
    fig.savefig(root / 'strict_negative_difficulty.png', dpi=150); plt.close(fig)
    print('NEGATIVE_ANALYSIS ' + str(result), flush=True)
    print('NEXT_TRIAL status=proposed_not_executed optimizer_steps_this_round=0', flush=True)


if __name__ == '__main__':
    main()
