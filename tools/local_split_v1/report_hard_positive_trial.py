"""Export compact, explicitly fitted-subset E1-R1 results without new training."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tree_learn.local_split_v1 import data
from tree_learn.mask_refinement.data import file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='work_dirs/treelearn_local_split_e1_r1_s42')
    parser.add_argument('--output', default='docs/experiments/assets/local_split_e1_r1_20261022')
    args = parser.parse_args()
    work, output = Path(args.work), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve the previous export; choose a fresh output.')
    result = data.read_json(work / 'audit/summary.json')
    verified = data.read_json(work / 'verified.json')
    if result['signature'] != verified['signature'] or not verified['immutable_inputs_verified']:
        raise ValueError('Incomplete or unverified R1 result.')
    if file_digest(result['checkpoint']['checkpoint']) != result['checkpoint']['checkpoint_sha256']:
        raise ValueError('Changed R1 checkpoint.')
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('manifest.json', 'trained.json', 'verified.json', 'history.json'):
        shutil.copyfile(work / filename, output / filename)
    shutil.copyfile(work / 'audit/summary.json', output / 'summary.json')
    shutil.copyfile(work / 'audit/edge_rates.json', output / 'edge_rates.json')
    original = data.read_json('docs/experiments/assets/local_split_failure_mechanisms_20261022/summary.json')
    old_diagnosis = data.read_json('docs/experiments/assets/local_split_v1_20261022/e1_diagnosis.json')
    rates = data.read_json(work / 'audit/edge_rates.json')
    old_parents = {(r['plot'], r['parent_id']): r for r in old_diagnosis['per_parent']}
    edge_rows = []
    for parent in rates['per_parent']:
        old = old_parents[parent['plot'], parent['parent_id']]
        row = {k: parent[k] for k in ('plot', 'parent_id', 'category')}
        for category in ('same', 'other_tree', 'tree_background'):
            row[category + '_edges'] = parent[category]['count']
            row[category + '_v1_cut'] = old[category + '_cut']
            row[category + '_r1_cut'] = parent[category]['cut']
        edge_rows.append(row)
    with (output / 'parent_edges.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(edge_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(edge_rows)
    forest_rows = []
    for method in ('learned', 'geometry'):
        for forest in result[method]['selected']['forests']:
            row = dict(method=method, plot=forest['plot'], accepted=forest['accepted'],
                changed_points=forest['changed_points'], recovered_gt_ids=str(forest['recovered_gt_ids']),
                lost_gt_ids=str(forest['lost_gt_ids']), fp_delta=forest['fp_delta'], f1_delta_pp=forest['f1_delta_pp'])
            for model in ('baseline', 'refined'):
                row.update({model + '_' + k: forest[model][k] for k in ('tp', 'fp', 'fn', 'f1')})
            forest_rows.append(row)
    with (output / 'forest_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(forest_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(forest_rows)

    hist = rates['score_histogram']
    cdf = np.cumsum(hist['same']) / sum(hist['same'])
    band = int(np.flatnonzero(cdf > .05)[0])
    other_cdf = np.cumsum(hist['other_tree']) / sum(hist['other_tree'])
    bound = dict(max_same_false_cut_rate=.05,
        edge_score_threshold_must_be_below=hist['bins'][band + 1],
        strict_other_tree_detection_upper_bound=float(other_cdf[band]),
        note='Monotonic bound from 0.025-wide score bins. No new threshold chosen or run.')
    write_json(output / 'threshold_bound.json', bound)
    compact = dict(stage='E1-R1', scope=result['scope'], gate=result['gate'],
        additional_optimizer_steps=500, total_v1_plus_r1_optimizer_steps=1000,
        edge_rates=result['edge_rates'], checkpoint_sha256=result['checkpoint']['checkpoint_sha256'],
        source_signature=result['source_signature'], trial_signature=result['signature'],
        thresholds={k: result[k]['selected']['threshold'] for k in ('learned', 'geometry')},
        subset_counts={k: dict(accepted=result[k]['selected']['accepted'],
            changed_points=result[k]['selected']['changed_points'],
            recovered=result[k]['selected']['gate']['recovered'], lost=result[k]['selected']['gate']['lost'],
            macro_f1=result[k]['selected']['gate']['refined_macro']['f1']) for k in ('learned', 'geometry')},
        decisions={k: dict(Counter(r['reason'] for rows in result[k]['partition_reports'].values() for r in rows))
                   for k in ('learned', 'geometry')},
        point_verification=result['point_verification'], threshold_bound=bound,
        eligible_for_advancement=False, e2_executed=False, external_executed=False)
    write_json(output / 'brief.json', compact)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    for axis, category, title, threshold in zip(axes, ('same', 'other_tree', 'tree_background'),
            ('Same-tree false cuts', 'Strict other-tree detection', 'Tree-background detection'), (.05, .8, None)):
        values = [original['totals'][category], rates['totals'][category]]
        y = [r['cut'] / r['count'] for r in values]
        axis.bar(['V1', 'R1'], y, color=['tab:blue', 'tab:orange'])
        for x, value in enumerate(y):
            axis.text(x, value + .02, f'{100 * value:.2f}%', ha='center')
        if threshold is not None:
            axis.axhline(threshold, color='red', linestyle='--', label='Locked gate')
            axis.legend(loc='upper right')
        axis.set(title=title, ylim=(0, 1.12), ylabel='Full-graph edge fraction')
    fig.suptitle('Identical 16 fitted parents | 500 steps each | R1 fails strict other-tree gate')
    fig.savefig(output / 'edge_tradeoff.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    history = data.read_json(work / 'history.json')['history']
    for field in ('positive', 'negative', 'hard_positive'):
        axes[0].plot([r['step'] for r in history], [r[field] for r in history], label=field)
    axes[0].set(title='R1 training BCE components', xlabel='Optimizer step', ylabel='BCE')
    axes[0].legend()
    centers = (np.array(hist['bins'][:-1]) + np.array(hist['bins'][1:])) / 2
    for category in ('same', 'other_tree', 'tree_background'):
        counts = np.array(hist[category])
        axes[1].plot(centers, counts / counts.sum(), label=category)
    axes[1].axvline(.5, color='black', linestyle='--')
    axes[1].set(title='R1 full-graph scores', xlabel='Same-tree connection score', ylabel='Within-class mass')
    axes[1].legend()
    fig.savefig(output / 'training_and_scores.png', dpi=150); plt.close(fig)
    write_json(output / 'delivery_manifest.json', dict(
        source_work=str(work), signature=result['signature'], posthoc=True,
        exporter_sha256=file_digest(__file__), checkpoint_sha256=result['checkpoint']['checkpoint_sha256'],
        files_sha256={p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print('R1_REPORT ' + str(compact), flush=True)


if __name__ == '__main__':
    main()
