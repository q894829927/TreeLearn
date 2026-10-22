"""Export compact source diagnostics/figures after a gated local-split run."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tree_learn.local_split_v1 import data
from tree_learn.mask_refinement.data import file_digest, load_cache, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/full.yaml')
    parser.add_argument('--output', default='docs/experiments/assets/local_split_v1_20261022')
    args = parser.parse_args()
    config = data.load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    work = Path(config['work_dir'])
    e0 = data.read_json(work / 'e0/summary.json')
    if e0['signature'] != data.signature(config):
        raise ValueError('Report does not match current source/config.')
    stages = dict(E0=e0)
    for stage in ('e1', 'e2', 'e3'):
        path = work / stage / 'summary.json'
        if path.exists():
            stages[stage.upper()] = data.read_json(path)
    for stage, result in stages.items():
        write_json(output / f'{stage.lower()}_summary.json', result)
    fields = ['split', 'plot', 'num_parents', 'num_candidates', 'votes', 'geometry',
              'reliable', 'recoverable_candidates', 'selected_original_tp']
    with (output / 'forests.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for forest in e0['forests']:
            writer.writerow({k: forest['entrances'].get(k, 0) if k in ('votes', 'geometry') else forest[k] for k in fields})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout='constrained')
    labels = [r['plot'] for r in e0['forests']]
    x = np.arange(len(labels))
    a = np.array([r['entrances'].get('votes', 0) for r in e0['forests']])
    b = np.array([r['entrances'].get('geometry', 0) for r in e0['forests']])
    axes[0].bar(x, a, label='Vote candidates')
    axes[0].bar(x, b, bottom=a, label='Geometry candidates')
    axes[1].bar(x - .2, [r['reliable'] for r in e0['forests']], .4, label='Reliable dual seeds (posthoc)')
    axes[1].bar(x + .2, [r['selected_original_tp'] for r in e0['forests']], .4, label='Selected original TP (risk, not lost)')
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45)
        axis.axvline(12.5, color='black', linestyle='--')
        axis.legend()
        axis.set_ylabel('Parents')
    fig.suptitle('E0 | 13 source training forests / 5 development validation forests')
    fig.savefig(output / 'e0_candidates.png', dpi=150)
    plt.close(fig)
    all_candidates = [c for f in e0['forests'] for c in f['candidates']]
    examples = []
    for split in ('train', 'validation'):
        for reliable in (True, False):
            options = [c for c in all_candidates if c['split'] == split and c['reliable'] == reliable
                       and (reliable or c['entrance'] == 'geometry')]
            if options:
                examples.append(options[0])
    manifests = []
    for case in examples:
        row, = [r for r in data.rows(config) if r['plot'] == case['plot']]
        prepared = data.load_prepared(row, config)
        record, = [r for r in prepared['records'] if r['parent_id'] == case['parent_id']]
        arrays, _ = load_cache(row['cache'], with_labels=True)
        try:
            payload = data.load_parent(row, record, arrays)
            indices = payload['indices']
            xyz = arrays['coords'][indices].astype(float)
            votes = arrays['votes_xy'][indices].astype(float)
            gt = arrays['labels'][indices]
            positive, counts = np.unique(gt[gt > 0], return_counts=True)
            important = positive[np.argsort(-counts, kind='stable')[:6]]
            color = np.full(len(gt), -1)
            color[gt == 0] = -2
            for k, identifier in enumerate(important):
                color[gt == identifier] = k
            palette = plt.get_cmap('tab10')
            colors = np.array([palette(k % 10) if k >= 0 else (.65, .65, .65, .5) if k == -1 else (0, 0, 0, .6) for k in color])
            center = np.median(xyz, axis=0)
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout='constrained')
            for axis, values, name in [(axes[0], xyz[:, [0, 1]] - center[[0, 1]], 'XY'),
                (axes[1], xyz[:, [0, 2]] - center[[0, 2]], 'XZ'),
                (axes[2], votes - center[:2], 'Official XY votes')]:
                axis.scatter(*values.T, s=1, c=colors, alpha=.65, rasterized=True)
                axis.set_title(name + ' | GT posthoc colors')
                axis.set_aspect('equal', adjustable='datalim')
            axes[3].scatter(*(xyz[:, [0, 2]] - center[[0, 2]]).T, s=1, c='.8', alpha=.5)
            for k in (0, 1):
                at = payload[f'seed{k}']
                axes[3].scatter(*(xyz[at][:, [0, 2]] - center[[0, 2]]).T, s=5, label=f'Observable seed {k}')
            axes[3].legend()
            axes[3].set_title('XZ | input-only seeds')
            axes[3].set_aspect('equal', adjustable='datalim')
            fig.suptitle(f'{case["split"]}/{case["plot"]} parent {case["parent_id"]} | {case["entrance"]} | reliable={case["reliable"]} | full core N={len(indices)}')
            filename = f'{case["plot"]}_{case["parent_id"]}_seeds.png'
            fig.savefig(output / filename, dpi=150)
            plt.close(fig)
            manifests.append(dict(plot=case['plot'], parent_id=case['parent_id'], file=filename,
                core_points=len(indices), gt_color_ids=important.tolist(), posthoc=True,
                payload_sha256=record['payload_sha256']))
        finally:
            data.close_arrays(arrays)
    log = Path('logs/local_split_v1/pipeline.log')
    if log.exists():
        values = re.findall(r'TRAIN e1 step=(\d+)/\d+ loss=([\d.]+)', log.read_text())
        if values:
            fig, axis = plt.subplots(figsize=(7, 4), layout='constrained')
            axis.plot([int(a) for a, _ in values], [float(b) for _, b in values], marker='.')
            axis.set(xlabel='Optimizer step', ylabel='Balanced BCE', title='E1 fixed-parent fit | training loss, not generalization')
            fig.savefig(output / 'e1_loss.png', dpi=150)
            plt.close(fig)
    write_json(output / 'figure_manifest.json', dict(signature=e0['signature'], examples=manifests,
        report_code_sha256=file_digest(__file__), posthoc=True))
    print('REPORT_ASSETS ' + str(output), flush=True)
    print('STAGES ' + str({k: v.get('gate') for k, v in stages.items()}), flush=True)


if __name__ == '__main__':
    main()
