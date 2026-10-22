"""Posthoc CPU diagnosis of a completed E1; no fitting or gate changes."""

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
import torch

from tree_learn.local_split_v1 import data, learning
from tree_learn.local_split_v1.audit import edge_targets
from tree_learn.local_split_v1.geometry import graph_edges
from tree_learn.local_split_v1.model import EdgeHead, predict_edges
from tree_learn.mask_refinement.data import file_digest, load_cache, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/full.yaml')
    parser.add_argument('--output', default='docs/experiments/assets/local_split_v1_20261022')
    args = parser.parse_args()
    config = data.load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    e1 = data.read_json(Path(config['work_dir']) / 'e1/summary.json')
    if e1['signature'] != data.signature(config):
        raise ValueError('E1 source/config mismatch.')
    checkpoint = e1['selected']['checkpoint']
    if file_digest(checkpoint) != e1['selected']['checkpoint_sha256']:
        raise ValueError('Changed E1 checkpoint.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    norm = saved['normalization']
    model = EdgeHead(norm['mean'], norm['scale'])
    model.load_state_dict(saved['state_dict'])
    torch.set_num_threads(4)
    allowed, reports = defaultdict(set), []
    bins = np.linspace(0, 1, 41)
    histograms = [np.zeros(40, dtype=np.int64), np.zeros(40, dtype=np.int64)]
    for record in e1['parent_records']:
        allowed[record['plot']].update(record['parents'])
    for row in [r for r in data.rows(config) if r['split'] == 'train' and r['plot'] in allowed]:
        arrays, _ = load_cache(row['cache'], with_labels=True)
        prepared = data.load_prepared(row, config)
        candidates = {r['parent_id'] for r in prepared['records'] if r['reason'] == 'candidate'}
        try:
            ids, starts, counts, order = data.source_groups(arrays['baseline'])
            cores = {int(i): order[s:s + c] for i, s, c in zip(ids, starts, counts)}
            for record in [r for r in e1['parent_records'] if r['plot'] == row['plot']]:
                parent = record['parents'][0]
                indices = cores[parent]
                xyz = arrays['coords'][indices].astype(float)
                edges = graph_edges(xyz, config['graph'])
                valid, targets = edge_targets(arrays['labels'][indices], edges)
                scores = predict_edges(model, arrays, indices, edges, torch.device('cpu'))
                result = dict(plot=row['plot'], parent_id=parent, category=record['category'],
                              candidate=parent in candidates, core_points=len(indices))
                for label, name in [(1, 'same'), (0, 'different')]:
                    selected = valid & (targets == label)
                    histograms[1 - label] += np.histogram(scores[selected], bins=bins)[0]
                    result[name + '_edges'] = int(selected.sum())
                    result[name + '_cut'] = int(np.count_nonzero(selected & (scores < .5)))
                left = arrays['labels'][indices[edges[:, 0]]]
                right = arrays['labels'][indices[edges[:, 1]]]
                for name, selected in [
                    ('other_tree', (left > 0) & (right > 0) & (left != right)),
                    ('tree_background', ((left == 0) & (right > 0)) | ((left > 0) & (right == 0)))]:
                    result[name + '_edges'] = int(selected.sum())
                    result[name + '_cut'] = int(np.count_nonzero(selected & (scores < .5)))
                result['same_false_cut_rate'] = result['same_cut'] / max(result['same_edges'], 1)
                result['different_detection_rate'] = result['different_cut'] / max(result['different_edges'], 1)
                reports.append(result)
                print('EDGE_AUDIT ' + str(result), flush=True)
        finally:
            data.close_arrays(arrays)
    previous = output / 'e1_diagnosis.json'
    if previous.exists() and data.read_json(previous)['signature'] == e1['signature']:
        geometry = data.read_json(previous)['geometric_control']
    else:
        geometry = learning.evaluate(config, allowed=allowed, split='train')
    result = dict(signature=e1['signature'], posthoc=True, training=False, cpu_only=True,
        original_e1_gate=e1['gate'], learned_selected_threshold=e1['selected']['threshold'],
        per_parent=reports, geometric_control=geometry,
        score_histogram=dict(bins=bins.tolist(), same=histograms[0].tolist(), different=histograms[1].tolist()),
        checkpoint_sha256=e1['selected']['checkpoint_sha256'], code_sha256=file_digest(__file__))
    write_json(output / 'e1_diagnosis.json', result)
    with (output / 'e1_edge_errors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(reports)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    centers = (bins[:-1] + bins[1:]) / 2
    for counts, label in zip(histograms, ['Same-GT edges', 'Different-GT / tree-background edges']):
        axes[0].plot(centers, counts / max(counts.sum(), 1), label=label)
    axes[0].axvline(.5, color='black', linestyle='--')
    axes[0].legend()
    axes[0].set(xlabel='Learned same-tree affinity', ylabel='Within-class probability mass', title='Full valid edges, fitted E1 parents')
    x = np.arange(len(reports))
    axes[1].bar(x, [r['same_false_cut_rate'] for r in reports], color=['tab:blue' if r['category'] == 'mixed' else 'tab:orange' for r in reports])
    axes[1].axhline(.05, color='red', linestyle='--', label='Locked E1 maximum = 5%')
    axes[1].set_xticks(x, [f'{r["plot"]}/{r["parent_id"]}' for r in reports], rotation=90)
    axes[1].set(ylabel='Same-GT false edge-cut rate', title='Blue=mixed; orange=correct multi-branch controls')
    axes[1].legend()
    fig.savefig(output / 'e1_edge_errors.png', dpi=150)
    plt.close(fig)
    print('GEOMETRIC_E1_CONTROL ' + str(geometry['gate']), flush=True)
    print('E1_DIAGNOSIS ' + str(output / 'e1_diagnosis.json'), flush=True)


if __name__ == '__main__':
    main()
