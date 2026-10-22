"""CPU-only audit of the identical E1 banks, negative subclasses and sampling.

Reconstruct every complete-parent negative edge, match its exact saved float32
inputs, then replay the 500-step sampler without a model update. No candidate,
prediction, threshold, optimizer or official network is changed.
"""

import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tree_learn.local_split_v1 import data, learning
from tree_learn.local_split_v1.geometry import graph_edges
from tree_learn.local_split_v1.model import EdgeHead, raw_edges
from tree_learn.instance_relation.data import stamp
from tree_learn.mask_refinement.data import file_digest, load_cache, write_json
from tools.local_split_v1.analyze_failure_mechanisms import band_indices, edge_classes

CLASS_NAMES = ['same', 'other_tree', 'tree_background']
DISTANCE_BINS = [0, .1, .2, .3, .4, .600001]
FEATURE_BINS = [0, .1, .25, .5, 1, 2, np.inf]


def row_keys(raw, symmetric=False):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 72 or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Exact matching requires finite float32 72-column raw inputs.')
    values = np.ascontiguousarray(raw).copy()
    if symmetric:
        first = np.ascontiguousarray(values[:, :32]).view(np.uint8).reshape(-1, 128)
        second = np.ascontiguousarray(values[:, 32:64]).view(np.uint8).reshape(-1, 128)
        different = first != second
        first_difference = different.argmax(axis=1)
        rows = np.arange(len(values))
        swap = different.any(axis=1) & (first[rows, first_difference] > second[rows, first_difference])
        values[swap, :32], values[swap, 32:64] = values[swap, 32:64].copy(), values[swap, :32].copy()
    return values.view('V288').ravel()


def match_negative_rows(complete_raw, bank_raw, semantics):
    """Match multisets, refusing an ambiguous assignment across GT subclasses/pairs."""
    full, saved = row_keys(complete_raw), row_keys(bank_raw)
    if len(full) != len(saved):
        raise ValueError('The saved bank does not contain the entire negative population.')
    full_order, saved_order = np.argsort(full, kind='stable'), np.argsort(saved, kind='stable')
    if not np.array_equal(full[full_order], saved[saved_order]):
        raise ValueError('Negative bank inputs differ from the complete graph, byte for byte.')
    duplicate = full[full_order][1:] == full[full_order][:-1]
    if np.any(duplicate & np.any(semantics[full_order][1:] != semantics[full_order][:-1], axis=1)):
        raise ValueError('Identical raw negative inputs have ambiguous GT subclass/pair; do not assign arbitrary labels.')
    mapping = np.empty(len(full), dtype=np.int64)
    mapping[saved_order] = full_order
    return mapping, int(duplicate.sum())


def canonical_conflicts(raw, target):
    """Identical symmetric raw inputs with conflicting targets are unlearnable pairs."""
    keys = row_keys(raw, symmetric=True)
    _, inverse = np.unique(keys, return_inverse=True)
    positive = np.bincount(inverse[target == 1], minlength=int(inverse.max()) + 1)
    negative = np.bincount(inverse[target == 0], minlength=len(positive))
    conflict = (positive > 0) & (negative > 0)
    return dict(conflicting_input_groups=int(conflict.sum()),
                positive_rows=int(positive[conflict].sum()), negative_rows=int(negative[conflict].sum()))


def replay_sampler(records, steps=500, edges_per_step=8192, seed=42):
    """Exact original RNG calls; return exposure counts without drawing gradients."""
    pools = defaultdict(lambda: defaultdict(list))
    visits = [np.zeros(len(r['kind']), dtype=np.int64) for r in records]
    for number, record in enumerate(records):
        pools[record['category']][record['plot']].append(number)
    rng = np.random.default_rng(seed)
    total, negative_shares = np.zeros(3, dtype=np.int64), np.zeros(2)
    for _ in range(steps):
        counts = np.zeros(3, dtype=np.int64)
        for category in ('mixed', 'mixed', 'correct', 'correct'):
            plots = sorted(pools[category])
            plot = plots[int(rng.integers(len(plots)))]
            index = pools[category][plot][int(rng.integers(len(pools[category][plot])))]
            at = rng.integers(len(records[index]['kind']), size=edges_per_step // 4)
            visits[index] += np.bincount(at, minlength=len(visits[index]))
            counts += np.bincount(records[index]['kind'][at], minlength=3)
        if not counts[1:].sum():
            raise ValueError('A replayed batch has no negative edges.')
        total += counts
        negative_shares += counts[1:] / counts[1:].sum()
    return dict(sampled_edges={name: int(total[i]) for i, name in enumerate(CLASS_NAMES)},
        mean_within_negative_batch_share={name: float(negative_shares[i - 1] / steps)
                                         for i, name in enumerate(CLASS_NAMES) if i > 0},
        steps=steps, rng_state=rng.bit_generator.state), visits


def predict(model, raw):
    with torch.no_grad():
        return model(torch.as_tensor(raw)).sigmoid().numpy().astype(float)


def statistics(mask, scores):
    count = int(mask.sum())
    clipped = np.clip(scores[mask], 1e-7, 1 - 1e-7)
    return dict(count=count, cut=int(np.count_nonzero(scores[mask] < .5)),
        mean_score=float(clipped.mean()) if count else None,
        mean_negative_bce=float(-np.log1p(-clipped).mean()) if count else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/full.yaml')
    parser.add_argument('--r1-work', default='work_dirs/treelearn_local_split_e1_r1_s42')
    parser.add_argument('--output', default='docs/experiments/assets/local_split_negative_audit_20261022')
    parser.add_argument('--sidecars', default='work_dirs/treelearn_local_split_negative_audit_s42/subclasses')
    args = parser.parse_args()
    config = data.load_config(args.config)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve the existing audit; choose a fresh output directory.')
    data.checkpoint_md5(config)
    e0 = learning.require_gate(config, 'e0')
    e1 = data.read_json(Path(config['work_dir']) / 'e1/summary.json')
    r1 = data.read_json(Path(args.r1_work) / 'audit/summary.json')
    bank_path = Path(config['work_dir']) / 'banks/bank.json'
    if not bank_path.exists():
        raise FileNotFoundError('Original banks required; no bank construction in this audit.')
    bank = learning.bank(config, e0)
    source_hashes = {str(path): file_digest(path) for path in [bank_path,
        Path(config['work_dir']) / 'e0/summary.json', Path(config['work_dir']) / 'e1/summary.json',
        Path(args.r1_work) / 'audit/summary.json']}
    label_sources = {r['plot']: stamp(Path(r['cache']) / 'labels.npy') for r in data.rows(config)}
    if label_sources != {f['plot']: f['label_source'] for f in e0['forests']}:
        raise ValueError('GT sources differ from original E0.')
    if e1['signature'] != data.signature(config) or r1['source_signature'] != e1['signature']:
        raise ValueError('V1/R1/source signature differs.')
    if data.read_json(Path(args.r1_work) / 'verified.json')['signature'] != r1['signature']:
        raise ValueError('R1 is not verified.')
    selected = {(r['plot'], r['category'], tuple(r['parents'])): r for r in bank['records']}
    records = [selected[r['plot'], r['category'], tuple(r['parents'])] for r in e1['parent_records']]
    paths = dict(v1=e1['selected']['checkpoint'], r1=r1['checkpoint']['checkpoint'])
    shas = dict(v1=e1['selected']['checkpoint_sha256'], r1=r1['checkpoint']['checkpoint_sha256'])
    models, saved = {}, {}
    torch.set_num_threads(4)
    for name, path in paths.items():
        if file_digest(path) != shas[name]:
            raise ValueError('Changed edge checkpoint.')
        saved[name] = torch.load(path, map_location='cpu', weights_only=False)
        if saved[name]['normalization'] != bank['normalization']:
            raise ValueError('Normalization differs between bank and checkpoints.')
        norm = bank['normalization']
        model = EdgeHead(norm['mean'], norm['scale']); model.load_state_dict(saved[name]['state_dict']); model.eval()
        models[name] = model
    output.mkdir(parents=True, exist_ok=True)
    sidecars = Path(args.sidecars)
    if sidecars.exists():
        raise FileExistsError('Preserve previous subtype sidecars; no overwrite.')
    sidecars.mkdir(parents=True)
    parent_rows, edge_rows, loaded, pairs = [], [], [], []
    bank_raw_all, bank_target_all = [], []
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        current = [r for r in records if r['plot'] == row['plot']]
        if not current:
            continue
        arrays, _ = load_cache(row['cache'], with_labels=True)
        try:
            ids, starts, counts, order = data.source_groups(arrays['baseline'])
            cores = {int(i): order[s:s + c] for i, s, c in zip(ids, starts, counts)}
            for record in current:
                parent = record['parents'][0]
                indices = cores[parent]
                labels = arrays['labels'][indices]
                edges = graph_edges(arrays['coords'][indices].astype(float), config['graph'])
                classes = edge_classes(labels, edges)
                neg_at = np.flatnonzero(classes['other_tree'] | classes['tree_background'])
                negative_edges = edges[neg_at]
                raw = raw_edges(arrays, indices, negative_edges)
                kind = np.where(classes['other_tree'][neg_at], 1, 2)
                gt_pair = np.sort(labels[negative_edges], axis=1)
                semantics = np.column_stack((kind, gt_pair))
                with np.load(record['path'], allow_pickle=False) as z:
                    bank_raw, target = z['raw'], z['target']
                at = np.flatnonzero(target == 0)
                if len(neg_at) > config['training']['bank_edges_per_parent'] // 2:
                    raise ValueError('A selected negative pool exceeds the cap; revise the audit rather than assume full coverage.')
                mapping, duplicates = match_negative_rows(raw, bank_raw[at], semantics)
                bank_kind = np.zeros(len(target), dtype=np.int8)
                bank_kind[at] = kind[mapping]
                bank_pair = np.full((len(target), 2), -1, dtype=np.int64)
                bank_pair[at] = gt_pair[mapping]
                sidecar = sidecars / f'{row["plot"]}_{parent}.npz'
                np.savez(sidecar, kind=bank_kind, negative_gt_pair=bank_pair, full_negative_edge=neg_at[mapping])
                entry = dict(record, kind=bank_kind, sidecar=str(sidecar), sidecar_sha256=file_digest(sidecar))
                loaded.append(entry)
                bank_raw_all.append(bank_raw); bank_target_all.append(target)
                score = {name: predict(model, raw) for name, model in models.items()}
                bank_scores = {name: predict(model, bank_raw) for name, model in models.items()}
                positive_ids, positive_counts = np.unique(labels[labels > 0], return_counts=True)
                dominant = sorted(positive_ids[np.argsort(-positive_counts, kind='stable')[:2]].tolist())
                dominant_pair = np.all(gt_pair == dominant, axis=1) if len(dominant) == 2 else np.zeros(len(kind), bool)
                distance = raw[:, 67].astype(float)
                feature = np.sqrt(np.square((raw[:, :32] - raw[:, 32:64]) / np.array(norm['scale'])).mean(1))
                bands = dict(distance=band_indices(distance, DISTANCE_BINS), feature_rms=band_indices(feature, FEATURE_BINS),
                             dominant_pair=dominant_pair.astype(int))
                result = dict(plot=row['plot'], parent_id=parent, category=record['category'],
                    same_full=int(classes['same'].sum()), other_tree_full=int(np.count_nonzero(kind == 1)),
                    background_full=int(np.count_nonzero(kind == 2)), same_bank=int(np.count_nonzero(target == 1)),
                    other_tree_bank=int(np.count_nonzero(bank_kind == 1)), background_bank=int(np.count_nonzero(bank_kind == 2)),
                    all_negative_rows_match=True, duplicate_negative_raw_rows=duplicates,
                    positive_gt_count=len(positive_ids), dominant_gt_ids=dominant)
                for name in models:
                    for i, category in enumerate(CLASS_NAMES):
                        mask = bank_kind == i
                        result[f'{name}_{category}_bank_cut'] = int(np.count_nonzero(mask & (bank_scores[name] < .5)))
                parent_rows.append(result)
                for dimension, band in bands.items():
                    for value in np.unique(band):
                        for i, category in [(1, 'other_tree'), (2, 'tree_background')]:
                            mask = (band == value) & (kind == i)
                            if not mask.any():
                                continue
                            detail = dict(plot=row['plot'], parent_id=parent, category=record['category'],
                                dimension=dimension, band=int(value), edge_class=category)
                            for name in models:
                                detail[name] = statistics(mask, score[name])
                            detail['v1_cut_r1_keep'] = int(np.count_nonzero(mask & (score['v1'] < .5) & (score['r1'] >= .5)))
                            detail['v1_keep_r1_cut'] = int(np.count_nonzero(mask & (score['v1'] >= .5) & (score['r1'] < .5)))
                            edge_rows.append(detail)
                unique_pairs, inverse = np.unique(gt_pair, axis=0, return_inverse=True)
                for number, pair in enumerate(unique_pairs):
                    mask = inverse == number
                    pairs.append(dict(plot=row['plot'], parent_id=parent, category=record['category'],
                        gt_ids=pair.tolist(), edge_class='other_tree' if pair[0] > 0 else 'tree_background',
                        count=int(mask.sum()), median_distance_m=float(np.median(distance[mask])),
                        median_feature_rms=float(np.median(feature[mask])),
                        v1_cut=int(np.count_nonzero(mask & (score['v1'] < .5))),
                        r1_cut=int(np.count_nonzero(mask & (score['r1'] < .5)))))
                print('NEGATIVE_PARENT ' + str(result), flush=True)
        finally:
            data.close_arrays(arrays)
    by_key = {(r['plot'], r['category'], tuple(r['parents'])): r for r in loaded}
    loaded = [by_key[r['plot'], r['category'], tuple(r['parents'])] for r in records]
    replay, visits = replay_sampler(loaded)
    if replay['rng_state'] != saved['v1']['rng_numpy'] or replay['rng_state'] != saved['r1']['rng_numpy']:
        raise ValueError('Sampler replay differs from the actual 500-step training runs.')
    seen = {name: 0 for name in CLASS_NAMES}
    sidecar_manifest = []
    for record, sampled in zip(loaded, visits):
        row, = [r for r in parent_rows if r['plot'] == record['plot'] and r['parent_id'] == record['parents'][0]]
        for i, category in enumerate(CLASS_NAMES):
            mask = record['kind'] == i
            row[category + '_sampled_occurrences'] = int(sampled[mask].sum())
            row[category + '_bank_rows_seen'] = int(np.count_nonzero(sampled[mask]))
            seen[category] += row[category + '_bank_rows_seen']
        sidecar_manifest.append({k: record[k] for k in ('plot', 'category', 'parents', 'path', 'sha256', 'sidecar', 'sidecar_sha256')})
    reference = data.read_json('docs/experiments/assets/local_split_failure_mechanisms_20261022/summary.json')
    for name, expected in [('v1', reference['totals']), ('r1', r1['edge_rates'])]:
        for category in ('other_tree', 'tree_background'):
            rows = [r for r in edge_rows if r['dimension'] == 'distance' and r['edge_class'] == category]
            actual = dict(count=sum(r[name]['count'] for r in rows), cut=sum(r[name]['cut'] for r in rows))
            if actual != expected[category]:
                raise ValueError(f'Negative full-graph evaluation failed to reproduce: {name}/{category}')
    conflicts = canonical_conflicts(np.concatenate(bank_raw_all), np.concatenate(bank_target_all))
    for row in data.rows(config):
        data.load_prepared(row, config)
    after = learning.require_gate(config, 'e0')
    if after != e0 or any(file_digest(path) != sha for path, sha in source_hashes.items()):
        raise ValueError('Original audit changed.')
    if label_sources != {r['plot']: stamp(Path(r['cache']) / 'labels.npy') for r in data.rows(config)}:
        raise ValueError('GT sources changed during the audit.')
    for record in bank['records']:
        if file_digest(record['path']) != record['sha256']:
            raise ValueError('Training bank changed during the audit.')
    for name, path in paths.items():
        if file_digest(path) != shas[name]:
            raise ValueError('Checkpoint changed during the audit.')
    write_json(output / 'parents.json', parent_rows)
    write_json(output / 'edge_difficulty.json', edge_rows)
    write_json(output / 'gt_pairs.json', pairs)
    write_json(output / 'subclass_manifest.json', dict(gt_supervision_only=True, kind_names=CLASS_NAMES,
               original_inputs_unchanged=True, records=sidecar_manifest))
    write_json(sidecars / 'manifest.json', dict(gt_supervision_only=True, kind_names=CLASS_NAMES, records=sidecar_manifest))
    with (output / 'parents.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parent_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(parent_rows)
    totals = dict(same_bank=sum(r['same_bank'] for r in parent_rows),
        strict_negative_full=sum(r['other_tree_full'] for r in parent_rows),
        background_negative_full=sum(r['background_full'] for r in parent_rows), bank_rows_seen=seen)
    result = dict(schema=1, source_signature=e1['signature'], cpu_only=True, optimizer_steps=0,
        eligible_for_advancement=False, full_negative_bank_multisets_match=True,
        sampler_reproduces_both_checkpoint_rng_states=True, totals=totals, sampler=replay,
        exact_symmetric_bank_input_conflicts=conflicts, checkpoint_sha256=shas,
        code_sha256=file_digest(__file__), source_bank_manifest_sha256=file_digest(bank_path),
        immutable_source_hashes=source_hashes,
        source_e1_summary_sha256=file_digest(Path(config['work_dir']) / 'e1/summary.json'),
        source_r1_summary_sha256=file_digest(Path(args.r1_work) / 'audit/summary.json'),
        gt_source_stamps={f['plot']: f['label_source'] for f in e0['forests']},
        original_sources_verified=True, external_executed=False)
    write_json(output / 'summary.json', result)
    plot(parent_rows, replay, output)
    print('NEGATIVE_AUDIT ' + str(result), flush=True)


def plot(parents, replay, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained')
    names = ['strict other-tree', 'tree-background']
    bank = [sum(r['other_tree_bank'] for r in parents), sum(r['background_bank'] for r in parents)]
    sample = [replay['sampled_edges'][k] for k in ('other_tree', 'tree_background')]
    for at, values, label in [(0, bank, 'Unique bank negatives'), (1, sample, 'Replayed 500-step negatives')]:
        fraction = np.array(values) / sum(values)
        axes[at].bar(names, fraction, color=['tab:orange', 'tab:green'])
        for x, y in enumerate(fraction):
            axes[at].text(x, y + .02, f'{100*y:.2f}%', ha='center')
        axes[at].set(title=label, ylabel='Within-negative fraction', ylim=(0, 1.1))
    for name, color in [('v1', 'tab:blue'), ('r1', 'tab:orange')]:
        values = [sum(r[f'{name}_other_tree_bank_cut'] for r in parents) / bank[0],
                  sum(r[f'{name}_tree_background_bank_cut'] for r in parents) / bank[1]]
        x = np.arange(2) + (-.18 if name == 'v1' else .18)
        axes[2].bar(x, values, width=.36, label=name.upper(), color=color)
    axes[2].set_xticks(range(2), names)
    axes[2].set(title='All saved negatives are full-graph negatives', ylabel='Detection rate', ylim=(0, 1.1))
    axes[2].legend()
    fig.suptitle('Fixed 16 fitted parents | source-only audit | 0 optimizer steps')
    fig.savefig(output / 'negative_composition.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
