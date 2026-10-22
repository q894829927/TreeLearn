"""Source-only CPU diagnosis of locked E1 scores and geometric seed surfaces.

No optimizer, candidate-policy update, prediction edit or Wytham access. GT is
used only for posthoc grouping. Keep the original experiment signature intact.
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
from scipy.sparse import coo_matrix

from tree_learn.local_split_v1 import data
from tree_learn.local_split_v1.geometry import graph_edges
from tree_learn.instance_relation.data import stamp
from tree_learn.mask_refinement.data import file_digest, load_cache, write_json

DISTANCE_BINS = [0, .1, .2, .3, .4, .600001]
HEIGHT_DIFFERENCE_BINS = [0, .1, .2, .4, .600001]
FEATURE_BINS = [0, .1, .25, .5, 1., 2., np.inf]
HEIGHT_BINS = [0, .25, .5, .75, 1.000001]
CONTACT_LABELS = ['incident_to_cross_tree_edge', 'one_hop', 'two_hops', 'outside_two_hops']


def band_indices(values, boundaries):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values < boundaries[0]) or np.any(values >= boundaries[-1]):
        raise ValueError('Diagnostic values outside declared half-open bins.')
    return np.searchsorted(boundaries, values, side='right') - 1


def edge_classes(labels, edges):
    a, b = labels[edges[:, 0]], labels[edges[:, 1]]
    return dict(same=(a == b) & (a > 0),
                other_tree=(a > 0) & (b > 0) & (a != b),
                tree_background=((a == 0) & (b > 0)) | ((a > 0) & (b == 0)))


def contact_bands(n, edges, cross_tree):
    """Hops on the fixed observable graph; cross-tree annotations only posthoc."""
    adjacency = coo_matrix((np.ones(2 * len(edges), dtype=np.int8),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    boundary = np.zeros(n, dtype=bool)
    boundary[edges[cross_tree].ravel()] = True
    distance = np.full(n, 3, dtype=np.int8)
    reached = boundary.copy()
    distance[boundary] = 0
    for hop in (1, 2):
        expanded = (adjacency @ reached.astype(np.int32)) > 0
        distance[expanded & ~reached] = hop
        reached |= expanded
    return np.minimum(distance[edges[:, 0]], distance[edges[:, 1]])


def accumulate(groups, dimension, bands, masks, scores, parent):
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    for category, mask in masks.items():
        for band in np.unique(bands[mask]):
            selected = mask & (bands == band)
            count = int(selected.sum())
            cut = int(np.count_nonzero(selected & (scores < .5)))
            loss = -np.log(clipped[selected] if category == 'same' else 1 - clipped[selected]).sum()
            key = (dimension, int(band), category)
            row = groups.setdefault(key, dict(count=0, cut=0, loss_sum=0., score_sum=0., parents=set()))
            row['count'] += count; row['cut'] += cut
            row['loss_sum'] += float(loss); row['score_sum'] += float(scores[selected].sum())
            row['parents'].add(parent)


def finish_groups(groups):
    result = []
    for (dimension, band, category), row in sorted(groups.items()):
        result.append(dict(dimension=dimension, band=band, category=category,
            count=row['count'], cut=row['cut'], cut_fraction=row['cut'] / row['count'],
            mean_bce=row['loss_sum'] / row['count'], mean_score=row['score_sum'] / row['count'],
            parents=len(row['parents'])))
    return result


def fit_circle(xy):
    """Algebraic fit, then geometric radial residual; reject ill-conditioned arcs."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 6 or xy.shape[1:] != (2,) or not np.isfinite(xy).all():
        return dict(valid=False, reason='insufficient_or_invalid_points')
    origin = xy.mean(0)
    scale = float(np.sqrt(np.square(xy - origin).sum(1).mean()))
    if scale < 1e-8:
        return dict(valid=False, reason='zero_extent')
    normalized = (xy - origin) / scale
    design = np.column_stack((2 * normalized, np.ones(len(xy))))
    coefficients, _, rank, singular = np.linalg.lstsq(design, np.square(normalized).sum(1), rcond=None)
    condition = float(singular[0] / max(singular[-1], 1e-30))
    squared_radius = coefficients[2] + np.square(coefficients[:2]).sum()
    if rank < 3 or condition > 1e4 or squared_radius <= 0:
        return dict(valid=False, reason='ill_conditioned_circle', condition=condition)
    center = origin + coefficients[:2] * scale
    radius = float(np.sqrt(squared_radius) * scale)
    residual = np.linalg.norm(xy - center, axis=1) - radius
    rmse = float(np.sqrt(np.square(residual).mean()))
    return dict(valid=True, center=center.tolist(), radius_m=radius,
                radial_rmse_m=rmse, relative_rmse=rmse / radius, condition=condition)


def circle_pair(first, second):
    fits = [fit_circle(first), fit_circle(second)]
    shared = fit_circle(np.r_[first, second])
    if not all(f['valid'] for f in fits) or not shared['valid']:
        return dict(valid=False, fits=fits, shared=shared)
    mean_separation = float(np.linalg.norm(np.mean(first, axis=0) - np.mean(second, axis=0)))
    fitted_separation = float(np.linalg.norm(np.array(fits[0]['center']) - fits[1]['center']))
    return dict(valid=True, fits=fits, shared=shared, centroid_separation_m=mean_separation,
                fitted_center_separation_m=fitted_separation,
                center_to_centroid_separation_ratio=fitted_separation / max(mean_separation, 1e-8))


def seed_relation(audit):
    good = all(r['known_points'] >= 20 and r['purity'] >= .9 and r['dominant_gt_id'] > 0 for r in audit['seeds'])
    if not good:
        return 'ambiguous_gt'
    return 'same_gt' if audit['seeds'][0]['dominant_gt_id'] == audit['seeds'][1]['dominant_gt_id'] else 'different_gt'


def analyze_edges(config, e1, model, output):
    import torch
    from tree_learn.local_split_v1.model import predict_edges
    groups, parents = {}, []
    joint_count = np.zeros((len(DISTANCE_BINS) - 1, len(FEATURE_BINS) - 1), dtype=np.int64)
    joint_cut = np.zeros_like(joint_count)
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        selected = [r for r in e1['parent_records'] if r['plot'] == row['plot']]
        if not selected:
            continue
        arrays, _ = load_cache(row['cache'], with_labels=True)
        try:
            ids, starts, counts, order = data.source_groups(arrays['baseline'])
            cores = {int(i): order[s:s + c] for i, s, c in zip(ids, starts, counts)}
            for record in selected:
                parent_id = record['parents'][0]
                indices = cores[parent_id]
                xyz = arrays['coords'][indices].astype(float)
                edges = graph_edges(xyz, config['graph'])
                masks = edge_classes(arrays['labels'][indices], edges)
                scores = predict_edges(model, arrays, indices, edges, torch.device('cpu'))
                delta = np.abs(xyz[edges[:, 0]] - xyz[edges[:, 1]])
                distances = np.linalg.norm(delta, axis=1)
                feature_distances = np.empty(len(edges))
                for start in range(0, len(edges), 65536):
                    a, b = indices[edges[start:start + 65536]].T
                    diff = (arrays['features'][a] - arrays['features'][b]) / model.scale.numpy()
                    feature_distances[start:start + len(a)] = np.sqrt(np.square(diff).mean(1))
                height = (xyz[:, 2] - xyz[:, 2].min()) / max(float(np.ptp(xyz[:, 2])), 1e-8)
                bands = dict(distance=band_indices(distances, DISTANCE_BINS),
                    abs_dz=band_indices(delta[:, 2], HEIGHT_DIFFERENCE_BINS),
                    feature_rms=band_indices(feature_distances, FEATURE_BINS),
                    relative_height=band_indices(height[edges].mean(1), HEIGHT_BINS),
                    cross_tree_contact=contact_bands(len(xyz), edges, masks['other_tree']))
                parent = f'{row["plot"]}/{parent_id}'
                for dimension, values in bands.items():
                    accumulate(groups, dimension, values, masks, scores, parent)
                accumulate(groups, 'parent_category', np.full(len(edges), int(record['category'] == 'correct')), masks, scores, parent)
                same = masks['same']
                joint_id = bands['distance'][same] * joint_count.shape[1] + bands['feature_rms'][same]
                joint_count += np.bincount(joint_id, minlength=joint_count.size).reshape(joint_count.shape)
                joint_cut += np.bincount(joint_id[scores[same] < .5], minlength=joint_cut.size).reshape(joint_cut.shape)
                parents.append(dict(plot=row['plot'], parent_id=parent_id, category=record['category'],
                                    points=len(indices), edges=len(edges)))
                print(f'EDGE_MECHANISM {parent} points={len(indices)} edges={len(edges)}', flush=True)
        finally:
            data.close_arrays(arrays)
    result = dict(groups=finish_groups(groups), parents=parents,
        same_joint_count=joint_count.tolist(), same_joint_cut=joint_cut.tolist(),
        bins=dict(distance=DISTANCE_BINS, abs_dz=HEIGHT_DIFFERENCE_BINS,
                  feature_rms=[0, .1, .25, .5, 1., 2., 'infinity'], relative_height=HEIGHT_BINS,
                  cross_tree_contact=CONTACT_LABELS, parent_category=['mixed', 'correct']))
    write_json(output / 'edge_mechanisms.json', result)
    with (output / 'edge_groups.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['groups'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(result['groups'])
    return result


def analyze_seeds(config, e0, output):
    records = []
    for row in data.rows(config):
        forest, = [f for f in e0['forests'] if f['plot'] == row['plot']]
        audits = {r['parent_id']: r for r in forest['candidates'] if r['entrance'] == 'geometry'}
        if not audits:
            continue
        prepared = data.load_prepared(row, config)
        arrays, _ = load_cache(row['cache'], with_labels=False)
        try:
            for record in prepared['records']:
                if record['parent_id'] not in audits:
                    continue
                payload = data.load_parent(row, record, arrays)
                xyz = arrays['coords'][payload['indices']].astype(float)
                z0 = np.quantile(xyz[:, 2], config['candidate']['lower_quantile'])
                levels = np.floor((xyz[:, 2] - z0) / config['candidate']['slab_m']).astype(int)
                layers = []
                for level in record['geometry_layers']:
                    groups = [xyz[at[levels[at] == level], :2] for at in [payload['seed0'], payload['seed1']]]
                    layers.append(dict(level=level, counts=list(map(len, groups)), **circle_pair(*groups)))
                valid = [v for v in layers if v['valid']]
                result = dict(plot=row['plot'], split=row['split'], parent_id=record['parent_id'],
                    relation=seed_relation(audits[record['parent_id']]), original_tp=audits[record['parent_id']]['original_tp'],
                    layers=layers, valid_layers=len(valid),
                    median_center_ratio=float(np.median([v['center_to_centroid_separation_ratio'] for v in valid])) if valid else None,
                    median_shared_relative_rmse=float(np.median([v['shared']['relative_rmse'] for v in valid])) if valid else None)
                records.append(result)
                print(f'SEED_SURFACE {row["plot"]}/{record["parent_id"]} relation={result["relation"]} circle_layers={len(valid)}', flush=True)
        finally:
            data.close_arrays(arrays)
    write_json(output / 'seed_surfaces.json', dict(records=records, deployment_rule_changed=False))
    return records


def figures(edges, seeds, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for axis, dimension, labels in [
        (axes[0], 'distance', ['0-.1', '.1-.2', '.2-.3', '.3-.4', '.4-.6']),
        (axes[1], 'feature_rms', ['0-.1', '.1-.25', '.25-.5', '.5-1', '1-2', '>=2']),
        (axes[2], 'cross_tree_contact', ['incident', '1 hop', '2 hops', '>2 hops'])]:
        rows = {r['band']: r for r in edges['groups'] if r['dimension'] == dimension and r['category'] == 'same'}
        x = np.arange(len(labels))
        axis.bar(x, [rows.get(i, {}).get('cut_fraction', 0) for i in x])
        axis.axhline(.05, color='red', linestyle='--')
        axis.set_xticks(x, labels, rotation=35)
        axis.set(xlabel=dimension, ylabel='Same-GT score < 0.5 fraction')
    fig.suptitle('Fixed E1 | full valid edges | association, not causal attribution')
    fig.savefig(output / 'edge_failure_groups.png', dpi=150); plt.close(fig)
    n, errors = np.array(edges['same_joint_count']), np.array(edges['same_joint_cut'])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    rates = np.divide(errors, n, out=np.full(n.shape, np.nan), where=n > 0)
    im = ax.imshow(rates, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='magma')
    ax.set_xticks(np.arange(6), ['0-.1', '.1-.25', '.25-.5', '.5-1', '1-2', '>=2'])
    ax.set_yticks(np.arange(5), ['0-.1', '.1-.2', '.2-.3', '.3-.4', '.4-.6'])
    ax.set(xlabel='Standardized frozen-feature RMS difference', ylabel='3D edge distance (m)', title='Same-GT score-error fraction | fixed E1')
    fig.colorbar(im, ax=ax)
    fig.savefig(output / 'distance_feature_errors.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for relation, marker in [('same_gt', 'o'), ('different_gt', '^'), ('ambiguous_gt', 'x')]:
        rows = [r for r in seeds if r['relation'] == relation and r['valid_layers']]
        ax.scatter([r['median_shared_relative_rmse'] for r in rows], [r['median_center_ratio'] for r in rows],
                   marker=marker, label=f'{relation} (n={len(rows)})', alpha=.7)
    ax.set(xlabel='One-circle radial RMSE / radius (median layers)',
           ylabel='Fitted-center separation / KMeans-centroid separation',
           title='All geometric candidates | GT only for posthoc colors')
    ax.legend()
    fig.savefig(output / 'seed_circle_evidence.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/full.yaml')
    parser.add_argument('--output', default='docs/experiments/assets/local_split_failure_mechanisms_20261022')
    args = parser.parse_args()
    config = data.load_config(args.config)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve the existing diagnosis; choose a fresh output directory.')
    output.mkdir(parents=True, exist_ok=True)
    work = Path(config['work_dir'])
    e0, e1 = [data.read_json(work / stage / 'summary.json') for stage in ('e0', 'e1')]
    signature = data.signature(config)
    if e0['signature'] != signature or e1['signature'] != signature:
        raise ValueError('Locked source experiment changed.')
    gt_sources = {}
    for row in data.rows(config):
        forest, = [f for f in e0['forests'] if f['plot'] == row['plot']]
        current = stamp(Path(row['cache']) / 'labels.npy')
        if current != forest['label_source']:
            raise ValueError(f'GT source changed since E0: {row["plot"]}')
        gt_sources[row['plot']] = current
    checkpoint = e1['selected']['checkpoint']
    expected_sha = e1['selected']['checkpoint_sha256']
    if file_digest(checkpoint) != expected_sha:
        raise ValueError('Changed local edge checkpoint.')
    import torch
    from tree_learn.local_split_v1.model import EdgeHead
    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['signature'] != signature or saved['step'] != 500:
        raise ValueError('Unexpected E1 checkpoint.')
    norm = saved['normalization']
    model = EdgeHead(norm['mean'], norm['scale']); model.load_state_dict(saved['state_dict']); model.eval()
    edges = analyze_edges(config, e1, model, output)
    seeds = analyze_seeds(config, e0, output)
    figures(edges, seeds, output)
    for row in data.rows(config):
        data.load_prepared(row, config)
        if stamp(Path(row['cache']) / 'labels.npy') != gt_sources[row['plot']]:
            raise ValueError('GT source changed during diagnosis.')
    if file_digest(checkpoint) != expected_sha:
        raise ValueError('Checkpoint changed during diagnosis.')
    totals = {category: dict(count=sum(r['count'] for r in edges['groups'] if r['dimension'] == 'distance' and r['category'] == category),
                            cut=sum(r['cut'] for r in edges['groups'] if r['dimension'] == 'distance' and r['category'] == category))
              for category in ('same', 'other_tree', 'tree_background')}
    same_rate = totals['same']['cut'] / totals['same']['count']
    other_rate = totals['other_tree']['cut'] / totals['other_tree']['count']
    strict_passed = (same_rate <= .05 and other_rate >= .8 and e1['gate']['recovered'] >= 1 and e1['gate']['lost'] == 0)
    result = dict(schema=1, posthoc=True, eligible_for_advancement=False, cpu_only=True, optimizer_steps=0,
        source_signature=signature, checkpoint_sha256=expected_sha, totals=totals,
        code_sha256=file_digest(__file__), source_e1_sha256=file_digest(work / 'e1/summary.json'),
        gt_source_stamps=gt_sources,
        strict_e1_gate=dict(same_false_cut=same_rate, other_tree_detection=other_rate,
            same_max=.05, other_tree_min=.8, passed=bool(strict_passed)),
        sources_unchanged=True, candidate_policy_changed=False, prediction_files_written=False,
        files_sha256={p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output / 'summary.json', result)
    print('MECHANISM_SUMMARY ' + str(result['totals']), flush=True)


if __name__ == '__main__':
    main()
