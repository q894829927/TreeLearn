"""Read-only CPU audit of complete local split cases; never writes predictions.

All graph/mode construction uses inputs only. GT is used afterwards for diagnosis.
No fitted classifier, proposed segmentation, network execution, or model metric.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.instance_relation_v3.export_split_cases import (
    INPUT_NAMES, audit_core, check_output, collect_cores, collect_context, select_cases)
from tree_learn.instance_relation import data as relation_data
from tree_learn.instance_relation.supervision import load_supervision
from tree_learn.mask_refinement.data import file_digest, write_json


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def verify_bundle(bundle, config):
    """Verify provenance, whole core membership, context and every selected value."""
    brief = json.loads((bundle / 'brief.json').read_text())
    if brief['export_code_sha256'] != file_digest(Path(__file__).with_name('export_split_cases.py')):
        raise ValueError('Exporter code differs; do not silently reuse.')
    for name, stamp in brief['inputs'].items():
        if relation_data.stamp(stamp['path']) != stamp:
            raise ValueError(f'Source stamp changed: {name}')
    for path, digest in brief['guarded_sha256'].items():
        if file_digest(path) != digest:
            raise ValueError(f'Source metadata changed: {path}')
    row, = [r for r in relation_data.entries(config, external=True) if r['plot'] == brief['plot']]
    cache = Path(row['cache'])
    prepared = Path(config['base_prepared_root']) / 'external' / row['plot']
    feature_marker = json.loads((prepared / 'features.json').read_text())
    if feature_marker['request'] != relation_data.feature_request(cache, dict(config['geometry'], seed=config['seed'])):
        raise ValueError('Geometry provenance differs from official cache/config.')
    reference = load_npz(bundle / 'reference_counts.npz')
    targets = load_supervision(prepared, config['supervision'])
    for name, array in reference.items():
        if not np.array_equal(array, targets[name]):
            raise ValueError(f'Reference counts differ: {name}')
    settings = brief['settings']
    cases, metrics = select_cases(relation_data.load_features(prepared), targets,
        [r['original_id'] for r in brief['cases'] if r['role'] == 'mixed_case'], settings['max_core_points'])
    if metrics != brief['baseline'] or [metrics[k] for k in ('tp', 'fp', 'fn')] != [596, 222, 281]:
        raise ValueError('Baseline counts or protocol differ.')
    for case, report in zip(cases, brief['cases']):
        if any(report[k] != v for k, v in case.items()):
            raise ValueError('Case/control selection differs.')
    arrays = {k: np.load(cache / f'{k}.npy', mmap_mode='r', allow_pickle=False)
              for k in (*INPUT_NAMES, 'labels')}
    cores = collect_cores(arrays['baseline'], cases, settings['chunk_points'], settings['max_core_points'])
    contexts = collect_context(arrays['coords'], arrays['baseline'], cases, cores,
        settings['chunk_points'], settings['context_points'], settings['strata'],
        settings['padding_xy'], settings['padding_z'], settings['seed'])
    verified = []
    for case in brief['cases']:
        identifier = case['original_id']
        folder = bundle / f'original_{identifier}'
        if json.loads((folder / 'case.json').read_text()) != case:
            raise ValueError('Case metadata does not match manifest.')
        for name in ('inputs', 'audit'):
            if file_digest(folder / f'{name}.npz') != case[f'{"input" if name == "inputs" else name}_sha256']:
                raise ValueError(f'Case file hash differs: {identifier}/{name}')
        data, audit = load_npz(folder / 'inputs.npz'), load_npz(folder / 'audit.npz')
        if set(data) != set(INPUT_NAMES) | {'is_core', 'source_indices'}:
            raise ValueError('Unexpected input fields.')
        indices = data['source_indices']
        context = contexts[identifier]
        expected_indices = np.sort(np.concatenate((cores[identifier], *context['indices'])))
        if (not np.array_equal(indices, expected_indices)
                or not np.array_equal(indices, audit['source_indices'])
                or not np.array_equal(indices[data['is_core']], cores[identifier])
                or not np.array_equal(data['is_core'], data['baseline'] == identifier)):
            raise ValueError('Full membership/context/alignment differs.')
        for name in INPUT_NAMES:
            if not np.array_equal(data[name], arrays[name][indices]):
                raise ValueError(f'Observable values differ: {identifier}/{name}')
        if not np.array_equal(audit['labels'], arrays['labels'][indices]):
            raise ValueError('Audit labels differ from source.')
        result = audit_core(audit['labels'], data['is_core'], targets, case['node'])
        if any(case[k] != v for k, v in result.items()):
            raise ValueError('Exact full core counts differ.')
        verified.append(dict(original_id=identifier, core_points=int(data['is_core'].sum()),
            selected_points=len(indices), inputs_sha256=case['input_sha256'], audit_sha256=case['audit_sha256']))
        print(f'VERIFIED original={identifier} all_core_and_context_values_exact', flush=True)
    archive = bundle / 'split_casepack.zip'
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Archive CRC failed.')
        expected = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and p != archive}
        if set(zipped.namelist()) != expected:
            raise ValueError('Archive membership differs.')
        import hashlib
        for name in expected:
            if hashlib.sha256(zipped.read(name)).hexdigest() != file_digest(bundle / name):
                raise ValueError(f'Archive differs: {name}')
    for stamp in brief['inputs'].values():
        if relation_data.stamp(stamp['path']) != stamp:
            raise ValueError('Source changed during verification.')
    for path, digest in brief['guarded_sha256'].items():
        if file_digest(path) != digest:
            raise ValueError('Metadata changed during verification.')
    return dict(reused_export=str(bundle), archive_sha256=file_digest(archive),
        baseline=metrics, verified=verified, source_stamps=brief['inputs'],
        guarded_sha256=brief['guarded_sha256'], export_code_sha256=brief['export_code_sha256'],
        checks='Code/config/stamps/metadata/compact counts/GT case plan; full-forest core/context index reconstruction; '
               'every selected observable and GT value; archive CRC/membership/byte hashes.'), brief


def knn_edges(xyz, k=12, radius=.6):
    if len(xyz) < 2:
        return np.empty((0, 2), dtype=np.int64)
    distances, neighbors = cKDTree(xyz).query(xyz, k=min(k + 1, len(xyz)), workers=4)
    left = np.broadcast_to(np.arange(len(xyz))[:, None], neighbors.shape)
    valid = (neighbors != left) & (distances <= radius)
    edges = np.sort(np.column_stack((left[valid], neighbors[valid])), axis=1)
    return np.unique(edges, axis=0)


def components(n, edges):
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    count, labels = connected_components(graph, directed=False)
    return labels, np.bincount(labels, minlength=count)


def composition(labels):
    ids, counts = np.unique(labels, return_counts=True)
    return {str(int(i)): int(c) for i, c in zip(ids, counts)}


def radius_summary(xyz, gt, radius):
    edges = knn_edges(xyz, radius=radius)
    labels, sizes = components(len(xyz), edges)
    largest = np.argsort(-sizes, kind='stable')[:6]
    known = (gt[edges[:, 0]] > 0) & (gt[edges[:, 1]] > 0)
    summary = dict(radius_m=radius, k=12, components=len(sizes),
        substantial_components=int((sizes >= max(20, .01 * len(xyz))).sum()),
        largest_fractions=(sizes[largest] / len(xyz)).tolist(),
        positive_edge_count=int(known.sum()),
        different_gt_positive_edge_fraction=float(np.mean(gt[edges[known, 0]] != gt[edges[known, 1]])) if known.any() else None,
        largest_components=[dict(points=int(sizes[i]), gt_counts=composition(gt[labels == i])) for i in largest])
    return summary, labels, edges


def layer_graph(xyz, gt, width=1., radius=.35):
    """GT-free XY components per 1m slab, bounded adjacent-slab 3D edges."""
    levels = np.floor((xyz[:, 2] - np.floor(xyz[:, 2].min())) / width).astype(int)
    nodes, assignment = [], np.full(len(xyz), -1, dtype=int)
    for level in np.unique(levels):
        indices = np.flatnonzero(levels == level)
        edges = knn_edges(xyz[indices, :2], k=8, radius=radius)
        labels, sizes = components(len(indices), edges)
        for component in np.argsort(-sizes, kind='stable')[:6]:
            if sizes[component] < 20:
                continue
            at = indices[labels == component]
            assignment[at] = len(nodes)
            positive = gt[at][gt[at] > 0]
            ids, counts = np.unique(positive, return_counts=True)
            nodes.append(dict(id=len(nodes), layer=int(level), points=len(at),
                center=xyz[at].mean(0).tolist(), gt_counts=composition(gt[at]),
                dominant_gt=int(ids[counts.argmax()]) if len(ids) else -1))
    links = []
    for level in np.unique(levels)[:-1]:
        lower = np.flatnonzero((levels == level) & (assignment >= 0))
        upper = np.flatnonzero((levels == level + 1) & (assignment >= 0))
        if not len(lower) or not len(upper):
            continue
        d, at = cKDTree(xyz[upper]).query(xyz[lower], k=min(4, len(upper)), workers=4)
        if d.ndim == 1:
            d, at = d[:, None], at[:, None]
        aa = np.broadcast_to(lower[:, None], at.shape)
        valid = d <= .6
        pairs = np.column_stack((assignment[aa[valid]], assignment[upper[at[valid]]]))
        pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        links.extend(dict(source=int(a), target=int(b), contacts=int(c)) for (a, b), c in zip(pairs, counts) if c >= 3)
    return dict(slab_width_m=width, xy_radius_m=radius, adjacent_3d_radius_m=.6,
        minimum_component_points=20, max_components_per_slab=6, minimum_contacts=3,
        retained_points=int((assignment >= 0).sum()), nodes=nodes, edges=links)


def vote_grid(votes, cell=.1):
    low = np.floor(votes.min(0) / cell) * cell - 1
    high = np.ceil(votes.max(0) / cell) * cell + 1
    bins = [np.arange(a, b + cell, cell) for a, b in zip(low, high)]
    if np.prod([len(b) for b in bins]) > 4_000_000:
        raise ValueError('Vote grid exceeds CPU bound.')
    return bins


def find_modes(votes, bins, sigma):
    density = gaussian_filter(np.histogram2d(*votes.T, bins=bins)[0], sigma / .1)
    candidates = np.argwhere((density == maximum_filter(density, size=5)) & (density >= .2 * density.max()))
    values = density[tuple(candidates.T)]
    order = np.argsort(-values, kind='stable')
    peaks = []
    for index in candidates[order]:
        center = np.array([(bins[d][index[d]] + bins[d][index[d] + 1]) / 2 for d in range(2)])
        if any(np.linalg.norm(center - np.array(p['xy'])) < .5 for p in peaks):
            continue
        peaks.append(dict(xy=center.tolist(), relative_density=float(density[tuple(index)] / density.max()),
            mass_within_0_5m=float(np.mean(np.linalg.norm(votes - center, axis=1) <= .5))))
        if len(peaks) == 6:
            break
    return peaks, density


def vote_summary(votes, source_indices, z):
    bins = vote_grid(votes)
    parity = (relation_data.priorities(source_indices, 42) & np.uint64(1)).astype(bool)
    summaries, grids = [], []
    for sigma in (.15, .3, .6):
        peaks, density = find_modes(votes, bins, sigma)
        halves = [find_modes(votes[parity == side], bins, sigma)[0] for side in (0, 1)]
        matches = [all(any(np.linalg.norm(np.array(p['xy']) - q['xy']) <= .3 for q in half) for half in halves) for p in peaks]
        summaries.append(dict(sigma_m=sigma, peaks=peaks, half_peak_counts=list(map(len, halves)),
            peaks_reproduced_within_0_3m=matches,
            height_quartile_peak_xy=[find_modes(votes[(z >= a) & (z <= b)], bins, sigma)[0]
                for a, b in zip(np.quantile(z, [0, .25, .5, .75]), np.quantile(z, [.25, .5, .75, 1]))]))
        grids.append(density)
    return summaries, bins, grids


def distance_auc(same, different):
    if not len(same) or not len(different):
        return None
    # P(different-GT distance > same-GT distance), a diagnostic rank statistic.
    ranks = rankdata(np.r_[same, different])
    return float((ranks[len(same):].sum() - len(different) * (len(different) + 1) / 2) / (len(same) * len(different)))


def representation_summary(xyz, votes, features, gt, edges, source):
    positive = gt > 0
    idx = np.flatnonzero(positive)
    idx = idx[np.argsort(relation_data.priorities(source[idx], 43), kind='stable')[:8000]]
    result = dict(nn_point_limit=8000, positive_sample_points=len(idx), nn={})
    for name, values in [('xyz', xyz), ('votes_xy', votes), ('frozen32_standardized', features)]:
        _, nn = cKDTree(values[idx]).query(values[idx], k=min(7, len(idx)), workers=4)
        # Remove self by index, including coincident-point ties.
        rows = np.broadcast_to(np.arange(len(idx))[:, None], nn.shape)
        valid = nn != rows
        valid &= np.cumsum(valid, axis=1) <= min(6, len(idx) - 1)
        same = gt[idx[nn]] == gt[idx, None]
        agreement = (same & valid).sum(1) / valid.sum(1)
        result['nn'][name] = dict(micro_same_gt_fraction=float(agreement.mean()),
            macro_same_gt_fraction=float(np.mean([agreement[gt[idx] == label].mean() for label in np.unique(gt[idx])])),
            per_gt={str(int(label)): float(agreement[gt[idx] == label].mean()) for label in np.unique(gt[idx])})
    a, b = edges.T
    valid = (gt[a] > 0) & (gt[b] > 0)
    a, b = a[valid], b[valid]
    same = gt[a] == gt[b]
    distances = dict(xyz=np.linalg.norm(xyz[a] - xyz[b], axis=1),
        votes_xy=np.linalg.norm(votes[a] - votes[b], axis=1),
        frozen32=np.linalg.norm(features[a] - features[b], axis=1))
    result['positive_local_edges'] = dict(same=int(same.sum()), different=int((~same).sum()))
    result['local_edge_distances'] = {}
    for name, d in distances.items():
        result['local_edge_distances'][name] = dict(
            same_median=float(np.median(d[same])) if same.any() else None,
            different_median=float(np.median(d[~same])) if (~same).any() else None,
            distance_auc=distance_auc(d[same], d[~same]))
    # Compare same/different edge distances within fixed physical-distance bins.
    # Geometry/height confounding is reduced, not removed; these are not classifier scores.
    result['distance_matched_feature_audit'] = []
    dz = np.abs(xyz[a, 2] - xyz[b, 2])
    for low, high in zip([0, .15, .3, .45], [.15, .3, .45, .6]):
        mask = (distances['xyz'] >= low) & (distances['xyz'] < high) & (dz <= .15)
        ss, dd = distances['frozen32'][mask & same], distances['frozen32'][mask & ~same]
        result['distance_matched_feature_audit'].append(dict(distance_range_m=[low, high], max_dz_m=.15,
            same_edges=len(ss), different_edges=len(dd), feature_distance_auc=distance_auc(ss, dd)))
    centered = features - features.mean(0)
    eig, vec = np.linalg.eigh(centered.T @ centered / len(features))
    pc1 = centered @ vec[:, -1]
    result['pca_pc1_fraction'] = float(eig[-1] / eig.sum())
    result['pca_pc1_z_correlation'] = float(np.corrcoef(pc1, xyz[:, 2])[0, 1])
    return result


def gt_spatial_summary(xyz, votes, gt, target_ids):
    low_z = xyz[:, 2].min() + .25 * np.ptp(xyz[:, 2])
    records = []
    for label in target_ids:
        mask = gt == label
        lower = mask & (xyz[:, 2] <= low_z)
        records.append(dict(gt_id=int(label), points=int(mask.sum()), lower_points=int(lower.sum()),
            xyz_quantiles=np.quantile(xyz[mask], [.05, .5, .95], axis=0).tolist(),
            lower_median_xy=np.median(xyz[lower, :2], axis=0).tolist() if lower.any() else None,
            vote_median=np.median(votes[mask], axis=0).tolist(),
            vote_rms=float(np.sqrt(np.mean(np.sum((votes[mask] - votes[mask].mean(0)) ** 2, axis=1))))))
    result = dict(lower_cut_z=float(low_z), lower_definition='min(core_z)+0.25*ptp(core_z); not terrain-normalized', records=records)
    if len(records) == 2:
        aa, bb = records
        result['vote_median_distance_m'] = float(np.linalg.norm(np.array(aa['vote_median']) - bb['vote_median']))
        result['xyz_median_xy_distance_m'] = float(np.linalg.norm(np.array(aa['xyz_quantiles'][1][:2]) - bb['xyz_quantiles'][1][:2]))
        result['lower_median_distance_m'] = float(np.linalg.norm(np.array(aa['lower_median_xy']) - bb['lower_median_xy'])) if aa['lower_points'] and bb['lower_points'] else None
        a, b = xyz[gt == target_ids[0]], xyz[gt == target_ids[1]]
        d = cKDTree(a).query(b, k=1, workers=4)[0]
        result['cross_gt_nearest_distances_m'] = np.quantile(d, [0, .05, .5]).tolist()
    return result


def height_pair_audit(xyz, gt, target_ids):
    """Posthoc within-slab target separation; does not select any graph nodes."""
    if len(target_ids) != 2:
        return []
    rows = []
    for z in np.arange(np.floor(xyz[:, 2].min()), xyz[:, 2].max(), 1.):
        layer = (xyz[:, 2] >= z) & (xyz[:, 2] < z + 1)
        a, b = [xyz[layer & (gt == label)] for label in target_ids]
        if min(len(a), len(b)) < 10:
            continue
        rows.append(dict(z_low=float(z), counts=[len(a), len(b)],
            median_xy_distance_m=float(np.linalg.norm(np.median(a[:, :2], 0) - np.median(b[:, :2], 0))),
            xy_rms_m=[float(np.sqrt(np.mean(np.sum((v[:, :2] - v[:, :2].mean(0)) ** 2, axis=1)))) for v in (a, b)],
            cross_nearest_q05_m=float(np.quantile(cKDTree(a).query(b, workers=4)[0], .05))))
    return rows


def lower_support_summary(xyz, gt):
    """Fine-scale support count sensitivity; no GT, no claim of tree count."""
    rows = []
    z_min = xyz[:, 2].min()
    for radius in (.15, .25, .35):
        counts, slabs = [], []
        for z in np.arange(z_min, z_min + 4, .5):
            mask = (xyz[:, 2] >= z) & (xyz[:, 2] < z + .5)
            points = xyz[mask, :2]
            if not len(points):
                counts.append(0)
                slabs.append(dict(z_low=float(z), components=[]))
                continue
            labels, sizes = components(len(points), knn_edges(points, k=8, radius=radius))
            counts.append(int((sizes >= 5).sum()))
            # Audit only after input-only membership is fixed.
            slabs.append(dict(z_low=float(z), components=[dict(points=int(sizes[c]),
                center_xy=points[labels == c].mean(0).tolist(), gt_counts=composition(gt[mask][labels == c]))
                for c in np.argsort(-sizes, kind='stable') if sizes[c] >= 5]))
        rows.append(dict(xy_radius_m=radius, slab_width_m=.5, minimum_points=5,
            component_counts_first_4m=counts, slabs=slabs))
    return rows


def diagnostic_colors(gt, target_ids):
    colors = np.full(len(gt), '#a0a0a0', dtype=object)
    colors[gt == 0] = '#252525'
    colors[(gt > 0) & ~np.isin(gt, target_ids)] = '#8e6db0'
    for label, color in zip(target_ids, ['#167cbd', '#e98019']):
        colors[gt == label] = color
    return colors.tolist()


def save_figures(output, records, payloads):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9})
    for mixed in [r for r in records if r['role'] == 'mixed_case']:
        control, = [r for r in records if r.get('control_for_original_id') == mixed['original_id']]
        fig, axes = plt.subplots(2, 4, figsize=(17, 10), constrained_layout=True)
        for row, record in enumerate([mixed, control]):
            p = payloads[record['original_id']]
            xyz, votes, gt = p['xyz'], p['votes'], p['gt']
            origin = xyz.mean(0)
            colors = diagnostic_colors(gt, record['target_gt_ids'])
            for col, dims in enumerate([(0, 1), (0, 2), (1, 2)]):
                ax = axes[row, col]
                context = p['context'] - origin
                ax.scatter(context[:, dims[0]], context[:, dims[1]], s=.12, c='#dddddd', rasterized=True)
                ax.scatter(xyz[:, dims[0]] - origin[dims[0]], xyz[:, dims[1]] - origin[dims[1]],
                    s=.6, c=colors, alpha=.65, rasterized=True)
                ax.set(xlabel='XYZ'[dims[0]] + ' (m, centered)', ylabel='XYZ'[dims[1]] + ' (m, centered)')
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_title(f"{record['role']} {record['original_id']} | {'XY XZ YZ'.split()[col]}")
            ax = axes[row, 3]
            ax.scatter(*(votes - origin[:2]).T, s=.8, c=colors, alpha=.5, rasterized=True)
            ax.set(xlabel='vote X (m, centered)', ylabel='vote Y (m, centered)', title='Official XY votes (all core points)')
            ax.set_aspect('equal', adjustable='datalim')
            handles = [Line2D([], [], marker='o', ls='', color=c, label=f'GT{g}') for g, c in zip(record['target_gt_ids'], ['#167cbd', '#e98019'])]
            handles += [Line2D([], [], marker='o', ls='', color=c, label=s) for c, s in [('#8e6db0', 'other GT>0'), ('#a0a0a0', 'GT<0 unknown'), ('#252525', 'GT=0 background')]]
            ax.legend(handles=handles, fontsize=7)
        fig.suptitle(f"Full original cores: {mixed['original_id']} and matched TP {control['original_id']} | GT colors are posthoc; pale context is sampled")
        fig.savefig(output / f"pair_{mixed['original_id']}_geometry.png", dpi=135)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for row, record in enumerate([mixed, control]):
            p = payloads[record['original_id']]
            for col, (summary, density) in enumerate(zip(record['votes'], p['grids'])):
                ax = axes[row, col]
                bins = p['bins']
                ax.imshow(density.T, origin='lower', extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]], cmap='magma', aspect='equal')
                for i, peak in enumerate(summary['peaks']):
                    ax.plot(*peak['xy'], '+', c='cyan')
                    ax.text(*peak['xy'], str(i + 1), color='cyan')
                ax.set(title=f"{record['original_id']} sigma={summary['sigma_m']}m | modes={len(summary['peaks'])}, half counts={summary['half_peak_counts']}", xlabel='vote X (m)', ylabel='vote Y (m)')
        fig.suptitle('GT-free vote density, 0.1m grid | peaks >=20% max, separated >=0.5m, max 6 | diagnostic settings, not a split policy')
        fig.savefig(output / f"pair_{mixed['original_id']}_votes.png", dpi=135)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        for row, record in enumerate([mixed, control]):
            p = payloads[record['original_id']]
            xyz = p['xyz']
            axes[row, 0].scatter(*xyz[:, :2].T, c=xyz[:, 2], cmap='viridis', s=1)
            axes[row, 0].set(title=f"{record['original_id']} XY by Z (observable)", xlabel='X (m)', ylabel='Y (m)')
            axes[row, 0].set_aspect('equal', adjustable='datalim')
            axes[row, 1].scatter(xyz[:, 0], xyz[:, 2], c=p['component_labels'], cmap='tab20', s=1)
            axes[row, 1].set(title='3D 12NN radius 0.35m components (observable)', xlabel='X (m)', ylabel='Z (m)')
            graph, ax = record['layer_graph'], axes[row, 2]
            for edge in graph['edges']:
                a, b = (graph['nodes'][edge[key]] for key in ('source', 'target'))
                ax.plot([a['center'][0], b['center'][0]], [a['center'][2], b['center'][2]], c='#aaaaaa', lw=.5)
            centers = np.array([n['center'] for n in graph['nodes']])
            colors = diagnostic_colors(np.array([n['dominant_gt'] for n in graph['nodes']]), record['target_gt_ids'])
            ax.scatter(centers[:, 0], centers[:, 2], s=np.sqrt([n['points'] for n in graph['nodes']]) * 3, c=colors)
            ax.set(title='1m slab XY components + adjacent 3D contacts\nGraph is GT-free; node majority color is posthoc', xlabel='component mean X (m)', ylabel='mean Z (m)')
        fig.suptitle('Local connectivity | no terrain normalization; side branches can form separate slab components; context excluded from graphs')
        fig.savefig(output / f"pair_{mixed['original_id']}_connections.png", dpi=135)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for row, record in enumerate([mixed, control]):
            p = payloads[record['original_id']]
            xyz, gt = p['xyz'], p['gt']
            z_min = xyz[:, 2].min()
            for col, (dims, height) in enumerate([((0, 1), 4), ((0, 2), 8), ((1, 2), 8)]):
                mask = xyz[:, 2] <= z_min + height
                ax = axes[row, col]
                ax.scatter(xyz[mask, dims[0]], xyz[mask, dims[1]], s=5,
                    c=diagnostic_colors(gt[mask], record['target_gt_ids']), alpha=.7)
                ax.set_aspect('equal', adjustable='datalim')
                ax.set(title=f"{record['original_id']} {'XY XZ YZ'.split()[col]} | bottom {height}m of core",
                    xlabel='XYZ'[dims[0]] + ' (m)', ylabel='XYZ'[dims[1]] + ' (m)')
        fig.suptitle('Lower support zoom, all core points within input-defined Z window | GT colors are posthoc; no ground normalization')
        fig.savefig(output / f"pair_{mixed['original_id']}_lower.png", dpi=135)
        plt.close(fig)
        print(f'PLOTS pair={mixed["original_id"]}/{control["original_id"]}', flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for col, record in enumerate([r for r in records if r['role'] == 'mixed_case']):
        p = payloads[record['original_id']]
        a, b = p['edges'].T
        valid = (p['gt'][a] > 0) & (p['gt'][b] > 0)
        a, b = a[valid], b[valid]
        same = p['gt'][a] == p['gt'][b]
        distance = np.linalg.norm(p['features'][a] - p['features'][b], axis=1)
        for row, selection in enumerate([np.ones(len(a), dtype=bool),
                (np.linalg.norm(p['xyz'][a] - p['xyz'][b], axis=1) >= .15)
                & (np.linalg.norm(p['xyz'][a] - p['xyz'][b], axis=1) < .3)
                & (np.abs(p['xyz'][a, 2] - p['xyz'][b, 2]) <= .15)]):
            ax = axes[row, col]
            for mask, color, label in [(selection & same, '#167cbd', 'same GT>0'), (selection & ~same, '#e98019', 'different GT>0')]:
                values = np.sort(distance[mask])
                ax.plot(values, np.arange(1, len(values) + 1) / len(values), c=color, label=f'{label} n={len(values)}')
            ax.set(title=f"{record['original_id']} | {'all local edges' if row == 0 else 'd=0.15-0.30m, |dz|<=0.15m'}",
                xlabel='Frozen 32D standardized distance', ylabel='Empirical CDF', xlim=(0, 4), ylim=(0, 1))
            ax.legend(fontsize=7)
    fig.suptitle('Posthoc feature distance audit on observable 12NN edges <=0.6m | correlated edges, no classifier fitted; unknown/background excluded')
    fig.savefig(output / 'feature_edge_audit.png', dpi=135)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--private-output', type=Path,
        help='Server-only detailed GT/coordinate audit; defaults below work_dir, outside published assets.')
    parser.add_argument('--config', default='configs/experiments/instance_relation_v3/full.yaml')
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    check_output(args.output, [args.bundle, config['external_cache_root'], config['base_prepared_root']])
    private_output = args.private_output or Path(config['work_dir']) / 'split_case_cpu_private_20261022'
    check_output(private_output, [args.bundle, args.output, config['external_cache_root'], config['base_prepared_root']])
    args.output.mkdir(parents=True, exist_ok=True)
    private_output.mkdir(parents=True, exist_ok=True)
    verification, brief = verify_bundle(args.bundle, config)
    write_json(args.output / 'verification.json', verification)
    norm = json.loads((args.bundle / 'normalization.json').read_text())
    records, payloads = [], {}
    for case in brief['cases']:
        identifier = case['original_id']
        print(f'ANALYZE original={identifier} core={case["core_points"]}', flush=True)
        directory = args.bundle / f'original_{identifier}'
        inputs, audit = load_npz(directory / 'inputs.npz'), load_npz(directory / 'audit.npz')
        core = inputs['is_core']
        xyz, votes, gt = inputs['coords'][core].astype(float), inputs['votes_xy'][core].astype(float), audit['labels'][core]
        features = (inputs['features'][core].astype(float) - norm['mean']) / norm['scale']
        target_ids = [r['gt_id'] for r in case['gt_components'][:2 if case['role'] == 'mixed_case' else 1]]
        source = inputs['source_indices'][core]
        graphs = [radius_summary(xyz, gt, radius) for radius in (.2, .35, .6)]
        vote_records, bins, grids = vote_summary(votes, source, xyz[:, 2])
        record = dict(original_id=identifier, role=case['role'], target_gt_ids=target_ids,
            core_points=len(xyz), context_points=int((~core).sum()), known_points=case['known_points'],
            ignored_points=case['ignored_points'], background_points=case['background_points'],
            z_range=[float(xyz[:, 2].min()), float(xyz[:, 2].max())],
            gt_components=case['gt_components'], gt_spatial=gt_spatial_summary(xyz, votes, gt, target_ids),
            target_pair_by_height=height_pair_audit(xyz, gt, target_ids), lower_support=lower_support_summary(xyz, gt),
            radius_graphs=[g[0] for g in graphs], layer_graph=layer_graph(xyz, gt), votes=vote_records,
            representations=representation_summary(xyz, votes, features, gt, graphs[-1][2], source))
        if 'control_for_original_id' in case:
            record['control_for_original_id'] = case['control_for_original_id']
        records.append(record)
        payloads[identifier] = dict(xyz=xyz, votes=votes, gt=gt, context=inputs['coords'][~core],
            features=features, edges=graphs[-1][2], component_labels=graphs[1][1], bins=bins, grids=grids)
        print(f'STATS original={identifier} radius_large={[g[0]["substantial_components"] for g in graphs]} '
              f'vote_modes={[len(v["peaks"]) for v in vote_records]}', flush=True)
    metadata = dict(schema=2, posthoc=True, model_executed=False,
        predictions_modified=False, eligible_for_advancement=False,
        protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5',
        analysis_code_sha256=file_digest(__file__), bundle_sha256=verification['archive_sha256'],
        normalization_note=norm['note'])
    write_json(private_output / 'analysis_full.json', dict(**metadata, cases=records))
    # Published statistics contain only per-case aggregates, no node/GT identities or coordinates.
    public_cases = []
    for r in records:
        public_cases.append(dict(original_id=r['original_id'], role=r['role'],
            core_points=r['core_points'], known_points=r['known_points'],
            ignored_points=r['ignored_points'], background_points=r['background_points'],
            z_span_m=float(np.ptp(r['z_range'])),
            substantial_components=[g['substantial_components'] for g in r['radius_graphs']],
            largest_component_fractions=[g['largest_fractions'][0] for g in r['radius_graphs']],
            vote_mode_counts=[len(v['peaks']) for v in r['votes']],
            vote_half_mode_counts=[v['half_peak_counts'] for v in r['votes']],
            positive_local_edges=r['representations']['positive_local_edges'],
            feature_edge_distance_audit=r['representations']['local_edge_distances']['frozen32'],
            geometry_matched_feature_audit=r['representations']['distance_matched_feature_audit']))
    write_json(args.output / 'analysis.json', dict(**metadata, scope='Aggregate diagnostic statistics only; full coordinate/GT audit stays on server.', cases=public_cases))
    with (args.output / 'summary.csv').open('w', newline='') as handle:
        fields = ['original_id', 'role', 'core_points', 'known_points', 'ignored_points', 'background_points',
                  'z_span_m', 'substantial_components_r035', 'substantial_components_r060',
                  'vote_modes_s015', 'vote_modes_s030', 'vote_modes_s060',
                  'xyz_nn_macro', 'votes_nn_macro', 'features_nn_macro', 'local_feature_distance_auc']
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for r in records:
            nn = r['representations']['nn']
            writer.writerow(dict(**{k: r[k] for k in fields[:6]}, z_span_m=np.ptp(r['z_range']),
                substantial_components_r035=r['radius_graphs'][1]['substantial_components'],
                substantial_components_r060=r['radius_graphs'][2]['substantial_components'],
                **{f'vote_modes_s{s}': len(v['peaks']) for s, v in zip(['015', '030', '060'], r['votes'])},
                xyz_nn_macro=nn['xyz']['macro_same_gt_fraction'], votes_nn_macro=nn['votes_xy']['macro_same_gt_fraction'],
                features_nn_macro=nn['frozen32_standardized']['macro_same_gt_fraction'],
                local_feature_distance_auc=r['representations']['local_edge_distances']['frozen32']['distance_auc']))
    save_figures(args.output, records, payloads)
    print(f'COMPLETE output={args.output.resolve()} cases={len(records)}', flush=True)


if __name__ == '__main__':
    main()
