"""CPU analysis of an exported casepack; no torch, models, or prediction writes.

GT is used only for posthoc matching and case inspection. No policy is fitted.
Dependencies: numpy, scipy; matplotlib only when --plots is requested.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment


NAMES = ('other', 'recover', 'destroy')


def ratio(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b > 0)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def arrays(bundle, name):
    with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def match(common, known, sizes):
    active = np.flatnonzero(known > 0)
    iou = ratio(common[active], known[active, None] + sizes - common[active])
    rows, cols = linear_sum_assignment(-iou)
    passed = iou[rows, cols] > .5
    detected = set(cols[passed].tolist())
    counted = ratio(common.sum(1), known) >= .5
    tp, fp, fn = len(detected), int(counted.sum()) - len(detected), len(sizes) - len(detected)
    if len(active):
        best = iou.argmax(0)
        col = np.arange(len(sizes))
        coverage = iou[best, col]
        precision = ratio(common[active][best, col], known[active][best])
        recall = ratio(common[active][best, col], sizes)
    else:
        coverage = precision = recall = np.zeros(len(sizes))
    metrics = dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / max(2 * tp + fp + fn, 1),
        completeness=tp / len(sizes), commission=fp / max(tp + fp, 1),
        coverage=float(coverage.mean()), point_precision=float(precision.mean()), point_recall=float(recall.mean()))
    return metrics, detected


def replay_pairs(counts, edges):
    common, known, sizes = (counts[key] for key in ('intersections', 'known', 'gt_counts'))
    before, detected = match(common, known, sizes)
    active = np.flatnonzero(known > 0)
    positions = np.full(len(known), -1, dtype=int)
    positions[active] = np.arange(len(active))
    compact = ratio(common[active], known[active, None] + sizes - common[active])
    counted = ratio(common.sum(1), known) >= .5
    outcomes = {}
    for number, edge in enumerate(edges):
        a, b = counts['pairs'][edge]
        union, mass = common[a] + common[b], known[a] + known[b]
        union_iou = ratio(union, mass + sizes - union)
        if positions[a] >= 0 and positions[b] >= 0:
            matrix = np.delete(compact, positions[b], axis=0)
            matrix[positions[a]] = union_iou
        else:
            remaining = known > 0
            remaining[a], remaining[b] = mass > 0, False
            indices = np.flatnonzero(remaining)
            matrix = ratio(common[indices], known[indices, None] + sizes - common[indices])
            if mass > 0:
                matrix[np.searchsorted(indices, a)] = union_iou
        r, c = linear_sum_assignment(-matrix)
        after = set(c[matrix[r, c] > .5].tolist())
        lost, recovered = sorted(detected - after), sorted(after - detected)
        eligible = int(counted.sum()) - int(counted[a]) - int(counted[b]) + int(union.sum() / max(mass, 1) >= .5)
        outcomes[int(edge)] = dict(outcome=2 if lost else 1 if recovered else 0,
            lost_gt=counts['gt_ids'][lost].tolist(), recovered_gt=counts['gt_ids'][recovered].tolist(),
            delta_tp=len(after) - len(detected), delta_fp=eligible - len(after) - before['fp'])
        if (number + 1) % 500 == 0:
            print(f'CPU_REPLAY {number + 1}/{len(edges)}', flush=True)
    return outcomes


def selected_edges(counts, scores, threshold):
    pairs = counts['pairs']
    action = scores['action']
    score = action[:, 1] - 3 * action[:, 2]
    eligible = (action.argmax(1) == 1) & (score >= threshold)
    order = sorted(np.flatnonzero(eligible), key=lambda e: (-float(score[e]), int(pairs[e, 0]), int(pairs[e, 1])))
    best = {}
    for edge in order:
        for node in pairs[edge]:
            best.setdefault(int(node), int(edge))
    accepted = [int(e) for e in order if all(best[int(n)] == e for n in pairs[e])]
    if not np.array_equal(pairs[accepted].reshape(-1, 2), scores['accepted']):
        raise ValueError('Saved actions differ from the reconstructed fixed V3 policy.')
    return accepted, best


def union_opportunity(counts):
    """Geometric TP ceiling for this fixed graph and disjoint one-round unions.

    Unchanged original masks and every possible allowed union are enumerated.
    This is a GT oracle diagnostic, not a learned policy or an F1 ceiling.
    """
    common, known, sizes = (counts[key] for key in ('intersections', 'known', 'gt_counts'))
    baseline, detected = match(common, known, sizes)
    missed = np.array(sorted(set(range(len(sizes))) - detected), dtype=np.int64)
    original_iou = ratio(common, known[:, None] + sizes - common)
    pairs = counts['pairs']
    joined, mass = common[pairs].sum(1), known[pairs].sum(1)
    union_iou = ratio(joined, mass[:, None] + sizes - joined)
    possible = union_iou[:, missed] > .5
    feasible_columns = missed[possible.any(0)]
    existing_columns = missed[(original_iou[:, missed] > .5).any(0)]
    eligible = set(feasible_columns.tolist()) | set(existing_columns.tolist())
    edges = np.flatnonzero(possible.any(1))
    # For any union J/(K+G-J) > .5 iff sum(3*j_i-k_i) > G.
    # Two largest margins test ALL pairs; all positive margins test arbitrary
    # unions, without materializing N*N*G. These ignore graph/safety conflicts.
    margins = 3 * common[:, missed] - known[:, None]
    best_two = np.sort(margins, axis=0)[-2:].sum(0) > sizes[missed] if len(common) >= 2 else np.zeros(len(missed), bool)
    any_union = np.maximum(margins, 0).sum(0) > sizes[missed]
    single_pure = 2 * common[:, missed].max(0) > sizes[missed]
    foreground_pure = 2 * common[:, missed].sum(0) > sizes[missed]
    result = dict(original_missed_gt=counts['gt_ids'][missed].tolist(),
        union_iou_feasible_missed_gt=counts['gt_ids'][feasible_columns].tolist(),
        unchanged_iou_feasible_missed_gt=counts['gt_ids'][existing_columns].tolist(),
        feasible_edges=edges.tolist(), feasible_original_ids=counts['ids'][pairs[edges]].tolist(),
        max_possible_new_detections=len(eligible),
        completeness_upper_bound=(baseline['tp'] + len(eligible)) / len(sizes),
        any_two_original_union_feasible_gt=counts['gt_ids'][missed[best_two]].tolist(),
        any_number_original_union_feasible_gt=counts['gt_ids'][missed[any_union]].tolist(),
        gt_pure_submask_in_one_original_feasible_gt=counts['gt_ids'][missed[single_pure]].tolist(),
        gt_pure_partition_existing_foreground_feasible_gt=counts['gt_ids'][missed[foreground_pure]].tolist(),
        insufficient_existing_foreground_gt=counts['gt_ids'][missed[~foreground_pure]].tolist(),
        conditions='Fixed original masks, fixed candidate graph, disjoint one-round two-mask unions; no reassignment or new candidates.')
    selected = pairs[edges]
    if selected.size and len(np.unique(selected)) == selected.size:
        merged, totals = common.copy(), known.copy()
        for a, b in selected:
            merged[a] += merged[b]
            totals[a] += totals[b]
            merged[b], totals[b] = 0, 0
        metrics, after = match(merged, totals, sizes)
        result['gt_oracle_joint_feasible_unions'] = dict(metrics=metrics,
            lost_gt=counts['gt_ids'][sorted(detected - after)].tolist(),
            recovered_gt=counts['gt_ids'][sorted(after - detected)].tolist(),
            deployable=False, note='Uses GT to choose actions; not a measured model improvement or a global F1 upper bound.')
    return result


def split_cases(counts):
    """Rank GT-selected mixed masks for subsequent inspection, not deployment."""
    common, known, sizes = (counts[key] for key in ('intersections', 'known', 'gt_counts'))
    _, detected = match(common, known, sizes)
    result = []
    for col, gt in enumerate(counts['gt_ids']):
        if col in detected:
            continue
        eligible = np.flatnonzero(2 * common[:, col] > sizes[col])
        if not len(eligible):
            continue
        iou = ratio(common[eligible, col], known[eligible] + sizes[col] - common[eligible, col])
        row = int(eligible[iou.argmax()])
        shared = int(common[row, col])
        minimum_other_to_separate = max(0, int(known[row] + sizes[col] - 3 * shared + 1))
        result.append(dict(gt_id=int(gt), original_id=int(counts['ids'][row]),
            gt_points=int(sizes[col]), target_points_in_original=shared,
            original_known_points=int(known[row]), original_total_points=int(counts['total'][row]),
            original_is_tp_for_another_gt=bool(counts['original_tp'][row]),
            current_iou=float(ratio(shared, known[row] + sizes[col] - shared)),
            target_recall=float(shared / sizes[col]), target_purity=float(shared / known[row]),
            min_other_points_to_separate_for_iou_gt_half=minimum_other_to_separate,
            min_other_fraction_of_known=float(minimum_other_to_separate / known[row])))
    return sorted(result, key=lambda r: (r['min_other_fraction_of_known'], -r['current_iou'], r['gt_id']))


def summarize_scores(probabilities, truth, mask):
    result = {}
    for category, name in enumerate(NAMES):
        at = mask & (truth == category)
        p = probabilities[at]
        result[name] = dict(count=int(at.sum()), predicted_recover=int((p.argmax(1) == 1).sum()) if len(p) else 0,
            mean_predicted_recover=float(p[:, 1].mean()) if len(p) else None,
            median_score=float(np.median(p[:, 1] - 3 * p[:, 2])) if len(p) else None)
    return result


def case_features(points, valid, weights, labels, mean, scale):
    p = [points[k, valid[k]].astype(float) for k in range(2)]
    w = [weights[k, valid[k]].astype(float) for k in range(2)]
    g = [labels[k, valid[k]] for k in range(2)]
    if any(not len(x) for x in p):
        raise ValueError('Case contains an empty original instance.')
    frozen = [(x[:, 6:] - mean) / scale for x in p]
    centers = [np.average(x[:, :3], weights=y, axis=0) for x, y in zip(p, w)]
    votes = [np.average(x[:, 3:5], weights=y, axis=0) for x, y in zip(p, w)]
    fm = [np.average(x, weights=y, axis=0) for x, y in zip(frozen, w)]
    lower, p_tree, dom, purity, fraction = [], [], [], [], []
    for x, y, gt in zip(p, w, g):
        at = x[:, 2] <= x[:, 2].min() + .25 * max(np.ptp(x[:, 2]), .1)
        lower.append(np.average(x[at, :2], weights=y[at], axis=0))
        p_tree.append(float(np.average(x[:, 5], weights=y)))
        positive = np.unique(gt[gt > 0])
        masses = np.array([y[gt == j].sum() for j in positive])
        dominant = int(positive[masses.argmax()]) if len(positive) else 0
        dom.append(dominant)
        purity.append(float(masses.max() / max(y[gt >= 0].sum(), 1)) if len(masses) else 0.)
        fraction.append(float(y[at].sum() / y.sum()))
    connection = dict(known=0., same=0., lower_known=0., lower_same=0., contact_known=0., contact_same=0.)
    for direction in range(2):
        x, y = p[direction], p[1 - direction]
        source, dest = g[direction], g[1 - direction]
        d = np.linalg.norm((x[:, None, :3] - y[None, :, :3]) / [1, 1, 2], axis=-1)
        vd = np.linalg.norm(x[:, None, 3:5] - y[None, :, 3:5], axis=-1)
        # The same 4 geometry + 2 vote neighbors as V3. Ties have a stable CPU
        # ordering; this inspection is not a bitwise torch.topk replay.
        edge = np.concatenate((np.argsort(d, axis=1, kind='stable')[:, :min(4, len(y))],
            np.argsort(vd, axis=1, kind='stable')[:, :min(2, len(y))]), axis=1)
        unique = np.ones(edge.shape, dtype=bool)
        for j in range(edge.shape[1]):
            unique[:, j] = ~np.any(edge[:, :j] == edge[:, j, None], axis=1)
        known = (source[:, None] >= 0) & (dest[edge] >= 0) & unique
        same = (source[:, None] > 0) & (source[:, None] == dest[edge])
        mass = np.broadcast_to(w[direction][:, None], known.shape)
        distance = np.linalg.norm(x[:, None, :3] - y[edge, :3], axis=-1)
        own_x = (x[:, 2] - x[:, 2].min()) / max(np.ptp(x[:, 2]), .1)
        own_y = (y[:, 2] - y[:, 2].min()) / max(np.ptp(y[:, 2]), .1)
        lower_at = (own_x[:, None] <= .25) & (own_y[edge] <= .25)
        for name, chosen in (('', known), ('lower_', known & lower_at), ('contact_', known & (distance <= 1.))):
            connection[name + 'known'] += float(mass[chosen].sum())
            connection[name + 'same'] += float(mass[chosen & same].sum())
    feature = np.concatenate(frozen).clip(-10, 10)
    feature -= feature.mean(0)
    _, singular, vectors = np.linalg.svd(feature, full_matrices=False)
    pc1 = feature @ vectors[0]
    z = np.concatenate(p)[:, 2]
    pca_z_correlation = float(np.corrcoef(pc1, z)[0, 1]) if np.std(z) > 0 and np.std(pc1) > 0 else None
    return dict(xy_center_distance=float(np.linalg.norm(centers[0][:2] - centers[1][:2])),
        lower_xy_center_distance=float(np.linalg.norm(lower[0] - lower[1])),
        vote_center_distance=float(np.linalg.norm(votes[0] - votes[1])),
        normalized_feature_centroid_distance=float(np.linalg.norm(fm[0] - fm[1])),
        sample_dominant_gt=dom, sample_dominant_purity=purity, sample_p_tree=p_tree,
        sample_lower_mass_fraction=fraction, valid_samples=[len(x) for x in p],
        per_case_pca_pc1_variance_fraction=float(singular[0] ** 2 / max(np.sum(singular ** 2), 1e-8)),
        per_case_pca_pc1_z_correlation=pca_z_correlation,
        gt_same_cross_edge_fraction=connection['same'] / max(connection['known'], 1e-8),
        gt_same_lower_edge_fraction=(connection['lower_same'] / connection['lower_known']
            if connection['lower_known'] else None),
        gt_same_contact_edge_fraction=(connection['contact_same'] / connection['contact_known']
            if connection['contact_known'] else None))


def plot_results(output, reports, observations, plot_data, normalization):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source_color, harmful_color, good_color = '#5c6c88', '#cf483f', '#147f78'
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    w = [x for x in observations if x['plot'] == 'wytham_vox0.1' and x['isolated_outcome'] != 'other']
    for category, color, marker in [('destroy', harmful_color, 'x'), ('recover', good_color, 'o')]:
        rows = [x for x in w if x['isolated_outcome'] == category]
        axes[0].scatter([x['vote_center_distance'] for x in rows], [x['score'] for x in rows],
            label=f'{category} (GT posthoc)', color=color, marker=marker, s=55)
        axes[1].scatter([x['gt_same_cross_edge_fraction'] for x in rows], [x['score'] for x in rows],
            label=category, color=color, marker=marker, s=55)
        for row in rows:
            if category == 'recover':
                label = row['groups'][0].replace('watched_gt_', 'GT')
                offset = (4, -12) if label == 'GT801' else (4, 7) if label == 'GT629' else (3, 3)
                axes[0].annotate(label, (row['vote_center_distance'], row['score']), fontsize=7, xytext=offset, textcoords='offset points')
                axes[1].annotate(label, (row['gt_same_cross_edge_fraction'], row['score']), fontsize=7, xytext=offset, textcoords='offset points')
    axes[0].set(xlabel='Official vote-center separation (m)', ylabel='Saved V3 score', title='Close votes are not proof of one tree')
    axes[1].set(xlabel='Same-GT fraction of known cross edges', title='Connection evidence vs saved decision')
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(.25, 2.25)
    axes[1].set_xlim(-.04, 1.12)
    for ax in axes[:2]:
        ax.axhline(.1, color=source_color, linestyle='--', linewidth=.9)
        ax.grid(alpha=.2)
    labels = [r['plot'] for r in reports]
    axes[2].bar(np.arange(len(labels)), [r['feature_summary']['mean_shift_l2'] for r in reports],
        color=[source_color] * (len(labels) - 1) + [good_color])
    axes[2].set(xticks=np.arange(len(labels)), xticklabels=labels, ylabel='L2 norm of per-channel standardized mean',
        title='Wytham has no uniquely large mean shift')
    axes[2].tick_params(axis='x', rotation=40)
    fig.suptitle('Posthoc case analysis, checkpoint 2000. GT edge labels are diagnostics, not deployable inputs.', fontsize=11)
    fig.savefig(output / 'decision_evidence.png', dpi=180)
    plt.close(fig)

    chosen = [('wytham_vox0.1', (308, 309)), ('wytham_vox0.1', (1025, 1026)), ('wytham_vox0.1', (1828, 1829))]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
    for row, (plot, identifiers) in enumerate(chosen):
        record, point_data = plot_data[(plot, identifiers)]
        points, valid, weights, labels = point_data
        flat = points[valid]
        xyz = flat[:, :3].astype(float)
        center = (xyz.min(0) + xyz.max(0)) / 2
        feature = ((flat[:, 6:] - normalization[0]) / normalization[1]).clip(-10, 10)
        feature -= feature.mean(0)
        _, _, vectors = np.linalg.svd(feature, full_matrices=False)
        pca = feature @ vectors[:2].T
        offset = 0
        gt_values = labels[valid]
        largest = [j for j, _ in Counter(gt_values[gt_values > 0]).most_common(2)]
        colors = {j: color for j, color in zip(largest, ['#147f78', '#d47728'])}
        for node, marker in enumerate(('o', '^')):
            m = valid[node]
            p, gt, mass = points[node, m], labels[node, m], weights[node, m]
            color = [colors.get(int(j), '#adb3bd') for j in gt]
            size = np.clip(mass / np.median(mass) * 8, 3, 24)
            axes[row, 0].scatter(p[:, 0] - center[0], p[:, 1] - center[1], c=color, marker=marker, s=size, alpha=.65)
            axes[row, 1].scatter(p[:, 0] - center[0], p[:, 2] - center[2], c=color, marker=marker, s=size, alpha=.65)
            axes[row, 2].scatter(pca[offset:offset + len(p), 0], pca[offset:offset + len(p), 1], c=color, marker=marker, s=8, alpha=.6)
            offset += len(p)
        caption = f'{identifiers[0]}+{identifiers[1]}: {record["isolated_outcome"]}; p(recover)={record["predicted_action"][1]:.4f}'
        axes[row, 0].set(title=caption, xlabel='relative X (m)', ylabel='relative Y (m)', aspect='equal')
        axes[row, 1].set(xlabel='relative X (m)', ylabel='relative Z (m)')
        axes[row, 2].set(xlabel='PC1', ylabel='PC2', title='Frozen 32D features; per-case PCA')
    fig.suptitle('Actual input samples. Color: dominant GT identities; marker: original instance.\n'
        'Gray: background / unknown / other GT. PCA overlap alone does not demonstrate feature collapse.', fontsize=11)
    fig.savefig(output / 'sampled_cases.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('casepack', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-valid-pairs', action='store_true', help='Exact CPU replay of all labeled natural candidates.')
    parser.add_argument('--all-pairs', action='store_true', help='Also replay annotation-insufficient candidates; diagnostic only.')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observations, reports, plot_data = [], [], {}
    with zipfile.ZipFile(args.casepack) as bundle:
        meta = json.loads(bundle.read('brief.json'))
        norm = json.loads(bundle.read('normalization.json'))
        mean, scale = np.array(norm['mean']), np.array(norm['scale'])
        if mean.shape != (32,) or scale.shape != (32,) or not np.isfinite([mean, scale]).all() or np.any(scale <= 0):
            raise ValueError('Invalid reconstructed normalization.')
        for exported in meta['forests']:
            plot = exported['plot']
            print('ANALYZE ' + plot, flush=True)
            case_meta = json.loads(bundle.read(plot + '/cases.json'))
            counts = arrays(bundle, plot + '/forest_counts.npz')
            scores = arrays(bundle, plot + '/saved_scores.npz')
            cases = arrays(bundle, plot + '/cases.npz')
            if len(scores['action']) != len(counts['pairs']) or not np.allclose(scores['action'].sum(1), 1, atol=1e-5):
                raise ValueError('Saved scores are not aligned categorical probabilities.')
            accepted, best = selected_edges(counts, scores, case_meta['threshold'])
            baseline, _ = match(counts['intersections'], counts['known'], counts['gt_counts'])
            for key, value in baseline.items():
                if not np.isclose(value, case_meta['baseline'][key], atol=1e-10, rtol=0):
                    raise ValueError('Baseline protocol mismatch: ' + plot + '/' + key)
            common, known = counts['intersections'].copy(), counts['known'].copy()
            for a, b in scores['accepted']:
                common[a] += common[b]
                known[a] += known[b]
                common[b], known[b] = 0, 0
            refined, _ = match(common, known, counts['gt_counts'])
            for key, value in refined.items():
                if not np.isclose(value, case_meta['refined'][key], atol=1e-10, rtol=0):
                    raise ValueError('Refined protocol mismatch: ' + plot + '/' + key)
            requested = set(accepted) | {c['edge_index'] for c in case_meta['cases']}
            if args.all_valid_pairs:
                requested.update(np.flatnonzero(counts['pair_valid']).tolist())
            if args.all_pairs:
                requested.update(range(len(counts['pairs'])))
            outcomes = replay_pairs(counts, sorted(requested))
            truth = np.full(len(counts['pairs']), -1, dtype=int)
            for edge, outcome in outcomes.items():
                truth[edge] = outcome['outcome']
            confusion = np.zeros((3, 3), dtype=int)
            evaluated = counts['pair_valid'] & (truth >= 0)
            np.add.at(confusion, (truth[evaluated], scores['action'][evaluated].argmax(1)), 1)
            stats = case_meta['feature_statistics']
            feature_summary = dict(mean_shift_l2=float(np.linalg.norm(stats['feature_z_mean'])),
                mean_shift_max_absolute=float(np.max(np.abs(stats['feature_z_mean']))),
                mean_channel_std=float(np.mean(stats['feature_z_std'])),
                clipped_value_fraction=stats['fraction_feature_values_clipped'])
            watched = []
            score = scores['action'][:, 1] - 3 * scores['action'][:, 2]
            for i, case in enumerate(case_meta['cases']):
                if not np.array_equal(case['original_ids'], cases['original_ids'][i]):
                    raise ValueError('Case metadata and points have different original IDs.')
                outcome = outcomes[case['edge_index']]
                if NAMES[outcome['outcome']] != case['isolated_outcome']:
                    raise ValueError('Exported isolated action outcome mismatch.')
                row = dict(plot=plot, **case, **case_features(cases['points'][i], cases['valid'][i],
                    cases['weights'][i], cases['point_gt'][i], mean, scale),
                    isolated_lost_gt=outcome['lost_gt'], isolated_recovered_gt=outcome['recovered_gt'])
                node = counts['pairs'][case['edge_index']]
                dom = counts['intersections'][node].argmax(1)
                row['full_dominant_gt'] = counts['gt_ids'][dom].tolist()
                row['full_dominant_purity'] = ratio(counts['intersections'][node].max(1), counts['known'][node]).tolist()
                observations.append(row)
                plot_data[(plot, tuple(case['original_ids']))] = (row, tuple(cases[k][i] for k in ('points', 'valid', 'weights', 'point_gt')))
                if any(group.startswith('watched_gt_') for group in case['groups']):
                    blockers = []
                    for edge in sorted({best[int(n)] for n in node if int(n) in best} - {case['edge_index']}):
                        if edge not in outcomes:
                            outcomes.update(replay_pairs(counts, [edge]))
                        blockers.append(dict(edge_index=edge, original_ids=counts['ids'][counts['pairs'][edge]].tolist(),
                            score=float(score[edge]), accepted=edge in accepted,
                            original_tp=counts['original_tp'][counts['pairs'][edge]].tolist(),
                            **outcomes[edge]))
                    watched.append(dict(original_ids=case['original_ids'], groups=case['groups'],
                        decision=case['decision'], score=case['score'], competing_edges=blockers))
            accepted_outcomes = dict(Counter(NAMES[outcomes[e]['outcome']] for e in accepted))
            safe_edges = np.flatnonzero(truth == 1)
            oracle_recovered = sorted({gt for e in safe_edges for gt in outcomes[int(e)]['recovered_gt']})
            reports.append(dict(plot=plot, baseline=baseline, refined=refined,
                all_valid_replayed=bool(args.all_valid_pairs or args.all_pairs), all_pairs_replayed=bool(args.all_pairs),
                evaluated_valid_pairs=int(evaluated.sum()), total_candidate_pairs=len(counts['pairs']),
                natural_classes={NAMES[j]: int(np.count_nonzero(evaluated & (truth == j))) for j in range(3)},
                all_replayed_classes={NAMES[j]: int(np.count_nonzero(truth == j)) for j in range(3)},
                isolated_safe_recovery_gt=oracle_recovered,
                isolated_safe_recovery_edges=safe_edges.tolist(),
                classifier_confusion_true_rows_predicted_columns=confusion.tolist(),
                score_summary=summarize_scores(scores['action'], truth, evaluated),
                accepted_pairs=len(accepted), accepted_isolated_outcomes=accepted_outcomes,
                accepted_both_original_tp=int(counts['original_tp'][scores['accepted']].all(1).sum()),
                accepted_isolated_lost=sum(len(outcomes[e]['lost_gt']) for e in accepted),
                accepted_isolated_recovered=sum(len(outcomes[e]['recovered_gt']) for e in accepted),
                feature_summary=feature_summary, watched=watched))
            reports[-1]['union_opportunity'] = union_opportunity(counts)
            reports[-1]['gt_selected_split_cases'] = split_cases(counts)
            split_rows = reports[-1]['gt_selected_split_cases']
            if split_rows:
                with (args.output / (plot + '_gt_selected_split_cases.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
                    writer.writeheader()
                    writer.writerows(split_rows)
            with (args.output / (plot + '_pair_outcomes.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                writer.writerow(['edge_index', 'original_a', 'original_b', 'pair_valid', 'p_other', 'p_recover', 'p_destroy',
                    'score', 'accepted', 'isolated_outcome', 'lost_gt', 'recovered_gt'])
                for edge in sorted(outcomes):
                    value = outcomes[edge]
                    writer.writerow([edge, *counts['ids'][counts['pairs'][edge]], int(counts['pair_valid'][edge]),
                        *scores['action'][edge], float(score[edge]), int(edge in accepted), NAMES[value['outcome']],
                        json.dumps(value['lost_gt']), json.dumps(value['recovered_gt'])])
    result = dict(schema=1, posthoc=True, eligible_for_advancement=False,
        casepack_sha256=hashlib.sha256(args.casepack.read_bytes()).hexdigest(),
        all_valid_pairs=bool(args.all_valid_pairs or args.all_pairs), all_pairs=bool(args.all_pairs), forests=reports, cases=observations,
        limitations=['No model execution or policy fitting; all metrics replay saved point counts.',
            'Cases are GT-selected, not a random sample; aggregate feature statistics cover all exported forest samples.',
            'GT cross-edge agreement is diagnostic supervision, never a deployable input.',
            '256 samples per instance do not establish full-resolution contact or stem geometry.',
            'Reconstructed normalization does not verify actual checkpoint buffers.',
            'CPU stable nearest-neighbor tie ordering need not equal torch.topk ordering.',
            'PCA is per-case, unweighted, and exploratory; it does not establish feature collapse.'])
    write_json(args.output / 'analysis.json', result)
    if args.plots:
        plot_results(args.output, reports, observations, plot_data, (mean, scale))
    for row in reports:
        print(json.dumps({key: row[key] for key in ('plot', 'natural_classes', 'classifier_confusion_true_rows_predicted_columns',
            'accepted_isolated_outcomes', 'accepted_isolated_lost', 'accepted_isolated_recovered')}, ensure_ascii=False), flush=True)
    print('SAVED ' + str(args.output / 'analysis.json'), flush=True)


if __name__ == '__main__':
    main()
