"""Observable-only candidates, bounded graphs and seeded harmonic ownership."""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree

from tree_learn.instance_relation.data import priorities


class BudgetExceeded(ValueError):
    """Per-parent refusal; never silently truncate an editable core."""


def modes(votes, bins, sigma, cfg):
    density = gaussian_filter(np.histogram2d(*votes.T, bins=bins)[0], sigma / cfg['cell_m'])
    if not density.max():
        return np.empty((0, 2))
    at = np.argwhere((density == maximum_filter(density, size=5)) &
                     (density >= cfg['relative_peak_density'] * density.max()))
    order = np.lexsort((at[:, 1], at[:, 0], -density[tuple(at.T)]))
    peaks = []
    for loc in at[order]:
        point = np.array([(bins[d][loc[d]] + bins[d][loc[d] + 1]) / 2 for d in (0, 1)])
        if not any(np.linalg.norm(point - other) < cfg['peak_separation_m'] for other in peaks):
            peaks.append(point)
        if len(peaks) == 3:  # three is already outside the binary action space
            break
    return np.asarray(peaks).reshape(-1, 2)


def match_modes(reference, other, tolerance):
    if len(reference) != len(other):
        return False
    distances = np.linalg.norm(reference[:, None] - other[None, :], axis=-1)
    a, b = linear_sum_assignment(distances)
    return bool(np.all(distances[a, b] <= tolerance))


def vote_seeds(xyz, votes, source, cfg, seed):
    cell = cfg['cell_m']
    low = np.floor(votes.min(0) / cell) * cell - cell
    high = np.ceil(votes.max(0) / cell) * cell + cell
    shape = np.ceil((high - low) / cell).astype(np.int64)
    if np.prod(shape.astype(float)) > cfg['max_grid_cells']:
        raise BudgetExceeded('vote_grid_budget')
    bins = [low[d] + np.arange(shape[d] + 1) * cell for d in (0, 1)]
    half = (priorities(source, seed) & np.uint64(1)).astype(bool)
    peaks = [modes(votes, bins, sigma, cfg) for sigma in cfg['sigmas_m']]
    diagnostic = dict(vote_mode_counts=list(map(len, peaks)))
    if all(len(p) >= 3 for p in peaks):
        return None, dict(diagnostic, vote_reason='third_stable_support')
    if any(len(p) != 2 for p in peaks) or not match_modes(peaks[0], peaks[1], cfg['peak_match_m']):
        return None, dict(diagnostic, vote_reason='no_scale_stable_pair')
    for sigma in cfg['sigmas_m']:
        for side in (False, True):
            other = modes(votes[half == side], bins, sigma, cfg)
            if not match_modes(peaks[0], other, cfg['peak_match_m']):
                return None, dict(diagnostic, vote_reason='hash_unstable')
    distances = np.linalg.norm(votes[:, None] - peaks[0][None], axis=2)
    nearest = distances.argmin(1)
    close = distances.min(1) <= cfg['vote_seed_radius_m']
    layers = np.floor(xyz[:, 2] - xyz[:, 2].min()).astype(int)
    supported = []
    for level in np.unique(layers):
        groups = [np.flatnonzero((layers == level) & close & (nearest == i)) for i in (0, 1)]
        if min(map(len, groups)) >= cfg['vote_layer_points']:
            supported.append(int(level))
    starts = [level for level in supported if level + 1 in supported]
    if not starts:
        return None, dict(diagnostic, vote_reason='no_shared_adjacent_height_support')
    # Earliest common two-layer support; no GT-based seed subset selection.
    selected = np.isin(layers, [starts[0], starts[0] + 1]) & close
    seeds = [np.flatnonzero(selected & (nearest == i)) for i in (0, 1)]
    return seeds, dict(diagnostic, vote_reason='accepted', vote_peaks=peaks[0].tolist(),
                       shared_layers=[starts[0], starts[0] + 1])


def two_means(xy):
    """Deterministic two-means with two axis starts; assess all slab points."""
    best = None
    for axis in (0, 1):
        order = np.argsort(xy[:, axis], kind='stable')
        centers = xy[order[[len(xy) // 4, 3 * len(xy) // 4]]].copy()
        for _ in range(30):
            assigned = np.square(xy[:, None] - centers[None]).sum(2).argmin(1)
            if any(not np.any(assigned == k) for k in (0, 1)):
                break
            updated = np.array([xy[assigned == k].mean(0) for k in (0, 1)])
            if np.max(np.abs(updated - centers)) < 1e-6:
                centers = updated
                break
            centers = updated
        assigned = np.square(xy[:, None] - centers[None]).sum(2).argmin(1)
        if len(np.unique(assigned)) != 2:
            continue
        sse = float(np.square(xy - centers[assigned]).sum())
        if best is None or sse < best[0]:
            best = sse, centers, assigned
    return best


def geometry_seeds(xyz, cfg):
    z0 = float(np.quantile(xyz[:, 2], cfg['lower_quantile']))
    levels = np.floor((xyz[:, 2] - z0) / cfg['slab_m']).astype(int)
    valid_layers = []
    run = []
    third_run = 0
    selected_run = None
    for level in range(int(cfg['lower_window_m'] / cfg['slab_m'])):
        at = np.flatnonzero(levels == level)
        fitted = two_means(xyz[at, :2]) if len(at) >= 2 * cfg['cluster_points'] else None
        acceptable = False
        if fitted is not None:
            sse, centers, groups = fitted
            counts = np.bincount(groups, minlength=2)
            rms = np.array([np.sqrt(np.square(xyz[at[groups == k], :2] - centers[k]).sum(1).mean()) for k in (0, 1)])
            distance = np.linalg.norm(centers[0] - centers[1])
            total = float(np.square(xyz[at, :2] - xyz[at, :2].mean(0)).sum())
            acceptable = (counts.min() >= cfg['cluster_points'] and distance >= cfg['separation_m']
                          and distance >= cfg['separation_rms'] * rms.max()
                          and sse <= cfg['sse_ratio'] * max(total, 1e-12))
        # Explicit third-support check; never interpret forced K=2 as tree count.
        third = False
        if fitted is not None and len(at) >= 3 * cfg['cluster_points']:
            xy = xyz[at, :2]
            residual = np.square(xy - centers[groups]).sum(1)
            centers3 = np.r_[centers, xy[[int(residual.argmax())]]]
            for _ in range(30):
                groups3 = np.square(xy[:, None] - centers3[None]).sum(2).argmin(1)
                if len(np.unique(groups3)) < 3:
                    break
                updated = np.array([xy[groups3 == k].mean(0) for k in range(3)])
                if np.max(np.abs(updated - centers3)) < 1e-6:
                    centers3 = updated
                    break
                centers3 = updated
            if len(np.unique(groups3)) == 3:
                sizes = np.bincount(groups3, minlength=3)
                rms3 = [np.sqrt(np.square(xy[groups3 == k] - centers3[k]).sum(1).mean()) for k in range(3)]
                separation3 = np.linalg.norm(centers3[:, None] - centers3[None], axis=-1)[np.triu_indices(3, 1)].min()
                sse3 = np.square(xy - centers3[groups3]).sum()
                third = (sizes.min() >= cfg['cluster_points'] and
                    separation3 >= max(cfg['separation_m'], cfg['separation_rms'] * max(rms3)) and
                    sse3 <= cfg['sse_ratio'] * max(sse, 1e-12))
        third_run = third_run + 1 if third else 0
        if third_run >= cfg['consecutive_layers']:
            return None, dict(geometry_reason='third_stable_support')
        if acceptable:
            if run:
                d = np.linalg.norm(run[-1][1][:, None] - centers[None], axis=2)
                a, b = linear_sum_assignment(d)
                if d[a, b].max() > cfg['slab_m']:
                    run = []
                else:
                    permutation = b[np.argsort(a)]
                    inverse = np.argsort(permutation)
                    centers, groups = centers[permutation], inverse[groups]
            if not run and level * cfg['slab_m'] > cfg['start_max_m']:
                continue
            run.append((level, centers, at, groups, rms))
            valid_layers.append(level)
            if len(run) >= cfg['consecutive_layers'] and selected_run is None:
                selected_run = list(run)
        else:
            run = []
    # Scan the entire lower window before accepting the earliest pair: a later
    # persistent third support must override an earlier binary hypothesis.
    if selected_run is not None:
        seeds = [np.concatenate([item[2][item[3] == k] for item in selected_run]) for k in (0, 1)]
        return seeds, dict(geometry_reason='accepted', geometry_layers=[v[0] for v in selected_run],
                           geometry_centers=[v[1].tolist() for v in selected_run])
    return None, dict(geometry_reason='no_persistent_lower_tracks', geometry_valid_layers=valid_layers)


def candidate(xyz, votes, source, cfg, seed=42):
    if len(xyz) > cfg['max_points']:
        return None, dict(reason='core_budget')
    if len(xyz) < cfg['min_points']:
        return None, dict(reason='small_core')
    if not np.isfinite(xyz).all() or not np.isfinite(votes).all():
        raise ValueError('Nonfinite observable candidate inputs.')
    try:
        seeds, details = vote_seeds(xyz, votes, source, cfg, seed)
    except BudgetExceeded as error:
        return None, dict(reason=str(error))
    # A third stable vote support is a binary-action refusal, not a fallback.
    if details['vote_reason'] == 'third_stable_support':
        return None, dict(details, reason='third_stable_support')
    entrance = 'votes'
    if seeds is None:
        seeds, extra = geometry_seeds(xyz, cfg)
        details.update(extra)
        entrance = 'geometry'
    if seeds is None:
        return None, dict(details, reason='third_stable_support' if details.get('geometry_reason') == 'third_stable_support' else 'no_seeds')
    center = np.median(votes, axis=0)
    positions = np.array([np.median(votes[at], axis=0) for at in seeds])
    anchor = int(np.argmin(np.linalg.norm(positions - center, axis=1)))
    seeds = [seeds[anchor], seeds[1 - anchor]]
    return seeds, dict(details, reason='candidate', entrance=entrance,
                       seed_counts=list(map(len, seeds)))


def graph_edges(xyz, cfg, radius=None):
    radius = cfg['radius_m'] if radius is None else radius
    n = len(xyz)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    distances, neighbors = cKDTree(xyz).query(xyz, k=min(cfg['neighbors'] + 1, n), workers=4)
    left = np.broadcast_to(np.arange(n)[:, None], neighbors.shape)
    good = (neighbors != left) & (distances <= radius)
    parts = [np.column_stack((left[good], neighbors[good]))]
    levels = np.floor((xyz[:, 2] - xyz[:, 2].min()) / cfg['slab_m']).astype(int)
    for level in np.unique(levels):
        aa, bb = np.flatnonzero(levels == level), np.flatnonzero(levels == level + 1)
        if not len(bb):
            continue
        d, j = cKDTree(xyz[bb]).query(xyz[aa], k=min(cfg['layer_neighbors'], len(bb)), workers=1)
        d, j = d.reshape(len(aa), -1), j.reshape(len(aa), -1)
        ii = np.broadcast_to(aa[:, None], j.shape)
        good = d <= radius
        parts.append(np.column_stack((ii[good], bb[j[good]])))
    edges = np.unique(np.sort(np.concatenate(parts), axis=1), axis=0)
    if len(edges) > cfg['max_edges']:
        raise BudgetExceeded('edge_budget')
    return edges


def harmonic_partition(xyz, edges, scores, seeds, cfg):
    """u=0 retains parent, u=1 separates. Seedless components stay parent."""
    n = len(xyz)
    if (len(scores) != len(edges) or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError('Invalid edge scores.')
    if len(np.intersect1d(*seeds)) or any(not len(at) for at in seeds):
        raise ValueError('Need two nonempty disjoint seed sets.')
    weights = np.exp(-np.square(np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)) /
                     (2 * cfg['spatial_sigma_m'] ** 2)) * np.maximum(scores, 1e-4) ** cfg['score_power']
    adjacency = coo_matrix((np.r_[weights, weights],
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    _, components = connected_components(adjacency, directed=False)
    u = np.zeros(n, dtype=float)
    fixed = np.full(n, -1, dtype=np.int8)
    fixed[seeds[0]], fixed[seeds[1]] = 0, 1
    dual, seedless, iterations = np.zeros(n, bool), 0, 0
    for component in np.unique(components):
        at = np.flatnonzero(components == component)
        kinds = np.unique(fixed[at][fixed[at] >= 0])
        if not len(kinds):
            seedless += len(at)
            continue
        if len(kinds) == 1:
            u[at] = kinds[0]
            continue
        dual[at] = True
        local = adjacency[at][:, at]
        laplacian = diags(np.asarray(local.sum(1)).ravel()) - local
        unknown, boundary = np.flatnonzero(fixed[at] < 0), np.flatnonzero(fixed[at] >= 0)
        u[at[boundary]] = fixed[at[boundary]]
        if not len(unknown):
            continue
        system = laplacian[unknown][:, unknown].tocsr()
        rhs = -laplacian[unknown][:, boundary] @ fixed[at[boundary]].astype(float)
        def tick(_):
            nonlocal iterations
            iterations += 1
        result, info = cg(system, rhs, M=diags(1 / np.maximum(system.diagonal(), 1e-30)),
                          rtol=cfg['cg_rtol'], atol=0., maxiter=cfg['cg_maxiter'], callback=tick)
        if info != 0 or not np.isfinite(result).all():
            return None, dict(reason='solver_failed', cg_info=int(info), iterations=iterations)
        u[at[unknown]] = np.clip(result, 0, 1)
    return u, dict(reason='solved', seedless_points=int(seedless), dual_points=int(dual.sum()),
                   dual_mask=dual, iterations=iterations)


def jaccard(a, b):
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))


def split_proposal(xyz, source, seeds, graph_cfg, policy, scorer, geometry_only=False):
    """scorer(xyz, edges) uses observable input only; threshold applied later."""
    partitions, base_scores, base_edges, report = [], None, None, {}
    parity = (priorities(source, 42) & np.uint64(1)).astype(int)
    variants = [(graph_cfg['radius_m'], seeds)]
    for side in (0, 1):
        subset = [at[parity[at] == side] for at in seeds]
        if any(not len(at) for at in subset):
            return None, dict(reason='empty_hash_seeds')
        variants.append((graph_cfg['radius_m'], subset))
    variants.append((policy['stability_radius_m'], seeds))
    for number, (radius, selected) in enumerate(variants):
        edges = graph_edges(xyz, graph_cfg, radius)
        scores = scorer(edges)
        u, details = harmonic_partition(xyz, edges, scores, selected, graph_cfg)
        if u is None:
            return None, details
        mask = u > .5
        if number == 0:
            dual = details.pop('dual_mask')
            ambiguous = float(np.mean(np.abs(u[dual] - .5) < policy['ambiguous_margin'])) if dual.any() else 0.
            report = dict(details, ambiguous_fraction=ambiguous)
            base_scores, base_edges = scores, edges
        partitions.append(mask)
    mask = partitions[0]
    stability = [jaccard(mask, other) for other in partitions[1:]]
    cut = mask[base_edges[:, 0]] != mask[base_edges[:, 1]]
    cut_scores = base_scores
    if geometry_only:
        cut_scores = np.exp(-np.square(np.linalg.norm(xyz[base_edges[:, 0]] - xyz[base_edges[:, 1]], axis=1)) /
                            (2 * graph_cfg['spatial_sigma_m'] ** 2))
    report.update(stability_jaccard=stability, separated_points=int(mask.sum()),
                  cut_edges=int(cut.sum()), mean_cut_score=float(cut_scores[cut].mean()) if cut.any() else 0.,
                  cut_score_kind='spatial_kernel' if geometry_only else 'learned_affinity',
                  geometry_disconnected=bool(not cut.any()))
    minimum = max(policy['min_child_points'], int(np.ceil(policy['min_child_fraction'] * len(xyz))))
    if min(mask.sum(), len(mask) - mask.sum()) < minimum:
        return None, dict(report, reason='small_child')
    if report['ambiguous_fraction'] > policy['max_ambiguous_fraction']:
        return None, dict(report, reason='ambiguous_ownership')
    if min(stability) < policy['min_stability_jaccard']:
        return None, dict(report, reason='unstable_partition')
    return mask, dict(report, reason='proposal')
