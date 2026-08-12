"""GT-free vote-space proposals for conservative base-seed completion."""

import numpy as np

from .seed_completion_diagnostics import (
    _stable_top_k,
    cluster_from_base_seed_mask,
    evaluate_instance_predictions,
)
from .seed_quality import candidate_seed_mask, tree_probabilities
from .omission_diagnostics import compute_gt_omission_diagnostics


PROPOSAL_FEATURE_NAMES = (
    'scale', 'shift_x', 'shift_y', 'semantic_count',
    'base_seed_count', 'added_seed_count',
    'tree_probability_mean', 'tree_probability_std',
    'verticality_mean', 'verticality_std',
    'offset_z_abs_mean', 'offset_z_abs_std',
    'vote_radius_rms', 'margin_mean', 'margin_std',
)


def _cell_inverse(votes_xy, scale, shift_x, shift_y):
    if float(scale) <= 0:
        raise ValueError('Proposal scale must be positive.')
    shift = np.asarray([shift_x, shift_y], dtype=np.float64)
    cells = np.floor(
        np.asarray(votes_xy, dtype=np.float64) / float(scale) - shift
    ).astype(np.int64)
    dtype = np.dtype([('x', '<i8'), ('y', '<i8')])
    keys = np.ascontiguousarray(cells).view(dtype).reshape(-1)
    unique, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    unique_cells = unique.view(np.int64).reshape(-1, 2)
    return unique_cells, inverse.astype(np.int64), counts.astype(np.int64)


def build_multiscale_seed_completion_proposals(
        coords, semantic_logits, offset_predictions, verticality,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0, tau_min=50,
        tree_class_index=0, scales=(0.6, 1.2),
        shift_fractions=(0.0, 0.5)):
    """Build proposals without reading point or instance GT labels."""
    xyz = np.asarray(coords, dtype=np.float32)
    logits = np.asarray(semantic_logits, dtype=np.float32)
    offsets = np.asarray(offset_predictions, dtype=np.float32)
    vertical = np.asarray(verticality, dtype=np.float32).reshape(-1)
    if xyz.shape != offsets.shape or xyz.shape != (len(vertical), 3):
        raise ValueError('Q5b0 point arrays are not aligned.')
    if len(logits) != len(xyz):
        raise ValueError('Q5b0 semantic logits are not aligned.')
    if not scales or not shift_fractions:
        raise ValueError('Q5b0 proposal grid is empty.')

    probabilities = tree_probabilities(logits, tree_class_index)
    semantic = probabilities >= float(tree_conf_thresh)
    baseline = candidate_seed_mask(
        logits, vertical, offsets,
        tree_conf_thresh=tree_conf_thresh,
        tau_vert=tau_vert, tau_off=tau_off,
        tree_class_index=tree_class_index)
    semantic_indices = np.flatnonzero(semantic)
    semantic_votes = (
        xyz[semantic_indices, :2] + offsets[semantic_indices, :2])
    point_margin = (
        probabilities.astype(np.float64) +
        np.clip(vertical.astype(np.float64), 0.0, 1.0) +
        np.exp(-np.abs(offsets[:, 2].astype(np.float64)) /
               max(float(tau_off), 1e-6)))

    proposals = []
    seen_selected_sets = set()
    proposal_id = 0
    for scale in map(float, scales):
        for shift_x in map(float, shift_fractions):
            for shift_y in map(float, shift_fractions):
                unique_cells, inverse, counts = _cell_inverse(
                    semantic_votes, scale, shift_x, shift_y)
                seed_counts = np.bincount(
                    inverse,
                    weights=baseline[semantic_indices].astype(np.int64),
                    minlength=len(counts)).astype(np.int64)
                candidate_cells = np.flatnonzero(
                    (counts >= int(tau_min)) &
                    (seed_counts < int(tau_min)))
                if len(candidate_cells) == 0:
                    continue
                order = np.argsort(inverse, kind='stable')
                starts = np.concatenate((
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(counts, dtype=np.int64)))
                for cell_id in candidate_cells:
                    members = semantic_indices[
                        order[starts[cell_id]:starts[cell_id + 1]]]
                    available = members[~baseline[members]]
                    need = int(tau_min) - int(seed_counts[cell_id])
                    if len(available) < need:
                        raise AssertionError(
                            'A semantic proposal cannot be topped up.')
                    selected = _stable_top_k(
                        available, point_margin[available], need)
                    signature = tuple(map(int, selected.tolist()))
                    if signature in seen_selected_sets:
                        continue
                    seen_selected_sets.add(signature)
                    votes = xyz[members, :2] + offsets[members, :2]
                    center = votes.mean(axis=0, dtype=np.float64)
                    radius = np.sqrt(np.mean(np.sum(
                        (votes.astype(np.float64) - center) ** 2,
                        axis=1)))
                    proposal_id += 1
                    proposals.append({
                        'proposal_id': proposal_id,
                        'scale': scale,
                        'shift_x': shift_x,
                        'shift_y': shift_y,
                        'cell_x': int(unique_cells[cell_id, 0]),
                        'cell_y': int(unique_cells[cell_id, 1]),
                        'semantic_count': int(len(members)),
                        'base_seed_count': int(seed_counts[cell_id]),
                        'added_seed_count': int(need),
                        'tree_probability_mean': float(
                            probabilities[members].mean()),
                        'tree_probability_std': float(
                            probabilities[members].std()),
                        'verticality_mean': float(vertical[members].mean()),
                        'verticality_std': float(vertical[members].std()),
                        'offset_z_abs_mean': float(
                            np.abs(offsets[members, 2]).mean()),
                        'offset_z_abs_std': float(
                            np.abs(offsets[members, 2]).std()),
                        'vote_radius_rms': float(radius),
                        'margin_mean': float(point_margin[members].mean()),
                        'margin_std': float(point_margin[members].std()),
                        'selected_indices': selected.astype(np.int64),
                    })
    return {
        'baseline_mask': baseline,
        'semantic_mask': semantic,
        'proposals': proposals,
    }


def select_oracle_seed_completion_proposals(
        proposals, baseline_mask, instance_labels, target_tree_ids,
        tau_min=50):
    """Use GT only to measure the upper bound of GT-free proposals."""
    baseline = np.asarray(baseline_mask, dtype=bool).reshape(-1)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    if len(baseline) != len(labels):
        raise ValueError('Q5b0 Oracle arrays are not aligned.')
    targets = sorted(set(map(int, target_tree_ids)))
    target_set = set(targets)
    annotations = []
    proposals_by_target = {tree_id: [] for tree_id in targets}
    for proposal in proposals:
        selected = proposal['selected_indices']
        selected_labels = labels[selected]
        positive = selected_labels[selected_labels > 0]
        if len(positive):
            ids, counts = np.unique(positive, return_counts=True)
            best = np.lexsort((ids, -counts))[0]
            dominant_id = int(ids[best])
            dominant_count = int(counts[best])
        else:
            dominant_id = 0
            dominant_count = 0
        row = {
            key: proposal[key]
            for key in (
                'proposal_id', 'scale', 'shift_x', 'shift_y',
                'cell_x', 'cell_y', 'semantic_count', 'base_seed_count',
                'added_seed_count', 'tree_probability_mean',
                'tree_probability_std', 'verticality_mean',
                'verticality_std', 'offset_z_abs_mean',
                'offset_z_abs_std', 'vote_radius_rms',
                'margin_mean', 'margin_std')}
        row.update({
            'oracle_dominant_tree_id': dominant_id,
            'oracle_dominant_added_points': dominant_count,
            'oracle_selected_purity': float(
                dominant_count / max(len(selected), 1)),
            'oracle_activated': False,
        })
        target_ids, target_counts = np.unique(
            selected_labels[np.isin(selected_labels, targets)],
            return_counts=True)
        target_contributions = {
            int(tree_id): int(count)
            for tree_id, count in zip(target_ids, target_counts)}
        row.update({
            'oracle_target_tree_ids': ';'.join(map(
                str, sorted(target_contributions))),
            'oracle_target_added_points': int(max(
                target_contributions.values(), default=0)),
            'oracle_target_proposal': bool(target_contributions),
        })
        annotations.append(row)
        for tree_id, target_count in target_contributions.items():
            proposals_by_target[tree_id].append(
                (proposal, row, target_count))

    output_mask = baseline.copy()
    per_target = []
    activated_ids = set()
    for tree_id in targets:
        current = int(np.count_nonzero(baseline & (labels == tree_id)))
        ranked = sorted(
            proposals_by_target[tree_id],
            key=lambda value: (
                -value[2],
                -value[1]['oracle_selected_purity'],
                value[0]['added_seed_count'],
                value[0]['proposal_id']))
        for proposal, row, _ in ranked:
            if int(np.count_nonzero(
                    output_mask & (labels == tree_id))) >= int(tau_min):
                break
            output_mask[proposal['selected_indices']] = True
            row['oracle_activated'] = True
            activated_ids.add(int(proposal['proposal_id']))
        final_count = int(np.count_nonzero(
            output_mask & (labels == tree_id)))
        per_target.append({
            'gt_tree_id': int(tree_id),
            'baseline_seed_count': current,
            'oracle_seed_count': final_count,
            'proposal_covered': bool(final_count >= int(tau_min)),
            'candidate_proposals': int(len(ranked)),
        })
    return {
        'seed_mask': output_mask,
        'proposal_rows': annotations,
        'per_target': per_target,
        'activated_proposal_ids': sorted(activated_ids),
    }


def proposal_rows_for_csv(rows):
    output = []
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            safe[key] = value
        output.append(safe)
    return output


def analyze_seed_completion_proposal_oracle(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, baseline_predictions,
        baseline_initial_predictions, target_tree_ids,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0,
        tau_min=50, tree_class_index=0, scales=(0.6, 1.2),
        shift_fractions=(0.0, 0.5), match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5,
        min_recall_for_undersegmentation=0.5,
        min_fragment_overlap_fraction=0.05,
        min_fragment_overlap_points=20, knn_chunk_size=200000,
        max_cluster_seed_points=None, logger=None):
    q3 = compute_gt_omission_diagnostics(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, baseline_predictions,
        baseline_initial_predictions,
        tree_conf_thresh=tree_conf_thresh, tau_vert=tau_vert,
        tau_off=tau_off, tau_min=tau_min,
        tree_class_index=tree_class_index,
        match_iou_threshold=match_iou_threshold,
        min_precision_for_counted_fp=min_precision_for_counted_fp,
        min_recall_for_undersegmentation=(
            min_recall_for_undersegmentation),
        min_fragment_overlap_fraction=min_fragment_overlap_fraction,
        min_fragment_overlap_points=min_fragment_overlap_points)
    expected_targets = sorted(set(map(int, target_tree_ids)))
    observed_targets = sorted(
        int(row['gt_tree_id']) for row in q3['rows']
        if row['category'] == 'base_seed_support_failure')
    if observed_targets != expected_targets:
        raise ValueError(
            'Q5b0 targets differ from fixed Q3: '
            f'observed={observed_targets}, expected={expected_targets}.')
    baseline_metrics = evaluate_instance_predictions(
        instance_labels, baseline_predictions,
        match_iou_threshold, min_precision_for_counted_fp)
    baseline_detected = set(
        baseline_metrics.pop('detected_gt_ids'))
    if any(int(baseline_metrics[key]) != int(q3['baseline'][key])
           for key in ('tp', 'fp', 'fn')):
        raise ValueError('Q5b0 baseline does not reproduce Q3.')
    generated = build_multiscale_seed_completion_proposals(
        coords, semantic_logits, offset_predictions, verticality,
        tree_conf_thresh=tree_conf_thresh, tau_vert=tau_vert,
        tau_off=tau_off, tau_min=tau_min,
        tree_class_index=tree_class_index, scales=scales,
        shift_fractions=shift_fractions)
    print(
        f'Q5b0: generated {len(generated["proposals"]):,} '
        'GT-free proposals.', flush=True)
    selected = select_oracle_seed_completion_proposals(
        generated['proposals'], generated['baseline_mask'],
        instance_labels, expected_targets, tau_min=tau_min)
    print(
        f'Q5b0: activated '
        f'{len(selected["activated_proposal_ids"]):,} proposals; '
        f'clustering {int(selected["seed_mask"].sum()):,} seeds...',
        flush=True)
    predictions, _ = cluster_from_base_seed_mask(
        coords, offset_predictions, semantic_logits,
        selected['seed_mask'], tree_conf_thresh=tree_conf_thresh,
        tau_min=tau_min, tree_class_index=tree_class_index,
        knn_chunk_size=knn_chunk_size,
        max_cluster_seed_points=max_cluster_seed_points,
        logger=logger)
    metrics = evaluate_instance_predictions(
        instance_labels, predictions,
        match_iou_threshold, min_precision_for_counted_fp)
    detected = set(metrics.pop('detected_gt_ids'))
    targets = set(expected_targets)
    recovered = sorted((detected - baseline_detected) & targets)
    lost = sorted(baseline_detected - detected)
    print(
        f'Q5b0: clustering finished; recovered={len(recovered)}, '
        f'lost={len(lost)}, F1={100.0 * metrics["f1"]:.3f}%.',
        flush=True)
    metrics.update({
        'recovered_target_tree_ids': recovered,
        'recovered_target_trees': int(len(recovered)),
        'lost_baseline_tree_ids': lost,
        'lost_baseline_trees': int(len(lost)),
    })
    return {
        'baseline': baseline_metrics,
        'num_target_trees': int(len(targets)),
        'num_proposals': int(len(generated['proposals'])),
        'num_activated_proposals': int(
            len(selected['activated_proposal_ids'])),
        'num_baseline_seeds': int(generated['baseline_mask'].sum()),
        'num_oracle_seeds': int(selected['seed_mask'].sum()),
        'proposal_covered_target_trees': int(sum(
            row['proposal_covered'] for row in selected['per_target'])),
        'per_target': selected['per_target'],
        'proposal_rows': proposal_rows_for_csv(selected['proposal_rows']),
        'oracle_metrics': metrics,
        'proposal_parameters': {
            'scales': [float(value) for value in scales],
            'shift_fractions': [float(value) for value in shift_fractions],
            'tau_min': int(tau_min),
        },
    }
