"""CPU-only posthoc audit of saved relation candidates, scores and merge actions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.forest_abc.diagnose import analyze_table, assert_metrics, compare_models
from tools.instance_relation.run import code_signature, config_read
from tree_learn.instance_relation.data import entries, feature_request, load_features
from tree_learn.instance_relation.policy import select_pairs
from tree_learn.instance_relation.supervision import load_supervision, merge_table
from tree_learn.mask_refinement.data import file_digest, write_json

WATCHLIST = {259: (1375, 1376), 375: (1750, 1751), 419: (1828, 1829),
             629: (1549, 1550), 720: (1025, 1026), 801: (1080, 1324)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pair_truth(target, pairs, original):
    """Exact union geometry; IoU feasibility is not a Hungarian detection result."""
    common, counts, known = target['intersections'], target['gt_counts'], target['known']
    tp_gt = np.asarray([p['matched_gt_id'] for p in original['predictions']], dtype=np.int64)
    detected = tp_gt[tp_gt > 0]
    node_iou = np.zeros(len(common), dtype=np.float64)
    for start in range(0, len(common), 2048):
        stop = min(start + 2048, len(common))
        union = known[start:stop, None] + counts[None] - common[start:stop]
        node_iou[start:stop] = np.divide(common[start:stop], union,
            out=np.zeros_like(union, dtype=float), where=union > 0).max(1)
    n = len(pairs)
    quality = np.zeros((n, 3), dtype=np.float64)
    best_gt = np.zeros(n, dtype=np.int64)
    new_tp_gt = np.zeros(n, dtype=np.int64)
    for start in range(0, n, 2048):
        stop = min(start + 2048, n)
        a, b = pairs[start:stop].T
        intersection = common[a] + common[b]
        mass = known[a] + known[b]
        union = mass[:, None] + counts[None] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)
        best = iou.argmax(1)
        hit = intersection[np.arange(stop - start), best]
        maximum = iou[np.arange(stop - start), best]
        quality[start:stop] = np.column_stack((hit / np.maximum(mass, 1), hit / counts[best], maximum))
        best_gt[start:stop] = target['gt_ids'][best]
        new_tp_gt[start:stop] = np.where(maximum > .5, target['gt_ids'][best], 0)
    a, b = pairs.T
    lost = ((tp_gt[a] > 0) & (tp_gt[a] != new_tp_gt)).astype(np.int64)
    lost += ((tp_gt[b] > 0) & (tp_gt[b] != new_tp_gt)).astype(np.int64)
    recovered = (new_tp_gt > 0) & ~np.isin(new_tp_gt, detected)
    return dict(quality=quality, best_gt=best_gt, new_tp_gt=new_tp_gt, lost=lost, recovered=recovered,
                tp_gt=tp_gt, node_iou=node_iou, potential_recovery=recovered & (lost == 0))


def gates(scores, pairs, policy):
    a, b = pairs.T
    return dict(merge=scores['merge'] >= policy['merge_threshold'],
        same=scores['same'] >= policy['same_threshold'],
        union_precision=scores['quality'][:, 0] >= policy['min_union_precision'],
        union_recall=scores['quality'][:, 1] >= policy['min_union_recall'],
        union_iou=scores['quality'][:, 2] >= policy['min_union_iou'],
        union_gain=scores['gain'] >= policy['min_iou_gain'],
        original_protection=(scores['node_quality'][a] < policy['protect_threshold'])
                            & (scores['node_quality'][b] < policy['protect_threshold']))


def check_scores(scores, values):
    n, p = len(values['ids']), len(values['pairs'])
    for key, shape in dict(merge=(p,), same=(p,), quality=(p, 3), gain=(p,), node_quality=(n,)).items():
        if scores[key].shape != shape or not np.isfinite(scores[key]).all():
            raise ValueError(f'Invalid saved score array: {key}')
        lo = -1 if key == 'gain' else 0
        if np.any((scores[key] < lo) | (scores[key] > 1)):
            raise ValueError(f'Saved {key} values outside their output range.')
    accepted = scores['accepted']
    if (accepted.ndim != 2 or accepted.shape[1] != 2 or not np.issubdtype(accepted.dtype, np.integer)
            or np.any(accepted < 0) or np.any(accepted >= n)):
        raise ValueError('Invalid saved accepted instance pairs.')


def audit(values, target, scores, policy, watch_pairs=None):
    ids, pairs = values['ids'], values['pairs']
    if not np.array_equal(ids, target['ids']):
        raise ValueError('Features and supervision instance IDs differ.')
    check_scores(scores, values)
    accepted = select_pairs(pairs, scores, scores['node_quality'], policy)
    if not np.array_equal(accepted, scores['accepted']):
        raise ValueError('Saved actions do not reproduce the saved policy and scores.')
    geometry, table = merge_table(target, [])
    original = analyze_table(geometry, table)
    _, merged = merge_table(target, accepted)
    refined = analyze_table(geometry, merged)
    rows, comparison = compare_models(geometry, dict(baseline=original, refined=refined))
    truth = pair_truth(target, pairs, original)
    lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
    accepted_edges = np.asarray([lookup[tuple(pair)] for pair in accepted], dtype=np.int64)
    selected = np.zeros(len(pairs), dtype=bool)
    selected[accepted_edges] = True
    gate = gates(scores, pairs, policy)
    positive = target['pair_valid'] & (target['pair_merge'] > 0)

    def counts(mask):
        recovery = mask & truth['potential_recovery']
        return dict(pairs=int(mask.sum()), beneficial_label_pairs=int(np.count_nonzero(mask & positive)),
                    potential_recovery_pairs=int(recovery.sum()),
                    potential_recovery_gt_ids=np.unique(truth['new_tp_gt'][recovery]).tolist())

    surviving = np.ones(len(pairs), dtype=bool)
    stages = dict(all_candidates=counts(surviving))
    for name, mask in gate.items():
        surviving &= mask
        stages[name] = counts(surviving)
    stages['mutual_selection'] = counts(selected)

    isolated = {}

    def edge_record(edge):
        a, b = pairs[edge]
        before = truth['tp_gt'][[a, b]]
        lost = before[(before > 0) & (before != truth['new_tp_gt'][edge])].tolist()
        if edge not in isolated:
            _, table = merge_table(target, pairs[[edge]])
            one = analyze_table(geometry, table)
            transitions, _ = compare_models(geometry, dict(baseline=original, refined=one))
            isolated[edge] = dict(
                isolated_lost_gt_ids=[r['gt_id'] for r in transitions if r['transition'] == 'lost'],
                isolated_recovered_gt_ids=[r['gt_id'] for r in transitions if r['transition'] == 'recovered'],
                isolated_delta_counts={k: one['metrics'][k] - original['metrics'][k] for k in ('tp', 'fp', 'fn')},
                isolated_f1_delta_pp=100 * (one['metrics']['f1'] - original['metrics']['f1']))
        return dict(original_ids=ids[[a, b]].tolist(), accepted=bool(selected[edge]),
            failed_gates=[name for name, mask in gate.items() if not mask[edge]],
            passed_all_gates_but_not_mutual=bool(surviving[edge] and not selected[edge]),
            predicted_merge=float(scores['merge'][edge]), predicted_same=float(scores['same'][edge]),
            predicted_union_quality=scores['quality'][edge].tolist(),
            actual_union_quality=truth['quality'][edge].tolist(),
            predicted_gain=float(scores['gain'][edge]), actual_gain=float(target['pair_gain'][edge]),
            predicted_original_iou=scores['node_quality'][[a, b]].tolist(),
            actual_original_iou=truth['node_iou'][[a, b]].tolist(),
            original_matched_gt=before.tolist(), actual_union_best_gt=int(truth['best_gt'][edge]),
            supervised_valid=bool(target['pair_valid'][edge]), beneficial_label=bool(positive[edge]),
            actual_same_tree_label=bool(target['pair_same'][edge]),
            actual_union_meets_predicted_quality_cutoffs=bool(
                truth['quality'][edge, 0] >= policy['min_union_precision']
                and truth['quality'][edge, 1] >= policy['min_union_recall']
                and truth['quality'][edge, 2] >= policy['min_union_iou']),
            original_tp_losing_iou_match=lost,
            potential_recovered_gt_ids=[int(truth['new_tp_gt'][edge])] if truth['recovered'][edge] else [],
            **isolated[edge])

    actions = []
    for number, edge in enumerate(accepted_edges):
        actions.append(edge_record(int(edge)))
        print(f'AUDIT_ACTIONS pairs={number + 1}/{len(accepted_edges)}', flush=True)
    # Recompute the global matcher for each reported counterfactual. Maximizing
    # total IoU before thresholding can differ from matching only IoU > .5 edges.
    lost_ids = sorted(r['gt_id'] for r in rows if r['transition'] == 'lost')
    recovered_ids = sorted(r['gt_id'] for r in rows if r['transition'] == 'recovered')

    watchlist = []
    gt_lookup = {int(identifier): j for j, identifier in enumerate(target['gt_ids'])}
    baseline_trees = {r['gt_id']: r for r in original['trees']}
    final_trees = {r['gt_id']: r for r in refined['trees']}
    for identifier, expected_ids in (WATCHLIST if watch_pairs is None else watch_pairs).items():
        if identifier not in gt_lookup:
            continue
        at = np.searchsorted(ids, expected_ids)
        present = bool(np.all(at < len(ids)) and np.array_equal(ids[at], expected_ids))
        edge = lookup.get(tuple(at)) if present else None
        record = dict(gt_id=identifier, original_case_pair=list(expected_ids), original_instances_present=present,
            case_pair_in_candidates=edge is not None, baseline=baseline_trees[identifier], refined=final_trees[identifier])
        if edge is not None:
            record['case_pair'] = edge_record(edge)
            col = gt_lookup[identifier]
            a, b = pairs[edge]
            hit = target['intersections'][a, col] + target['intersections'][b, col]
            record['case_pair']['actual_iou_for_watched_gt'] = float(hit / (target['known'][a] + target['known'][b] + target['gt_counts'][col] - hit))
        recoveries = truth['potential_recovery'] & (truth['new_tp_gt'] == identifier)
        record['potential_recovery_candidate_pairs'] = int(recoveries.sum())
        record['potential_recovery_candidates_accepted'] = int(np.count_nonzero(recoveries & selected))
        watchlist.append(record)

    selected_nodes = np.unique(accepted)
    selected_tp = selected_nodes[truth['tp_gt'][selected_nodes] > 0]
    first_block = Counter()
    for edge in np.flatnonzero(truth['potential_recovery'] & ~selected):
        failed = next((name for name, mask in gate.items() if not mask[edge]), 'mutual_selection')
        first_block[failed] += 1
    brief = dict(schema=1, posthoc=True, eligible_for_advancement=False,
        scope='saved_candidates_scores_and_exact_union_counts', official_model_loaded=False,
        baseline=original['metrics'], refined=refined['metrics'], accepted_pairs=len(accepted),
        lost_gt_ids=lost_ids, recovered_gt_ids=recovered_ids,
        accepted_harmful_pairs_in_isolation=sum(bool(r['isolated_lost_gt_ids']) for r in actions),
        accepted_beneficial_label_pairs=int(np.count_nonzero(positive & selected)),
        accepted_pairs_covering_two_original_tp=sum(all(i > 0 for i in r['original_matched_gt']) for r in actions),
        selected_original_tp_masks=len(selected_tp),
        selected_tp_predicted_below_protection=int(np.count_nonzero(scores['node_quality'][selected_tp] < policy['protect_threshold'])),
        selected_tp_actual_below_protection=int(np.count_nonzero(truth['node_iou'][selected_tp] < policy['protect_threshold'])),
        selected_tp_actual_at_or_above_protection=int(np.count_nonzero(truth['node_iou'][selected_tp] >= policy['protect_threshold'])),
        potential_recovery_candidates=counts(truth['potential_recovery']),
        potential_recovery_first_rejection=dict(first_block), stages=stages,
        potential_recovery_note='IoU-feasible candidate union, not a guaranteed global Hungarian detection; reported actions and watched pairs have exact isolated replays.',
        harmful_actions=[{key: action[key] for key in (
            'original_ids', 'original_matched_gt', 'isolated_lost_gt_ids',
            'predicted_merge', 'predicted_original_iou', 'actual_original_iou',
            'predicted_union_quality', 'actual_union_quality', 'predicted_gain', 'actual_gain',
            'actual_same_tree_label')}
            for action in actions if action['isolated_lost_gt_ids']],
        isolated_pair_sum_matches_combined_counts=all(
            sum(r['isolated_delta_counts'][k] for r in actions) == refined['metrics'][k] - original['metrics'][k]
            for k in ('tp', 'fp', 'fn')),
        transition_counts=comparison['transitions'])
    return brief, actions, watchlist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/instance_relation/full.yaml')
    parser.add_argument('--source', action='store_true', help='Audit five validation forests instead of Wytham.')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    # Same repository-relative paths as the foreground training entry point.
    import os
    os.chdir(ROOT)
    config = config_read(config_path)
    work = Path(config['work_dir'])
    output = work / ('evaluation' if args.source else 'external_evaluation')
    rows = entries(config, external=not args.source)
    if args.source:
        rows = [r for r in rows if r['split'] == 'validation']
    source_counts = []
    for row in entries(config):
        path = Path(row['prepared']) / 'supervision.json'
        meta = read_json(path)
        source_counts.append(dict(plot=row['plot'], split=row['split'], valid_pairs=meta['num_valid_pairs'],
                                  merge_positive_pairs=meta['num_merge_positive'], valid_nodes=meta['num_valid_nodes']))
    for row in rows:
        prepared, saved = Path(row['prepared']), output / row['plot']
        print(f'AUDIT {row["plot"]}: saved arrays only; no forward pass or full-forest scan', flush=True)
        guarded = (prepared / 'features.npz', prepared / 'supervision.npz', saved / 'scores.npz',
                   saved / 'prediction.json', saved / 'metrics.json', work / 'best.pth')
        before = {str(path): file_digest(path) for path in guarded}
        feature_meta, marker = read_json(prepared / 'features.json'), read_json(saved / 'prediction.json')
        if (feature_meta['request'] != feature_request(row['cache'], dict(config['geometry'], seed=config['seed']))
                or marker['request']['signature'] != code_signature(config)
                or marker['request']['features'] != feature_meta['npz_sha256']
                or marker['request']['checkpoint_sha256'] != file_digest(work / 'best.pth')
                or marker['scores_sha256'] != file_digest(saved / 'scores.npz')):
            raise ValueError('Saved results belong to changed features/model/code/scores.')
        if marker['changed_foreground_points'] or marker['changed_unselected_points']:
            raise ValueError('Saved result failed the required point-set invariants.')
        values = load_features(prepared)
        target = load_supervision(prepared, config['supervision'])
        with np.load(saved / 'scores.npz', allow_pickle=False) as stored:
            scores = {key: stored[key].copy() for key in stored.files}
        brief, actions, watchlist = audit(values, target, scores, marker['request']['policy'],
                                         WATCHLIST if row['plot'] == 'wytham_vox0.1' else {})
        recorded = read_json(saved / 'metrics.json')
        for name in ('baseline', 'refined'):
            assert_metrics(recorded[name], brief[name])
        if (sorted(recorded['lost_gt_ids']) != brief['lost_gt_ids']
                or sorted(recorded['recovered_gt_ids']) != brief['recovered_gt_ids']
                or marker['accepted_pairs'] != brief['accepted_pairs']):
            raise ValueError('Replayed transitions/actions differ from the saved evaluation.')
        if any(file_digest(path) != before[str(path)] for path in guarded):
            raise ValueError('Saved artifacts changed while the audit was running; retry after the workflow exits.')
        brief.update(checkpoint_step=marker['request']['step'], plot=row['plot'], policy=marker['request']['policy'],
            source_supervision_counts=source_counts,
            provenance=dict(prediction_marker_sha256=file_digest(saved / 'prediction.json'),
                scores_sha256=marker['scores_sha256'], supervision_sha256=file_digest(prepared / 'supervision.npz'),
                audit_code_sha256=file_digest(__file__)),
            watched_cases=[dict(gt_id=r['gt_id'], case_pair_in_candidates=r['case_pair_in_candidates'],
                failed_gates=r.get('case_pair', {}).get('failed_gates'),
                potential_recovery_candidate_pairs=r['potential_recovery_candidate_pairs'],
                accepted=r.get('case_pair', {}).get('accepted', False)) for r in watchlist])
        directory = work / 'relation_diagnostics' / row['plot']
        write_json(directory / 'accepted_pair_audit.json', actions)
        write_json(directory / 'watchlist.json', watchlist)
        write_json(directory / 'brief.json', brief)
        print(json.dumps(brief, indent=2, ensure_ascii=False), flush=True)
        print(f'SAVED {directory / "brief.json"}', flush=True)


if __name__ == '__main__':
    main()
