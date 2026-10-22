"""CPU posthoc audit of saved V2 decisions; no model, inference or policy search."""

import argparse
from collections import Counter
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.forest_abc.diagnose import analyze_table, assert_metrics, compare_models
from tools.instance_relation_v2.run import config_read, read_json, signature
from tree_learn.instance_relation.data import load_features
from tree_learn.instance_relation.supervision import load_supervision, merge_table
from tree_learn.instance_relation_v2 import data, policy
from tree_learn.instance_relation_v2.supervision import utility
from tree_learn.mask_refinement.data import file_digest, write_json

WATCHLIST = {259: (1375, 1376), 375: (1750, 1751), 419: (1828, 1829),
             629: (1549, 1550), 720: (1025, 1026), 801: (1080, 1324)}


def check_scores(values, scores):
    n, p = len(values['ids']), len(values['pairs'])
    pairs = values['pairs']
    if (pairs.shape != (p, 2) or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0) or np.any(pairs >= n) or np.any(pairs[:, 0] >= pairs[:, 1])
            or len(np.unique(pairs, axis=0)) != p):
        raise ValueError('Require unique sorted original candidate pairs.')
    for key, shape in dict(action=(p, 3), risk=(p,), utility=(p,), quality=(p, 3)).items():
        value = scores[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid saved {key} scores.')
        lo = -1 if key == 'utility' else 0
        if np.any((value < lo) | (value > 1)):
            raise ValueError(f'Saved {key} scores outside output range.')
    if not np.allclose(scores['action'].sum(1), 1, rtol=0, atol=1e-5):
        raise ValueError('Action probabilities must sum to one.')
    accepted = scores['accepted']
    if (accepted.ndim != 2 or accepted.shape[1] != 2
            or not np.issubdtype(accepted.dtype, np.integer)):
        raise ValueError('Invalid saved accepted pairs.')


def replay(targets, accepted, original=None):
    geometry, table = merge_table(targets, accepted)
    refined = analyze_table(geometry, table)
    if original is None:
        return refined
    trees, comparison = compare_models(geometry, dict(baseline=original, refined=refined))
    return refined, dict(lost=[r['gt_id'] for r in trees if r['transition'] == 'lost'],
                         recovered=[r['gt_id'] for r in trees if r['transition'] == 'recovered'])


def audit(values, targets, scores, settings, threshold, watch_pairs=None):
    """Replay selected/watched pairs individually and all selected pairs jointly."""
    check_scores(values, scores)
    ids, pairs = values['ids'], values['pairs']
    if not np.array_equal(ids, targets['ids']):
        raise ValueError('Candidate and contingency-table instance IDs differ.')
    accepted = policy.select_pairs(pairs, scores, settings, threshold)
    if not np.array_equal(accepted, scores['accepted']):
        raise ValueError('Saved actions do not reproduce the saved policy and scores.')
    original = replay(targets, [])
    refined, transitions = replay(targets, accepted, original)
    lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
    selected = {lookup[tuple(pair)] for pair in accepted}
    combined = policy.action_score(scores, settings)
    ranked = sorted(np.flatnonzero(combined >= threshold).tolist(),
                    key=lambda i: (-float(combined[i]), int(pairs[i, 0]), int(pairs[i, 1])))
    best = {}
    for edge in ranked:
        for node in pairs[edge]:
            best.setdefault(int(node), edge)
    predictions = {r['pred_id']: r for r in original['predictions']}
    records = {}

    def record(edge):
        if edge in records:
            return records[edge]
        pair = pairs[edge]
        isolated, change = replay(targets, [pair], original)
        delta = {key: isolated['metrics'][key] - original['metrics'][key] for key in ('tp', 'fp', 'fn')}
        valid = bool(targets['pair_valid'][edge])
        same = bool(targets['pair_same'][edge])
        safe = valid and targets['pair_merge'][edge] > 0 and not change['lost']
        actual_action = ('recover' if change['recovered'] else 'complete') if safe else 'keep'
        blockers = sorted({best[int(node)] for node in pair if int(node) in best and best[int(node)] != edge})
        joined = targets['intersections'][pair].sum(0)
        mass = int(targets['known'][pair].sum())
        denom = mass + targets['gt_counts'] - joined
        iou = np.divide(joined, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
        col = int(iou.argmax())
        matched = [int(predictions[int(i)]['matched_gt_id']) for i in ids[pair]]
        action = scores['action'][edge]
        result = dict(original_ids=ids[pair].tolist(), accepted=edge in selected,
            decision=('accepted' if edge in selected else 'below_threshold' if combined[edge] < threshold
                      else 'mutual_selection'), score=float(combined[edge]), threshold=float(threshold),
            score_terms=dict(recover=float(action[1]), complete=float(settings['completion_credit'] * action[2]),
                utility=float(settings['utility_weight'] * scores['utility'][edge]),
                risk_penalty=float(-settings['risk_weight'] * scores['risk'][edge])),
            predicted_action=action.tolist(), predicted_risk=float(scores['risk'][edge]),
            predicted_utility=float(scores['utility'][edge]), auxiliary_quality=scores['quality'][edge].tolist(),
            blocking_pairs=[dict(original_ids=ids[pairs[i]].tolist(), score=float(combined[i]),
                                 accepted=i in selected) for i in blockers],
            original_matched_gt=matched, two_original_tp=all(i > 0 for i in matched),
            original_masks=[predictions[int(i)] for i in ids[pair]],
            actual_label=actual_action if valid else 'unlabeled', same_tree_label=same if valid else None,
            isolated_delta_counts=delta, isolated_lost_gt_ids=change['lost'],
            isolated_recovered_gt_ids=change['recovered'],
            actual_risk=bool(change['lost'] or not same) if valid else None,
            actual_utility=float(utility(len(change['recovered']), len(change['lost']), delta['fp'])),
            actual_union_best_gt_id=int(targets['gt_ids'][col]),
            actual_union_quality=[float(joined[col] / max(mass, 1)),
                                  float(joined[col] / targets['gt_counts'][col]), float(iou[col])])
        records[edge] = result
        return result

    actions = []
    for number, pair in enumerate(accepted):
        actions.append(record(lookup[tuple(pair)]))
        print(f'AUDIT_ACTIONS {number + 1}/{len(accepted)}', flush=True)
    watchlist = []
    baseline_trees = {r['gt_id']: r for r in original['trees']}
    final_trees = {r['gt_id']: r for r in refined['trees']}
    for gt_id, expected in (watch_pairs or {}).items():
        at = np.searchsorted(ids, expected)
        present = bool(np.all(at < len(ids)) and np.array_equal(ids[at], expected))
        edge = lookup.get(tuple(at)) if present else None
        item = dict(gt_id=gt_id, original_case_pair=list(expected), case_pair_in_candidates=edge is not None,
                    baseline=baseline_trees.get(gt_id), refined=final_trees.get(gt_id))
        if edge is not None:
            item['pair'] = record(edge)
        watchlist.append(item)

    def distribution(group, key):
        array = np.asarray([r[key] for r in group], dtype=float)
        return dict(n=len(array), minimum=float(array.min()), median=float(np.median(array)),
                    maximum=float(array.max())) if len(array) else dict(n=0)

    harmful = [r for r in actions if r['isolated_lost_gt_ids']]
    recovery = [r for r in actions if r['isolated_recovered_gt_ids'] and not r['isolated_lost_gt_ids']]
    # Strict action labels and exact action benefit are different quantities.
    beneficial = [r for r in actions if not r['isolated_lost_gt_ids'] and
                  (r['isolated_recovered_gt_ids'] or r['isolated_delta_counts']['fp'] < 0)]
    brief = dict(schema=2, posthoc=True, eligible_for_advancement=False,
        scope='saved_v2_scores_and_exact_union_counts', official_model_loaded=False,
        baseline=original['metrics'], refined=refined['metrics'], accepted_pairs=len(actions),
        lost_gt_ids=transitions['lost'], recovered_gt_ids=transitions['recovered'],
        accepted_two_original_tp=sum(r['two_original_tp'] for r in actions),
        accepted_harmful_pairs_in_isolation=len(harmful), accepted_beneficial_pairs_in_isolation=len(beneficial),
        selected_strict_label_counts=dict(Counter(r['actual_label'] for r in actions)),
        accepted_keep_label_but_beneficial_in_isolation=sum(r['actual_label'] == 'keep' for r in beneficial),
        isolated_pair_sum_matches_combined_counts=all(
            sum(r['isolated_delta_counts'][k] for r in actions) == refined['metrics'][k] - original['metrics'][k]
            for k in ('tp', 'fp', 'fn')),
        risk_and_value_by_outcome={name: {key: distribution(group, key) for key in (
            'predicted_risk', 'predicted_utility', 'score')} for name, group in (
                ('harmful_in_isolation', harmful), ('safe_recovery_in_isolation', recovery), ('all_accepted', actions))},
        harmful_actions=[{key: r[key] for key in ('original_ids', 'original_matched_gt', 'isolated_lost_gt_ids',
            'predicted_risk', 'predicted_utility', 'score', 'same_tree_label')} for r in harmful],
        watched_cases=[dict(gt_id=r['gt_id'], case_pair_in_candidates=r['case_pair_in_candidates'],
            decision=r.get('pair', {}).get('decision'), score=r.get('pair', {}).get('score'),
            isolated_lost_gt_ids=r.get('pair', {}).get('isolated_lost_gt_ids'),
            isolated_recovered_gt_ids=r.get('pair', {}).get('isolated_recovered_gt_ids')) for r in watchlist])
    return brief, actions, watchlist


def calibration_history(work):
    """Expose existing source validation history; never retune using external GT."""
    result = []
    for path in sorted((Path(work) / 'calibration').glob('step_*.json')):
        report = read_json(path)
        selected = report['selected']
        result.append(dict(step=report['step'], threshold=selected['threshold'],
            f1=selected['macro']['f1'], lost=selected['lost'], recovered=selected['recovered'],
            accepted_pairs=selected['accepted_pairs'],
            recovery_classifier=report['natural_recovery_classifier'],
            selected_label_counts=selected['selected_labeled_counts']))
    return result


def audit_saved(config, row, evaluation, output, expected_step=None):
    prepared, base = Path(row['prepared']), Path(row['base_prepared'])
    saved = Path(evaluation) / row['plot']
    guarded = [base / name for name in ('features.json', 'features.npz', 'supervision.json', 'supervision.npz')]
    guarded += [prepared / name for name in ('points.json', 'points.npz')]
    guarded += [saved / name for name in ('scores.npz', 'prediction.json', 'metrics.json',
                                        'accepted_merges.json', 'action_diagnostics.json')]
    guarded += [Path(config['work_dir']) / 'best.pth']
    before = {str(path): file_digest(path) for path in guarded}
    marker, point_meta = read_json(saved / 'prediction.json'), read_json(prepared / 'points.json')
    request = marker['request']
    if expected_step is not None and request['step'] != expected_step:
        raise ValueError(f'Expected step {expected_step}, found {request["step"]}.')
    if (point_meta['request'] != data.request(row, config)
            or point_meta['sha256'] != before[str(prepared / 'points.npz')]
            or request['points_sha256'] != point_meta['sha256']
            or request['signature'] != signature(config)
            or request['plot'] != row['plot'] or request['policy'] != config['policy']
            or request['checkpoint_sha256'] != before[str(Path(config['work_dir']) / 'best.pth')]
            or marker['scores_sha256'] != before[str(saved / 'scores.npz')]):
        raise ValueError('Changed V2 inputs, model, code, configuration or saved scores.')
    if marker['changed_foreground_points'] or marker['changed_unselected_points'] or marker['gt_opened']:
        raise ValueError('Saved prediction violates required point-set or GT-isolation invariants.')
    # Load compact original descriptors/counts only. No full point-array scan or torch.load.
    values, targets = load_features(base), load_supervision(base, config['supervision'])
    with np.load(saved / 'scores.npz', allow_pickle=False) as stored:
        scores = {key: stored[key].copy() for key in stored.files}
    brief, actions, watchlist = audit(values, targets, scores, request['policy'], request['threshold'],
                                     WATCHLIST if row['plot'] == 'wytham_vox0.1' else {})
    recorded = read_json(saved / 'metrics.json')
    for name in ('baseline', 'refined'):
        assert_metrics(recorded[name], brief[name])
    for name in ('lost_gt_ids', 'recovered_gt_ids', 'accepted_pairs', 'accepted_two_original_tp'):
        if recorded[name] != brief[name]:
            raise ValueError('Saved evaluation differs from replay: ' + name)
    if marker['accepted_pairs'] != brief['accepted_pairs']:
        raise ValueError('Prediction marker and replay have different accepted counts.')
    prediction_actions = {tuple(r['original_ids']): r for r in read_json(saved / 'accepted_merges.json')}
    actual_actions = {tuple(r['original_ids']): r for r in read_json(saved / 'action_diagnostics.json')}
    keys = {tuple(r['original_ids']) for r in actions}
    if set(prediction_actions) != keys or set(actual_actions) != keys:
        raise ValueError('Saved per-action records have different instance IDs.')
    for action in actions:
        key = tuple(action['original_ids'])
        predicted, actual = prediction_actions[key], actual_actions[key]
        for name in ('score', 'predicted_risk', 'predicted_utility'):
            if not np.isclose(predicted[name], action[name], rtol=0, atol=1e-6):
                raise ValueError('Saved per-action scores disagree with scores.npz.')
        expected = dict(isolated_delta_tp=action['isolated_delta_counts']['tp'],
            isolated_delta_fp=action['isolated_delta_counts']['fp'],
            isolated_lost=len(action['isolated_lost_gt_ids']), isolated_recovered=len(action['isolated_recovered_gt_ids']))
        if any(actual[name] != value for name, value in expected.items()) or not np.allclose(
                actual['actual_union_quality'], action['actual_union_quality'], rtol=0, atol=1e-6):
            raise ValueError('Saved action diagnosis disagrees with exact isolated replay.')
    if any(file_digest(path) != before[str(path)] for path in guarded):
        raise ValueError('Inputs changed during the audit; retry after the workflow exits.')
    brief.update(plot=row['plot'], checkpoint_step=request['step'], threshold=request['threshold'],
        provenance=dict(prediction_marker_sha256=before[str(saved / 'prediction.json')],
            scores_sha256=marker['scores_sha256'], audit_code_sha256=file_digest(__file__)),
        verification='Saved-score/count replay; original point-set invariants are read from prediction.json, not rescanned.')
    directory = Path(output) / row['plot']
    write_json(directory / 'accepted_pair_audit.json', actions)
    write_json(directory / 'watchlist.json', watchlist)
    write_json(directory / 'brief.json', brief)
    print('SAVED ' + str(directory / 'brief.json'), flush=True)
    return brief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/instance_relation_v2/full.yaml')
    parser.add_argument('--split', choices=('both', 'source', 'external'), default='both')
    parser.add_argument('--expect-step', type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    os.chdir(ROOT)
    config = config_read(config_path)
    work = Path(config['work_dir'])
    output = work / 'action_audit'
    forests = []
    for external in (False, True):
        if (external and args.split == 'source') or (not external and args.split == 'external'):
            continue
        rows = data.rows(config, external=external)
        for row in rows:
            if external or row['split'] == 'validation':
                print(f'AUDIT {row["plot"]}: saved scores and counts, no GPU/model', flush=True)
                forests.append(audit_saved(config, row,
                    work / ('external_evaluation' if external else 'evaluation'), output, args.expect_step))
    report = dict(schema=2, posthoc=True, eligible_for_advancement=False,
        split=args.split, forests=forests, source_calibration_history=calibration_history(work),
        notes=['No model, forward pass, training, full-forest scan or policy change.',
               'Pair classifier TP/FP are not tree-instance TP/FP.',
               'Strict keep labels do not imply every selected pair is harmful; inspect exact action effects.',
               'Isolated action effects can differ from combined effects under global Hungarian matching.',
               'Watched pairs are GT-selected posthoc examples, not a deployable decision rule.'])
    write_json(output / 'brief.json', report)
    print('SAVED ' + str(output / 'brief.json'), flush=True)


if __name__ == '__main__':
    main()
