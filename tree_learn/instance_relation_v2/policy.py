"""One source-calibrated action score, then the unchanged disjoint mutual rule."""

import numpy as np

from tree_learn.instance_relation.policy import write_prediction


def action_score(scores, settings):
    action = scores['action']
    result = action[:, 1] + settings['completion_credit'] * action[:, 2]
    result = result + settings['utility_weight'] * scores['utility'] - settings['risk_weight'] * scores['risk']
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite action scores.')
    return result


def select_pairs(pairs, scores, settings, threshold):
    score = action_score(scores, settings)
    order = np.flatnonzero(score >= threshold)
    order = sorted(order.tolist(), key=lambda i: (-float(score[i]), int(pairs[i, 0]), int(pairs[i, 1])))
    best = {}
    for edge in order:
        for node in pairs[edge]:
            best.setdefault(int(node), edge)
    chosen = [edge for edge in order if all(best[int(node)] == edge for node in pairs[edge])]
    accepted = pairs[chosen].copy()
    if len(np.unique(accepted)) != 2 * len(accepted):
        raise RuntimeError('Repeated or transitive V2 merge actions.')
    return accepted
