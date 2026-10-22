"""Choose predicted recovery actions; one score from one outcome distribution."""

import numpy as np

from tree_learn.instance_relation.policy import write_prediction


def action_score(scores, settings):
    probabilities = scores['action']
    if (probabilities.ndim != 2 or probabilities.shape[1] != 3
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Require other/recover/destroy probabilities.')
    return probabilities[:, 1] - settings['destruction_cost'] * probabilities[:, 2]


def select_pairs(pairs, scores, settings, threshold):
    score = action_score(scores, settings)
    eligible = (scores['action'].argmax(1) == 1) & (score >= threshold)
    order = sorted(np.flatnonzero(eligible).tolist(),
                   key=lambda i: (-float(score[i]), int(pairs[i, 0]), int(pairs[i, 1])))
    best = {}
    for edge in order:
        for node in pairs[edge]:
            best.setdefault(int(node), edge)
    accepted = pairs[[edge for edge in order if all(best[int(node)] == edge for node in pairs[edge])]].copy()
    if len(np.unique(accepted)) != accepted.size:
        raise RuntimeError('Only disjoint one-round original instance unions are allowed.')
    return accepted
