"""Explicit candidate-score semantics shared by inference and diagnostics."""

import torch


def proposal_scores(mask_logits, quality_logits, config):
    probability = mask_logits.sigmoid()
    membership = probability >= config['mask_threshold']
    confidence = (probability * membership).sum(-1) / membership.sum(-1).clamp_min(1)
    quality = quality_logits.sigmoid()
    revision = config.get('score_revision', 1)
    if revision == 1:
        score = quality * confidence
    elif revision == 2:
        # The quality head estimates proposal IoU directly. Multiplying another
        # confidence estimate changes its calibration and can remove valid masks.
        score = quality
    else:
        raise ValueError(f'Unknown candidate score revision: {revision}')
    return score, membership, quality, confidence
