"""Real and synthetic actions use identical exact detection-outcome semantics."""

import json
from pathlib import Path

import numpy as np

from tree_learn.instance_relation_v2 import supervision as previous
from tree_learn.mask_refinement.data import file_digest, write_json
from .data import point_row

NAMES = ('other', 'recover', 'destroy')


def outcomes(lost, recovered):
    """Destruction has priority even if another tree is recovered simultaneously."""
    lost, recovered = np.asarray(lost), np.asarray(recovered)
    return np.where(lost > 0, 2, np.where(recovered > 0, 1, 0)).astype(np.int64)


def action_targets(*args, **kwargs):
    return previous.action_targets(*args, **kwargs)


def request(row, config):
    return dict(schema=3, code=file_digest(__file__), settings=config['supervision'],
        reused_targets=file_digest(Path(row['point_prepared']) / 'targets.json'),
        points=file_digest(Path(row['prepared']) / 'points.json'))


def load(row, config):
    output = Path(row['prepared'])
    meta = json.loads((output / 'targets.json').read_text(encoding='utf-8'))
    if meta['request'] != request(row, config) or meta['sha256'] != file_digest(output / 'targets.npz'):
        raise ValueError('V3 targets changed since preparation.')
    # Validate original code, GT stamps, point samples and full-count provenance.
    previous.load(point_row(row), config)
    with np.load(output / 'targets.npz', allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def prepare(row, values, config):
    output = Path(row['prepared'])
    if (output / 'targets.json').exists():
        return load(row, config)
    wanted = request(row, config)
    old = previous.load(point_row(row), config)
    if not np.array_equal(values['ids'], old['ids']):
        raise ValueError('Frozen point samples and exact counts have different instance IDs.')
    result = {key: value.copy() for key, value in old.items()}
    result['action'] = outcomes(old['lost_count'], old['recovered_count'])
    result['risk'] = (result['action'] == 2).astype(np.float32)
    # Reported signed outcome, not an independent regressor or an FP-reduction reward.
    result['utility'] = ((result['action'] == 1).astype(np.float32) - result['risk'])
    counts = {name: int(np.count_nonzero(old['valid'] & (result['action'] == i))) for i, name in enumerate(NAMES)}
    if request(row, config) != wanted:
        raise ValueError('V2 counterfactual artifacts changed while preparing V3.')
    np.savez(output / 'targets.npz', **result)
    write_json(output / 'targets.json', dict(request=wanted, sha256=file_digest(output / 'targets.npz'),
        action_counts=counts, valid_pairs=int(old['valid'].sum()),
        newly_recovery_labeled=int(np.count_nonzero(old['valid'] & (result['action'] == 1) & (old['action'] != 1))),
        quality_auxiliary_only=True, no_fp_cleanup_reward=True))
    return result
