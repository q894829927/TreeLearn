"""Reuse immutable V2 samples; keep V3 metadata and targets in a new directory."""

import json
from pathlib import Path

from tree_learn.instance_relation_v2 import data as previous
from tree_learn.mask_refinement.data import file_digest, write_json


def rows(config, external=False):
    result = previous.rows(dict(config, prepared_root=config['point_prepared_root']), external)
    for row in result:
        row['point_prepared'] = row['prepared']
        row['prepared'] = str(Path(config['prepared_root']) / row['split'] / row['plot'])
    return result


def point_row(row):
    return dict(row, prepared=row['point_prepared'])


def request(row):
    path = Path(row['point_prepared']) / 'points.json'
    return dict(schema=3, code=file_digest(__file__), reused_points_json=file_digest(path),
                source=str(path.resolve()))


def load(row, config):
    meta = json.loads((Path(row['prepared']) / 'points.json').read_text(encoding='utf-8'))
    if meta['request'] != request(row):
        raise ValueError('V3 sample reference changed; use a new experiment directory.')
    values = previous.load(point_row(row), config)
    source = json.loads((Path(row['point_prepared']) / 'points.json').read_text(encoding='utf-8'))
    if meta['sha256'] != source['sha256']:
        raise ValueError('V3 reference and original frozen samples differ.')
    return values


def prepare(row, config):
    output = Path(row['prepared']) / 'points.json'
    if output.exists():
        return load(row, config)
    # V2 completed these samples already. Do not regenerate or overwrite them.
    wanted = request(row)
    values = previous.load(point_row(row), config)
    source = json.loads((Path(row['point_prepared']) / 'points.json').read_text(encoding='utf-8'))
    if request(row) != wanted:
        raise ValueError('Original point samples changed while preparing the V3 reference.')
    write_json(output, dict(request=wanted, sha256=source['sha256'],
        num_nodes=len(values['ids']), num_pairs=len(values['pairs']),
        gt_opened=False, candidates_exactly_v1=True, samples_reused=True))
    return values
