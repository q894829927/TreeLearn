"""Prediction-only preparation, immutable provenance and bounded forest access."""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from tree_learn.instance_relation.data import entries, stamp
from tree_learn.mask_refinement.data import digest, file_digest, load_cache, write_json
from .geometry import BudgetExceeded, candidate, graph_edges

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if config['protocol'] != PROTOCOL or config['seed'] != 42:
        raise ValueError('V1 protocol and seed are locked.')
    inputs = [Path(config[k]).resolve() for k in ('cache_root', 'external_cache_root', 'base_prepared_root')]
    outputs = [Path(config[k]).resolve() for k in ('prepared_root', 'work_dir')]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(inputs + outputs)
           for b in (inputs + outputs)[i + 1:] if a in outputs or b in outputs):
        raise ValueError('Outputs must be separate from every immutable input and each other.')
    if 'wytham' in str(config['splits']).lower() or set(config['splits']['train']) & set(config['splits']['validation']):
        raise ValueError('Disjoint source splits required; Wytham cannot train or calibrate.')
    if len(config['splits']['train']) != 13 or len(config['splits']['validation']) != 5:
        raise ValueError('Use the locked 13/5 forest split.')
    if config['training']['steps'] > 3000 or config['training']['overfit_steps'] > 500:
        raise ValueError('Training exceeds the authorized budget.')
    if config['policy']['cut_thresholds'] != [.1, .2, .3, .4, .5]:
        raise ValueError('Cut threshold search is locked.')
    return config


def signature(config):
    files = sorted(Path(__file__).parent.glob('*.py')) + [ROOT / 'tools/local_split_v1/run.py']
    files += [ROOT / p for p in ('tree_learn/mask_refinement/data.py',
        'tree_learn/mask_refinement/evaluation.py', 'tree_learn/instance_relation/data.py',
        'tree_learn/instance_relation/supervision.py', 'tools/forest_abc/diagnose.py')]
    return digest(dict(config=config, code={p.relative_to(ROOT).as_posix(): file_digest(p) for p in files}))


def provenance(row, config):
    cache = Path(row['cache'])
    return dict(signature=signature(config), cache_metadata_sha256=file_digest(cache / 'cache.json'),
                arrays={k: stamp(cache / f'{k}.npy') for k in
                        ('coords', 'votes_xy', 'tree_probability', 'baseline', 'features')})


def close_arrays(arrays):
    for array in arrays.values():
        if getattr(array, '_mmap', None) is not None:
            array._mmap.close()


def rows(config, external=False):
    return entries(config, external)


def source_groups(baseline):
    """One complete foreground index; no point or parent truncation."""
    at = np.flatnonzero(baseline > 0)
    order = at[np.argsort(baseline[at], kind='stable')]
    ids, starts, counts = np.unique(baseline[order], return_index=True, return_counts=True)
    return ids, starts, counts, order


def prepare_forest(row, config):
    directory = Path(row['prepared'])
    request = provenance(row, config)
    marker = directory / 'candidates.json'
    if marker.exists():
        saved = load_prepared(row, config)
        print(f'REUSE {row["plot"]}: candidates={saved["num_candidates"]}', flush=True)
        return saved
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Incomplete preparation retained for diagnosis: {directory}; use a new output root.')
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'parents').mkdir()
    arrays, metadata = load_cache(row['cache'], with_labels=False)
    try:
        ids, starts, counts, order = source_groups(arrays['baseline'])
        records = []
        for number, (identifier, start, count) in enumerate(zip(ids, starts, counts)):
            indices = order[start:start + count]
            xyz = arrays['coords'][indices].astype(float)
            votes = arrays['votes_xy'][indices].astype(float)
            seeds, record = candidate(xyz, votes, indices, config['candidate'], config['seed'])
            record.update(parent_id=int(identifier), core_points=int(count))
            if seeds is not None:
                try:
                    edges = graph_edges(xyz, config['graph'])
                    record['graph_edges'] = len(edges)
                    path = directory / 'parents' / f'{int(identifier)}.npz'
                    np.savez(path, indices=indices, seed0=seeds[0], seed1=seeds[1])
                    record['payload_sha256'] = file_digest(path)
                except BudgetExceeded as error:
                    record['reason'] = str(error)
            records.append(record)
            if number == 0 or (number + 1) % 25 == 0 or number + 1 == len(ids):
                accepted = sum(r['reason'] == 'candidate' for r in records)
                print(f'PREPARE {row["split"]}/{row["plot"]} parents={number + 1}/{len(ids)} candidates={accepted}', flush=True)
        if provenance(row, config) != request:
            raise ValueError('Observable sources changed during preparation.')
        result = dict(schema=1, protocol=PROTOCOL, request=request, num_points=metadata['num_points'],
                      num_parents=len(records), num_candidates=sum(r['reason'] == 'candidate' for r in records),
                      records=records)
        write_json(marker, result)
        return result
    finally:
        close_arrays(arrays)


def load_prepared(row, config):
    result = read_json(Path(row['prepared']) / 'candidates.json')
    if result['request'] != provenance(row, config):
        raise ValueError(f'Stale observable preparation: {row["plot"]}; use a new output root.')
    for record in result['records']:
        if record['reason'] == 'candidate':
            path = Path(row['prepared']) / 'parents' / f'{record["parent_id"]}.npz'
            if file_digest(path) != record['payload_sha256']:
                raise ValueError(f'Changed parent payload: {path}')
    return result


def load_parent(row, record, arrays=None):
    path = Path(row['prepared']) / 'parents' / f'{record["parent_id"]}.npz'
    if file_digest(path) != record['payload_sha256']:
        raise ValueError('Parent payload digest mismatch.')
    with np.load(path, allow_pickle=False) as z:
        result = {key: z[key] for key in z.files}
    indices = result['indices']
    if len(indices) != record['core_points'] or len(np.unique(indices)) != len(indices):
        raise ValueError('Incomplete or duplicate parent indices.')
    if arrays is not None and not np.all(arrays['baseline'][indices] == record['parent_id']):
        raise ValueError('Parent payload belongs to another official instance.')
    return result


def checkpoint_md5(config):
    value = hashlib.md5()
    with Path(config['checkpoint']).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    if value.hexdigest() != config['checkpoint_md5']:
        raise ValueError('Official checkpoint identity changed; never load replacement weights.')


def apply_edits(baseline, edits):
    """Return a separate array; each edit is a subset of exactly one parent."""
    result = np.asarray(baseline).copy()
    next_id = int(result.max()) + 1
    seen = set()
    for parent, separated in sorted(edits, key=lambda item: item[0]):
        indices = np.asarray(separated, dtype=np.int64)
        if parent in seen or parent <= 0 or len(np.unique(indices)) != len(indices):
            raise ValueError('Duplicate or invalid edit.')
        seen.add(parent)
        if not len(indices) or not np.all(baseline[indices] == parent):
            raise ValueError('Cross-parent or empty edit.')
        if len(indices) >= np.count_nonzero(baseline == parent):
            raise ValueError('An edit must leave a nonempty original parent.')
        result[indices] = next_id
        next_id += 1
    if not np.array_equal(result > 0, np.asarray(baseline) > 0):
        raise AssertionError('Foreground changed.')
    return result
