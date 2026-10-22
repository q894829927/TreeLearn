"""Posthoc GT auditing and exact full-forest matching; never candidate inputs."""

from collections import Counter
from pathlib import Path

import numpy as np

from tools.forest_abc.diagnose import analyze_table
from tree_learn.instance_relation.data import stamp
from tree_learn.instance_relation.supervision import load_supervision, merge_table
from tree_learn.mask_refinement.data import file_digest, load_cache, write_json
from tree_learn.mask_refinement.evaluation import macro_metrics
from .data import close_arrays, load_parent, load_prepared, rows, signature
from .geometry import graph_edges
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def baseline_table(row, config):
    directory = Path(config['base_prepared_root']) / row['split'] / row['plot']
    supervision = load_supervision(directory)
    geometry, table = merge_table(supervision, [])
    return geometry, table, analyze_table(geometry, table)


def seed_audit(labels, seeds, cfg):
    records = []
    for at in seeds:
        local = labels[at]
        known = local[local >= 0]
        ids, counts = np.unique(known[known > 0], return_counts=True)
        best = int(np.argmax(counts)) if len(counts) else None
        records.append(dict(known_points=len(known), ignored_points=int(np.count_nonzero(local < 0)),
            dominant_gt_id=int(ids[best]) if best is not None else 0,
            purity=float(counts[best] / len(known)) if best is not None else 0.))
    reliable = (all(r['known_points'] >= cfg['min_known_seed_points'] and
                    r['purity'] >= cfg['min_seed_purity'] and r['dominant_gt_id'] > 0 for r in records)
                and records[0]['dominant_gt_id'] != records[1]['dominant_gt_id'])
    return dict(seeds=records, reliable=bool(reliable))


def edge_targets(labels, edges):
    a, b = labels[edges[:, 0]], labels[edges[:, 1]]
    valid = (a >= 0) & (b >= 0) & ((a > 0) | (b > 0))
    return valid, ((a == b) & (a > 0)).astype(np.float32)


def split_table(geometry, base, edits, labels):
    """Update sufficient counts, then rematch the entire forest (not local IoU)."""
    table = {key: np.asarray(value).copy() for key, value in base.items()}
    next_id = int(table['ids'].max()) + 1
    appended = {key: [] for key in table}
    seen = set()
    for parent, indices in sorted(edits, key=lambda item: item[0]):
        if parent in seen or len(np.unique(indices)) != len(indices):
            raise ValueError('Overlapping/duplicate parent edits.')
        seen.add(parent)
        row = int(np.searchsorted(table['ids'], parent))
        if row >= len(table['ids']) or table['ids'][row] != parent:
            raise ValueError('Unknown parent.')
        local = labels[indices]
        positive = local[local > 0]
        common = np.bincount(np.searchsorted(geometry['ids'], positive), minlength=len(geometry['ids']))
        values = dict(ids=next_id, total=len(local), known=int(np.count_nonzero(local >= 0)), intersections=common)
        if values['total'] <= 0 or values['total'] >= table['total'][row]:
            raise ValueError('Empty child in count update.')
        for key in ('total', 'known', 'intersections'):
            table[key][row] -= values[key]
            if np.any(table[key][row] < 0):
                raise ValueError('Child counts exceed parent counts.')
        for key in table:
            appended[key].append(values[key])
        next_id += 1
    if edits:
        table = {key: np.concatenate((value, np.asarray(appended[key], dtype=value.dtype)), axis=0)
                 for key, value in table.items()}
    return table


def comparison(geometry, base, refined):
    before, after = analyze_table(geometry, base), analyze_table(geometry, refined)
    old = {r['gt_id'] for r in before['trees'] if r['detected']}
    new = {r['gt_id'] for r in after['trees'] if r['detected']}
    return dict(baseline=before['metrics'], refined=after['metrics'], recovered_gt_ids=sorted(new - old),
                lost_gt_ids=sorted(old - new), recovered=len(new - old), lost=len(old - new),
                f1_delta_pp=100 * (after['metrics']['f1'] - before['metrics']['f1']),
                fp_delta=after['metrics']['fp'] - before['metrics']['fp'])


def advancement(reports):
    recovered, lost = sum(r['recovered'] for r in reports), sum(r['lost'] for r in reports)
    before = macro_metrics([r['baseline'] for r in reports])
    after = macro_metrics([r['refined'] for r in reports])
    nonnegative = sum(r['refined']['f1'] >= r['baseline']['f1'] - 1e-12 for r in reports)
    return dict(passed=bool(recovered > 3 * lost and after['f1'] > before['f1'] + 1e-12 and nonnegative >= 4),
                recovered=recovered, lost=lost, utility=recovered - 3 * lost,
                nonnegative_forests=nonnegative, baseline_macro=before, refined_macro=after)


def audit_all(config):
    output = Path(config['work_dir']) / 'e0'
    output.mkdir(parents=True, exist_ok=True)
    summaries, all_candidates = [], []
    for row in rows(config):
        prepared = load_prepared(row, config)  # fixed before GT is opened
        geometry, table, baseline = baseline_table(row, config)
        if ([r['parent_id'] for r in prepared['records']] != table['ids'].tolist() or
                [r['core_points'] for r in prepared['records']] != table['total'].tolist()):
            raise ValueError('Official complete parent counts changed.')
        arrays, _ = load_cache(row['cache'], with_labels=True)
        label_stamp = stamp(Path(row['cache']) / 'labels.npy')
        try:
            candidates = []
            detected = {r['gt_id'] for r in baseline['trees'] if r['detected']}
            for record in prepared['records']:
                if record['reason'] != 'candidate':
                    continue
                payload = load_parent(row, record, arrays)
                indices = payload['indices']
                seeds = [payload['seed0'], payload['seed1']]
                local = arrays['labels'][indices]
                result = seed_audit(local, seeds, config['audit'])
                at = int(np.searchsorted(table['ids'], record['parent_id']))
                recoverable = geometry['ids'][(table['intersections'][at] / geometry['counts'] > .5)
                                              & ~np.isin(geometry['ids'], list(detected))]
                seed_ids = [r['dominant_gt_id'] for r in result['seeds']]
                opportunities = sorted(set(recoverable.tolist()) & set(seed_ids)) if result['reliable'] else []
                xyz = arrays['coords'][indices].astype(float)
                edges = graph_edges(xyz, config['graph'])
                adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(xyz), len(xyz))).tocsr()
                num_components, groups = connected_components(adjacency, directed=False)
                seeded = np.unique(groups[np.concatenate(seeds)])
                valid, target = edge_targets(local, edges)
                pred = baseline['predictions'][at]
                result.update(parent_id=record['parent_id'], entrance=record['entrance'], plot=row['plot'], split=row['split'],
                    original_tp=pred['category'] == 'true_positive', recoverable_gt_ids=opportunities,
                    core_points=len(indices), graph_edges=len(edges), components=int(num_components),
                    seedless_points=int(np.count_nonzero(~np.isin(groups, seeded))),
                    same_edges=int(np.count_nonzero(valid & (target == 1))),
                    different_edges=int(np.count_nonzero(valid & (target == 0))))
                candidates.append(result)
                print(f'AUDIT {row["plot"]}/{record["parent_id"]} entrance={record["entrance"]} reliable={result["reliable"]} recoverable={opportunities}', flush=True)
            if stamp(Path(row['cache']) / 'labels.npy') != label_stamp:
                raise ValueError('GT changed during audit.')
            summary = dict(plot=row['plot'], split=row['split'], baseline=baseline['metrics'],
                num_parents=prepared['num_parents'], num_candidates=len(candidates),
                reasons=dict(Counter(r['reason'] for r in prepared['records'])),
                entrances=dict(Counter(r['entrance'] for r in candidates)),
                reliable=sum(r['reliable'] for r in candidates),
                recoverable_candidates=sum(bool(r['recoverable_gt_ids']) for r in candidates),
                selected_original_tp=sum(r['original_tp'] for r in candidates),
                reliable_original_tp=sum(r['original_tp'] and r['reliable'] for r in candidates),
                label_source=label_stamp, candidates=candidates)
            write_json(output / f'{row["plot"]}.json', summary)
            summaries.append(summary)
            all_candidates.extend(candidates)
        finally:
            close_arrays(arrays)
    validation = [r['baseline'] for r in summaries if r['split'] == 'validation']
    if abs(macro_metrics(validation)['f1'] - config['expected_validation_baseline_f1']) > 1e-12:
        raise ValueError('Official five-forest baseline changed.')
    train = [r for r in all_candidates if r['split'] == 'train' and r['reliable']]
    val = [r for r in all_candidates if r['split'] == 'validation' and r['recoverable_gt_ids']]
    gate = dict(reliable_train=len(train), reliable_train_forests=len({r['plot'] for r in train}),
                recoverable_validation=len(val))
    gate['passed'] = (len(train) >= config['audit']['min_train_reliable'] and
        gate['reliable_train_forests'] >= config['audit']['min_train_forests'] and
        len(val) >= config['audit']['min_validation_recoverable'])
    result = dict(schema=1, signature=signature(config), protocol=config['protocol'], stage='E0',
        model_executed=False, predictions_modified=False, gate=gate, forests=summaries,
        preparation_sha256={row['plot']: file_digest(Path(row['prepared']) / 'candidates.json') for row in rows(config)},
        validation_baseline_macro=macro_metrics(validation))
    write_json(output / 'summary.json', result)
    print('E0_GATE ' + str(gate), flush=True)
    return result
