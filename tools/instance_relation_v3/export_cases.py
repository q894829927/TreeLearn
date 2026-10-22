"""Export actual V3 model observations and saved decisions for CPU case analysis."""

import argparse
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.forest_abc.diagnose import analyze_table, assert_metrics
from tools.instance_relation_v3.run import config_read, read_json, signature
from tree_learn.instance_relation.supervision import load_supervision, merge_table
from tree_learn.instance_relation_v3 import data, policy, supervision
from tree_learn.mask_refinement.data import file_digest, write_json

WATCHLIST = {259: (1375, 1376), 375: (1750, 1751), 419: (1828, 1829),
             629: (1549, 1550), 720: (1025, 1026), 801: (1080, 1324)}


def source_normalization(config):
    """Reproduce normalize() statistics from train inputs; do not load checkpoints."""
    count, total, square, provenance = 0, np.zeros(32), np.zeros(32), {}
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        print('SOURCE_FEATURE_STATS ' + row['plot'], flush=True)
        marker = Path(row['prepared']) / 'points.json'
        before = file_digest(marker)
        values = data.load(row, config)
        for start in range(0, len(values['points']), 256):
            block = values['points'][start:start + 256, :, 6:][values['valid'][start:start + 256]].astype(np.float64)
            count += len(block)
            total += block.sum(0)
            square += np.square(block).sum(0)
        if file_digest(marker) != before:
            raise ValueError('Training input metadata changed during statistics export.')
        provenance[row['plot']] = before
    if count == 0:
        raise ValueError('No train observations for normalization.')
    mean = total / count
    scale = np.maximum(np.sqrt(np.maximum(square / count - mean**2, 0)), .05)
    return dict(mean=mean.astype(np.float32).tolist(), scale=scale.astype(np.float32).tolist(),
        train_observations=count, train_points_json_sha256=provenance,
        note='Reconstructed train-sample normalization; not an inspection of checkpoint buffers.')


def decision(probabilities, score, threshold, accepted):
    if accepted:
        return 'accepted'
    if int(np.argmax(probabilities)) != 1:
        return 'predicted_' + supervision.NAMES[int(np.argmax(probabilities))]
    return 'below_threshold' if score < threshold else 'mutual_selection'


def choose_cases(values, targets, scores, settings, threshold, watchlist=None, controls=2):
    """GT selects posthoc cases, never deployable actions. Only compact counts replay."""
    pairs, ids = values['pairs'], values['ids']
    if not np.array_equal(ids, targets['ids']):
        raise ValueError('Sample and supervision instance IDs differ.')
    accepted = policy.select_pairs(pairs, scores, settings, threshold)
    if not np.array_equal(accepted, scores['accepted']):
        raise ValueError('Saved V3 actions differ from saved scores/policy.')
    combined = policy.action_score(scores, settings)
    lookup = {tuple(pair): edge for edge, pair in enumerate(pairs)}
    accepted_edges = [lookup[tuple(pair)] for pair in accepted]
    selected, exact, watched = {}, {}, []
    replayed = supervision.action_targets(targets['intersections'], targets['known'], targets['gt_counts'], accepted)
    other_controls = []
    for i, edge in enumerate(accepted_edges):
        exact[edge] = {key: value[i] for key, value in replayed.items()}
        outcome = int(supervision.outcomes(replayed['lost_count'][i], replayed['recovered_count'][i]))
        if outcome == 2:
            selected[edge] = ['accepted_destroy']
        elif outcome == 1:
            selected[edge] = ['accepted_recover']
        else:
            other_controls.append(edge)
    for edge in sorted(other_controls, key=lambda e: (-float(combined[e]), e))[:controls]:
        selected[edge] = ['accepted_other_control']
    for gt_id, expected in (watchlist or {}).items():
        at = np.searchsorted(ids, expected)
        present = bool(np.all(at < len(ids)) and np.array_equal(ids[at], expected))
        edge = lookup.get(tuple(at)) if present else None
        watched.append(dict(gt_id=gt_id, original_ids=list(expected), edge_index=edge))
        if edge is not None:
            selected.setdefault(edge, []).append('watched_gt_' + str(gt_id))
    missing = [edge for edge in selected if edge not in exact]
    if missing:
        replayed = supervision.action_targets(targets['intersections'], targets['known'], targets['gt_counts'], pairs[missing])
        for i, edge in enumerate(missing):
            exact[edge] = {key: value[i] for key, value in replayed.items()}
    geometry, table = merge_table(targets, [])
    original = analyze_table(geometry, table)
    _, table = merge_table(targets, accepted)
    refined = analyze_table(geometry, table)
    original_matches = {r['pred_id']: r['matched_gt_id'] for r in original['predictions']}
    records = []
    for edge in sorted(selected):
        pair = pairs[edge]
        truth = exact[edge]
        records.append(dict(edge_index=int(edge), original_ids=ids[pair].tolist(), groups=selected[edge],
            predicted_action=scores['action'][edge].tolist(), score=float(combined[edge]),
            decision=decision(scores['action'][edge], combined[edge], threshold, edge in accepted_edges),
            original_matched_gt=[int(original_matches[int(i)]) for i in ids[pair]],
            full_instance_points=targets['total'][pair].tolist(),
            full_known_fractions=(targets['known'][pair] / np.maximum(targets['total'][pair], 1)).tolist(),
            supervised_pair_valid=bool(targets['pair_valid'][edge]),
            isolated_lost=int(truth['lost_count']), isolated_recovered=int(truth['recovered_count']),
            isolated_delta_tp=int(truth['delta_tp']), isolated_delta_fp=int(truth['delta_fp']),
            isolated_outcome=supervision.NAMES[int(supervision.outcomes(truth['lost_count'], truth['recovered_count']))],
            actual_union_quality=truth['quality'].tolist()))
    return records, watched, original['metrics'], refined['metrics']


def export_one(config, row, output, normalization, expect_step):
    saved = Path(config['work_dir']) / ('external_evaluation' if row['split'] == 'external' else 'evaluation') / row['plot']
    guarded = [saved / name for name in ('prediction.json', 'scores.npz', 'metrics.json')]
    guarded += [Path(row['prepared']) / 'points.json', Path(row['base_prepared']) / 'supervision.npz',
                Path(config['work_dir']) / 'best.pth']
    before = {str(path): file_digest(path) for path in guarded}
    marker = read_json(saved / 'prediction.json')
    request = marker['request']
    point_meta = read_json(Path(row['prepared']) / 'points.json')
    if (request['signature'] != signature(config) or request['policy'] != config['policy']
            or request['points_sha256'] != point_meta['sha256'] or request['plot'] != row['plot']
            or request['checkpoint_sha256'] != before[str(Path(config['work_dir']) / 'best.pth')]
            or marker['scores_sha256'] != before[str(saved / 'scores.npz')]):
        raise ValueError('Saved V3 inputs, checkpoint or scores changed.')
    if expect_step is not None and request['step'] != expect_step:
        raise ValueError(f'Expected step {expect_step}, found {request["step"]}.')
    if marker['changed_foreground_points'] or marker['changed_unselected_points'] or marker['gt_opened']:
        raise ValueError('Prediction did not preserve the required deployment invariants.')
    values = data.load(row, config)
    targets = load_supervision(row['base_prepared'], config['supervision'])
    with np.load(saved / 'scores.npz', allow_pickle=False) as stored:
        scores = {key: stored[key].copy() for key in stored.files}
    records, watched, baseline, refined = choose_cases(values, targets, scores, config['policy'], request['threshold'],
        WATCHLIST if row['plot'] == 'wytham_vox0.1' else {})
    recorded = read_json(saved / 'metrics.json')
    assert_metrics(recorded['baseline'], baseline)
    assert_metrics(recorded['refined'], refined)
    for category in ('destroy', 'recover'):
        observed = sum('accepted_' + category in record['groups'] for record in records)
        if observed != recorded['accepted_observed_outcome_counts'][category]:
            raise ValueError('Saved per-action outcome counts differ from exact replay.')
    edges = np.asarray([r['edge_index'] for r in records], dtype=np.int64)
    pairs = values['pairs'][edges]
    points, valid = values['points'][pairs], values['valid'][pairs]
    labels = np.full(valid.shape, -1, dtype=np.int64)
    label_path = Path(row['cache']) / 'labels.npy'
    label_stamp = label_path.stat()
    source_gt = np.load(label_path, mmap_mode='r', allow_pickle=False)
    try:
        # Read GT only for the actual sampled observations, not all 37M points.
        labels[valid] = source_gt[values['indices'][pairs][valid]]
    finally:
        source_gt._mmap.close()
    if (label_path.stat().st_size, label_path.stat().st_mtime_ns) != (label_stamp.st_size, label_stamp.st_mtime_ns):
        raise ValueError('GT changed during case export.')
    mean, scale = np.asarray(normalization['mean']), np.asarray(normalization['scale'])
    raw = values['points'][..., 6:][values['valid']]
    standardized = (raw.astype(np.float64) - mean) / scale
    feature_stats = dict(sample_observations=len(raw), feature_z_mean=standardized.mean(0).tolist(),
        feature_z_std=standardized.std(0).tolist(), fraction_feature_values_clipped=float((np.abs(standardized) > 10).mean()))
    if any(file_digest(path) != before[str(path)] for path in guarded):
        raise ValueError('Saved artifacts changed during case export.')
    directory = Path(output) / row['plot']
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'cases.npz', original_ids=values['ids'][pairs], points=points,
        valid=valid, weights=values['weights'][pairs], indices=values['indices'][pairs], point_gt=labels)
    # Compact exact counts and all saved scores make later CPU analyses reproducible
    # without repeated server requests or another network forward pass.
    np.savez_compressed(directory / 'forest_counts.npz', pairs=values['pairs'], **{key: targets[key] for key in (
        'ids', 'gt_ids', 'gt_counts', 'z_span', 'ignored_points', 'num_points', 'total', 'known',
        'intersections', 'original_tp', 'pair_valid')})
    np.savez_compressed(directory / 'saved_scores.npz', **scores)
    report = dict(schema=1, posthoc=True, eligible_for_advancement=False, plot=row['plot'],
        checkpoint_step=request['step'], threshold=request['threshold'], cases=records, watched=watched,
        feature_statistics=feature_stats, baseline=baseline, refined=refined,
        scores_sha256=marker['scores_sha256'], point_sample_sha256=point_meta['sha256'],
        export_code_sha256=file_digest(__file__))
    write_json(directory / 'cases.json', report)
    return report, [directory / name for name in ('cases.npz', 'cases.json', 'forest_counts.npz', 'saved_scores.npz')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/instance_relation_v3/full.yaml')
    parser.add_argument('--expect-step', type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    os.chdir(ROOT)
    config = config_read(config_path)
    output = Path(config['work_dir']) / 'case_analysis'
    normalization = source_normalization(config)
    reports, files = [], []
    for row in data.rows(config) + data.rows(config, external=True):
        if row['split'] == 'train':
            continue
        print('EXPORT_CASES ' + row['plot'] + ': saved observations, no model or GPU', flush=True)
        report, produced = export_one(config, row, output, normalization, args.expect_step)
        reports.append(report)
        files.extend(produced)
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        if file_digest(Path(row['prepared']) / 'points.json') != normalization['train_points_json_sha256'][row['plot']]:
            raise ValueError('Train normalization inputs changed during export.')
    write_json(output / 'normalization.json', normalization)
    write_json(output / 'brief.json', dict(schema=1, posthoc=True, eligible_for_advancement=False,
        forests=reports, normalization='normalization.json', notes=[
            'Model inputs only: 256 sampled points per original instance, not complete tree clouds.',
            'Each point: xyz(3), votes_xy(2), tree_probability(1), frozen features(32).',
            'GT-selected posthoc cases; negative GT is unknown and known background is 0.',
            'forest_counts and saved_scores are compact full candidate data for posthoc CPU replay, not GT-free deployment inputs.',
            'No network/torch/checkpoint tensors loaded; no predictions, thresholds or labels changed.',
            'Feature statistics describe saved instance samples, not population-weighted whole-forest points.',
            'PCA or feature drift alone cannot prove feature collapse or the cause of detection errors.']))
    files += [output / 'normalization.json', output / 'brief.json']
    temporary = output / 'casepack.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(output).as_posix())
    temporary.replace(output / 'casepack.zip')
    print(f'SAVED {output / "casepack.zip"} bytes={(output / "casepack.zip").stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
