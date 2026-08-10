"""E3a fixed-validation multi-scale HDBSCAN proposal Oracle.

This diagnostic is intentionally independent from the failed E2a/E2b vote
refinement hypotheses.  It never changes votes or the candidate seed set.
Instead, it asks whether a fixed set of HDBSCAN minimum-cluster sizes contains
complementary tree proposals that the original scale (50) misses.  Ground
truth is used only to measure the proposal-pool upper bound.  Wytham is
explicitly forbidden.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.diagnostics.diagnose_cellwise_vote_refinement_oracle import (  # noqa: E402
    _array_digest,
    cluster_votes,
    load_validation_rows,
    load_vote_artifact,
    maximize_assignment,
    seed_cluster_detection_metrics,
)


LOCKED_CLUSTER_SIZES = [25, 35, 50, 75, 100]


def proposal_iou_table(ground_truth_labels, predicted_labels):
    """Return proposal-by-tree IoU without changing the clustering labels."""
    gt = np.asarray(ground_truth_labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
    if len(gt) != len(pred):
        raise ValueError('GT and predicted cluster labels are not aligned.')
    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred >= 0])
    if len(gt_ids) == 0:
        raise ValueError('Multi-scale Oracle requires positive GT trees.')
    if len(pred_ids) == 0:
        return pred_ids, gt_ids, np.zeros((0, len(gt_ids)), dtype=np.float64)

    gt_position = np.searchsorted(gt_ids, gt)
    pred_position = np.searchsorted(pred_ids, pred)
    valid = (gt > 0) & (pred >= 0)
    flat = (
        pred_position[valid].astype(np.int64) * len(gt_ids) +
        gt_position[valid].astype(np.int64))
    intersection = np.bincount(
        flat, minlength=len(pred_ids) * len(gt_ids)).reshape(
            len(pred_ids), len(gt_ids))
    pred_sizes = np.bincount(
        pred_position[pred >= 0], minlength=len(pred_ids)).reshape(-1, 1)
    gt_sizes = np.bincount(
        gt_position[gt > 0], minlength=len(gt_ids)).reshape(1, -1)
    union = pred_sizes + gt_sizes - intersection
    iou = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)
    return pred_ids, gt_ids, iou


def accepted_tree_ids(ground_truth_labels, predicted_labels, min_iou):
    """Return the GT ids accepted by the same Hungarian rule as E2a."""
    _, gt_ids, iou = proposal_iou_table(
        ground_truth_labels, predicted_labels)
    if iou.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    rows, columns = maximize_assignment(iou)
    accepted = iou[rows, columns] > float(min_iou)
    return gt_ids[columns[accepted]]


def multiscale_proposal_oracle(
        ground_truth_labels, clusterings, base_cluster_size, min_iou):
    """Measure the upper bound of selecting from a fixed proposal pool.

    Every GT tree independently keeps its best-IoU proposal.  A proposal cannot
    exceed IoU 0.5 with two disjoint GT instances, so with the locked strict
    IoU > 0.5 acceptance rule this is also a valid one-proposal-per-tree upper
    bound.  Ties are resolved in favour of the original cluster size so that
    unchanged proposals are not credited to an alternative scale.
    """
    if base_cluster_size not in clusterings:
        raise ValueError('The original cluster size must be in the pool.')
    ordered_sizes = list(clusterings)
    if len(set(ordered_sizes)) != len(ordered_sizes):
        raise ValueError('Cluster sizes must be unique.')
    gt = np.asarray(ground_truth_labels, dtype=np.int64).reshape(-1)
    tables = []
    proposal_scales = []
    gt_ids = None
    scale_metrics = {}
    scale_offsets = {}
    offset = 0
    for size in ordered_sizes:
        prediction = np.asarray(clusterings[size], dtype=np.int64).reshape(-1)
        if len(prediction) != len(gt):
            raise ValueError('Multi-scale clustering arrays are not aligned.')
        _, observed_gt_ids, iou = proposal_iou_table(gt, prediction)
        if gt_ids is None:
            gt_ids = observed_gt_ids
        elif not np.array_equal(gt_ids, observed_gt_ids):
            raise AssertionError('GT ids changed between clustering scales.')
        tables.append(iou)
        proposal_scales.extend([int(size)] * iou.shape[0])
        scale_offsets[int(size)] = (offset, offset + iou.shape[0])
        offset += iou.shape[0]
        scale_metrics[str(size)] = seed_cluster_detection_metrics(
            gt, prediction, min_iou)

    pooled_iou = np.concatenate(tables, axis=0)
    proposal_scales = np.asarray(proposal_scales, dtype=np.int64)
    best_rows = np.argmax(pooled_iou, axis=0)
    best_iou = pooled_iou[best_rows, np.arange(len(gt_ids))]

    base_start, base_end = scale_offsets[int(base_cluster_size)]
    base_iou = pooled_iou[base_start:base_end]
    base_best_rows = np.argmax(base_iou, axis=0) + base_start
    base_best_iou = pooled_iou[base_best_rows, np.arange(len(gt_ids))]
    tied_with_base = base_best_iou >= best_iou - 1e-12
    best_rows[tied_with_base] = base_best_rows[tied_with_base]
    best_iou = pooled_iou[best_rows, np.arange(len(gt_ids))]
    best_scales = proposal_scales[best_rows]

    recoverable = best_iou > float(min_iou)
    base_tree_ids = accepted_tree_ids(
        gt, clusterings[base_cluster_size], min_iou)
    base_matched = np.isin(gt_ids, base_tree_ids)
    newly_recovered = recoverable & ~base_matched
    tp = int(recoverable.sum())
    fn = int(len(gt_ids) - tp)
    oracle_f1 = float(2 * tp / max(2 * tp + fn, 1))
    contribution = {
        str(size): int(np.sum(recoverable & (best_scales == size)))
        for size in ordered_sizes
    }
    recovered_contribution = {
        str(size): int(np.sum(newly_recovered & (best_scales == size)))
        for size in ordered_sizes
    }
    base_metrics = scale_metrics[str(base_cluster_size)]
    total_proposals = int(len(proposal_scales))
    return {
        'num_gt_trees': int(len(gt_ids)),
        'num_candidate_proposals': total_proposals,
        'proposal_inflation_vs_base': float(
            total_proposals / max(base_metrics['num_pred_clusters'], 1)),
        'scale_metrics': scale_metrics,
        'oracle': {
            'tp': tp,
            'fp': 0,
            'fn': fn,
            'tree_recall': float(tp / len(gt_ids)),
            'commission': 0.0,
            'f1': oracle_f1,
            'mean_selected_iou': float(
                best_iou[recoverable].mean() if tp else 0.0),
        },
        'num_newly_recovered_trees': int(newly_recovered.sum()),
        'newly_recovered_tree_rate': float(
            newly_recovered.sum() / len(gt_ids)),
        'non_base_best_tree_rate': float(np.mean(
            recoverable & (best_scales != int(base_cluster_size)))),
        'selected_scale_contribution': contribution,
        'recovered_scale_contribution': recovered_contribution,
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_e2a_base_references(settings, expected_plots):
    summary_path = Path(settings['e2a_summary_path'])
    plot_dir = Path(settings['e2a_plot_dir'])
    if not summary_path.is_file() or not plot_dir.is_dir():
        raise FileNotFoundError('E2a summary and plot caches are required.')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    observed = summary.get('settings', {})
    expected = {
        'run_hdbscan': True,
        'min_cluster_size': settings['base_cluster_size'],
        'min_iou': settings['min_iou'],
        'n_jobs': settings['n_jobs'],
    }
    for name, value in expected.items():
        actual = observed.get(name)
        matches = (
            np.isclose(float(actual), value)
            if isinstance(value, float) and actual is not None
            else actual == value)
        if not matches:
            raise ValueError(
                f'E2a setting {name}={actual!r} differs from E3a {value!r}.')
    references = {}
    hashes = {}
    for plot in expected_plots:
        path = plot_dir / f'{plot}.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing E2a plot cache: {path}')
        row = json.loads(path.read_text(encoding='utf-8'))
        cluster = row.get('modes', {}).get('base', {}).get('cluster')
        digest = row.get('cluster_digests', {}).get('base')
        if row.get('source_plot') != plot or cluster is None or digest is None:
            raise ValueError(f'Incomplete E2a base cache for {plot}.')
        references[plot] = {
            'artifact_identity': row.get('artifact_identity'),
            'cluster': cluster,
            'cluster_digest': digest,
        }
        hashes[plot] = _file_sha256(path)
    reference_digest = hashlib.sha256(json.dumps({
        'summary': _file_sha256(summary_path),
        'plots': hashes,
    }, sort_keys=True).encode('utf-8')).hexdigest()
    return references, reference_digest


def _cluster_cache_path(cluster_dir, plot, cluster_size):
    return Path(cluster_dir) / f'{plot}_mcs{int(cluster_size):03d}.npz'


def load_or_cluster(
        votes, plot, cluster_size, cluster_dir, cache_identity,
        n_jobs, force=False, logger=print):
    path = _cluster_cache_path(cluster_dir, plot, cluster_size)
    expected_meta = {
        'plot': plot,
        'cluster_size': int(cluster_size),
        'n_jobs': int(n_jobs),
        'num_candidates': int(len(votes)),
        'artifact_identity': cache_identity,
    }
    if path.is_file() and not force:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata'].item()))
            predictions = data['predictions'].astype(np.int64)
        if metadata == expected_meta and predictions.shape == (len(votes),):
            logger(
                f'  SKIP HDBSCAN mcs={cluster_size}: validated cache.',
                flush=True)
            return predictions
    start = time.time()
    logger(
        f'  HDBSCAN mcs={cluster_size}: {len(votes):,} seeds...',
        flush=True)
    predictions = cluster_votes(votes, cluster_size, n_jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=predictions.astype(np.int64),
        metadata=np.asarray(json.dumps(expected_meta, sort_keys=True)),
    )
    logger(
        f'    clusters={len(np.unique(predictions[predictions >= 0])):,}, '
        f'time={time.time() - start:.1f}s', flush=True)
    return predictions


def _cluster_metrics_match(observed, expected):
    integer_names = ('num_gt_trees', 'num_pred_clusters', 'tp', 'fp', 'fn')
    float_names = ('tree_recall', 'commission', 'f1', 'mean_matched_iou')
    return (
        all(int(observed[name]) == int(expected[name]) for name in integer_names)
        and all(np.isclose(
            float(observed[name]), float(expected[name]), atol=1e-12)
            for name in float_names))


def analyze_plot(row, settings, reference, cluster_dir, force=False):
    arrays, artifact_identity = load_vote_artifact(row, settings['data_root'])
    if reference['artifact_identity'] != artifact_identity:
        raise ValueError(
            f'E2a artifact identity differs for {row["source_plot"]}.')
    votes = arrays['base_votes_xy'].astype(np.float64)
    labels = arrays['target_tree_id'].astype(np.int64)
    clusterings = {}
    for size in settings['cluster_sizes']:
        clusterings[size] = load_or_cluster(
            votes, row['source_plot'], size, cluster_dir,
            artifact_identity, settings['n_jobs'], force=force)
    base = clusterings[settings['base_cluster_size']]
    base_digest = _array_digest(base)
    base_metrics = seed_cluster_detection_metrics(
        labels, base, settings['min_iou'])
    reproduction = (
        base_digest == reference['cluster_digest'] and
        _cluster_metrics_match(base_metrics, reference['cluster']))
    if not reproduction:
        raise RuntimeError(
            f'E3a baseline reproduction failed for {row["source_plot"]}.')
    result = multiscale_proposal_oracle(
        labels, clusterings, settings['base_cluster_size'],
        settings['min_iou'])
    result.update({
        'source_plot': row['source_plot'],
        'num_candidates': int(len(votes)),
        'num_tree_candidates': int(arrays['target_is_tree'].sum()),
        'artifact_identity': artifact_identity,
        'candidate_set_sha256': _array_digest(
            arrays['candidate_indices'].astype(np.int64)),
        'base_cluster_digest': base_digest,
        'baseline_reproduced': reproduction,
    })
    base_recall = result['scale_metrics'][
        str(settings['base_cluster_size'])]['tree_recall']
    result['effects'] = {
        'oracle_recall_gain_pp': float(
            100 * (result['oracle']['tree_recall'] - base_recall)),
        'oracle_f1_gain_pp': float(100 * (
            result['oracle']['f1'] - result['scale_metrics'][
                str(settings['base_cluster_size'])]['f1'])),
        'strict_recovery': bool(result['num_newly_recovered_trees'] > 0),
    }
    return result


def aggregate_reports(reports, settings):
    base_key = str(settings['base_cluster_size'])
    scale_metrics = {}
    for size in settings['cluster_sizes']:
        key = str(size)
        scale_metrics[key] = {
            name: float(np.mean([
                row['scale_metrics'][key][name] for row in reports]))
            for name in ('tree_recall', 'commission', 'f1')
        }
    base = scale_metrics[base_key]
    oracle = {
        name: float(np.mean([row['oracle'][name] for row in reports]))
        for name in ('tree_recall', 'commission', 'f1')
    }
    total_gt = sum(row['num_gt_trees'] for row in reports)
    total_non_base_best = sum(
        round(row['non_base_best_tree_rate'] * row['num_gt_trees'])
        for row in reports)
    effects = {
        'oracle_tree_recall_gain_pp': float(
            100 * (oracle['tree_recall'] - base['tree_recall'])),
        'oracle_f1_gain_pp': float(100 * (oracle['f1'] - base['f1'])),
        'total_newly_recovered_trees': int(sum(
            row['num_newly_recovered_trees'] for row in reports)),
        'strict_recovery_plots': int(sum(
            row['effects']['strict_recovery'] for row in reports)),
        'non_base_best_tree_rate': float(
            total_non_base_best / max(total_gt, 1)),
        'mean_proposal_inflation': float(np.mean([
            row['proposal_inflation_vs_base'] for row in reports])),
    }
    rule = settings['gate']
    gate = {
        'expected_validation_plots': (
            len(reports) == int(rule['expected_validation_plots'])),
        'baseline_reproduced': all(
            row['baseline_reproduced'] for row in reports),
        'scale_grid_locked': (
            settings['cluster_sizes'] == LOCKED_CLUSTER_SIZES),
        'oracle_recall_gain_passed': (
            effects['oracle_tree_recall_gain_pp'] >=
            float(rule['min_oracle_tree_recall_gain_pp'])),
        'recovered_tree_count_passed': (
            effects['total_newly_recovered_trees'] >=
            int(rule['min_newly_recovered_trees'])),
        'per_plot_recovery_passed': (
            effects['strict_recovery_plots'] >=
            int(rule['min_strict_recovery_plots'])),
        'scale_diversity_passed': (
            effects['non_base_best_tree_rate'] >=
            float(rule['min_non_base_best_tree_rate'])),
        'proposal_inflation_passed': (
            effects['mean_proposal_inflation'] <=
            float(rule['max_mean_proposal_inflation'])),
    }
    gate['passed'] = bool(all(gate.values()))
    aggregate = {
        'num_validation_plots': int(len(reports)),
        'num_candidates': int(sum(row['num_candidates'] for row in reports)),
        'num_gt_trees': int(total_gt),
        'scale_metrics_macro': scale_metrics,
        'oracle_macro': oracle,
    }
    return aggregate, effects, gate


def format_markdown(report):
    aggregate = report['aggregate']
    effects = report['effects']
    settings = report['settings']
    lines = [
        '# E3a 复杂度感知多尺度分组 Proposal Oracle', '',
        '- 数据：固定五个 validation forests；未读取 Wytham。',
        '- 不修改 TreeLearn votes、种子集合或特征。',
        f'- 固定 HDBSCAN `min_cluster_size`：{settings["cluster_sizes"]}',
        f'- 原版尺度：{settings["base_cluster_size"]}',
        f'- 候选种子：{aggregate["num_candidates"]:,}',
        f'- GT 树：{aggregate["num_gt_trees"]:,}', '',
        '## Macro 单尺度与 Proposal Oracle', '',
        '| Mode | Tree recall | Commission | F1 |',
        '|---|---:|---:|---:|',
    ]
    for size in settings['cluster_sizes']:
        metric = aggregate['scale_metrics_macro'][str(size)]
        label = f'mcs_{size}' + (
            ' (base)' if size == settings['base_cluster_size'] else '')
        lines.append(
            f'| {label} | {100 * metric["tree_recall"]:.3f}% | '
            f'{100 * metric["commission"]:.3f}% | '
            f'{100 * metric["f1"]:.3f}% |')
    oracle = aggregate['oracle_macro']
    lines.append(
        f'| proposal_oracle | {100 * oracle["tree_recall"]:.3f}% | '
        f'{100 * oracle["commission"]:.3f}% | {100 * oracle["f1"]:.3f}% |')
    lines.extend([
        '', '## Oracle 效果', '',
        f'- Tree recall 增益：{effects["oracle_tree_recall_gain_pp"]:+.3f} pp',
        f'- F1 上限增益：{effects["oracle_f1_gain_pp"]:+.3f} pp',
        f'- 新恢复 GT 树：{effects["total_newly_recovered_trees"]}',
        f'- 有新增恢复的森林：{effects["strict_recovery_plots"]}/5',
        f'- 最佳尺度非原版的 GT 比例：{100 * effects["non_base_best_tree_rate"]:.3f}%',
        f'- 平均 proposal 膨胀倍数：{effects["mean_proposal_inflation"]:.3f}x',
        '', '## 每森林', '',
        '| Plot | Base recall | Oracle recall | Gain | Recovered | Non-base best | Inflation |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    base_key = str(settings['base_cluster_size'])
    for row in report['plots']:
        base = row['scale_metrics'][base_key]['tree_recall']
        lines.append(
            f'| {row["source_plot"]} | {100 * base:.3f}% | '
            f'{100 * row["oracle"]["tree_recall"]:.3f}% | '
            f'{row["effects"]["oracle_recall_gain_pp"]:+.3f} pp | '
            f'{row["num_newly_recovered_trees"]} | '
            f'{100 * row["non_base_best_tree_rate"]:.3f}% | '
            f'{row["proposal_inflation_vs_base"]:.3f}x |')
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend(['', (
        'PASS：进入 E3b 多尺度 proposal 特征与参数匹配选择器；先做 MLP，'
        '再做关系注意力，仍不读取 Wytham。'
        if report['gate']['passed'] else
        'STOP：多尺度分组没有足够的验证集恢复上限；关闭复杂度感知分组路线，'
        '不使用 Wytham 调参。')])
    return '\n'.join(lines) + '\n'


def write_per_plot_csv(path, reports, base_cluster_size):
    rows = []
    for report in reports:
        base = report['scale_metrics'][str(base_cluster_size)]
        rows.append({
            'source_plot': report['source_plot'],
            'num_gt_trees': report['num_gt_trees'],
            'base_tree_recall': base['tree_recall'],
            'oracle_tree_recall': report['oracle']['tree_recall'],
            'oracle_recall_gain_pp': report['effects']['oracle_recall_gain_pp'],
            'num_newly_recovered_trees': report['num_newly_recovered_trees'],
            'non_base_best_tree_rate': report['non_base_best_tree_rate'],
            'proposal_inflation_vs_base': report['proposal_inflation_vs_base'],
        })
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config_fingerprint(settings):
    relevant = {
        name: settings[name] for name in (
            'cluster_sizes', 'base_cluster_size', 'min_iou', 'n_jobs',
            'gate', 'e2a_reference_digest')}
    return hashlib.sha256(json.dumps(
        relevant, sort_keys=True).encode('utf-8')).hexdigest()


def run_settings(raw, config_path='<memory>', force=False):
    required = {
        'data_root', 'manifest_path', 'output_dir',
        'e2a_summary_path', 'e2a_plot_dir',
        'expected_validation_plots', 'cluster_sizes', 'base_cluster_size',
        'min_iou', 'n_jobs', 'gate'}
    missing = required - set(raw)
    if missing:
        raise ValueError(f'Missing E3a config fields: {sorted(missing)}')
    cluster_sizes = [int(value) for value in raw['cluster_sizes']]
    if cluster_sizes != LOCKED_CLUSTER_SIZES:
        raise ValueError(
            f'E3a cluster sizes are locked to {LOCKED_CLUSTER_SIZES}.')
    settings = {
        'data_root': raw['data_root'],
        'e2a_summary_path': raw['e2a_summary_path'],
        'e2a_plot_dir': raw['e2a_plot_dir'],
        'cluster_sizes': cluster_sizes,
        'base_cluster_size': int(raw['base_cluster_size']),
        'min_iou': float(raw['min_iou']),
        'n_jobs': int(raw['n_jobs']),
        'gate': raw['gate'],
    }
    if settings['base_cluster_size'] != 50:
        raise ValueError('E3a original HDBSCAN scale is locked to 50.')
    if not np.isclose(settings['min_iou'], 0.5):
        raise ValueError('E3a IoU threshold is locked to 0.5.')
    expected_plots = list(raw['expected_validation_plots'])
    rows = load_validation_rows(raw['manifest_path'], expected_plots)
    references, reference_digest = load_e2a_base_references(
        settings, expected_plots)
    settings['e2a_reference_digest'] = reference_digest
    fingerprint = _config_fingerprint(settings)

    output_dir = Path(raw['output_dir'])
    plot_dir = output_dir / 'plots'
    cluster_dir = output_dir / 'clusters'
    plot_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, row in enumerate(rows, start=1):
        plot = row['source_plot']
        cache_path = plot_dir / f'{plot}.json'
        artifact_path = Path(row['artifact_path']).resolve()
        stat = artifact_path.stat()
        identity = {
            'path': str(artifact_path), 'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns)}
        if cache_path.is_file() and not force:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            if (cached.get('config_fingerprint') == fingerprint and
                    cached.get('artifact_identity') == identity):
                print(
                    f'[{index}/{len(rows)}] SKIP {plot}: validated cache.',
                    flush=True)
                reports.append(cached)
                continue
        print(f'[{index}/{len(rows)}] Analyzing {plot}...', flush=True)
        report = analyze_plot(
            row, settings, references[plot], cluster_dir, force=force)
        report['config_fingerprint'] = fingerprint
        cache_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding='utf-8')
        reports.append(report)
        print(
            f'  recovered={report["num_newly_recovered_trees"]}, '
            f'recall gain={report["effects"]["oracle_recall_gain_pp"]:+.3f} pp',
            flush=True)

    aggregate, effects, gate = aggregate_reports(reports, settings)
    report = {
        'config': str(config_path),
        'settings': {
            name: settings[name] for name in (
                'cluster_sizes', 'base_cluster_size', 'min_iou', 'n_jobs',
                'e2a_reference_digest')},
        'plots': reports,
        'aggregate': aggregate,
        'effects': effects,
        'gate': gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    write_per_plot_csv(
        output_dir / 'per_plot_metrics.csv', reports,
        settings['base_cluster_size'])
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'E3a multi-scale proposal gate failed; close grouping route.')
    return report


def run(config_path, force=False):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        raw = yaml.safe_load(file)
    return run_settings(raw, config_path=config_path, force=force)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the E3a fixed-validation multi-scale grouping Oracle.')
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--force', action='store_true',
        help='Recompute all HDBSCAN scales and per-plot caches.')
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
