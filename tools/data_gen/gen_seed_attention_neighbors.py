"""Generate fixed same-forest KNN caches for E1c seed relation models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def query_fixed_radius_neighbors(
        base_votes_xy, num_neighbors=8, radius=0.6,
        query_chunk_size=200000, workers=-1):
    """Return local int32 neighbours; column zero is always the query itself."""
    from scipy.spatial import cKDTree

    votes = np.asarray(base_votes_xy, dtype=np.float64)
    if votes.ndim != 2 or votes.shape[1] != 2 or len(votes) == 0:
        raise ValueError('base_votes_xy must have non-empty shape [N, 2].')
    k = int(num_neighbors)
    chunk_size = int(query_chunk_size)
    radius = float(radius)
    if k < 2 or chunk_size < 1 or radius <= 0:
        raise ValueError('Neighbour configuration is invalid.')

    tree = cKDTree(votes)
    output = np.full((len(votes), k), -1, dtype=np.int32)
    output[:, 0] = np.arange(len(votes), dtype=np.int32)
    query_k = min(k, len(votes))
    for start in range(0, len(votes), chunk_size):
        end = min(start + chunk_size, len(votes))
        query_indices = np.arange(start, end, dtype=np.int64)
        try:
            distances, indices = tree.query(
                votes[start:end], k=query_k,
                distance_upper_bound=radius, workers=int(workers))
        except TypeError:
            distances, indices = tree.query(
                votes[start:end], k=query_k,
                distance_upper_bound=radius)
        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        distances[indices == query_indices[:, None]] = np.inf
        order = np.argsort(distances, axis=1, kind='stable')
        take = min(k - 1, query_k)
        chosen_order = order[:, :take]
        chosen_indices = np.take_along_axis(
            indices, chosen_order, axis=1)
        chosen_distances = np.take_along_axis(
            distances, chosen_order, axis=1)
        valid = (
            np.isfinite(chosen_distances) &
            (chosen_distances <= radius) &
            (chosen_indices >= 0) &
            (chosen_indices < len(votes)))
        output[start:end, 1:1 + take] = np.where(
            valid, chosen_indices, -1).astype(np.int32)
        print(
            f'  queried {end:,}/{len(votes):,} candidates '
            f'({100 * end / len(votes):.1f}%)', flush=True)
    return output


def validate_neighbor_cache(
        cache_path, source_artifact_path, num_neighbors, radius):
    source_path = Path(source_artifact_path)
    with np.load(source_path, allow_pickle=False) as source:
        source_candidates = source['candidate_indices']
        votes = source['base_votes_xy'].astype(np.float64)
        critical = source['target_coverage_critical'].astype(bool)
    with np.load(cache_path, allow_pickle=False) as cache:
        if set(cache.files) != {'candidate_indices', 'neighbor_indices'}:
            raise ValueError(f'Unexpected arrays in {cache_path}.')
        candidate_indices = cache['candidate_indices']
        neighbours = cache['neighbor_indices']
    count = len(source_candidates)
    if not np.array_equal(candidate_indices, source_candidates):
        raise ValueError('Neighbour cache candidate indices are misaligned.')
    if neighbours.shape != (count, int(num_neighbors)):
        raise ValueError('Neighbour cache shape is invalid.')
    if neighbours.dtype != np.int32:
        raise ValueError('Neighbour indices must use int32 storage.')
    if np.any((neighbours < -1) | (neighbours >= count)):
        raise ValueError('Neighbour indices are out of bounds.')
    if not np.array_equal(
            neighbours[:, 0], np.arange(count, dtype=np.int32)):
        raise ValueError('Every candidate must be its own first support.')
    valid = neighbours >= 0
    rows, columns = np.where(valid[:, 1:])
    if len(rows):
        columns = columns + 1
        distances = np.linalg.norm(
            votes[neighbours[rows, columns]] - votes[rows], axis=1)
        if np.any(distances > float(radius) + 1e-5):
            raise ValueError('Neighbour cache contains out-of-radius edges.')
    has_relation = valid[:, 1:].any(axis=1)
    return {
        'num_candidates': int(count),
        'num_neighbors': int(num_neighbors),
        'radius': float(radius),
        'relation_coverage': float(has_relation.mean()),
        'critical_relation_coverage': float(
            has_relation[critical].mean()) if critical.any() else 0.0,
        'mean_valid_neighbors': float(valid.sum(axis=1).mean()),
        'all_self_edges_valid': True,
        'all_indices_in_bounds': True,
        'all_edges_within_radius': True,
    }


def load_manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('The E1a manifest is empty.')
    if {row['split'] for row in rows} != {'train', 'validation'}:
        raise ValueError('Both fixed E1a splits are required.')
    return rows


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_markdown(summary):
    lines = [
        '# E1c-a Seed Relation 邻域缓存审计', '',
        f'- 森林数：{summary["num_artifacts"]}',
        f'- 候选数：{summary["num_candidates"]:,}',
        f'- K：{summary["num_neighbors"]}',
        f'- 半径：{summary["radius"]:.3f} m',
        f'- 总体 relation coverage：'
        f'{100 * summary["relation_coverage"]:.3f}%',
        f'- Critical relation coverage：'
        f'{100 * summary["critical_relation_coverage"]:.3f}%',
        f'- 平均有效支持点：{summary["mean_valid_neighbors"]:.3f}', '',
        '## Gate', '',
    ]
    for name, passed in summary['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend([
        '',
        ('PASS：可以训练参数匹配的 Neighborhood-MLP 与 '
         'Relation-Attention。'
         if summary['gate']['passed'] else
         'STOP：邻域缓存未通过，不能开始 E1c 训练。'),
    ])
    return '\n'.join(lines) + '\n'


def run(config_path, force=False):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    rows = load_manifest(settings['manifest_path'])
    data_root = Path(settings['data_root']).resolve()
    output_root = Path(settings['output_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    k = int(settings['num_neighbors'])
    radius = float(settings['radius'])
    output_rows = []
    summaries = []
    for position, row in enumerate(rows, start=1):
        source_path = Path(row['artifact_path']).resolve()
        if data_root not in source_path.parents or not source_path.is_file():
            raise ValueError(f'Invalid E1a artifact path: {source_path}')
        destination = (
            output_root / row['split'] /
            f'{row["source_plot"]}_neighbors.npz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and not force:
            print(
                f'SKIP {row["source_plot"]} ({position}/{len(rows)}): '
                'validating existing cache...', flush=True)
        else:
            print(
                f'RUN {row["source_plot"]} ({position}/{len(rows)}): '
                f'{int(row["num_candidates"]):,} candidates', flush=True)
            with np.load(source_path, allow_pickle=False) as source:
                candidates = source['candidate_indices'].copy()
                votes = source['base_votes_xy'].copy()
            neighbours = query_fixed_radius_neighbors(
                votes, k, radius,
                int(settings['query_chunk_size']),
                int(settings.get('workers', -1)))
            np.savez_compressed(
                destination,
                candidate_indices=candidates,
                neighbor_indices=neighbours)
        validation = validate_neighbor_cache(
            destination, source_path, k, radius)
        summaries.append(validation)
        output_rows.append({
            'source_plot': row['source_plot'],
            'split': row['split'],
            'source_artifact_path': str(source_path),
            'neighbor_artifact_path': str(destination),
            **validation,
        })
        print(
            f'DONE {row["source_plot"]}: relation coverage='
            f'{100 * validation["relation_coverage"]:.2f}%, '
            f'critical={100 * validation["critical_relation_coverage"]:.2f}%',
            flush=True)

    manifest_path = output_root / 'manifest.csv'
    write_csv(manifest_path, output_rows)
    total_candidates = sum(row['num_candidates'] for row in summaries)
    relation_coverage = sum(
        row['relation_coverage'] * row['num_candidates']
        for row in summaries) / max(total_candidates, 1)
    critical_weights = []
    for output_row in output_rows:
        with np.load(
                output_row['source_artifact_path'],
                allow_pickle=False) as source:
            critical_count = int(source['target_coverage_critical'].sum())
        critical_weights.append(critical_count)
    total_critical = sum(critical_weights)
    critical_coverage = sum(
        row['critical_relation_coverage'] * weight
        for row, weight in zip(summaries, critical_weights)
    ) / max(total_critical, 1)
    mean_valid = sum(
        row['mean_valid_neighbors'] * row['num_candidates']
        for row in summaries) / max(total_candidates, 1)
    gate_cfg = settings['gate']
    gate = {
        'all_fixed_artifacts_present': (
            len(summaries) == int(gate_cfg['expected_artifacts'])),
        'candidate_count_matches': (
            total_candidates == sum(
                int(row['num_candidates']) for row in rows)),
        'relation_coverage_passed': (
            relation_coverage >= float(
                gate_cfg['min_relation_coverage'])),
        'critical_relation_coverage_passed': (
            critical_coverage >= float(
                gate_cfg['min_critical_relation_coverage'])),
        'self_edges_passed': all(
            row['all_self_edges_valid'] for row in summaries),
        'index_bounds_passed': all(
            row['all_indices_in_bounds'] for row in summaries),
        'radius_passed': all(
            row['all_edges_within_radius'] for row in summaries),
    }
    gate['passed'] = bool(all(gate.values()))
    summary = {
        'num_artifacts': len(summaries),
        'num_candidates': int(total_candidates),
        'num_neighbors': k,
        'radius': radius,
        'relation_coverage': float(relation_coverage),
        'critical_relation_coverage': float(critical_coverage),
        'mean_valid_neighbors': float(mean_valid),
        'manifest_path': str(manifest_path),
        'gate': gate,
    }
    with (output_root / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    markdown = format_markdown(summary)
    (output_root / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError('E1c-a neighbour cache gate failed.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate E1c fixed seed-relation neighbours.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config, force=args.force)


if __name__ == '__main__':
    main()
