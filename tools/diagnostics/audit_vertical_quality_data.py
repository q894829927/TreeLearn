import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np


SUPPORTED_EXTENSIONS = {'.las', '.laz', '.npz', '.npy', '.txt'}


def json_default(value):
    """Convert NumPy containers/scalars used by audit checks to JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(
        f'Object of type {value.__class__.__name__} is not JSON serializable')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit full-forest data for vertical quality scoring.')
    parser.add_argument(
        '--config',
        default=(
            'configs/experiments/vertical_instance_quality/'
            'e0_data_audit.yaml'))
    return parser.parse_args()


def load_yaml(path):
    import yaml

    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_value(*args):
    try:
        return subprocess.check_output(
            ['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_record(path, include_sha256=False):
    path = Path(path)
    record = {
        'path': str(path),
        'exists': path.is_file(),
    }
    if not record['exists']:
        return record
    stat = path.stat()
    record.update({
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    })
    if include_sha256:
        record['sha256'] = sha256_file(path)
    return record


def _empty_accumulator():
    return {
        'point_count': 0,
        'tree_point_count': 0,
        'non_tree_point_count': 0,
        'unlabeled_point_count': 0,
        'label_conflict_count': 0,
        'tree_counts': {},
        'boundary_tree_ids': set(),
    }


def _update_tree_counts(accumulator, tree_ids):
    if len(tree_ids) == 0:
        return
    unique_ids, counts = np.unique(tree_ids, return_counts=True)
    for tree_id, count in zip(unique_ids, counts):
        key = int(tree_id)
        accumulator['tree_counts'][key] = (
            accumulator['tree_counts'].get(key, 0) + int(count))


def _accumulate_labels(
        accumulator, labels, x=None, y=None, bounds=None,
        edge_margin_m=0.5, classes=None):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    tree_mask = labels > 0
    if classes is None:
        non_tree_mask = labels == 0
        unlabeled_mask = labels < 0
        conflict_mask = np.zeros(len(labels), dtype=bool)
    else:
        classes = np.asarray(classes).reshape(-1)
        class_non_tree = np.isin(classes, [1, 2])
        conflict_mask = tree_mask & class_non_tree
        non_tree_mask = (~tree_mask) & class_non_tree
        unlabeled_mask = (~tree_mask) & (~class_non_tree)

    accumulator['point_count'] += int(len(labels))
    accumulator['tree_point_count'] += int(tree_mask.sum())
    accumulator['non_tree_point_count'] += int(non_tree_mask.sum())
    accumulator['unlabeled_point_count'] += int(unlabeled_mask.sum())
    accumulator['label_conflict_count'] += int(conflict_mask.sum())
    _update_tree_counts(accumulator, labels[tree_mask])

    if bounds is not None and x is not None and y is not None:
        min_x, min_y, max_x, max_y = bounds
        edge_mask = (
            (np.asarray(x) <= min_x + edge_margin_m) |
            (np.asarray(x) >= max_x - edge_margin_m) |
            (np.asarray(y) <= min_y + edge_margin_m) |
            (np.asarray(y) >= max_y - edge_margin_m))
        boundary_ids = np.unique(labels[tree_mask & edge_mask])
        accumulator['boundary_tree_ids'].update(
            int(value) for value in boundary_ids)


def _finalize_summary(path, accumulator, bounds, label_source,
                      dimensions=None):
    point_count = accumulator['point_count']
    labeled_count = (
        accumulator['tree_point_count'] +
        accumulator['non_tree_point_count'])
    tree_counts = np.asarray(
        list(accumulator['tree_counts'].values()), dtype=np.int64)
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    x_span = max_x - min_x
    y_span = max_y - min_y
    xy_area = max(x_span * y_span, 1e-9)
    summary = {
        'path': str(path),
        'exists': True,
        'format': Path(path).suffix.lower(),
        'size_bytes': int(Path(path).stat().st_size),
        'label_source': label_source,
        'dimensions': list(dimensions or []),
        'point_count': int(point_count),
        'tree_point_count': accumulator['tree_point_count'],
        'non_tree_point_count': accumulator['non_tree_point_count'],
        'unlabeled_point_count': accumulator['unlabeled_point_count'],
        'label_conflict_count': accumulator['label_conflict_count'],
        'label_coverage_rate': (
            float(labeled_count / point_count) if point_count else 0.0),
        'tree_point_rate': (
            float(accumulator['tree_point_count'] / point_count)
            if point_count else 0.0),
        'num_trees': int(len(tree_counts)),
        'boundary_tree_count_bbox': int(
            len(accumulator['boundary_tree_ids'])),
        'bounds': {
            'min': [float(min_x), float(min_y), float(min_z)],
            'max': [float(max_x), float(max_y), float(max_z)],
        },
        'xy_bbox_area_m2': float(xy_area),
        'point_density_bbox_m2': float(point_count / xy_area),
        'tree_points_per_instance': {
            'min': int(tree_counts.min()) if len(tree_counts) else 0,
            'median': float(np.median(tree_counts)) if len(tree_counts) else 0.0,
            'p90': float(np.percentile(tree_counts, 90))
            if len(tree_counts) else 0.0,
            'max': int(tree_counts.max()) if len(tree_counts) else 0,
        },
    }
    return summary, set(accumulator['tree_counts'])


def summarize_labeled_arrays(
        path, coords, labels, edge_margin_m=0.5,
        label_source='fourth_column'):
    coords = np.asarray(coords)
    if len(coords) == 0:
        raise ValueError(f'Point cloud is empty: {path}')
    mins = coords[:, :3].min(axis=0)
    maxs = coords[:, :3].max(axis=0)
    bounds_xy = (mins[0], mins[1], maxs[0], maxs[1])
    accumulator = _empty_accumulator()
    _accumulate_labels(
        accumulator, labels, coords[:, 0], coords[:, 1], bounds_xy,
        edge_margin_m=edge_margin_m)
    bounds = (*mins.tolist(), *maxs.tolist())
    return _finalize_summary(
        path, accumulator, bounds, label_source)


def scan_las(path, chunk_size, edge_margin_m):
    import laspy

    accumulator = _empty_accumulator()
    with laspy.open(path) as reader:
        header = reader.header
        dimensions = [str(name) for name in header.point_format.dimension_names]
        dimension_lookup = {name.lower(): name for name in dimensions}
        tree_dimension = dimension_lookup.get('treeid')
        class_dimension = dimension_lookup.get('classification')
        mins = np.asarray(header.mins, dtype=float)
        maxs = np.asarray(header.maxs, dtype=float)
        bounds_xy = (mins[0], mins[1], maxs[0], maxs[1])
        for points in reader.chunk_iterator(chunk_size):
            if tree_dimension is None:
                labels = -np.ones(len(points), dtype=np.int64)
            else:
                labels = np.asarray(points[tree_dimension], dtype=np.int64)
            classes = (
                np.asarray(points[class_dimension])
                if class_dimension is not None else None)
            _accumulate_labels(
                accumulator, labels,
                np.asarray(points.x), np.asarray(points.y), bounds_xy,
                edge_margin_m=edge_margin_m, classes=classes)
    label_source = (
        'treeID+classification' if tree_dimension and class_dimension
        else 'treeID_only' if tree_dimension else 'missing_treeID')
    bounds = (*mins.tolist(), *maxs.tolist())
    return _finalize_summary(
        path, accumulator, bounds, label_source, dimensions)


def scan_array_file(path, edge_margin_m):
    suffix = Path(path).suffix.lower()
    if suffix == '.npz':
        archive = np.load(path)
        if 'points' not in archive:
            raise ValueError(f'NPZ lacks points: {path}')
        coords = archive['points']
        labels = archive['labels'] if 'labels' in archive else (
            -np.ones(len(coords), dtype=np.int64))
    elif suffix == '.npy':
        data = np.load(path, mmap_mode='r')
        coords = data[:, :3]
        labels = data[:, 3] if data.shape[1] >= 4 else (
            -np.ones(len(coords), dtype=np.int64))
    else:
        data = np.loadtxt(path)
        coords = data[:, :3]
        labels = data[:, -1] if data.shape[1] >= 4 else (
            -np.ones(len(coords), dtype=np.int64))
    return summarize_labeled_arrays(
        path, coords, labels, edge_margin_m=edge_margin_m)


def scan_point_cloud(path, chunk_size, edge_margin_m):
    path = Path(path)
    if not path.is_file():
        return ({'path': str(path), 'exists': False}, set())
    if path.suffix.lower() in {'.las', '.laz'}:
        return scan_las(path, chunk_size, edge_margin_m)
    return scan_array_file(path, edge_margin_m)


def read_feature_ids(path):
    ids = set()
    row_count = 0
    with open(path, newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            row_count += 1
            ids.add(int(row['instance_id']))
    return ids, row_count


def read_evaluation_prediction_ids(path):
    import torch

    try:
        results = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        results = torch.load(path, map_location='cpu')
    detection = results['detection_results']
    keys = [
        'matched_preds', 'non_matched_preds',
        'non_matched_preds_filtered']
    values = {}
    for key in keys:
        values[key] = set(
            np.asarray(detection[key], dtype=np.int64).tolist())
    return values


def audit_diagnostic_artifact(
        artifact, chunk_size, edge_margin_m, scan_cache):
    paths = {
        key: Path(artifact[key])
        for key in [
            'prediction', 'ground_truth', 'features', 'metadata',
            'evaluation']}
    records = {key: file_record(path) for key, path in paths.items()}
    checks = {
        'all_files_exist': all(item['exists'] for item in records.values()),
    }
    output = {
        'name': artifact['name'],
        'files': records,
        'checks': checks,
    }
    if not checks['all_files_exist']:
        checks['passed'] = False
        return output

    prediction_key = str(paths['prediction'].resolve())
    if prediction_key not in scan_cache:
        scan_cache[prediction_key] = scan_point_cloud(
            paths['prediction'], chunk_size, edge_margin_m)
    prediction_summary, prediction_ids = scan_cache[prediction_key]
    feature_ids, feature_rows = read_feature_ids(paths['features'])
    with open(paths['metadata'], encoding='utf-8') as file:
        metadata = json.load(file)
    evaluation_ids = read_evaluation_prediction_ids(paths['evaluation'])
    referenced_ids = (
        evaluation_ids['matched_preds'] |
        evaluation_ids['non_matched_preds'])

    checks.update({
        'feature_rows_match_metadata': (
            feature_rows == int(metadata.get('num_instances', -1))),
        'feature_ids_match_prediction_ids': feature_ids == prediction_ids,
        'evaluation_ids_exist_in_features': referenced_ids <= feature_ids,
        'evaluation_belongs_to_prediction_dir': (
            paths['evaluation'].parent.parent.resolve() ==
            paths['prediction'].parent.resolve()),
        'evaluation_not_older_than_prediction': (
            paths['evaluation'].stat().st_mtime_ns >=
            paths['prediction'].stat().st_mtime_ns),
        'plot_name_matches': (
            str(metadata.get('plot_name')) ==
            str(artifact.get('expected_plot_name'))),
        'seed_keep_ratio_matches': np.isclose(
            float(metadata.get('seed_keep_ratio', np.nan)),
            float(artifact.get('expected_seed_keep_ratio', 1.0))),
        'checkpoint_exists': Path(
            metadata.get('checkpoint', '')).is_file(),
    })
    checks['passed'] = all(checks.values())
    output.update({
        'prediction_summary': prediction_summary,
        'metadata_payload': metadata,
        'feature_rows': feature_rows,
        'evaluation_counts': {
            key: len(value) for key, value in evaluation_ids.items()},
    })
    return output


def expand_source_files(source):
    candidates = []
    for path in source.get('paths', []):
        candidates.append(path)
    pattern = source.get('glob')
    if pattern:
        candidates.extend(glob(pattern, recursive=True))
    files = sorted({
        str(Path(path)) for path in candidates
        if Path(path).is_file() and
        Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
    })
    return files


def forest_eligibility(row, settings):
    """Return strict full-label and role-specific quality-data eligibility."""
    base_requirements = bool(
        row.get('num_trees', 0) > 0 and
        row.get('non_tree_point_count', 0) > 0 and
        row.get('label_conflict_count', 0) == 0)
    coverage = row.get('label_coverage_rate', 0.0)
    full_forest = bool(
        base_requirements and coverage >= settings['min_label_coverage'])

    if row.get('role') == 'validation':
        role_eligible = bool(
            base_requirements and
            coverage >= settings['min_validation_label_coverage'] and
            row.get('num_trees', 0) >= settings['min_validation_trees'])
        threshold = settings['min_validation_label_coverage']
        minimum_trees = settings['min_validation_trees']
    else:
        role_eligible = full_forest
        threshold = settings['min_label_coverage']
        minimum_trees = 1
    return full_forest, role_eligible, threshold, minimum_trees


def audit_forest_sources(config, scan_cache):
    settings = config['audit']
    rows = []
    for source in config.get('forest_sources', []):
        for path in expand_source_files(source):
            resolved = str(Path(path).resolve())
            if resolved not in scan_cache:
                scan_cache[resolved] = scan_point_cloud(
                    path, settings['chunk_size'],
                    settings['edge_margin_m'])
            summary, _ = scan_cache[resolved]
            row = dict(summary)
            row.update({
                key: source.get(key) for key in [
                    'name', 'role', 'sensor', 'country', 'forest_type',
                    'label_quality', 'split_group', 'source']
            })
            if row.get('split_group') == 'filename':
                row['split_group'] = Path(path).stem
            full, role_eligible, threshold, minimum_trees = (
                forest_eligibility(row, settings))
            row['eligible_full_forest'] = full
            row['eligible_for_role'] = role_eligible
            row['role_label_coverage_threshold'] = threshold
            row['role_minimum_trees'] = minimum_trees
            rows.append(row)
    return rows


def audit_external_candidates(config):
    rows = []
    for source in config.get('external_test_candidates', []):
        files = [
            path for path in glob(source['glob'], recursive=True)
            if Path(path).is_file()]
        rows.append({
            'name': source['name'],
            'glob': source['glob'],
            'num_files': len(files),
            'available': bool(files),
            'required_now': bool(source.get('required_now', False)),
            'source': source.get('source'),
        })
    return rows


def audit_reproducibility(config):
    reproducibility = config['reproducibility']
    configs = [
        file_record(path, include_sha256=True)
        for path in reproducibility.get('fixed_configs', [])]
    checkpoints = []
    for item in reproducibility.get('checkpoints', []):
        record = file_record(
            item['path'], include_sha256=bool(item.get('sha256', False)))
        record['name'] = item['name']
        checkpoints.append(record)
    status = git_value('status', '--porcelain')
    return {
        'git': {
            'branch': git_value('branch', '--show-current'),
            'commit': git_value('rev-parse', 'HEAD'),
            'status_clean': status == '',
            'status_porcelain': status or '',
        },
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'numpy': np.__version__,
        },
        'random_seeds': reproducibility.get('random_seeds', []),
        'fixed_configs': configs,
        'checkpoints': checkpoints,
    }


def build_gate(config, forests, diagnostics, reproducibility):
    settings = config['audit']
    eligible_train = [
        item for item in forests
        if item.get('role') == 'train' and
        item.get('eligible_for_role', item.get('eligible_full_forest'))]
    eligible_validation = [
        item for item in forests
        if item.get('role') == 'validation' and
        item.get('eligible_for_role', item.get('eligible_full_forest'))]
    manual_validation = [
        item for item in eligible_validation
        if 'manual' in str(item.get('label_quality', '')).lower()]
    known_metadata = all(
        item.get('sensor') not in (None, '', 'unknown') and
        item.get('forest_type') not in (None, '', 'unknown')
        for item in forests)
    required_files = (
        reproducibility['fixed_configs'] +
        reproducibility['checkpoints'])
    gate = {
        'expected_branch': (
            reproducibility['git']['branch'] ==
            settings['expected_branch']),
        'enough_complete_training_forests': (
            len(eligible_train) >= settings['min_training_forests']),
        'enough_usable_validation_forests': (
            len(eligible_validation) >= settings['min_validation_forests']),
        'manual_validation_available': bool(manual_validation),
        'forest_metadata_complete': known_metadata,
        'diagnostic_artifacts_consistent': (
            bool(diagnostics) and
            all(item['checks'].get('passed', False)
                for item in diagnostics)),
        'fixed_files_exist': all(item['exists'] for item in required_files),
    }
    gate['passed'] = all(gate.values())
    return gate


def format_markdown(report):
    rows = [
        '# E0 垂直实例质量数据审计', '',
        f"- Git branch：`{report['reproducibility']['git']['branch']}`",
        f"- Git commit：`{report['reproducibility']['git']['commit']}`",
        f"- Git clean：{report['reproducibility']['git']['status_clean']}",
        '', '## 完整森林', '',
        '| Role | Plot | Points | Trees | Label coverage | Boundary trees | Role eligible |',
        '|---|---|---:|---:|---:|---:|---|']
    for item in report['forests']:
        rows.append(
            f"| {item.get('role')} | {Path(item['path']).stem} | "
            f"{item.get('point_count', 0):,} | {item.get('num_trees', 0):,} | "
            f"{100 * item.get('label_coverage_rate', 0):.3f}% | "
            f"{item.get('boundary_tree_count_bbox', 0):,} | "
            f"{item.get('eligible_for_role', item.get('eligible_full_forest', False))} |")
    rows.extend(['', '## 诊断 artifact 一致性', ''])
    for item in report['diagnostic_artifacts']:
        rows.append(
            f"- {item['name']}: **{item['checks'].get('passed', False)}**")
        for name, passed in item['checks'].items():
            rows.append(f'  - {name}: {passed}')
    rows.extend(['', '## 外部测试候选', ''])
    for item in report['external_test_candidates']:
        rows.append(
            f"- {item['name']}: available={item['available']}, "
            f"files={item['num_files']}")
    rows.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        rows.append(f'- {name}: **{passed}**')
    rows.append('')
    rows.append(
        'PASS：可以实现 E1 Oracle。' if report['gate']['passed'] else
        'STOP：先修复失败项，不进入 E1。')
    rows.append('')
    return '\n'.join(rows)


def main():
    args = parse_args()
    config = load_yaml(args.config)
    scan_cache = {}
    reproducibility = audit_reproducibility(config)
    forests = audit_forest_sources(config, scan_cache)
    diagnostics = [
        audit_diagnostic_artifact(
            artifact,
            config['audit']['chunk_size'],
            config['audit']['edge_margin_m'],
            scan_cache)
        for artifact in config.get('diagnostic_artifacts', [])]
    external = audit_external_candidates(config)
    gate = build_gate(config, forests, diagnostics, reproducibility)
    report = {
        'config': args.config,
        'reproducibility': reproducibility,
        'forests': forests,
        'diagnostic_artifacts': diagnostics,
        'external_test_candidates': external,
        'gate': gate,
    }

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'e0_data_audit.json'
    markdown_path = output_dir / 'e0_data_audit.md'
    json_path.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False, default=json_default),
        encoding='utf-8')
    markdown = format_markdown(report)
    markdown_path.write_text(markdown, encoding='utf-8')
    print(markdown)
    print(f'Saved JSON: {json_path}')
    print(f'Saved Markdown: {markdown_path}')


if __name__ == '__main__':
    main()
