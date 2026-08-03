import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized not in {'true', 'false'}:
        raise ValueError(f'Expected a boolean value, got {value!r}.')
    return normalized == 'true'


def _as_string(value):
    array = np.asarray(value)
    return str(array.item())


def _load_rows(sample_path):
    with Path(sample_path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'No rows found in audit sample: {sample_path}')
    required = {
        'instance_id', 'target_max_iou', 'target_labeled_fraction',
        'target_tree_point_fraction', 'target_is_edge', 'target_valid',
        'target_classification_valid', 'target_is_true_tree',
        'source_plot', 'split', 'artifact_path',
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f'Audit sample misses columns: {sorted(missing)}')
    return rows


def _load_artifact(path, cache):
    resolved = Path(path).resolve()
    if resolved not in cache:
        if not resolved.is_file():
            raise FileNotFoundError(f'Missing quality artifact: {resolved}')
        with np.load(resolved, allow_pickle=False) as data:
            cache[resolved] = {
                name: np.asarray(data[name]).copy()
                for name in data.files
            }
    return cache[resolved], resolved


def _value_matches(csv_value, array_value, tolerance=1e-6):
    return abs(float(csv_value) - float(array_value)) <= tolerance


def _audit_row(row, settings, cache):
    errors = []
    data_root = Path(settings['data_root']).resolve()
    artifact, artifact_path = _load_artifact(row['artifact_path'], cache)
    if data_root not in artifact_path.parents:
        errors.append('artifact_outside_data_root')

    split = row['split']
    plot = row['source_plot']
    expected_path = (
        data_root / split / f'{plot}.npz').resolve()
    if artifact_path != expected_path:
        errors.append('artifact_path_mismatch')
    if _as_string(artifact['source_plot']) != plot:
        errors.append('npz_source_plot_mismatch')
    if _as_string(artifact['split']) != split:
        errors.append('npz_split_mismatch')

    instance_id = int(row['instance_id'])
    instance_ids = artifact['instance_ids'].astype(np.int64)
    matches = np.flatnonzero(instance_ids == instance_id)
    if len(matches) != 1:
        errors.append('instance_id_not_unique')
        return errors, None
    index = int(matches[0])

    scalar_float_fields = (
        'target_max_iou',
        'target_labeled_fraction',
        'target_tree_point_fraction',
    )
    scalar_bool_fields = (
        'target_is_edge',
        'target_valid',
        'target_classification_valid',
        'target_is_true_tree',
    )
    for field in scalar_float_fields:
        if not _value_matches(row[field], artifact[field][index]):
            errors.append(f'{field}_csv_npz_mismatch')
    for field in scalar_bool_fields:
        if _as_bool(row[field]) != bool(artifact[field][index]):
            errors.append(f'{field}_csv_npz_mismatch')

    iou = float(artifact['target_max_iou'][index])
    labeled_fraction = float(artifact['target_labeled_fraction'][index])
    tree_fraction = float(artifact['target_tree_point_fraction'][index])
    is_edge = bool(artifact['target_is_edge'][index])
    valid = bool(artifact['target_valid'][index])
    classification_valid = bool(
        artifact['target_classification_valid'][index])
    is_true = bool(artifact['target_is_true_tree'][index])
    positive_threshold = float(settings['thresholds']['positive_iou'])
    negative_threshold = float(settings['thresholds']['negative_iou'])

    if not 0.0 <= iou <= 1.0:
        errors.append('iou_out_of_range')
    if not 0.0 <= labeled_fraction <= 1.0:
        errors.append('labeled_fraction_out_of_range')
    if not 0.0 <= tree_fraction <= labeled_fraction + 1e-6:
        errors.append('tree_fraction_inconsistent')
    if is_true != (iou >= positive_threshold):
        errors.append('positive_label_inconsistent')
    expected_classification_valid = valid and (
        iou >= positive_threshold or iou < negative_threshold)
    if classification_valid != expected_classification_valid:
        errors.append('classification_mask_inconsistent')
    if is_edge and valid:
        errors.append('edge_instance_marked_valid')

    tokens = artifact['vertical_tokens'][index]
    layer_mask = artifact['layer_valid_mask'][index].astype(bool)
    global_features = artifact['global_feature_values'][index]
    if not np.isfinite(tokens).all():
        errors.append('non_finite_vertical_token')
    if not np.isfinite(global_features).all():
        errors.append('non_finite_global_feature')
    if not np.any(layer_mask):
        errors.append('no_valid_vertical_layer')
    if np.any(tokens[~layer_mask] != 0):
        errors.append('nonzero_empty_vertical_layer')

    token_feature_names = artifact['token_feature_names'].astype(str).tolist()
    if 'occupancy_fraction' not in token_feature_names:
        errors.append('occupancy_feature_missing')
    else:
        occupancy_index = token_feature_names.index('occupancy_fraction')
        occupancy_sum = float(tokens[:, occupancy_index].sum())
        if not np.isclose(occupancy_sum, 1.0, atol=1e-5):
            errors.append('occupancy_does_not_sum_to_one')

    if iou >= positive_threshold:
        target_class = 'positive'
    elif iou < negative_threshold:
        target_class = 'negative'
    else:
        target_class = 'ambiguous'
    audited = {
        'source_plot': plot,
        'split': split,
        'instance_id': instance_id,
        'target_class': target_class,
        'target_max_iou': iou,
        'target_labeled_fraction': labeled_fraction,
        'target_tree_point_fraction': tree_fraction,
        'target_is_edge': is_edge,
        'target_valid': valid,
        'target_classification_valid': classification_valid,
        'num_valid_layers': int(layer_mask.sum()),
        'errors': ';'.join(errors),
    }
    return errors, audited


def audit_artifacts(settings):
    rows = _load_rows(settings['sample_path'])
    cache = {}
    audited_rows = []
    errors = []
    for row_number, row in enumerate(rows, start=2):
        try:
            row_errors, audited = _audit_row(row, settings, cache)
        except Exception as error:
            row_errors = [f'exception:{type(error).__name__}:{error}']
            audited = None
        if row_errors:
            errors.append({
                'csv_row': row_number,
                'source_plot': row.get('source_plot'),
                'instance_id': row.get('instance_id'),
                'errors': row_errors,
            })
        if audited is not None:
            audited_rows.append(audited)

    plots = {row['source_plot'] for row in audited_rows}
    splits = {row['split'] for row in audited_rows}
    class_counts = Counter(row['target_class'] for row in audited_rows)
    edge_count = sum(row['target_is_edge'] for row in audited_rows)
    gate_config = settings['gate']
    gate = {
        'enough_samples': (
            len(audited_rows) >= int(gate_config['min_sample_count'])),
        'enough_plots': (
            len(plots) >= int(gate_config['min_plot_count'])),
        'train_present': (
            not gate_config.get('require_train', True) or 'train' in splits),
        'validation_present': (
            not gate_config.get('require_validation', True)
            or 'validation' in splits),
        'positive_present': class_counts['positive'] > 0,
        'negative_present': class_counts['negative'] > 0,
        'ambiguous_present': (
            not gate_config.get('require_ambiguous', True)
            or class_counts['ambiguous'] > 0),
        'edge_present': (
            not gate_config.get('require_edge', True) or edge_count > 0),
        'all_rows_audited': len(audited_rows) == len(rows),
        'all_consistency_checks_passed': not errors,
    }
    gate['passed'] = all(gate.values())
    return {
        'sample_path': str(settings['sample_path']),
        'sample_count': len(rows),
        'audited_count': len(audited_rows),
        'plot_count': len(plots),
        'plots': sorted(plots),
        'splits': sorted(splits),
        'class_counts': dict(sorted(class_counts.items())),
        'edge_count': int(edge_count),
        'error_count': len(errors),
        'errors': errors,
        'gate': gate,
    }, audited_rows


def save_report(report, audited_rows, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / 'audit_results.json'
    csv_path = output / 'audited_instances.csv'
    markdown_path = output / 'summary.md'
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8')
    if audited_rows:
        with csv_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file, fieldnames=list(audited_rows[0]))
            writer.writeheader()
            writer.writerows(audited_rows)
    gate_lines = '\n'.join(
        f'- {name}: **{value}**'
        for name, value in report['gate'].items())
    class_lines = '\n'.join(
        f'- {name}: {count}'
        for name, count in report['class_counts'].items())
    markdown = (
        "# E2 fixed 50-instance artifact audit\n\n"
        f"- Samples: {report['sample_count']}\n"
        f"- Plots: {report['plot_count']}\n"
        f"- Splits: {', '.join(report['splits'])}\n"
        f"- Edge instances: {report['edge_count']}\n"
        f"- Consistency errors: {report['error_count']}\n\n"
        "## Class distribution\n\n"
        f"{class_lines}\n\n"
        "## Gate\n\n"
        f"{gate_lines}\n\n"
        "This audit checks CSV/NPZ agreement, target thresholds, edge "
        "validity, source split, layer occupancy, empty layers, and finite "
        "values. Per-point IoU correctness is covered by the E1 Oracle "
        "alignment check and E2 unit tests.\n")
    markdown_path.write_text(markdown, encoding='utf-8')
    return json_path, csv_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit the fixed E2 instance-quality artifact sample.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    import yaml

    args = parse_args()
    with open(args.config, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    report, audited_rows = audit_artifacts(settings)
    paths = save_report(report, audited_rows, settings['output_dir'])
    print(paths[2].read_text(encoding='utf-8'), flush=True)
    print(f'Saved JSON: {paths[0]}', flush=True)
    print(f'Saved audited rows: {paths[1]}', flush=True)
    if not report['gate']['passed']:
        raise RuntimeError('E2 artifact audit gate failed.')
    print(
        'PASS: inspect audited_instances.csv, then explicitly confirm E2.',
        flush=True)


if __name__ == '__main__':
    main()
