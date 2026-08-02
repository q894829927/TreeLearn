import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np




SUPPORTED_FOREST_SUFFIXES = {'.las', '.laz', '.npy', '.npz', '.txt'}
ARTIFACT_NAMES = (
    'quality_instances.npz',
    'quality_targets.csv',
    'metadata.json',
)


def plot_group(plot_name):
    match = re.fullmatch(r'(A1|G[1-4]|L[12]|O1)[NW]', plot_name)
    return match.group(1) if match else plot_name


def validate_splits(splits):
    required = {'train', 'validation'}
    if set(splits) != required:
        raise ValueError(
            f'splits must contain exactly {sorted(required)}, got '
            f'{sorted(splits)}.')
    train = list(splits['train'])
    validation = list(splits['validation'])
    if len(train) != len(set(train)) or len(validation) != len(set(validation)):
        raise ValueError('Duplicate plot names are not allowed within a split.')
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f'Plots leak across splits: {sorted(overlap)}')
    train_groups = {plot_group(name) for name in train}
    validation_groups = {plot_group(name) for name in validation}
    group_overlap = train_groups & validation_groups
    if group_overlap:
        raise ValueError(
            f'Paired plot groups leak across splits: {sorted(group_overlap)}')
    return {
        'train': train,
        'validation': validation,
        'train_groups': sorted(train_groups),
        'validation_groups': sorted(validation_groups),
    }


def locate_forest(source_root, plot_name):
    root = Path(source_root)
    matches = sorted(
        path for path in root.glob(f'{plot_name}.*')
        if path.is_file() and path.suffix.lower() in SUPPORTED_FOREST_SUFFIXES
        and path.stem == plot_name)
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected exactly one supported forest for {plot_name} under '
            f'{root}, found {[str(path) for path in matches]}.')
    return matches[0]


def artifact_destinations(output_root, split, plot_name):
    directory = Path(output_root) / split
    return {
        'npz': directory / f'{plot_name}.npz',
        'targets': directory / f'{plot_name}_targets.csv',
        'metadata': directory / f'{plot_name}_metadata.json',
    }


def artifacts_complete(destinations):
    return all(path.is_file() for path in destinations.values())


def write_resolved_config(
        template_path, forest_path, pipeline_base_dir, split,
        checkpoint, destination):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(template_path))
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(pipeline_base_dir)
    config.quality_split = split
    config.pretrain = str(checkpoint)
    payload = munch_to_dict(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            payload, file, sort_keys=False, allow_unicode=True)
    return payload


def validate_quality_artifact(npz_path, metadata_path):
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            'instance_ids',
            'global_feature_values',
            'vertical_tokens',
            'layer_valid_mask',
            'target_max_iou',
            'target_labeled_fraction',
            'target_tree_point_fraction',
            'target_valid',
            'target_classification_valid',
            'target_is_true_tree',
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f'{npz_path} misses arrays: {sorted(missing)}')
        count = len(data['instance_ids'])
        if count == 0:
            raise ValueError(f'{npz_path} contains no instances.')
        for name in required - {'instance_ids'}:
            if len(data[name]) != count:
                raise ValueError(
                    f'{name} has {len(data[name])} rows, expected {count}.')
        tokens = data['vertical_tokens']
        masks = data['layer_valid_mask']
        if tokens.ndim != 3 or tokens.shape[1] != 8:
            raise ValueError(
                f'Expected [N, 8, D] tokens, got {tokens.shape}.')
        if masks.shape != tokens.shape[:2]:
            raise ValueError(
                f'Layer mask {masks.shape} does not match {tokens.shape[:2]}.')
        if not np.isfinite(tokens).all():
            raise ValueError(f'{npz_path} contains non-finite tokens.')
        if not np.isfinite(data['global_feature_values']).all():
            raise ValueError(f'{npz_path} contains non-finite global features.')
        iou = data['target_max_iou']
        if not np.isfinite(iou).all() or np.any((iou < 0) | (iou > 1)):
            raise ValueError(f'{npz_path} contains invalid IoU targets.')
        if len(np.unique(data['instance_ids'])) != count:
            raise ValueError(f'{npz_path} contains duplicate instance IDs.')

    with Path(metadata_path).open(encoding='utf-8') as file:
        metadata = json.load(file)
    if metadata['num_instances'] != count:
        raise ValueError(
            f'Metadata count {metadata["num_instances"]} != {count}.')
    return metadata


def safe_cleanup_runtime(runtime_directory, runtime_root):
    runtime = Path(runtime_directory).resolve()
    root = Path(runtime_root).resolve()
    if runtime == root or root not in runtime.parents:
        raise ValueError(
            f'Refusing to clean unsafe runtime directory: {runtime}')
    if runtime.exists():
        shutil.rmtree(runtime)


def run_plot(
        plot_name, split, settings, force=False):
    output_root = Path(settings['output_root'])
    runtime_root = output_root / 'runtime'
    runtime_directory = runtime_root / plot_name
    destinations = artifact_destinations(output_root, split, plot_name)

    if artifacts_complete(destinations) and not force:
        metadata = validate_quality_artifact(
            destinations['npz'], destinations['metadata'])
        print(f'SKIP {plot_name}: validated existing artifact.', flush=True)
        return metadata

    if force and runtime_directory.exists():
        safe_cleanup_runtime(runtime_directory, runtime_root)

    forest_path = locate_forest(settings['source_root'], plot_name)
    resolved_config_path = (
        output_root / 'runtime_configs' / f'{plot_name}.yaml')
    payload = write_resolved_config(
        settings['pipeline_template'],
        forest_path,
        runtime_directory,
        split,
        settings['checkpoint'],
        resolved_config_path,
    )
    log_path = output_root / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        '-u',
        'tools/pipeline/pipeline.py',
        '--config',
        str(resolved_config_path),
    ]
    print(
        f'RUN {plot_name} ({split}) from {forest_path}; log={log_path}',
        flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f'Pipeline failed for {plot_name} with exit code '
            f'{completed.returncode}. Inspect {log_path}.')

    source_directory = (
        runtime_directory
        / payload['save_cfg']['results_dir']
        / 'instance_quality'
    )
    sources = {
        'npz': source_directory / ARTIFACT_NAMES[0],
        'targets': source_directory / ARTIFACT_NAMES[1],
        'metadata': source_directory / ARTIFACT_NAMES[2],
    }
    missing = [
        str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Pipeline completed but artifacts are missing: {missing}')

    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    for key, source in sources.items():
        shutil.copy2(source, destinations[key])

    metadata = validate_quality_artifact(
        destinations['npz'], destinations['metadata'])
    if settings.get('cleanup_intermediates', True):
        safe_cleanup_runtime(runtime_directory, runtime_root)
    print(
        f'DONE {plot_name}: {metadata["num_instances"]} candidates, '
        f'{metadata["num_positive_instances"]} positive, '
        f'{metadata["num_negative_instances"]} negative.',
        flush=True)
    return metadata


def load_manifest_rows(output_root):
    rows = []
    root = Path(output_root)
    for split in ('train', 'validation'):
        for target_path in sorted((root / split).glob('*_targets.csv')):
            plot_name = target_path.name[:-len('_targets.csv')]
            npz_path = root / split / f'{plot_name}.npz'
            metadata_path = root / split / f'{plot_name}_metadata.json'
            if not npz_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    f'Incomplete output triplet for {plot_name}.')
            validate_quality_artifact(npz_path, metadata_path)
            with target_path.open(newline='', encoding='utf-8') as file:
                for row in csv.DictReader(file):
                    if row['source_plot'] != plot_name:
                        raise ValueError(
                            f'{target_path} contains source_plot='
                            f'{row["source_plot"]}, expected {plot_name}.')
                    if row['split'] != split:
                        raise ValueError(
                            f'{target_path} contains split={row["split"]}, '
                            f'expected {split}.')
                    row['source_group'] = plot_group(row['source_plot'])
                    row['artifact_path'] = str(npz_path)
                    rows.append(row)
    return rows


def write_manifest(rows, output_root):
    if not rows:
        raise ValueError('No quality artifacts were found for the manifest.')
    fieldnames = list(rows[0])
    path = Path(output_root) / 'manifest.csv'
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def bool_value(value):
    return str(value).lower() == 'true'


def summarize(rows, settings, selected_plots, pilot, manual_audit_confirmed):
    valid_rows = [row for row in rows if bool_value(row['target_valid'])]
    class_rows = [
        row for row in rows
        if bool_value(row['target_classification_valid'])]
    positives = [
        row for row in class_rows
        if bool_value(row['target_is_true_tree'])]
    negatives = [
        row for row in class_rows
        if not bool_value(row['target_is_true_tree'])]
    train_plots = {
        row['source_plot'] for row in rows if row['split'] == 'train'}
    validation_plots = {
        row['source_plot'] for row in rows if row['split'] == 'validation'}
    train_groups = {plot_group(name) for name in train_plots}
    validation_groups = {plot_group(name) for name in validation_plots}
    no_plot_leakage = not (train_plots & validation_plots)
    no_group_leakage = not (train_groups & validation_groups)
    gate_config = settings['gate']
    audit_count = min(
        len(valid_rows), int(gate_config['min_random_audit_instances']))

    if pilot:
        selected_present = set(selected_plots) <= (train_plots | validation_plots)
        gate = {
            'selected_plots_present': selected_present,
            'artifacts_nonempty': all(
                any(row['source_plot'] == plot for row in rows)
                for plot in selected_plots),
            'no_plot_leakage': no_plot_leakage,
            'no_group_leakage': no_group_leakage,
        }
        gate['passed'] = all(gate.values())
    else:
        expected_plots = set(
            settings['splits']['train'] +
            settings['splits']['validation'])
        observed_plots = train_plots | validation_plots
        gate = {
            'all_fixed_plots_present': expected_plots <= observed_plots,
            'enough_positive_instances': (
                len(positives) >= int(
                    gate_config['min_positive_instances'])),
            'enough_negative_instances': (
                len(negatives) >= int(
                    gate_config['min_negative_instances'])),
            'enough_train_groups': (
                len(train_groups) >= int(
                    gate_config['min_independent_groups'])),
            'validation_available': len(validation_groups) >= 1,
            'no_plot_leakage': no_plot_leakage,
            'no_group_leakage': no_group_leakage,
            'random_audit_sample_prepared': (
                audit_count >= int(
                    gate_config['min_random_audit_instances'])),
            'manual_audit_confirmed': bool(manual_audit_confirmed),
        }
        gate['passed'] = all(gate.values())

    return {
        'mode': 'pilot' if pilot else 'full',
        'selected_plots': list(selected_plots),
        'num_instances': len(rows),
        'num_valid_instances': len(valid_rows),
        'num_classification_valid_instances': len(class_rows),
        'num_positive_instances': len(positives),
        'num_negative_instances': len(negatives),
        'train_plots': sorted(train_plots),
        'validation_plots': sorted(validation_plots),
        'train_groups': sorted(train_groups),
        'validation_groups': sorted(validation_groups),
        'manual_audit_sample_size': audit_count,
        'gate': gate,
    }


def write_audit_sample(rows, output_root, sample_size, seed=42):
    valid_rows = [row for row in rows if bool_value(row['target_valid'])]
    if not valid_rows:
        return None
    generator = np.random.default_rng(seed)
    indices = generator.choice(
        len(valid_rows),
        size=min(sample_size, len(valid_rows)),
        replace=False,
    )
    sampled = [valid_rows[int(index)] for index in indices]
    path = Path(output_root) / 'manual_audit_sample.csv'
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(sampled[0]))
        writer.writeheader()
        writer.writerows(sampled)
    return path


def write_summary(summary, output_root):
    root = Path(output_root)
    json_path = root / 'generation_summary.json'
    md_path = root / 'generation_summary.md'
    with json_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    gate_lines = '\n'.join(
        f'- {name}: **{value}**'
        for name, value in summary['gate'].items())
    markdown = (
        '# E2 candidate-instance data generation\n\n'
        f'- mode: {summary["mode"]}\n'
        f'- instances: {summary["num_instances"]}\n'
        f'- valid: {summary["num_valid_instances"]}\n'
        f'- classification valid: '
        f'{summary["num_classification_valid_instances"]}\n'
        f'- positive: {summary["num_positive_instances"]}\n'
        f'- negative: {summary["num_negative_instances"]}\n'
        f'- train plots: {", ".join(summary["train_plots"])}\n'
        f'- validation plots: '
        f'{", ".join(summary["validation_plots"])}\n\n'
        '## Gate\n\n'
        f'{gate_lines}\n'
    )
    with md_path.open('w', encoding='utf-8') as file:
        file.write(markdown)
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate forest-level instance-quality training data.')
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--pilot', action='store_true',
        help='Run only the two configured pilot forests.')
    parser.add_argument(
        '--plots', nargs='+',
        help='Run only named configured plots (for recovery/debugging).')
    parser.add_argument(
        '--force', action='store_true',
        help='Explicitly replace outputs for selected plots.')
    parser.add_argument(
        '--manual-audit-confirmed', action='store_true',
        help='Record that the fixed random audit sample was manually checked.')
    return parser.parse_args()


def main():
    import yaml

    args = parse_args()
    with open(args.config, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    split_info = validate_splits(settings['splits'])
    settings['pipeline_template'] = str(settings['pipeline_template'])
    all_plots = split_info['train'] + split_info['validation']
    if args.pilot:
        selected = list(settings['pilot_plots'])
    elif args.plots:
        selected = list(args.plots)
    else:
        selected = all_plots
    unknown = set(selected) - set(all_plots)
    if unknown:
        raise ValueError(f'Plots are not in fixed splits: {sorted(unknown)}')
    if not Path(settings['checkpoint']).is_file():
        raise FileNotFoundError(
            f'Frozen checkpoint does not exist: {settings["checkpoint"]}')
    if not Path(settings['pipeline_template']).is_file():
        raise FileNotFoundError(
            f'Pipeline template does not exist: '
            f'{settings["pipeline_template"]}')

    # Fail before any expensive pipeline run if a source is missing or
    # ambiguous. This also catches accidental use of the wrong server dataset.
    for plot_name in selected:
        locate_forest(settings['source_root'], plot_name)

    Path(settings['output_root']).mkdir(parents=True, exist_ok=True)
    for plot_name in selected:
        split = (
            'train' if plot_name in split_info['train'] else 'validation')
        run_plot(
            plot_name, split, settings, force=args.force)

    rows = load_manifest_rows(settings['output_root'])
    manifest_path = write_manifest(rows, settings['output_root'])
    audit_path = write_audit_sample(
        rows,
        settings['output_root'],
        int(settings['gate']['min_random_audit_instances']),
    )
    summary = summarize(
        rows,
        settings,
        selected,
        pilot=args.pilot,
        manual_audit_confirmed=args.manual_audit_confirmed,
    )
    summary['manifest_path'] = str(manifest_path)
    summary['manual_audit_sample_path'] = (
        str(audit_path) if audit_path else None)
    summary_paths = write_summary(summary, settings['output_root'])
    print(Path(summary_paths[1]).read_text(encoding='utf-8'), flush=True)
    print(f'Saved manifest: {manifest_path}', flush=True)
    if not summary['gate']['passed']:
        if not args.pilot and all(
                value for name, value in summary['gate'].items()
                if name not in {'passed', 'manual_audit_confirmed'}):
            print(
                'DATA GATE PASS; manually inspect the fixed 50-instance '
                'audit sample before confirming E2.',
                flush=True)
        else:
            raise RuntimeError('E2 generation gate failed.')
    else:
        print(
            'PASS: E2 data are ready for the next experiment stage.',
            flush=True)


if __name__ == '__main__':
    main()
