"""Generate fixed Q5b1 train/validation proposal-learning artifacts."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tree_learn.util.seed_completion_learning import (
    ARTIFACT_VERSION,
    PROPOSAL_FEATURE_NAMES,
    METADATA_NAME,
    PROPOSAL_CSV_NAME,
    save_learning_artifact,
    validate_learning_artifact,
    validate_proposal_rows,
)


SUPPORTED_SUFFIXES = {'.las', '.laz', '.npy', '.npz', '.txt'}


def plot_group(plot_name):
    match = re.fullmatch(r'(A1|G[1-4]|L[12]|O1)[NW]', plot_name)
    return match.group(1) if match else plot_name


def validate_splits(splits):
    if set(splits) != {'train', 'validation'}:
        raise ValueError('Q5b1 splits must contain train and validation.')
    train = list(splits['train'])
    validation = list(splits['validation'])
    if len(train) != len(set(train)) or len(validation) != len(
            set(validation)):
        raise ValueError('Duplicate Q5b1 plots are not allowed.')
    if set(train) & set(validation):
        raise ValueError('Q5b1 plots leak across train and validation.')
    train_groups = {plot_group(name) for name in train}
    validation_groups = {plot_group(name) for name in validation}
    if train_groups & validation_groups:
        raise ValueError('Q5b1 paired plot groups leak across splits.')
    if any('wytham' in name.lower() for name in train + validation):
        raise ValueError('Wytham is forbidden in Q5b1.')
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
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and
        path.stem == plot_name)
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected one forest for {plot_name}, found {matches}.')
    return matches[0]


def artifact_paths(output_root, split, plot_name):
    directory = Path(output_root) / split / plot_name
    return (
        directory / PROPOSAL_CSV_NAME,
        directory / METADATA_NAME,
    )


def safe_cleanup(path, root):
    target = Path(path).resolve()
    allowed = Path(root).resolve()
    if target == allowed or allowed not in target.parents:
        raise ValueError(f'Refusing unsafe cleanup: {target}')
    if target.exists():
        shutil.rmtree(target)


def write_resolved_config(
        settings, split, plot_name, forest_path, runtime_dir, output_dir):
    import yaml
    from tree_learn.util import get_config, munch_to_dict

    config = get_config(str(settings['pipeline_template']))
    config.forest_path = str(forest_path)
    config.pipeline_base_dir = str(runtime_dir)
    config.pretrain = str(settings['checkpoint'])
    config.seed_completion_learning_output_dir = str(output_dir)
    config.seed_completion_learning_split = str(split)
    diagnostic = dict(settings['proposal'])
    diagnostic['expected_target_tree_ids'] = None
    config.seed_completion_learning = diagnostic
    path = (
        Path(settings['output_root']) / 'runtime_configs' /
        f'{plot_name}.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            munch_to_dict(config), file, sort_keys=False,
            allow_unicode=True)
    return path


def import_validation_artifact(settings, plot_name, force=False):
    destination_csv, destination_metadata = artifact_paths(
        settings['output_root'], 'validation', plot_name)
    if destination_csv.is_file() and destination_metadata.is_file() and not force:
        metadata, _, _ = validate_learning_artifact(
            destination_csv, destination_metadata)
        if metadata['source_plot'] != plot_name or metadata[
                'split'] != 'validation':
            raise ValueError('Imported Q5b1 validation identity differs.')
        if str(metadata.get('checkpoint')) != str(settings['checkpoint']):
            raise ValueError('Imported Q5b1 checkpoint differs.')
        expected = {
            'scales': list(map(float, settings['proposal']['scales'])),
            'shift_fractions': list(map(
                float, settings['proposal']['shift_fractions'])),
            'tau_min': 50,
        }
        if metadata.get('proposal_parameters') != expected:
            raise ValueError('Imported Q5b1 proposal grid differs.')
        print(
            f'SKIP {plot_name}: validated imported Q5b0 artifact.',
            flush=True)
        return metadata

    root = Path(settings['q5b0_reference_root'])
    summary_path = root / 'summary.json'
    source_dir = root / 'validation' / plot_name
    source_csv = source_dir / 'proposals.csv'
    source_summary = source_dir / 'summary.json'
    if not summary_path.is_file() or not source_csv.is_file() or not (
            source_summary.is_file()):
        raise FileNotFoundError(
            f'Missing passed Q5b0 artifact for {plot_name}.')
    with summary_path.open(encoding='utf-8') as file:
        aggregate = json.load(file)
    if not aggregate['gate']['passed']:
        raise ValueError('Q5b0 reference gate did not pass.')
    with source_summary.open(encoding='utf-8') as file:
        report = json.load(file)
    with source_csv.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    arrays = validate_proposal_rows(rows)
    target_ids = sorted(
        int(row['gt_tree_id']) for row in report['per_target'])
    activated_ids = sorted(map(
        int, report['oracle_metrics'].get(
            'activated_proposal_ids',
            arrays['proposal_ids'][arrays['activation_target']].tolist())))
    observed_activated = sorted(map(
        int, arrays['proposal_ids'][arrays['activation_target']]))
    if activated_ids != observed_activated:
        raise ValueError(
            f'Q5b0 activation IDs differ for {plot_name}.')
    metadata = {
        'artifact_version': ARTIFACT_VERSION,
        'source_plot': plot_name,
        'split': 'validation',
        'feature_names': list(PROPOSAL_FEATURE_NAMES),
        'num_proposals': int(len(rows)),
        'num_activation_positive': int(
            arrays['activation_target'].sum()),
        'num_coverage_positive': int(arrays['coverage_target'].sum()),
        'num_target_trees': int(len(target_ids)),
        'num_proposal_covered_target_trees': int(sum(
            bool(row['proposal_covered']) for row in report['per_target'])),
        'target_tree_ids': target_ids,
        'activated_proposal_ids': observed_activated,
        'baseline': report['baseline'],
        'proposal_parameters': report['proposal_parameters'],
        'checkpoint': str(settings['checkpoint']),
        'source_artifact': str(source_csv),
        'imported_from_passed_q5b0': True,
    }
    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    save_learning_artifact(
        {'rows': rows, 'metadata': metadata}, destination_csv.parent)
    metadata, _, _ = validate_learning_artifact(
        destination_csv, destination_metadata)
    print(
        f'IMPORTED {plot_name}: {metadata["num_proposals"]:,} proposals, '
        f'{metadata["num_activation_positive"]} activated.', flush=True)
    return metadata


def run_train_plot(settings, plot_name, force=False):
    csv_path, metadata_path = artifact_paths(
        settings['output_root'], 'train', plot_name)
    if csv_path.is_file() and metadata_path.is_file() and not force:
        metadata, _, _ = validate_learning_artifact(csv_path, metadata_path)
        if metadata['source_plot'] != plot_name or metadata[
                'split'] != 'train':
            raise ValueError(f'Q5b1 artifact identity differs for {plot_name}.')
        if str(metadata.get('checkpoint')) != str(settings['checkpoint']):
            raise ValueError(f'Q5b1 checkpoint differs for {plot_name}.')
        expected = {
            'scales': list(map(float, settings['proposal']['scales'])),
            'shift_fractions': list(map(
                float, settings['proposal']['shift_fractions'])),
            'tau_min': 50,
        }
        if metadata.get('proposal_parameters') != expected:
            raise ValueError(f'Q5b1 proposal grid differs for {plot_name}.')
        print(
            f'SKIP {plot_name}: validated existing Q5b1 artifact.',
            flush=True)
        return metadata

    forest = locate_forest(settings['source_root'], plot_name)
    runtime_root = Path(settings['output_root']) / 'runtime'
    runtime_dir = runtime_root / plot_name
    output_dir = csv_path.parent
    if force:
        safe_cleanup(output_dir, settings['output_root'])
        safe_cleanup(runtime_dir, runtime_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_resolved_config(
        settings, 'train', plot_name, forest, runtime_dir, output_dir)
    log_path = Path(settings['output_root']) / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-u',
        'tools/data_gen/run_seed_completion_learning_pipeline.py',
        '--config', str(config_path)]
    print(f'RUN {plot_name} (train); log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q5b1 pipeline failed for {plot_name}; inspect {log_path}.')
    metadata, _, _ = validate_learning_artifact(csv_path, metadata_path)
    if settings.get('cleanup_intermediates', True):
        safe_cleanup(runtime_dir, runtime_root)
    print(
        f'DONE {plot_name}: {metadata["num_proposals"]:,} proposals, '
        f'{metadata["num_activation_positive"]} activated, '
        f'{metadata["num_target_trees"]} target trees.', flush=True)
    return metadata


def write_manifest(rows, path):
    fields = [
        'source_plot', 'split', 'num_proposals',
        'num_activation_positive', 'num_coverage_positive',
        'num_target_trees', 'num_proposal_covered_target_trees',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in rows)


def summarize(settings, split_info, rows, mode):
    by_split = {
        split: [row for row in rows if row['split'] == split]
        for split in ('train', 'validation')}
    totals = {
        split: {
            key: int(sum(int(row[key]) for row in values))
            for key in (
                'num_proposals', 'num_activation_positive',
                'num_coverage_positive', 'num_target_trees',
                'num_proposal_covered_target_trees')}
        for split, values in by_split.items()}
    expected = (
        set(settings['pilot_plots']) if mode == 'pilot' else
        set(split_info['train'] + split_info['validation']))
    observed = {row['source_plot'] for row in rows}
    if mode == 'pilot':
        gate = {
            'all_pilot_plots_present': observed == expected,
            'both_splits_present': {
                row['split'] for row in rows} == {'train', 'validation'},
            'all_pilot_artifacts_nonempty': all(
                int(row['num_proposals']) > 0 for row in rows),
            'activation_positive_present': sum(
                int(row['num_activation_positive']) for row in rows) > 0,
            'all_features_finite': True,
        }
    else:
        rule = settings['gate']
        train = totals['train']
        validation = totals['validation']
        gate = {
            'all_fixed_plots_present': observed == expected,
            'enough_train_proposals': (
                train['num_proposals'] >=
                int(rule['min_train_proposals'])),
            'enough_train_activation_positive': (
                train['num_activation_positive'] >=
                int(rule['min_train_activation_positive'])),
            'enough_train_coverage_positive': (
                train['num_coverage_positive'] >=
                int(rule['min_train_coverage_positive'])),
            'enough_train_target_trees': (
                train['num_target_trees'] >=
                int(rule['min_train_target_trees'])),
            'validation_proposals_reproduced': (
                validation['num_proposals'] ==
                int(rule['expected_validation_proposals'])),
            'validation_activation_reproduced': (
                validation['num_activation_positive'] ==
                int(rule['expected_validation_activation_positive'])),
            'validation_targets_reproduced': (
                validation['num_target_trees'] ==
                int(rule['expected_validation_target_trees'])),
            'validation_coverage_reproduced': (
                validation['num_proposal_covered_target_trees'] ==
                int(rule['expected_validation_covered_targets'])),
            'no_plot_leakage': not (
                set(split_info['train']) & set(split_info['validation'])),
            'no_group_leakage': not (
                set(split_info['train_groups']) &
                set(split_info['validation_groups'])),
            'all_features_finite': True,
        }
    gate['passed'] = all(gate.values())
    result = {
        'mode': mode,
        'totals': totals,
        'rows': rows,
        'gate': gate,
    }
    output = Path(settings['output_root'])
    name = 'pilot_summary' if mode == 'pilot' else 'generation_summary'
    (output / f'{name}.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8')
    lines = [
        f'# Q5b1 Seed-Completion data generation: {mode}', '',
        '- Uses only the fixed train/validation forests; Wytham is forbidden.',
        '- Validation data are imported from the passed Q5b0 proposal artifacts.',
        '- Train proposals are generated from frozen TreeLearn predictions.', '',
        '| Split | Proposals | Activated | Coverage positive | Target trees | Covered targets |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ('train', 'validation'):
        row = totals[split]
        lines.append(
            f'| {split} | {row["num_proposals"]:,} | '
            f'{row["num_activation_positive"]:,} | '
            f'{row["num_coverage_positive"]:,} | '
            f'{row["num_target_trees"]:,} | '
            f'{row["num_proposal_covered_target_trees"]:,} |')
    lines.extend(['', '## Gate', ''])
    lines.extend(
        f'- {name}: **{passed}**' for name, passed in gate.items())
    lines.extend(['', 'PASS' if gate['passed'] else 'STOP'])
    (output / f'{name}.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result


def load_settings(config_path):
    import yaml
    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    if 'wytham' in json.dumps(settings).lower():
        raise ValueError('Q5b1 generation must not reference Wytham.')
    return settings


def run(config_path, pilot=False, force=False):
    settings = load_settings(config_path)
    split_info = validate_splits(settings['splits'])
    selected = (
        set(settings['pilot_plots']) if pilot else
        set(split_info['train'] + split_info['validation']))
    rows = []
    for plot in split_info['train']:
        if plot in selected:
            rows.append(run_train_plot(settings, plot, force=force))
    for plot in split_info['validation']:
        if plot in selected:
            rows.append(import_validation_artifact(
                settings, plot, force=force))
    output = Path(settings['output_root'])
    write_manifest(
        rows, output / (
            'pilot_manifest.csv' if pilot else 'manifest.csv'))
    result = summarize(
        settings, split_info, rows, 'pilot' if pilot else 'full')
    if not result['gate']['passed']:
        raise RuntimeError(
            'Q5b1 data gate failed; do not train proposal activation heads.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate fixed Q5b1 proposal-learning data.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, pilot=args.pilot, force=args.force)


if __name__ == '__main__':
    main()
