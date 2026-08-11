"""Generate fixed train/validation Q4b1 three-head learning artifacts."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_SUFFIXES = {'.las', '.laz', '.npy', '.npz', '.txt'}


def plot_group(plot_name):
    match = re.fullmatch(r'(A1|G[1-4]|L[12]|O1)[NW]', plot_name)
    return match.group(1) if match else plot_name


def validate_splits(splits):
    if set(splits) != {'train', 'validation'}:
        raise ValueError('splits must contain train and validation.')
    train = list(splits['train'])
    validation = list(splits['validation'])
    if len(train) != len(set(train)) or len(validation) != len(
            set(validation)):
        raise ValueError('Duplicate plots are not allowed.')
    if set(train) & set(validation):
        raise ValueError('Plots leak across train and validation.')
    train_groups = {plot_group(name) for name in train}
    validation_groups = {plot_group(name) for name in validation}
    if train_groups & validation_groups:
        raise ValueError('Paired plot groups leak across splits.')
    forbidden = [name for name in train + validation if 'wytham' in name.lower()]
    if forbidden:
        raise ValueError(f'Wytham is forbidden in Q4b1: {forbidden}.')
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
        directory / 'instance_split_learning.npz',
        directory / 'metadata.json',
    )


def q3_undersegmented_ids(q3_root, plot_name):
    path = Path(q3_root) / 'validation' / plot_name / 'gt_trees.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing fixed Q3 artifact: {path}')
    with path.open(newline='', encoding='utf-8') as file:
        return sorted(
            int(row['gt_tree_id']) for row in csv.DictReader(file)
            if row['category'] == 'undersegmentation')


def safe_cleanup(path, root):
    target = Path(path).resolve()
    allowed_root = Path(root).resolve()
    if target == allowed_root or allowed_root not in target.parents:
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
    config.split_learning_output_dir = str(output_dir)
    config.split_learning_split = str(split)
    if split == 'validation':
        config.split_learning.expected_undersegmented_gt_ids = (
            q3_undersegmented_ids(settings['q3_reference_root'], plot_name))
    else:
        config.split_learning.expected_undersegmented_gt_ids = None
    path = (
        Path(settings['output_root']) / 'runtime_configs' /
        f'{plot_name}.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            munch_to_dict(config), file, sort_keys=False,
            allow_unicode=True)
    return path


def validate_artifact(npz_path, metadata_path):
    from tree_learn.util.instance_split_learning import (
        validate_instance_split_learning_artifact,
    )
    return validate_instance_split_learning_artifact(
        npz_path, metadata_path)


def run_plot(settings, split, plot_name, force=False):
    npz_path, metadata_path = artifact_paths(
        settings['output_root'], split, plot_name)
    if npz_path.is_file() and metadata_path.is_file() and not force:
        try:
            metadata = validate_artifact(npz_path, metadata_path)
            if metadata['source_plot'] != plot_name or metadata['split'] != split:
                raise ValueError(f'Artifact identity differs for {plot_name}.')
            if int(metadata.get('artifact_version', 0)) != 1:
                raise ValueError('Q4b1 artifact version is stale.')
            if str(metadata.get('checkpoint')) != str(settings['checkpoint']):
                raise ValueError('Q4b1 checkpoint differs from the config.')
            if split == 'validation' and metadata[
                    'undersegmented_gt_ids'] != q3_undersegmented_ids(
                        settings['q3_reference_root'], plot_name):
                raise ValueError('Q4b1 validation labels differ from Q3.')
            print(
                f'SKIP {plot_name}: validated existing artifact.',
                flush=True)
            return metadata
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            print(
                f'STALE {plot_name}: {error} Rebuilding artifact.',
                flush=True)

    forest_path = locate_forest(settings['source_root'], plot_name)
    runtime_root = Path(settings['output_root']) / 'runtime'
    runtime_dir = runtime_root / plot_name
    if force:
        safe_cleanup(runtime_dir, runtime_root)
    output_dir = npz_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_resolved_config(
        settings, split, plot_name, forest_path, runtime_dir, output_dir)
    log_path = Path(settings['output_root']) / 'logs' / f'{plot_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-u',
        'tools/data_gen/run_instance_split_learning_pipeline.py',
        '--config', str(config_path),
    ]
    print(f'RUN {plot_name} ({split}); log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Q4b1 pipeline failed for {plot_name}; inspect {log_path}.')
    metadata = validate_artifact(npz_path, metadata_path)
    if settings.get('cleanup_intermediates', True):
        safe_cleanup(runtime_dir, runtime_root)
    print(
        f'DONE {plot_name}: {metadata["num_instances"]} instances, '
        f'{metadata["num_candidate_parents"]} candidates, '
        f'{metadata["num_proposals"]} proposals.', flush=True)
    return metadata


def write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'source_plot', 'split', 'num_instances', 'num_candidate_parents',
        'num_undersegmented_gt_trees', 'num_known_k_safe_parents',
        'num_proposals', 'num_safe_proposals',
        'num_child_count_overflow',
    ]
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def summarize(settings, split_info, rows, mode):
    by_split = {
        split: [row for row in rows if row['split'] == split]
        for split in ('train', 'validation')}
    totals = {
        split: {
            key: int(sum(int(row[key]) for row in values))
            for key in (
                'num_instances', 'num_candidate_parents',
                'num_undersegmented_gt_trees',
                'num_known_k_safe_parents', 'num_proposals',
                'num_safe_proposals', 'num_child_count_overflow')}
        for split, values in by_split.items()}
    expected = (
        set(settings['pilot_plots']) if mode == 'pilot' else
        set(split_info['train'] + split_info['validation']))
    observed = {row['source_plot'] for row in rows}
    if mode == 'pilot':
        gate = {
            'all_pilot_plots_present': observed == expected,
            'all_features_validated': True,
            'no_child_count_overflow': sum(
                item['num_child_count_overflow']
                for item in totals.values()) == 0,
        }
    else:
        rule = settings['gate']
        validation = totals['validation']
        gate = {
            'all_fixed_plots_present': observed == expected,
            'enough_train_candidate_parents': (
                totals['train']['num_candidate_parents'] >=
                int(rule['min_train_candidate_parents'])),
            'validation_candidate_count_reproduced': (
                validation['num_candidate_parents'] ==
                int(rule['expected_validation_candidate_parents'])),
            'validation_undersegmentation_reproduced': (
                validation['num_undersegmented_gt_trees'] ==
                int(rule['expected_validation_undersegmented_gt_trees'])),
            'validation_known_k_safety_reproduced': (
                validation['num_known_k_safe_parents'] ==
                int(rule['expected_validation_known_k_safe_parents'])),
            'no_child_count_overflow': sum(
                item['num_child_count_overflow']
                for item in totals.values()) == 0,
            'no_plot_leakage': not (
                set(split_info['train']) & set(split_info['validation'])),
            'no_group_leakage': not (
                set(split_info['train_groups']) &
                set(split_info['validation_groups'])),
            'all_features_validated': True,
        }
    gate['passed'] = all(gate.values())
    summary = {'mode': mode, 'totals': totals, 'gate': gate, 'rows': rows}
    output_root = Path(settings['output_root'])
    name = 'pilot_summary' if mode == 'pilot' else 'generation_summary'
    (output_root / f'{name}.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        f'# Q4b1 三头拆分数据生成：{mode}', '',
        '- 数据：固定 train/validation forests；未读取 Wytham。',
        '- 特征、标签和 Raw-XY proposal 均来自同一次 TreeLearn 运行。', '',
        '| Split | Instances | Candidates | Underseg GT | Known-K safe | Proposals | Safe proposals |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split in ('train', 'validation'):
        item = totals[split]
        lines.append(
            f'| {split} | {item["num_instances"]:,} | '
            f'{item["num_candidate_parents"]:,} | '
            f'{item["num_undersegmented_gt_trees"]:,} | '
            f'{item["num_known_k_safe_parents"]:,} | '
            f'{item["num_proposals"]:,} | '
            f'{item["num_safe_proposals"]:,} |')
    lines.extend(['', '## Gate', ''])
    lines.extend(
        f'- {key}: **{value}**' for key, value in gate.items())
    lines.extend(['', 'PASS' if gate['passed'] else 'STOP'])
    (output_root / f'{name}.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return summary


def load_settings(config_path):
    import yaml
    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    text = json.dumps(settings).lower()
    if 'wytham' in text:
        raise ValueError('Q4b1 generation must not reference Wytham.')
    return settings


def run(config_path, pilot=False, force=False):
    settings = load_settings(config_path)
    split_info = validate_splits(settings['splits'])
    selected = (
        set(settings['pilot_plots']) if pilot else
        set(split_info['train'] + split_info['validation']))
    rows = []
    ordered = split_info['train'] + split_info['validation']
    split_by_plot = {
        plot: split for split in ('train', 'validation')
        for plot in split_info[split]}
    for index, plot_name in enumerate(
            [name for name in ordered if name in selected], start=1):
        print(f'[{index}/{len(selected)}] {plot_name}', flush=True)
        rows.append(run_plot(
            settings, split_by_plot[plot_name], plot_name, force=force))
    output_root = Path(settings['output_root'])
    write_manifest(rows, output_root / (
        'pilot_manifest.csv' if pilot else 'manifest.csv'))
    summary = summarize(
        settings, split_info, rows, 'pilot' if pilot else 'full')
    if not summary['gate']['passed']:
        raise RuntimeError('Q4b1 data gate failed; do not train the heads.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate fixed Q4b1 split-learning artifacts.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.config, pilot=args.pilot, force=args.force)


if __name__ == '__main__':
    main()