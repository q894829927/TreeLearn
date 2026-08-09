"""Generate and audit frozen-backbone candidate-seed training artifacts."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from gen_instance_quality_data import (  # noqa: E402
    locate_forest,
    plot_group,
    safe_cleanup_runtime,
    validate_splits,
)


def artifact_destinations(output_root, split, plot_name):
    directory = Path(output_root) / split
    return {
        'npz': directory / f'{plot_name}.npz',
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
    config.seed_quality_split = str(split)
    config.pretrain = str(checkpoint)
    payload = munch_to_dict(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            payload, file, sort_keys=False, allow_unicode=True)
    return payload


def validate_seed_artifact(npz_path, metadata_path):
    required = {
        'candidate_indices',
        'coords',
        'base_votes_xy',
        'backbone_features',
        'scalar_features',
        'target_valid',
        'target_is_tree',
        'target_tree_id',
        'target_vote_error_xy',
        'target_vote_cell_purity',
        'target_utility',
        'target_reliable',
        'target_cell_representative',
        'target_tree_quota',
        'target_coverage_critical',
    }
    with np.load(npz_path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f'{npz_path} misses arrays: {sorted(missing)}')
        count = len(data['candidate_indices'])
        if count == 0:
            raise ValueError(f'{npz_path} contains no candidate seeds.')
        for name in required - {'candidate_indices'}:
            if len(data[name]) != count:
                raise ValueError(
                    f'{name} has {len(data[name])} rows, expected {count}.')
        if data['coords'].shape != (count, 3):
            raise ValueError('coords must have shape [N, 3].')
        if data['base_votes_xy'].shape != (count, 2):
            raise ValueError('base_votes_xy must have shape [N, 2].')
        vote_target_names = {
            'target_base_vote_xy', 'target_vote_residual_xy'}
        present_vote_targets = vote_target_names & set(data.files)
        if present_vote_targets and present_vote_targets != vote_target_names:
            raise ValueError(
                'Vector vote targets must be present as a complete pair.')
        if present_vote_targets:
            for name in vote_target_names:
                if data[name].shape != (count, 2):
                    raise ValueError(f'{name} must have shape [N, 2].')
                if not np.isfinite(data[name]).all():
                    raise ValueError(f'{name} contains non-finite values.')
            tree_targets = data['target_is_tree'].astype(bool)
            if not np.allclose(
                    data['base_votes_xy'][tree_targets] +
                    data['target_vote_residual_xy'][tree_targets],
                    data['target_base_vote_xy'][tree_targets], atol=1e-5):
                raise ValueError('Vector vote targets are inconsistent.')
        if data['backbone_features'].ndim != 2:
            raise ValueError('backbone_features must be two-dimensional.')
        if data['scalar_features'].ndim != 2:
            raise ValueError('scalar_features must be two-dimensional.')
        if len(np.unique(data['candidate_indices'])) != count:
            raise ValueError('candidate_indices must be unique.')
        if np.any(np.diff(data['candidate_indices']) <= 0):
            raise ValueError('candidate_indices must be strictly increasing.')
        finite_arrays = [
            'coords', 'base_votes_xy', 'backbone_features',
            'scalar_features', 'target_vote_error_xy',
            'target_vote_cell_purity', 'target_utility',
        ]
        if any(not np.isfinite(data[name]).all() for name in finite_arrays):
            raise ValueError(f'{npz_path} contains non-finite values.')
        utility = data['target_utility']
        purity = data['target_vote_cell_purity']
        if np.any((utility < 0) | (utility > 1)):
            raise ValueError('target_utility must be in [0, 1].')
        if np.any((purity < 0) | (purity > 1)):
            raise ValueError('target_vote_cell_purity must be in [0, 1].')
        tree = data['target_is_tree'].astype(bool)
        critical = data['target_coverage_critical'].astype(bool)
        if np.any(critical & ~tree):
            raise ValueError('Only supervised tree seeds may be critical.')
        tree_ids = set(np.unique(data['target_tree_id'][tree]).tolist())
        critical_tree_ids = set(np.unique(
            data['target_tree_id'][critical]).tolist())
        all_trees_covered = tree_ids <= critical_tree_ids
        observed = {
            'num_candidates': count,
            'num_valid_candidates': int(data['target_valid'].sum()),
            'num_tree_candidates': int(tree.sum()),
            'num_non_tree_candidates': int(np.count_nonzero(
                data['target_valid'] & ~tree)),
            'num_unknown_candidates': int(np.count_nonzero(
                ~data['target_valid'])),
            'num_reliable_candidates': int(
                data['target_reliable'].sum()),
            'num_critical_candidates': int(critical.sum()),
            'num_supervised_trees': len(tree_ids),
            'num_critical_trees': len(critical_tree_ids),
            'backbone_dim': int(data['backbone_features'].shape[1]),
            'scalar_dim': int(data['scalar_features'].shape[1]),
            'all_supervised_trees_covered': bool(all_trees_covered),
        }

    with Path(metadata_path).open(encoding='utf-8') as file:
        metadata = json.load(file)
    for name, value in observed.items():
        if name == 'all_supervised_trees_covered':
            continue
        if int(metadata[name]) != int(value):
            raise ValueError(
                f'Metadata {name}={metadata[name]} != observed {value}.')
    if len(metadata['scalar_feature_names']) != observed['scalar_dim']:
        raise ValueError('scalar_feature_names do not match scalar_dim.')
    metadata = dict(metadata)
    metadata.update(observed)
    return metadata


def run_plot(plot_name, split, settings, force=False):
    output_root = Path(settings['output_root'])
    runtime_root = output_root / 'runtime'
    runtime_directory = runtime_root / plot_name
    destinations = artifact_destinations(output_root, split, plot_name)
    if artifacts_complete(destinations) and not force:
        metadata = validate_seed_artifact(
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
        resolved_config_path)
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
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Pipeline failed for {plot_name} with exit code '
            f'{completed.returncode}. Inspect {log_path}.')

    source_directory = (
        runtime_directory /
        payload['save_cfg']['results_dir'] /
        'seed_quality')
    sources = {
        'npz': source_directory / 'seed_candidates.npz',
        'metadata': source_directory / 'metadata.json',
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Pipeline completed but artifacts are missing: {missing}')
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, destinations[name])
    metadata = validate_seed_artifact(
        destinations['npz'], destinations['metadata'])
    if settings.get('cleanup_intermediates', True):
        safe_cleanup_runtime(runtime_directory, runtime_root)
    print(
        f'DONE {plot_name}: {metadata["num_candidates"]:,} candidates, '
        f'{metadata["num_critical_candidates"]:,} critical, '
        f'{metadata["num_non_tree_candidates"]:,} non-tree.',
        flush=True)
    return metadata


def load_manifest_rows(output_root):
    rows = []
    root = Path(output_root)
    for split in ('train', 'validation'):
        for metadata_path in sorted((root / split).glob('*_metadata.json')):
            plot_name = metadata_path.name[:-len('_metadata.json')]
            npz_path = root / split / f'{plot_name}.npz'
            if not npz_path.is_file():
                raise FileNotFoundError(
                    f'Missing seed artifact for {plot_name}: {npz_path}')
            metadata = validate_seed_artifact(npz_path, metadata_path)
            if metadata['source_plot'] != plot_name:
                raise ValueError(
                    f'{metadata_path} source_plot does not match filename.')
            if metadata['split'] != split:
                raise ValueError(
                    f'{metadata_path} split does not match directory.')
            candidates = int(metadata['num_candidates'])
            row = {
                'source_plot': plot_name,
                'source_group': plot_group(plot_name),
                'split': split,
                'artifact_path': str(npz_path),
                'num_candidates': candidates,
                'num_valid_candidates': int(
                    metadata['num_valid_candidates']),
                'num_tree_candidates': int(
                    metadata['num_tree_candidates']),
                'num_non_tree_candidates': int(
                    metadata['num_non_tree_candidates']),
                'num_unknown_candidates': int(
                    metadata['num_unknown_candidates']),
                'num_reliable_candidates': int(
                    metadata['num_reliable_candidates']),
                'num_critical_candidates': int(
                    metadata['num_critical_candidates']),
                'num_supervised_trees': int(
                    metadata['num_supervised_trees']),
                'num_critical_trees': int(
                    metadata['num_critical_trees']),
                'all_supervised_trees_covered': bool(
                    metadata['all_supervised_trees_covered']),
                'backbone_dim': int(metadata['backbone_dim']),
                'scalar_dim': int(metadata['scalar_dim']),
                'known_label_rate': (
                    int(metadata['num_valid_candidates']) /
                    max(candidates, 1)),
                'critical_rate': (
                    int(metadata['num_critical_candidates']) /
                    max(int(metadata['num_tree_candidates']), 1)),
            }
            rows.append(row)
    return rows


def write_manifest(rows, output_root):
    if not rows:
        raise ValueError('No seed artifacts were found.')
    path = Path(output_root) / 'manifest.csv'
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _audit_record(row, data, index, stratum):
    scalar = data['scalar_features'][index]
    return {
        'source_plot': row['source_plot'],
        'split': row['split'],
        'row_index': int(index),
        'candidate_index': int(data['candidate_indices'][index]),
        'stratum': stratum,
        'target_tree_id': int(data['target_tree_id'][index]),
        'target_is_tree': bool(data['target_is_tree'][index]),
        'target_reliable': bool(data['target_reliable'][index]),
        'target_coverage_critical': bool(
            data['target_coverage_critical'][index]),
        'target_utility': float(data['target_utility'][index]),
        'target_vote_error_xy': float(
            data['target_vote_error_xy'][index]),
        'tree_probability': float(scalar[0]),
        'verticality': float(scalar[1]),
    }


def write_audit_sample(rows, output_root, sample_size=50, seed=42):
    generator = np.random.default_rng(int(seed))
    pools = {
        'critical': [],
        'non_tree': [],
        'reliable': [],
        'unreliable_tree': [],
        'random': [],
    }
    for row in rows:
        with np.load(row['artifact_path'], allow_pickle=False) as data:
            tree = data['target_is_tree'].astype(bool)
            masks = {
                'critical': data['target_coverage_critical'].astype(bool),
                'non_tree': data['target_valid'].astype(bool) & ~tree,
                'reliable': data['target_reliable'].astype(bool),
                'unreliable_tree': (
                    tree & ~data['target_reliable'].astype(bool)),
                'random': np.ones(len(tree), dtype=bool),
            }
            for name, mask in masks.items():
                indices = np.flatnonzero(mask)
                if len(indices) == 0:
                    continue
                sample_count = min(
                    int(sample_size) if name == 'random' else 6,
                    len(indices))
                chosen = generator.choice(
                    indices, size=sample_count, replace=False)
                pools[name].extend(
                    _audit_record(row, data, int(index), name)
                    for index in np.atleast_1d(chosen))

    selected = []
    seen = set()
    quota = max(1, int(sample_size) // 4)
    for name in ('critical', 'non_tree', 'reliable', 'unreliable_tree'):
        pool = pools[name]
        if len(pool) > quota:
            chosen = generator.choice(len(pool), size=quota, replace=False)
            pool = [pool[int(index)] for index in chosen]
        for record in pool:
            key = record['source_plot'], record['row_index']
            if key not in seen:
                selected.append(record)
                seen.add(key)
    random_pool = list(pools['random'])
    generator.shuffle(random_pool)
    for record in random_pool:
        if len(selected) >= int(sample_size):
            break
        key = record['source_plot'], record['row_index']
        if key not in seen:
            selected.append(record)
            seen.add(key)
    selected = selected[:int(sample_size)]
    if not selected:
        return None, 0
    path = Path(output_root) / 'audit_sample.csv'
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return path, len(selected)


def summarize(rows, settings, selected_plots, pilot, audit_size):
    train_rows = [row for row in rows if row['split'] == 'train']
    validation_rows = [
        row for row in rows if row['split'] == 'validation']
    observed_plots = {row['source_plot'] for row in rows}
    train_plots = {row['source_plot'] for row in train_rows}
    validation_plots = {
        row['source_plot'] for row in validation_rows}
    train_groups = {row['source_group'] for row in train_rows}
    validation_groups = {
        row['source_group'] for row in validation_rows}
    total_candidates = sum(row['num_candidates'] for row in rows)
    total_valid = sum(row['num_valid_candidates'] for row in rows)
    total_tree = sum(row['num_tree_candidates'] for row in rows)
    total_critical = sum(
        row['num_critical_candidates'] for row in rows)
    totals = {
        'train_candidates': sum(
            row['num_candidates'] for row in train_rows),
        'validation_candidates': sum(
            row['num_candidates'] for row in validation_rows),
        'train_critical_candidates': sum(
            row['num_critical_candidates'] for row in train_rows),
        'train_non_tree_candidates': sum(
            row['num_non_tree_candidates'] for row in train_rows),
        'known_label_rate': total_valid / max(total_candidates, 1),
        'critical_rate': total_critical / max(total_tree, 1),
    }
    gate_settings = settings['gate']
    no_plot_leakage = not (train_plots & validation_plots)
    no_group_leakage = not (train_groups & validation_groups)
    dimensions_match = all(
        row['backbone_dim'] == int(
            gate_settings['expected_backbone_dim']) and
        row['scalar_dim'] == int(
            gate_settings['expected_scalar_dim'])
        for row in rows)
    all_trees_covered = all(
        row['all_supervised_trees_covered'] for row in rows)

    if pilot:
        gate = {
            'selected_plots_present': (
                set(selected_plots) <= observed_plots),
            'artifacts_nonempty': all(
                any(
                    row['source_plot'] == plot and
                    row['num_candidates'] > 0
                    for row in rows)
                for plot in selected_plots),
            'dimensions_match': dimensions_match,
            'all_supervised_trees_covered': all_trees_covered,
            'audit_sample_prepared': audit_size >= min(
                20, int(gate_settings['min_audit_candidates'])),
            'no_plot_leakage': no_plot_leakage,
            'no_group_leakage': no_group_leakage,
        }
    else:
        expected_plots = set(
            settings['splits']['train'] +
            settings['splits']['validation'])
        gate = {
            'all_fixed_plots_present': (
                observed_plots == expected_plots),
            'enough_train_candidates': (
                totals['train_candidates'] >= int(
                    gate_settings['min_train_candidates'])),
            'enough_validation_candidates': (
                totals['validation_candidates'] >= int(
                    gate_settings['min_validation_candidates'])),
            'enough_critical_targets': (
                totals['train_critical_candidates'] >= int(
                    gate_settings['min_train_critical_candidates'])),
            'enough_non_tree_targets': (
                totals['train_non_tree_candidates'] >= int(
                    gate_settings['min_train_non_tree_candidates'])),
            'known_label_rate_passed': (
                totals['known_label_rate'] >= float(
                    gate_settings['min_known_label_rate'])),
            'critical_rate_passed': (
                float(gate_settings['min_critical_rate']) <=
                totals['critical_rate'] <=
                float(gate_settings['max_critical_rate'])),
            'enough_independent_groups': (
                len(train_groups) >= int(
                    gate_settings['min_independent_groups'])),
            'dimensions_match': dimensions_match,
            'all_supervised_trees_covered': all_trees_covered,
            'audit_sample_prepared': (
                audit_size >= int(
                    gate_settings['min_audit_candidates'])),
            'no_plot_leakage': no_plot_leakage,
            'no_group_leakage': no_group_leakage,
        }
    gate['passed'] = bool(all(gate.values()))
    return {
        'mode': 'pilot' if pilot else 'full',
        'selected_plots': list(selected_plots),
        'num_artifacts': len(rows),
        'train_plots': sorted(train_plots),
        'validation_plots': sorted(validation_plots),
        'train_groups': sorted(train_groups),
        'validation_groups': sorted(validation_groups),
        'audit_sample_size': int(audit_size),
        **totals,
        'gate': gate,
    }


def write_summary(summary, output_root):
    root = Path(output_root)
    json_path = root / 'generation_summary.json'
    markdown_path = root / 'generation_summary.md'
    with json_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    gate_lines = '\n'.join(
        f'- {name}: **{value}**'
        for name, value in summary['gate'].items())
    markdown = (
        '# E1a seed reliability/coverage data audit\n\n'
        f'- mode: {summary["mode"]}\n'
        f'- artifacts: {summary["num_artifacts"]}\n'
        f'- train candidates: {summary["train_candidates"]:,}\n'
        f'- validation candidates: '
        f'{summary["validation_candidates"]:,}\n'
        f'- train critical targets: '
        f'{summary["train_critical_candidates"]:,}\n'
        f'- train non-tree targets: '
        f'{summary["train_non_tree_candidates"]:,}\n'
        f'- known-label rate: '
        f'{100 * summary["known_label_rate"]:.3f}%\n'
        f'- critical-target rate: '
        f'{100 * summary["critical_rate"]:.3f}%\n'
        f'- train plots: {", ".join(summary["train_plots"])}\n'
        f'- validation plots: '
        f'{", ".join(summary["validation_plots"])}\n\n'
        '## Gate\n\n'
        f'{gate_lines}\n')
    with markdown_path.open('w', encoding='utf-8') as file:
        file.write(markdown)
    return json_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate frozen TreeLearn candidate-seed data.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--plots', nargs='+')
    parser.add_argument(
        '--force', action='store_true',
        help='Explicitly replace selected plot artifacts.')
    return parser.parse_args()


def main():
    import yaml

    args = parse_args()
    with open(args.config, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    split_info = validate_splits(settings['splits'])
    all_plots = split_info['train'] + split_info['validation']
    if args.pilot:
        selected = list(settings['pilot_plots'])
    elif args.plots:
        selected = list(args.plots)
    else:
        selected = all_plots
    unknown = set(selected) - set(all_plots)
    if unknown:
        raise ValueError(
            f'Plots are not in fixed splits: {sorted(unknown)}')
    if not Path(settings['checkpoint']).is_file():
        raise FileNotFoundError(
            f'Frozen checkpoint does not exist: {settings["checkpoint"]}')
    if not Path(settings['pipeline_template']).is_file():
        raise FileNotFoundError(
            f'Pipeline template does not exist: '
            f'{settings["pipeline_template"]}')
    for plot_name in selected:
        locate_forest(settings['source_root'], plot_name)

    Path(settings['output_root']).mkdir(parents=True, exist_ok=True)
    for plot_name in selected:
        split = (
            'train' if plot_name in split_info['train']
            else 'validation')
        run_plot(plot_name, split, settings, force=args.force)

    rows = load_manifest_rows(settings['output_root'])
    manifest_path = write_manifest(rows, settings['output_root'])
    audit_path, audit_size = write_audit_sample(
        rows,
        settings['output_root'],
        sample_size=int(settings['gate']['min_audit_candidates']),
        seed=int(settings.get('audit_seed', 42)))
    summary = summarize(
        rows, settings, selected, args.pilot, audit_size)
    summary['manifest_path'] = str(manifest_path)
    summary['audit_sample_path'] = (
        str(audit_path) if audit_path else None)
    _, markdown_path = write_summary(
        summary, settings['output_root'])
    print(markdown_path.read_text(encoding='utf-8'), flush=True)
    print(f'Saved manifest: {manifest_path}', flush=True)
    print(f'Saved audit sample: {audit_path}', flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError('E1a seed-data gate failed.')
    print('PASS: E1a seed data are ready for E1b MLP.', flush=True)


if __name__ == '__main__':
    main()
