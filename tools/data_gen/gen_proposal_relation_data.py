"""Generate fixed train/validation proposal-relation graphs (P1)."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from tree_learn.util import get_config, munch_to_dict
from tree_learn.util.proposal_relation import (
    build_learning_targets, build_proposal_graph,
    build_vertical_micro_proposals, coordinate_sha256,
    load_inference_graph, save_inference_graph)


class CaptureComplete(RuntimeError):
    pass


def _pipeline_module():
    path = Path(__file__).resolve().parents[1] / 'pipeline' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location('proposal_relation_p1_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forest(root, plot):
    paths = [p for ext in ('.las', '.laz') for p in Path(root).glob(f'{plot}{ext}')]
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected exactly one forest for {plot}: {paths}')
    return paths[0]


def _remove(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Refusing unsafe cleanup: {path}')
    if path.exists():
        shutil.rmtree(path)


def _validate(directory, plot, split):
    graph, metadata = load_inference_graph(Path(directory) / 'graph.npz')
    with np.load(Path(directory) / 'learning.npz', allow_pickle=False) as data:
        required = {'node_gt', 'node_purity', 'edge_label', 'edge_valid',
                    'covered_gt_ids', 'point_gt', 'baseline_predictions'}
        if required - set(data.files):
            raise ValueError(f'Learning artifact misses {sorted(required-set(data.files))}.')
        if len(data['node_gt']) != len(graph['node_features']):
            raise ValueError('Node targets are not aligned.')
        if len(data['edge_label']) != graph['edge_index'].shape[1]:
            raise ValueError('Edge targets are not aligned.')
    if metadata.get('source_plot') != plot or metadata.get('split') != split:
        raise ValueError('Artifact source metadata differs.')
    return graph, metadata


def _worker(config_path):
    config = get_config(config_path)
    settings = config.proposal_relation_data
    output = Path(str(config.proposal_relation_output_dir))
    plot, split = str(settings.source_plot), str(settings.split)
    pipeline = _pipeline_module()
    original = np.savez_compressed
    completed = {'value': False}

    def intercept(file, *args, **kwargs):
        if Path(str(file)).name != 'pointwise_results.npz':
            return original(file, *args, **kwargs)
        required = {'coords', 'semantic_prediction_logits', 'instance_labels',
                    'instance_preds', 'input_feats'}
        if required - set(kwargs):
            raise ValueError(f'Pointwise payload misses {sorted(required-set(kwargs))}.')
        coords = np.asarray(kwargs['coords'])
        proposals = build_vertical_micro_proposals(
            coords, kwargs['semantic_prediction_logits'],
            verticality=np.asarray(kwargs['input_feats'])[:, -1],
            tree_probability_threshold=float(settings.tree_probability_threshold),
            xy_cell_size=float(settings.xy_cell_size),
            z_bin_size=float(settings.z_bin_size),
            vertical_split_gap=float(settings.vertical_split_gap),
            tree_class_index=int(settings.tree_class_index))
        graph = build_proposal_graph(
            proposals,
            max_xy_distance=float(settings.max_xy_distance),
            max_xy_bbox_gap=float(settings.max_xy_bbox_gap),
            max_vertical_gap=float(settings.max_vertical_gap),
            max_neighbors=int(settings.max_neighbors))
        targets = build_learning_targets(
            graph, kwargs['instance_labels'],
            minimum_node_purity=float(settings.minimum_node_purity))
        output.mkdir(parents=True, exist_ok=True)
        save_inference_graph(output / 'graph.npz', graph, {
            'source_plot': plot, 'split': split,
            'coordinate_sha256': coordinate_sha256(coords),
            'checkpoint': str(config.pretrain),
            'tree_probability_threshold': float(settings.tree_probability_threshold),
            'xy_cell_size': float(settings.xy_cell_size),
            'z_bin_size': float(settings.z_bin_size),
            'vertical_split_gap': float(settings.vertical_split_gap),
            'max_xy_distance': float(settings.max_xy_distance),
            'max_xy_bbox_gap': float(settings.max_xy_bbox_gap),
            'max_vertical_gap': float(settings.max_vertical_gap),
            'max_neighbors': int(settings.max_neighbors)})
        original(
            output / 'learning.npz',
            node_gt=targets['node_gt'], node_purity=targets['node_purity'],
            edge_label=targets['edge_label'], edge_valid=targets['edge_valid'],
            covered_gt_ids=targets['covered_gt_ids'],
            point_gt=np.asarray(kwargs['instance_labels'], np.int64),
            baseline_predictions=np.asarray(kwargs['instance_preds'], np.int64))
        summary = {
            'source_plot': plot, 'split': split,
            'num_points': int(len(coords)),
            'num_nodes': int(len(graph['node_features'])),
            'num_edges': int(graph['edge_index'].shape[1]),
            'positive_edges': int(np.sum(targets['edge_label'])),
            'hard_negative_edges': int(np.sum(targets['edge_valid'] & ~targets['edge_label'].astype(bool))),
            'covered_gt_trees': int(len(targets['covered_gt_ids'])),
            'all_features_finite': bool(
                np.all(np.isfinite(graph['node_features'])) and
                np.all(np.isfinite(graph['edge_features']))),
        }
        (output / 'summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        completed['value'] = True
        print(f'P1 saved {plot}: {summary["num_nodes"]:,} nodes, '
              f'{summary["num_edges"]:,} edges.', flush=True)
        raise CaptureComplete()

    np.savez_compressed = intercept
    try:
        pipeline.run_treelearn_pipeline(config, config_path)
    except CaptureComplete:
        pass
    finally:
        np.savez_compressed = original
    if not completed['value']:
        raise RuntimeError('Pipeline did not emit the intercepted pointwise artifact.')


def _runtime_config(settings, plot, split, destination):
    config = get_config(settings['pipeline_template'])
    train = get_config(settings['train_config'])
    runtime = Path(settings['output_root']) / 'runtime' / split / plot
    config.forest_path = str(_forest(settings['source_root'], plot))
    config.pipeline_base_dir = str(runtime)
    config.pretrain = str(settings['checkpoint'])
    config.model = train.model
    config.proposal_relation_output_dir = str(destination)
    payload = dict(settings['proposal_relation'])
    payload.update({'source_plot': plot, 'split': split})
    config.proposal_relation_data = payload
    config.save_cfg.results_dir = f'results_p1_{split}_{plot}'
    config.save_cfg.save_pointwise = True
    config.save_cfg.save_full_forest = False
    config.save_cfg.save_formats = ['npz']
    config.save_cfg.return_type = 'voxelized_and_filtered'
    config.tile_generation = not (runtime / 'tiles' / 'npz').is_dir()
    path = Path(settings['output_root']) / 'runtime_configs' / split / f'{plot}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(
        munch_to_dict(config), sort_keys=False, allow_unicode=True), encoding='utf-8')
    return path, runtime


def _run_plot(settings, plot, split, force):
    if 'wytham' in plot.lower():
        raise ValueError('P1 is forbidden from reading Wytham.')
    destination = Path(settings['data_root']) / split / plot
    if destination.is_dir() and not force:
        try:
            _validate(destination, plot, split)
        except Exception as error:
            print(f'STALE {plot}: {error}; rebuilding.', flush=True)
            _remove(destination, settings['data_root'])
        else:
            print(f'SKIP {plot}: validated existing graph.', flush=True)
            return
    if force:
        _remove(destination, settings['data_root'])
    config, runtime = _runtime_config(settings, plot, split, destination)
    log = Path(settings['output_root']) / 'logs' / split / f'{plot}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'RUN {plot} ({split}); log={log}', flush=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(
            [sys.executable, '-u', __file__, '--worker-config', str(config)],
            stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'P1 failed for {plot}; inspect {log}.')
    _validate(destination, plot, split)
    _remove(runtime, Path(settings['output_root']) / 'runtime')


def _aggregate(settings):
    rows = []
    for split, plots in (('train', settings['train_plots']),
                         ('validation', settings['validation_plots'])):
        for plot in plots:
            directory = Path(settings['data_root']) / split / plot
            _validate(directory, plot, split)
            rows.append(json.loads(
                (directory / 'summary.json').read_text(encoding='utf-8')))
    train = [row for row in rows if row['split'] == 'train']
    valid = [row for row in rows if row['split'] == 'validation']
    gate_settings = settings['gate']
    gate = {
        'all_fixed_plots_present': len(rows) == len(settings['train_plots']) + len(settings['validation_plots']),
        'enough_train_positive_edges': sum(r['positive_edges'] for r in train) >= int(gate_settings['minimum_train_positive_edges']),
        'enough_train_hard_negatives': sum(r['hard_negative_edges'] for r in train) >= int(gate_settings['minimum_train_hard_negatives']),
        'all_supervised_trees_covered': all(r['covered_gt_trees'] > 0 for r in rows),
        'all_features_finite': all(r['all_features_finite'] for r in rows),
        'no_plot_leakage': not (set(settings['train_plots']) & set(settings['validation_plots'])),
    }
    gate['passed'] = all(gate.values())
    result = {'rows': rows, 'gate': gate}
    data_root = Path(settings['data_root'])
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / 'generation_summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# P1 proposal-relation graph data', '',
             '| Split | Graphs | Nodes | Edges | Positive | Hard negative |',
             '|---|---:|---:|---:|---:|---:|']
    for split, split_rows in (('train', train), ('validation', valid)):
        lines.append(f'| {split} | {len(split_rows)} | '
                     f'{sum(r["num_nodes"] for r in split_rows):,} | '
                     f'{sum(r["num_edges"] for r in split_rows):,} | '
                     f'{sum(r["positive_edges"] for r in split_rows):,} | '
                     f'{sum(r["hard_negative_edges"] for r in split_rows):,} |')
    lines += ['', '## Gate', ''] + [
        f'- {key}: **{value}**' for key, value in gate.items()]
    lines += ['', 'PASS' if gate['passed'] else 'STOP', '']
    (data_root / 'generation_summary.md').write_text(
        '\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    if not gate['passed']:
        raise RuntimeError('P1 data gate failed; do not train P2.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--worker-config')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.worker_config:
        _worker(args.worker_config)
        return
    if not args.config:
        parser.error('--config is required.')
    settings = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['proposal_relation_data']
    for split, plots in (('train', settings['train_plots']),
                         ('validation', settings['validation_plots'])):
        for index, plot in enumerate(plots, 1):
            print(f'[{index}/{len(plots)}] {plot}', flush=True)
            _run_plot(settings, plot, split, args.force)
    _aggregate(settings)


if __name__ == '__main__':
    main()
