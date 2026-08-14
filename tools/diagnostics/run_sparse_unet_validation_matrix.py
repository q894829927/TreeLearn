"""Run fixed five-forest pipeline/evaluation without reading Wytham."""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml
import torch

from tree_learn.util import get_config, munch_to_dict


METRICS = (
    'Completeness', 'Commission Error Rate', 'F1 Score',
    'Precision', 'Recall', 'Coverage',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def locate_forest(root, plot):
    matches = []
    for suffix in ('.laz', '.las'):
        matches.extend(Path(root).glob(f'{plot}{suffix}'))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected one forest for {plot}, found {matches}.')
    return matches[0]


def parse_metrics(text):
    result = {}
    for metric in METRICS:
        matches = re.findall(
            rf'{re.escape(metric)}:\s*([0-9.]+)%', text)
        if matches:
            key = metric.lower().replace(' ', '_')
            result[key] = float(matches[-1])
    return result


def checkpoint_enhancement_parameter_count(checkpoint):
    try:
        payload = torch.load(
            checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location='cpu')
    state = payload.get('net', payload)
    return sum(
        tensor.numel() for name, tensor in state.items()
        if 'skip_enhancement' in name or 'post_enhancement' in name)


def exact_metrics(result_path, fallback):
    if not result_path.is_file():
        return fallback
    try:
        result = torch.load(
            result_path, map_location='cpu', weights_only=False)
    except TypeError:
        result = torch.load(result_path, map_location='cpu')
    detection = result['detection_results']
    tp = len(detection['matched_gts'])
    fp = len(detection['non_matched_preds_filtered'])
    fn = len(detection['non_matched_gts'])
    completeness = tp / max(tp + fn, 1)
    commission = fp / max(tp + fp, 1)
    precision = 1.0 - commission
    f1 = (
        2 * precision * completeness /
        max(precision + completeness, 1e-12))
    segmentation = result['segmentation_results']
    return {
        'completeness': 100 * completeness,
        'commission_error_rate': 100 * commission,
        'f1_score': 100 * f1,
        'precision': 100 * float(segmentation['precision']),
        'recall': 100 * float(segmentation['recall']),
        'coverage': 100 * float(segmentation['iou']),
    }


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(
            f'Command failed ({process.returncode}); inspect {log_path}.')


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True),
        encoding='utf-8')


def main():
    args = parse_args()
    settings = yaml.safe_load(
        Path(args.config).read_text(encoding='utf-8'))
    plots = list(settings['plots'])
    if any('wytham' in plot.lower() for plot in plots):
        raise ValueError('Validation matrix must never read Wytham.')
    output_root = Path(settings['output_root'])
    runtime_root = Path(settings['runtime_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for run_name, spec in settings['models'].items():
        model_name = spec.get('model', run_name)
        checkpoint = Path(spec['checkpoint'])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        train_config = get_config(spec['train_config'])
        enhancement_parameter_count = (
            checkpoint_enhancement_parameter_count(checkpoint))
        for plot in plots:
            print(f'===== {model_name} seed {spec["seed"]}: {plot} =====',
                  flush=True)
            forest = locate_forest(settings['source_root'], plot)
            plot_runtime = runtime_root / plot
            result_name = (
                f'results_{settings["stage"]}_{model_name}_'
                f'seed{spec["seed"]}')
            pipeline_config = get_config(settings['pipeline_template'])
            pipeline_config.forest_path = str(forest)
            pipeline_config.pipeline_base_dir = str(plot_runtime)
            pipeline_config.pretrain = str(checkpoint)
            pipeline_config.model = train_config.model
            pipeline_config.save_cfg.results_dir = result_name
            tile_dir = plot_runtime / 'tiles' / 'npz'
            pipeline_config.tile_generation = not tile_dir.is_dir()
            config_dir = output_root / 'runtime_configs'
            pipeline_path = config_dir / (
                f'pipeline_{run_name}_seed{spec["seed"]}_{plot}.yaml')
            write_yaml(pipeline_path, munch_to_dict(pipeline_config))

            prediction = (
                plot_runtime / result_name / 'full_forest' /
                f'{plot}.laz')
            pipeline_log = output_root / 'pipeline_logs' / (
                f'{run_name}_seed{spec["seed"]}_{plot}.log')
            if args.force or not prediction.is_file():
                run_logged([
                    sys.executable, '-u', 'tools/pipeline/pipeline.py',
                    '--config', str(pipeline_path),
                ], pipeline_log)

            evaluation_config = get_config(
                settings['evaluation_template'])
            evaluation_config.paths.gt_forest_path = str(forest)
            evaluation_config.paths.pred_forest_path = str(prediction)
            evaluation_path = config_dir / (
                f'evaluate_{run_name}_seed{spec["seed"]}_{plot}.yaml')
            write_yaml(
                evaluation_path, munch_to_dict(evaluation_config))
            evaluation_log = output_root / 'evaluation_logs' / (
                f'{run_name}_seed{spec["seed"]}_{plot}.log')
            if args.force or not evaluation_log.is_file():
                run_logged([
                    sys.executable, '-u', 'tools/evaluation/evaluate.py',
                    '--config', str(evaluation_path),
                ], evaluation_log)
            metrics = parse_metrics(
                evaluation_log.read_text(encoding='utf-8'))
            metrics = exact_metrics(
                prediction.parent / 'evaluation' /
                'evaluation_results.pt',
                metrics)
            if len(metrics) != len(METRICS):
                raise ValueError(
                    f'Incomplete metrics in {evaluation_log}: {metrics}')
            rows.append({
                'model': model_name,
                'seed': int(spec['seed']),
                'plot': plot,
                'train_config': spec['train_config'],
                'checkpoint': str(checkpoint),
                'training_log': spec.get('training_log', ''),
                'parameter_count': enhancement_parameter_count,
                'pipeline_log': str(pipeline_log),
                'evaluation_log': str(evaluation_log),
                **metrics,
            })

    csv_path = output_root / 'records.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / 'records.json').write_text(
        json.dumps(rows, indent=2), encoding='utf-8')
    print(f'Saved {csv_path}')


if __name__ == '__main__':
    main()
