"""Run one TreeLearn forest and intercept its pointwise payload for Q4b0."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from tree_learn.util import get_config
from tree_learn.util.instance_split_diagnostics import (
    analyze_instance_split_deployability,
)


class DiagnosticComplete(RuntimeError):
    """Internal control flow used to avoid a large pointwise NPZ write."""


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / 'pipeline' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location(
        'treelearn_q4b0_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path):
    config = get_config(config_path)
    output_dir = Path(str(config.split_deployability_output_dir))
    source_plot = Path(str(config.forest_path)).stem
    pipeline = load_pipeline_module()
    original_savez = np.savez_compressed
    completed = {'value': False}

    def intercept_savez(file, *args, **kwargs):
        if Path(str(file)).name != 'pointwise_results.npz':
            return original_savez(file, *args, **kwargs)
        required = {
            'coords', 'offset_predictions', 'instance_labels',
            'instance_preds',
        }
        missing = required - set(kwargs)
        if missing:
            raise ValueError(
                f'Intercepted Q4b0 payload misses {sorted(missing)}.')
        settings = config.split_deployability
        result = analyze_instance_split_deployability(
            coords=kwargs['coords'],
            offset_predictions=kwargs['offset_predictions'],
            instance_labels=kwargs['instance_labels'],
            instance_predictions=kwargs['instance_preds'],
            match_iou_threshold=float(settings.match_iou_threshold),
            min_precision_for_counted_fp=float(
                settings.min_precision_for_counted_fp),
            min_recall_for_undersegmentation=float(
                settings.min_recall_for_undersegmentation),
            max_fit_points=int(settings.max_fit_points),
            random_state=int(settings.random_state),
            allowed_undersegmented_gt_ids=list(
                settings.expected_undersegmented_gt_ids))
        result.update({
            'source_plot': source_plot,
            'split': 'validation',
        })
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'summary.json'
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding='utf-8')
        completed['value'] = True
        print(
            f'Q4b0 saved for {source_plot}: '
            f'{result["num_undersegmented_gt_trees"]} GT targets, '
            f'{result["num_split_target_predictions"]} parents.',
            flush=True)
        raise DiagnosticComplete()

    np.savez_compressed = intercept_savez
    try:
        pipeline.run_treelearn_pipeline(config, config_path)
    except DiagnosticComplete:
        pass
    finally:
        np.savez_compressed = original_savez
    if not completed['value']:
        raise RuntimeError(
            'TreeLearn finished without producing the intercepted Q4b0 '
            'payload.')


def main():
    parser = argparse.ArgumentParser(
        description='Run one fixed-validation Q4b0 deployability pipeline.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
