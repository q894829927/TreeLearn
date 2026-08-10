"""Run TreeLearn once and intercept its pointwise save for Q3 diagnostics."""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

from tree_learn.util import get_config
from tree_learn.util.omission_diagnostics import (
    compute_gt_omission_diagnostics,
    save_gt_omission_diagnostics,
)


class DiagnosticComplete(RuntimeError):
    """Internal control flow used to avoid writing the large pointwise NPZ."""


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / 'pipeline' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location(
        'treelearn_q3_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path):
    config = get_config(config_path)
    output_dir = str(config.omission_output_dir)
    source_plot = Path(str(config.forest_path)).stem
    split = str(getattr(config, 'omission_split', 'validation'))
    pipeline = load_pipeline_module()
    original_savez = np.savez_compressed
    completed = {'value': False}

    def intercept_savez(file, *args, **kwargs):
        if Path(str(file)).name != 'pointwise_results.npz':
            return original_savez(file, *args, **kwargs)
        required = {
            'coords', 'semantic_prediction_logits', 'offset_predictions',
            'instance_labels', 'instance_preds',
            'instance_preds_after_initial_clustering', 'input_feats',
        }
        missing = required - set(kwargs)
        if missing:
            raise ValueError(
                f'Intercepted pointwise payload misses {sorted(missing)}.')
        result = compute_gt_omission_diagnostics(
            coords=kwargs['coords'],
            semantic_logits=kwargs['semantic_prediction_logits'],
            offset_predictions=kwargs['offset_predictions'],
            verticality=kwargs['input_feats'][:, -1],
            instance_labels=kwargs['instance_labels'],
            instance_predictions=kwargs['instance_preds'],
            initial_instance_predictions=kwargs[
                'instance_preds_after_initial_clustering'],
            tree_conf_thresh=float(config.grouping.tree_conf_thresh),
            tau_vert=float(config.grouping.tau_vert),
            tau_off=float(config.grouping.tau_off),
            tau_min=int(config.grouping.tau_min),
            tree_class_index=int(getattr(
                config, 'omission_tree_class_index', 0)),
            match_iou_threshold=float(getattr(
                config, 'omission_match_iou_threshold', 0.5)),
            min_precision_for_counted_fp=float(getattr(
                config, 'omission_min_precision_for_counted_fp', 0.5)),
            min_recall_for_undersegmentation=float(getattr(
                config, 'omission_min_recall_for_undersegmentation', 0.5)),
            min_fragment_overlap_fraction=float(getattr(
                config, 'omission_min_fragment_overlap_fraction', 0.05)),
            min_fragment_overlap_points=int(getattr(
                config, 'omission_min_fragment_overlap_points', 20)))
        paths = save_gt_omission_diagnostics(
            result, output_dir, source_plot, split)
        completed['value'] = True
        print(
            f'Q3 diagnostic saved for {source_plot}: '
            f'{result["baseline"]["tp"]} TP, '
            f'{result["baseline"]["fp"]} FP, '
            f'{result["baseline"]["fn"]} FN; {paths[0]}',
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
            'TreeLearn finished without producing the intercepted Q3 payload.')


def main():
    parser = argparse.ArgumentParser(
        description='Run one GT omission diagnostic pipeline.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
