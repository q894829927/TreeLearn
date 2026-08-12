"""Run one TreeLearn forest and intercept its pointwise payload for Q5a."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from tree_learn.util import get_config
from tree_learn.util.seed_completion_diagnostics import (
    analyze_seed_completion_oracle,
)


class DiagnosticComplete(RuntimeError):
    """Internal control flow used to avoid a large pointwise NPZ write."""


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / 'pipeline' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location(
        'treelearn_q5a_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path):
    config = get_config(config_path)
    output_dir = Path(str(config.seed_completion_output_dir))
    source_plot = Path(str(config.forest_path)).stem
    pipeline = load_pipeline_module()
    original_savez = np.savez_compressed
    completed = {'value': False}

    def intercept_savez(file, *args, **kwargs):
        if Path(str(file)).name != 'pointwise_results.npz':
            return original_savez(file, *args, **kwargs)
        required = {
            'coords', 'semantic_prediction_logits', 'offset_predictions',
            'offset_labels', 'instance_labels', 'instance_preds',
            'instance_preds_after_initial_clustering', 'input_feats',
        }
        missing = required - set(kwargs)
        if missing:
            raise ValueError(
                f'Intercepted Q5a payload misses {sorted(missing)}.')
        settings = config.seed_completion
        grouping = config.grouping
        if float(grouping.upper_anchor_weight) != 0.0 or bool(getattr(
                grouping, 'use_axis_fusion', False)) or bool(getattr(
                    grouping, 'use_upper_seeds', False)):
            raise ValueError('Q5a requires locked base-only 2D grouping.')
        if bool(getattr(grouping, 'use_seed_confidence_filter', False)):
            raise ValueError(
                'Q5a requires seed-confidence filtering to be disabled.')
        print(
            f'Q5a analyzing {source_plot}: '
            f'{len(settings.expected_target_tree_ids)} fixed Q3 targets.',
            flush=True)
        result = analyze_seed_completion_oracle(
            coords=kwargs['coords'],
            semantic_logits=kwargs['semantic_prediction_logits'],
            offset_predictions=kwargs['offset_predictions'],
            offset_labels=kwargs['offset_labels'],
            verticality=kwargs['input_feats'][:, -1],
            instance_labels=kwargs['instance_labels'],
            baseline_predictions=kwargs['instance_preds'],
            baseline_initial_predictions=kwargs[
                'instance_preds_after_initial_clustering'],
            expected_target_tree_ids=list(
                settings.expected_target_tree_ids),
            tree_conf_thresh=float(grouping.tree_conf_thresh),
            tau_vert=float(grouping.tau_vert),
            tau_off=float(grouping.tau_off),
            tau_min=int(grouping.tau_min),
            match_iou_threshold=float(settings.match_iou_threshold),
            min_precision_for_counted_fp=float(
                settings.min_precision_for_counted_fp),
            min_recall_for_undersegmentation=float(
                settings.min_recall_for_undersegmentation),
            min_fragment_overlap_fraction=float(
                settings.min_fragment_overlap_fraction),
            min_fragment_overlap_points=int(
                settings.min_fragment_overlap_points),
            knn_chunk_size=int(getattr(
                grouping, 'knn_chunk_size', 200000)),
            max_cluster_seed_points=getattr(
                grouping, 'max_cluster_seed_points', None))
        result.update({'source_plot': source_plot, 'split': 'validation'})
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / 'summary.json'
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding='utf-8')
        completed['value'] = True
        print(
            f'Q5a saved for {source_plot}: '
            f'{result["num_target_trees"]} support-failure targets; {path}',
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
            'TreeLearn finished without producing the intercepted Q5a '
            'payload.')


def main():
    parser = argparse.ArgumentParser(
        description='Run one fixed-validation Q5a seed-completion Oracle.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
