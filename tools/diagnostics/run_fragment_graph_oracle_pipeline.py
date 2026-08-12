"""Run one frozen TreeLearn forest and intercept its Q6a pointwise payload."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from tree_learn.util import get_config
from tree_learn.util.fragment_graph_diagnostics import (
    analyze_fragment_graph_oracle,
)


class DiagnosticComplete(RuntimeError):
    """Internal control flow used to avoid writing a large pointwise NPZ."""


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / 'pipeline' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location(
        'treelearn_q6a_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path):
    config = get_config(config_path)
    output_dir = Path(str(config.fragment_graph_output_dir))
    source_plot = Path(str(config.forest_path)).stem
    settings = config.fragment_graph_oracle
    pipeline = load_pipeline_module()
    original_savez = np.savez_compressed
    completed = {'value': False}

    def intercept_savez(file, *args, **kwargs):
        if Path(str(file)).name != 'pointwise_results.npz':
            return original_savez(file, *args, **kwargs)
        required = {
            'coords', 'semantic_prediction_logits', 'offset_predictions',
            'instance_labels', 'instance_preds', 'input_feats',
        }
        missing = required - set(kwargs)
        if missing:
            raise ValueError(
                f'Intercepted Q6a payload misses {sorted(missing)}.')
        result = analyze_fragment_graph_oracle(
            coords=kwargs['coords'],
            semantic_logits=kwargs['semantic_prediction_logits'],
            offset_predictions=kwargs['offset_predictions'],
            verticality=kwargs['input_feats'][:, -1],
            instance_labels=kwargs['instance_labels'],
            instance_predictions=kwargs['instance_preds'],
            expected_fragmentation_gt_ids=list(
                settings.expected_fragmentation_gt_ids),
            adjacency_settings={
                'max_vote_center_distance': float(
                    settings.max_vote_center_distance),
                'max_raw_center_distance': float(
                    settings.max_raw_center_distance),
                'max_xy_bbox_gap': float(settings.max_xy_bbox_gap),
                'max_vertical_gap': float(settings.max_vertical_gap),
                'max_candidate_edges': int(settings.max_candidate_edges),
            },
            tree_class_index=int(settings.tree_class_index),
            match_iou_threshold=float(settings.match_iou_threshold),
            min_precision_for_counted_fp=float(
                settings.min_precision_for_counted_fp),
            min_fragment_overlap_fraction=float(
                settings.min_fragment_overlap_fraction),
            min_fragment_overlap_points=int(
                settings.min_fragment_overlap_points))
        result.update({'source_plot': source_plot, 'split': 'validation'})
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'summary.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding='utf-8')
        completed['value'] = True
        print(
            f'Q6a saved for {source_plot}: '
            f'{result["num_fragmentation_gt_trees"]} fragmentation targets, '
            f'{result["num_candidate_edges"]:,} GT-free graph edges.',
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
            'TreeLearn finished without producing the intercepted Q6a payload.')


def main():
    parser = argparse.ArgumentParser(
        description='Run one fixed-validation Q6a fragment-graph Oracle.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
