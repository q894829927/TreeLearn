"""Run one forest and intercept its Q5b1 proposal-learning artifact."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from tree_learn.util import get_config
from tree_learn.util.seed_completion_learning import (
    build_seed_completion_learning_artifact,
    save_learning_artifact,
)


class ArtifactComplete(RuntimeError):
    """Internal control flow used to avoid writing a large pointwise NPZ."""


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / 'pipeline' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location(
        'treelearn_q5b1_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path):
    config = get_config(config_path)
    output_dir = Path(str(config.seed_completion_learning_output_dir))
    source_plot = Path(str(config.forest_path)).stem
    split = str(config.seed_completion_learning_split)
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
                f'Intercepted Q5b1 payload misses {sorted(missing)}.')
        grouping = config.grouping
        if float(grouping.upper_anchor_weight) != 0.0 or bool(getattr(
                grouping, 'use_axis_fusion', False)) or bool(getattr(
                grouping, 'use_upper_seeds', False)):
            raise ValueError('Q5b1 requires locked base-only 2D grouping.')
        if bool(getattr(grouping, 'use_seed_confidence_filter', False)):
            raise ValueError(
                'Q5b1 requires seed-confidence filtering to be disabled.')
        settings = config.seed_completion_learning
        expected_ids = getattr(settings, 'expected_target_tree_ids', None)
        artifact = build_seed_completion_learning_artifact(
            coords=kwargs['coords'],
            semantic_logits=kwargs['semantic_prediction_logits'],
            offset_predictions=kwargs['offset_predictions'],
            verticality=kwargs['input_feats'][:, -1],
            instance_labels=kwargs['instance_labels'],
            baseline_predictions=kwargs['instance_preds'],
            baseline_initial_predictions=kwargs[
                'instance_preds_after_initial_clustering'],
            source_plot=source_plot,
            split=split,
            tree_conf_thresh=float(grouping.tree_conf_thresh),
            tau_vert=float(grouping.tau_vert),
            tau_off=float(grouping.tau_off),
            tau_min=int(grouping.tau_min),
            scales=list(settings.scales),
            shift_fractions=list(settings.shift_fractions),
            match_iou_threshold=float(settings.match_iou_threshold),
            min_precision_for_counted_fp=float(
                settings.min_precision_for_counted_fp),
            min_recall_for_undersegmentation=float(
                settings.min_recall_for_undersegmentation),
            min_fragment_overlap_fraction=float(
                settings.min_fragment_overlap_fraction),
            min_fragment_overlap_points=int(
                settings.min_fragment_overlap_points),
            expected_target_tree_ids=(
                None if expected_ids is None else list(expected_ids)))
        artifact['metadata'].update({
            'checkpoint': str(config.pretrain),
            'source_forest': str(config.forest_path),
        })
        paths = save_learning_artifact(artifact, output_dir)
        completed['value'] = True
        metadata = artifact['metadata']
        print(
            f'Q5b1 saved for {source_plot}: '
            f'{metadata["num_proposals"]:,} proposals, '
            f'{metadata["num_activation_positive"]} activated, '
            f'{metadata["num_target_trees"]} target trees; {paths[0]}',
            flush=True)
        raise ArtifactComplete()

    np.savez_compressed = intercept_savez
    try:
        pipeline.run_treelearn_pipeline(config, config_path)
    except ArtifactComplete:
        pass
    finally:
        np.savez_compressed = original_savez
    if not completed['value']:
        raise RuntimeError(
            'TreeLearn finished without producing the intercepted Q5b1 '
            'artifact.')


def main():
    parser = argparse.ArgumentParser(
        description='Run one Q5b1 proposal-learning pipeline.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
