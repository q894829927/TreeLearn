"""Run one TreeLearn forest and intercept its Q4b1 learning artifact."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from tree_learn.util import get_config
from tree_learn.util.instance_split_learning import (
    build_instance_split_learning_artifact,
    save_instance_split_learning_artifact,
)


class ArtifactComplete(RuntimeError):
    """Internal control flow used to avoid a large pointwise NPZ write."""


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / 'pipeline' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location(
        'treelearn_q4b1_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path):
    config = get_config(config_path)
    output_dir = Path(str(config.split_learning_output_dir))
    source_plot = Path(str(config.forest_path)).stem
    split = str(config.split_learning_split)
    pipeline = load_pipeline_module()
    original_savez = np.savez_compressed
    completed = {'value': False}

    def intercept_savez(file, *args, **kwargs):
        if Path(str(file)).name != 'pointwise_results.npz':
            return original_savez(file, *args, **kwargs)
        required = {
            'coords', 'semantic_prediction_logits', 'offset_predictions',
            'instance_labels', 'instance_preds',
            'instance_preds_after_initial_clustering', 'backbone_feats',
            'input_feats',
        }
        missing = required - set(kwargs)
        if missing:
            raise ValueError(
                f'Intercepted Q4b1 payload misses {sorted(missing)}.')
        if kwargs['backbone_feats'] is None:
            raise ValueError('Q4b1 needs frozen backbone features.')
        log_variance = kwargs.get('axis_log_variances')
        confidence = (
            None if log_variance is None else
            1.0 / (1.0 + np.exp(np.asarray(log_variance))))
        settings = config.split_learning
        expected_ids = getattr(
            settings, 'expected_undersegmented_gt_ids', None)
        artifact = build_instance_split_learning_artifact(
            coords=kwargs['coords'],
            semantic_logits=kwargs['semantic_prediction_logits'],
            offset_predictions=kwargs['offset_predictions'],
            instance_labels=kwargs['instance_labels'],
            instance_predictions=kwargs['instance_preds'],
            initial_instance_predictions=kwargs[
                'instance_preds_after_initial_clustering'],
            backbone_features=kwargs['backbone_feats'],
            verticality=kwargs['input_feats'][:, -1],
            axis_confidence=confidence,
            split=split,
            num_layers=int(settings.num_layers),
            min_parent_points=int(settings.min_parent_points),
            max_children=int(settings.max_children),
            max_fit_points=int(settings.max_fit_points),
            random_state=int(settings.random_state),
            max_train_negative_parents_per_plot=int(
                settings.max_train_negative_parents_per_plot),
            tree_conf_thresh=float(config.grouping.tree_conf_thresh),
            tau_vert=float(config.grouping.tau_vert),
            tau_off=float(config.grouping.tau_off),
            tau_min=int(config.grouping.tau_min),
            match_iou_threshold=float(settings.match_iou_threshold),
            min_precision_for_counted_fp=float(
                settings.min_precision_for_counted_fp),
            min_recall_for_undersegmentation=float(
                settings.min_recall_for_undersegmentation),
            min_fragment_overlap_fraction=float(
                settings.min_fragment_overlap_fraction),
            min_fragment_overlap_points=int(
                settings.min_fragment_overlap_points),
            expected_undersegmented_gt_ids=(
                None if expected_ids is None else list(expected_ids)))
        paths = save_instance_split_learning_artifact(
            artifact, output_dir, source_plot, split,
            metadata={
                'checkpoint': str(config.pretrain),
                'source_forest': str(config.forest_path),
            },
            savez_function=original_savez)
        completed['value'] = True
        metadata = artifact['metadata']
        print(
            f'Q4b1 saved for {source_plot}: '
            f'{metadata["num_instances"]} instances, '
            f'{metadata["num_candidate_parents"]} candidates, '
            f'{metadata["num_proposals"]} proposals; {paths[0]}',
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
            'TreeLearn finished without producing the intercepted Q4b1 '
            'artifact.')


def main():
    parser = argparse.ArgumentParser(
        description='Run one Q4b1 split-learning pipeline.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()