"""TreeLearn pipeline entrypoint with offset/HDBSCAN-free grouping."""

import argparse
import importlib.util
from pathlib import Path
from tree_learn.util import get_config
from tree_learn.util.proposal_relation_inference import (
    predict_proposal_relation_instances)


def _load_pipeline_module():
    path = Path(__file__).with_name('pipeline.py')
    spec = importlib.util.spec_from_file_location(
        'treelearn_relation_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path):
    config = get_config(config_path)
    if str(getattr(config.grouping, 'method', 'hdbscan')) != 'proposal_relation':
        raise ValueError(
            'This entrypoint requires grouping.method=proposal_relation.')
    pipeline = _load_pipeline_module()

    def relation_get_instances(
            coords, offset_predictions, upper_offset_predictions,
            semantic_prediction_logits, grouping_cfg, verticality,
            tree_class_index, non_tree_label, not_assigned_label,
            start_num_preds, logger=None, **kwargs):
        del offset_predictions, upper_offset_predictions, grouping_cfg
        del non_tree_label, not_assigned_label, start_num_preds, kwargs
        output = (Path(str(config.pipeline_base_dir)) /
                  str(config.save_cfg.results_dir) / 'proposal_relation')
        predictions, _ = predict_proposal_relation_instances(
            coords, semantic_prediction_logits, verticality,
            config.proposal_relation, output_dir=output, logger=logger)
        return predictions

    pipeline.get_instances = relation_get_instances
    return pipeline.run_treelearn_pipeline(config, config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
