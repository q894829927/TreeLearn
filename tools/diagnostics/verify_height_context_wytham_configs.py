"""Verify that the locked Wytham A0/A1/A2 runs differ only as intended."""

from copy import deepcopy
from pathlib import Path

from tree_learn.util import get_config, munch_to_dict


PIPELINES = {
    'a0_official': 'configs/experiments/height_context_attention/pipeline_wytham_a0_official_locked.yaml',
    'a1_height_mlp': 'configs/experiments/height_context_attention/pipeline_wytham_a1_height_mlp_locked.yaml',
    'a2_hsca': 'configs/experiments/height_context_attention/pipeline_wytham_a2_hsca_locked.yaml',
}
EVALUATIONS = {
    name: path.replace('pipeline_wytham_', 'evaluate_wytham_')
    for name, path in PIPELINES.items()
}
EXPECTED_MODELS = {
    'a0_official': (False, 'attention'),
    'a1_height_mlp': (True, 'mlp'),
    'a2_hsca': (True, 'attention'),
}
EXPECTED_CHECKPOINTS = {
    'a0_official': 'data/model_weights/model_weights_with_small_20241213.pth',
    'a1_height_mlp': 'work_dirs/train_a1_height_mlp_frozen/best_height_mlp.pth',
    'a2_hsca': 'work_dirs/train_a2_hsca_frozen/best_hsca.pth',
}


def without_path(mapping, *path):
    mapping = deepcopy(mapping)
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor[key]
    cursor.pop(path[-1], None)
    return mapping


def normalized_pipeline(config):
    result = munch_to_dict(config)
    for path in (
            ('pretrain',),
            ('save_cfg', 'results_dir'),
            ('model', 'use_height_context_adapter'),
            ('model', 'height_context_type')):
        result = without_path(result, *path)
    return result


def normalized_evaluation(config):
    return without_path(munch_to_dict(config), 'paths', 'pred_forest_path')


def main():
    pipelines = {name: get_config(path) for name, path in PIPELINES.items()}
    evaluations = {name: get_config(path) for name, path in EVALUATIONS.items()}
    reference_pipeline = normalized_pipeline(pipelines['a0_official'])
    reference_evaluation = normalized_evaluation(evaluations['a0_official'])

    for name, config in pipelines.items():
        assert normalized_pipeline(config) == reference_pipeline, (
            f'{name} contains an unintended pipeline configuration change.'
        )
        assert normalized_evaluation(evaluations[name]) == reference_evaluation, (
            f'{name} contains an unintended evaluation configuration change.'
        )

        enabled, adapter_type = EXPECTED_MODELS[name]
        assert config.model.use_height_context_adapter is enabled
        assert config.model.height_context_type == adapter_type
        assert config.pretrain == EXPECTED_CHECKPOINTS[name]
        assert config.forest_path == 'data/pipeline/wytham/forest/wytham_vox0.1.laz'
        assert config.tile_generation is False
        assert config.save_cfg.return_type == 'voxelized_and_filtered'
        assert config.dataset_test.base_anchor_mode == 'legacy'
        assert config.grouping.max_cluster_seed_points == 1500000
        assert config.grouping.tau_min == 50
        assert config.grouping.use_upper_seeds is False
        assert config.grouping.use_axis_fusion is False
        assert config.grouping.use_seed_confidence_filter is False
        assert config.grouping.seed_confidence_keep_ratio == 1.0

        for required_path in (
                config.forest_path,
                config.pretrain,
                evaluations[name].paths.gt_forest_path):
            assert Path(required_path).is_file(), (
                f'{name}: required file does not exist: {required_path}'
            )

        print(
            f'{name}: adapter={enabled}/{adapter_type}, '
            f'checkpoint={config.pretrain}, '
            f'results={config.save_cfg.results_dir}'
        )

    print('PASS: Wytham A0/A1/A2 configs are locked and directly comparable.')


if __name__ == '__main__':
    main()
