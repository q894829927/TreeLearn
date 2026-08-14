"""Generate immutable L1W and Wytham configs for the locked winner."""

import argparse
from pathlib import Path

import yaml

from tree_learn.util import get_config
from tree_learn.util.parser import munch_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--winner_config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tag', default='winner_locked')
    parser.add_argument(
        '--output_dir',
        default='configs/experiments/sparse_attention_unet/locked')
    return parser.parse_args()


def write(path, payload):
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def pipeline_payload(common, enhancement, checkpoint, results):
    return {
        'default_args': [
            common,
            'configs/_modular/sample_generation.yaml',
            'configs/_modular/model.yaml',
            'configs/_modular/grouping.yaml',
            'configs/_modular/dataset_test.yaml',
        ],
        'model': {
            'use_height_context_adapter': False,
            'use_axis_branch': False,
            'unet_enhancement': enhancement,
            'unet_finetune': {'enabled': False},
        },
        'pretrain': checkpoint,
        'save_cfg': {'results_dir': results},
    }


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = get_config(args.winner_config)
    enhancement = munch_to_dict(config.model.unet_enhancement)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    l1w_results = f'results_sparse_unet_{args.tag}_l1w'
    wytham_results = f'results_sparse_unet_{args.tag}_wytham'
    write(
        output / 'pipeline_l1w_winner_locked.yaml',
        pipeline_payload(
            'configs/experiments/height_context_attention/'
            '_pipeline_l1w_common.yaml',
            enhancement, str(checkpoint), l1w_results))
    write(
        output / 'evaluate_l1w_winner_locked.yaml',
        {
            'default_args': [
                'configs/experiments/_evaluation_common.yaml'],
            'paths': {
                'pred_forest_path':
                    f'data/pipeline/L1W/{l1w_results}/'
                    'full_forest/L1W.laz',
            },
        })
    write(
        output / 'pipeline_wytham_winner_locked.yaml',
        pipeline_payload(
            'configs/experiments/height_context_attention/'
            '_pipeline_wytham_locked_common.yaml',
            enhancement, str(checkpoint), wytham_results))
    write(
        output / 'evaluate_wytham_winner_locked.yaml',
        {
            'default_args': [
                'configs/experiments/wytham/_evaluation_common.yaml'],
            'paths': {
                'pred_forest_path':
                    f'data/pipeline/wytham/{wytham_results}/'
                    'full_forest/wytham_vox0.1.laz',
            },
        })
    lock = {
        'winner_config': args.winner_config,
        'checkpoint': str(checkpoint),
        'enhancement': enhancement,
        'tag': args.tag,
        'wytham_is_final_no_tuning': True,
    }
    write(output / 'lock.yaml', lock)
    print(f'Saved locked evaluation configs and lock file to {output}')


if __name__ == '__main__':
    main()
