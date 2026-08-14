"""Create the locked winner + seed-preservation configuration."""

import argparse
from pathlib import Path

import yaml

from tree_learn.util import get_config
from tree_learn.util.parser import munch_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--winner_config', required=True)
    parser.add_argument(
        '--output',
        default='configs/experiments/sparse_attention_unet/'
                'train_t4_winner_preservation_seed42.yaml')
    parser.add_argument('--weight', type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.weight <= 0:
        raise ValueError('--weight must be positive.')
    config = get_config(args.winner_config)
    enhancement = munch_to_dict(config.model.unet_enhancement)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'default_args': [
            'configs/experiments/sparse_attention_unet/_train_common.yaml',
            'configs/_modular/model.yaml',
            'configs/_modular/dataset_train.yaml',
            'configs/_modular/dataset_test.yaml',
        ],
        'seed': 42,
        'epochs': 50,
        'early_stopping_patience': 12,
        'scheduler': {'t_initial': 50},
        'model': {
            'unet_enhancement': enhancement,
            'unet_finetune': {
                'enabled': True,
                'freeze_input_conv': True,
                'freeze_encoder': True,
                'train_decoder_levels': [0, 1],
                'train_heads': True,
                'keep_frozen_batchnorm_eval': True,
                'teacher_seed_preservation_weight': args.weight,
                'teacher_seed_tree_conf_thresh': 0.5,
                'teacher_seed_tau_vert': 0.6,
                'teacher_seed_tau_off': 4.0,
            },
        },
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    print(f'Saved locked preservation config: {output}')


if __name__ == '__main__':
    main()
