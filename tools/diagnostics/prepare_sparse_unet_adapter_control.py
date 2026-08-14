"""Prepare a <=5%-parameter residual-adapter control for a locked winner."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from tree_learn.model import TreeLearn
from tree_learn.util import get_config
from tree_learn.util.parser import munch_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--winner_config', required=True)
    parser.add_argument(
        '--output',
        default='configs/experiments/sparse_attention_unet/'
                'train_t3_b2_adapter_seed42.yaml')
    parser.add_argument('--tolerance', type=float, default=0.05)
    return parser.parse_args()


def enhancement_count(model):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if 'skip_enhancement' in name or 'post_enhancement' in name)


def adapter_parameter_count(levels, ratio, channels=32, num_levels=7):
    count = 0
    for level in levels:
        width = channels * (level + 1)
        hidden = max(1, int(round(width * ratio)))
        if level < num_levels - 1:
            # E->H, D->H and H->E, including biases and gamma.
            count += 3 * width * hidden + 2 * hidden + width + 1
        else:
            # Bottleneck E->H->E, including biases and gamma.
            count += 2 * width * hidden + hidden + width + 1
    return count


def main():
    args = parse_args()
    config = get_config(args.winner_config)
    winner_settings = munch_to_dict(config.model)
    winner = TreeLearn(**winner_settings)
    target = enhancement_count(winner)
    enhancement = winner_settings['unet_enhancement']
    levels = [int(level) for level in enhancement['levels']]
    if not enhancement.get('enabled', False) or not levels:
        raise ValueError('winner_config must enable an enhancement.')

    candidates = []
    for step in range(1, 10001):
        ratio = step / 100.0
        count = adapter_parameter_count(
            levels, ratio,
            channels=int(winner_settings.get('channels', 32)),
            num_levels=int(winner_settings.get('num_blocks', 7)))
        relative_difference = abs(count - target) / max(target, 1)
        candidates.append((relative_difference, ratio, count))
    difference, ratio, adapter_count = min(candidates)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
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
            'unet_enhancement': {
                'enabled': True,
                'type': 'residual_adapter',
                'levels': levels,
                'hidden_ratio': ratio,
                'min_hidden_channels': 1,
            },
        },
    }
    output.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding='utf-8')
    report = {
        'winner_config': args.winner_config,
        'winner_type': enhancement['type'],
        'levels': levels,
        'winner_enhancement_parameters': target,
        'adapter_enhancement_parameters': adapter_count,
        'adapter_hidden_ratio': ratio,
        'relative_parameter_difference': difference,
        'tolerance': args.tolerance,
        'passed': difference <= args.tolerance,
        'output': str(output),
    }
    report_path = output.with_suffix('.parameter_match.json')
    report_path.write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError(
            'No residual-adapter configuration met the parameter gate.')


if __name__ == '__main__':
    main()
