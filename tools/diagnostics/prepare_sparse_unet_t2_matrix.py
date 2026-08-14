"""Generate a T2 five-forest matrix for B1 plus locked finalists."""

import argparse
from pathlib import Path

import yaml


VALID_MODELS = {
    'b1_partial',
    'm1_sparse_se',
    'm2_selective_kernel',
    'm3_hcag',
    'm4_window_attention',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models', nargs='+', required=True,
        help='Include b1_partial and only T1 finalists.')
    parser.add_argument(
        '--output',
        default='configs/experiments/sparse_attention_unet/'
                't2_validation_matrix.yaml')
    return parser.parse_args()


def main():
    args = parse_args()
    models = list(dict.fromkeys(args.models))
    invalid = set(models) - VALID_MODELS
    if invalid:
        raise ValueError(f'Unknown models: {sorted(invalid)}')
    if 'b1_partial' not in models:
        raise ValueError('T2 matrix must include b1_partial.')
    attention_models = [m for m in models if m != 'b1_partial']
    if not 1 <= len(attention_models) <= 2:
        raise ValueError('T2 must contain one or two T1 finalists.')

    specs = {}
    for model in models:
        for seed in (42, 43, 44):
            run_name = f'{model}_seed{seed}'
            config_name = f'train_t2_{model}_seed{seed}'
            specs[run_name] = {
                'model': model,
                'seed': seed,
                'train_config': (
                    'configs/experiments/sparse_attention_unet/'
                    f'{config_name}.yaml'),
                'checkpoint': (
                    f'work_dirs/{config_name}/best_unet.pth'),
                'training_log': (
                    f'logs/sparse_attention_unet/{config_name}.log'),
            }
    payload = {
        'stage': 't2',
        'source_root': 'data/train/forests',
        'runtime_root':
            'data/pipeline/sparse_attention_unet_validation',
        'output_root': 'logs/sparse_attention_unet/t2_validation',
        'pipeline_template': (
            'configs/experiments/sparse_attention_unet/'
            '_pipeline_validation_common.yaml'),
        'evaluation_template':
            'configs/experiments/_evaluation_common.yaml',
        'plots': ['G4N', 'G4W', 'L1N', 'O1N', 'O1W'],
        'models': specs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    print(f'Saved {output} for models: {", ".join(models)}')


if __name__ == '__main__':
    main()
