"""T0 structural, identity, gradient, FP16, memory and runtime smoke tests."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from tree_learn.model.blocks import ResidualBlock, UBlock


CANDIDATES = {
    'identity': {'enabled': False, 'type': 'identity', 'levels': []},
    'residual_adapter': {
        'enabled': True, 'type': 'residual_adapter', 'levels': [0, 1, 2]},
    'sparse_se': {
        'enabled': True, 'type': 'sparse_se', 'levels': [0, 1, 2]},
    'selective_kernel': {
        'enabled': True, 'type': 'selective_kernel', 'levels': [1, 2]},
    'hcag': {'enabled': True, 'type': 'hcag', 'levels': [0, 1]},
    'window_attention': {
        'enabled': True,
        'type': 'window_attention',
        'levels': [2],
        'window_size': [4, 4, 4],
        'num_heads': 4,
        'ffn_ratio': 2,
        'max_windows_per_chunk': 256,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        default='logs/sparse_attention_unet/t0_smoke')
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--memory_limit_gb', type=float, default=23.0)
    return parser.parse_args()


def sparse_input(num_points, device, seed=20260814):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    batch = torch.randint(0, 2, (num_points, 1), generator=generator)
    spatial = torch.randint(0, 32, (num_points, 3), generator=generator)
    indices = torch.cat([batch, spatial], dim=1)
    indices = torch.unique(indices, dim=0)
    features = torch.randn(
        len(indices), 8, generator=generator).to(device)
    return spconv.SparseConvTensor(
        features, indices.int().to(device), [32, 32, 32], 2)


def build_unet(config, device):
    norm_fn = lambda channels: nn.BatchNorm1d(
        channels, eps=1e-4, momentum=0.1)
    return UBlock(
        [8, 16, 24], norm_fn, 2, ResidualBlock, 3,
        enhancement_config={
            **config,
            'residual_gamma_init': 0.0,
            'return_attention_stats': True,
            'hidden_ratio': 0.5,
            'min_hidden_channels': 4,
        },
        total_levels=3).to(device).eval()


def enhancement_parameters(model):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if 'skip_enhancement' in name or 'post_enhancement' in name
    ]


def run_candidate(name, config, base, input_factory, repeats, device):
    model = build_unet(config, device)
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f'{name}: unexpected official keys: {unexpected}')
    allowed_missing = [
        key for key in incompatible.missing_keys
        if 'enhancement' in key
    ]
    if len(allowed_missing) != len(incompatible.missing_keys):
        raise RuntimeError(
            f'{name}: non-enhancement missing keys: '
            f'{set(incompatible.missing_keys) - set(allowed_missing)}')

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        reference_input = input_factory()
        base_output = base(reference_input)
        candidate_input = input_factory()
        output = model(candidate_input)
    identity_error = float(
        (output.features.float() -
         base_output.features.float()).abs().max().item())
    indices_equal = torch.equal(output.indices, base_output.indices)
    metadata_equal = (
        tuple(output.spatial_shape) == tuple(base_output.spatial_shape) and
        output.batch_size == base_output.batch_size)
    finite = bool(torch.isfinite(output.features).all().item())

    parameters = enhancement_parameters(model)
    for module in model.modules():
        if hasattr(module, 'gamma'):
            module.gamma.data.fill_(1.0)
    model.train()
    gradient_input = input_factory()
    with torch.cuda.amp.autocast(enabled=True):
        loss = model(gradient_input).features.float().square().mean()
    loss.backward()
    gradient_tensors = sum(
        parameter.grad is not None and
        torch.isfinite(parameter.grad).all().item()
        for parameter in parameters)
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            current = input_factory()
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=True):
                model(current)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    parameter_count = sum(p.numel() for p in model.parameters())
    enhancement_count = sum(p.numel() for p in parameters)
    stats = model.enhancement_statistics()
    attention_values = [
        value for key, value in stats.items()
        if 'attention_' in key and not key.endswith('sum_error')
    ]
    attention_finite = all(
        torch.isfinite(torch.tensor(value)).item()
        for value in attention_values)
    return {
        'model': name,
        'identity_max_abs_error': identity_error,
        'indices_equal': indices_equal,
        'metadata_equal': metadata_equal,
        'fp16_finite': finite,
        'parameter_count': parameter_count,
        'enhancement_parameter_count': enhancement_count,
        'enhancement_gradient_tensors': int(gradient_tensors),
        'enhancement_parameter_tensors': len(parameters),
        'attention_stats_finite': attention_finite,
        'peak_memory_gb': peak_gb,
        'mean_forward_seconds': sum(times) / len(times),
        'stats': stats,
    }


def write_markdown(path, rows, gate):
    lines = [
        '# T0 Sparse U-Net enhancement smoke',
        '',
        '| Model | Identity error | Params added | Grad tensors | '
        'Peak GB | Forward s | Pass |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | "
            f"{row['identity_max_abs_error']:.3e} | "
            f"{row['enhancement_parameter_count']:,} | "
            f"{row['enhancement_gradient_tensors']}/"
            f"{row['enhancement_parameter_tensors']} | "
            f"{row['peak_memory_gb']:.3f} | "
            f"{row['mean_forward_seconds']:.4f} | "
            f"{row['passed']} |")
    lines += ['', '## Gate', '']
    for key, value in gate.items():
        lines.append(f'- {key}: **{value}**')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('T0 requires CUDA/spconv.')
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = build_unet(CANDIDATES['identity'], device)
    factory = lambda: sparse_input(args.num_points, device)
    rows = []
    for name, config in CANDIDATES.items():
        print(f'===== T0 {name} =====', flush=True)
        row = run_candidate(
            name, config, base, factory, args.repeats, device)
        row['passed'] = bool(
            row['identity_max_abs_error'] == 0.0 and
            row['indices_equal'] and row['metadata_equal'] and
            row['fp16_finite'] and row['attention_stats_finite'] and
            row['peak_memory_gb'] < args.memory_limit_gb and
            (
                row['enhancement_parameter_tensors'] == 0 or
                row['enhancement_gradient_tensors'] ==
                row['enhancement_parameter_tensors']
            ))
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    gate = {
        'all_candidates_passed': all(row['passed'] for row in rows),
        'memory_limit_gb': args.memory_limit_gb,
        'passed': all(row['passed'] for row in rows),
    }
    report = {'rows': rows, 'gate': gate}
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    write_markdown(output_dir / 'summary.md', rows, gate)
    if not gate['passed']:
        raise RuntimeError('T0 gate failed; do not start T1.')


if __name__ == '__main__':
    main()
