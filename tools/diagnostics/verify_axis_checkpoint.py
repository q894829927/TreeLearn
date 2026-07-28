"""Verify that branch-only training did not change original TreeLearn tensors."""

import argparse
import sys

import torch


AXIS_PREFIXES = (
    'axis_input_projection.',
    'axis_point_transformer.',
    'axis_xy_head.',
    'axis_uncertainty_head.',
)


def load_network(path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'net' not in checkpoint:
        raise KeyError(f'{path} does not contain a "net" state dict.')
    return checkpoint['net']


def main():
    parser = argparse.ArgumentParser(
        description='Check that all shared non-axis tensors are bitwise frozen.')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()

    reference = load_network(args.reference)
    candidate = load_network(args.candidate)
    changed = []
    for name, candidate_tensor in candidate.items():
        if name.startswith(AXIS_PREFIXES) or name not in reference:
            continue
        if not torch.equal(reference[name], candidate_tensor):
            changed.append(name)

    candidate_only_non_axis = [
        name for name in candidate
        if name not in reference and not name.startswith(AXIS_PREFIXES)
    ]
    reference_only_non_axis = [
        name for name in reference
        if name not in candidate and not name.startswith(AXIS_PREFIXES)
    ]
    axis_tensors = [
        name for name in candidate if name.startswith(AXIS_PREFIXES)
    ]

    print(f'axis tensors in candidate: {len(axis_tensors)}')
    print(f'changed shared non-axis tensors: {len(changed)}')
    print(f'candidate-only non-axis tensors: {len(candidate_only_non_axis)}')
    print(
        'reference-only tensors ignored as unused source keys: '
        f'{len(reference_only_non_axis)}')
    if changed:
        print('changed:', ', '.join(changed[:20]))
    if candidate_only_non_axis:
        print('candidate-only:', ', '.join(candidate_only_non_axis[:20]))
    if reference_only_non_axis:
        print('reference-only:', ', '.join(reference_only_non_axis[:20]))

    if changed or candidate_only_non_axis:
        sys.exit(1)
    print('PASS: original TreeLearn tensors are bitwise unchanged.')


if __name__ == '__main__':
    main()
