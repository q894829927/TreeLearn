"""Verify that a height-adapter checkpoint leaves original TreeLearn intact."""

import argparse

import torch


def load_network(path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'net' not in checkpoint:
        raise KeyError(f'{path} does not contain a net state dict.')
    return checkpoint['net']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()

    reference = load_network(args.reference)
    candidate = load_network(args.candidate)
    height_keys = sorted(
        key for key in candidate
        if key.startswith('height_context_adapter.'))
    if not height_keys:
        raise RuntimeError('Candidate contains no height-context tensors.')

    changed_shared = [
        key for key in sorted(set(reference) & set(candidate))
        if not key.startswith('height_context_adapter.') and
        not torch.equal(reference[key], candidate[key])
    ]
    candidate_only_non_height = sorted(
        key for key in set(candidate) - set(reference)
        if not key.startswith('height_context_adapter.'))
    missing_reference = sorted(set(reference) - set(candidate))

    print('height tensors in candidate:', len(height_keys))
    print('changed shared non-height tensors:', len(changed_shared))
    print('candidate-only non-height tensors:', len(candidate_only_non_height))
    print('reference-only tensors ignored as unused source keys:',
          len(missing_reference))
    if changed_shared:
        print('first changed shared tensors:', changed_shared[:20])
    if candidate_only_non_height:
        print('candidate-only non-height tensors:',
              candidate_only_non_height[:20])
    if changed_shared or candidate_only_non_height:
        raise RuntimeError('Frozen TreeLearn checkpoint verification failed.')
    print('PASS: original TreeLearn tensors are bitwise unchanged.')


if __name__ == '__main__':
    main()