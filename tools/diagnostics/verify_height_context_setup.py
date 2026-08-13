"""Verify frozen TreeLearn height-adapter initialization and optimizer scope."""

import argparse
import logging
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tree_learn.model import TreeLearn
from tree_learn.util import build_optimizer, get_config, load_checkpoint


LOGGER = logging.getLogger('verify_height_context_setup')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--reference', required=True)
    return parser.parse_args()


def clone_original_state(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith('height_context_adapter.')
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the integration diagnostic.')

    config = get_config(args.config)
    if not config.model.use_height_context_adapter:
        raise ValueError('Config does not enable the height-context adapter.')

    model = TreeLearn(**config.model).cuda()
    load_checkpoint(args.reference, LOGGER, model)
    model.eval()

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name for name in trainable_names
        if not name.startswith('height_context_adapter.')
    ]
    if unexpected:
        raise RuntimeError(
            'Unexpected trainable original parameters: ' +
            ', '.join(unexpected))

    reference = torch.load(args.reference, map_location='cpu')['net']
    candidate = model.state_dict()
    candidate_only = sorted(set(candidate) - set(reference))
    non_height_candidate_only = [
        name for name in candidate_only
        if not name.startswith('height_context_adapter.')
    ]
    if non_height_candidate_only:
        raise RuntimeError(
            'Candidate-only non-height tensors: ' +
            ', '.join(non_height_candidate_only))
    changed_shared = [
        name for name in sorted(set(candidate) & set(reference))
        if not torch.equal(candidate[name].detach().cpu(), reference[name])
    ]
    if changed_shared:
        raise RuntimeError(
            'Shared checkpoint tensors changed during load: ' +
            ', '.join(changed_shared[:20]))

    num_points = 128
    channels = int(config.model.channels)
    features = torch.randn(num_points, channels, device='cuda')
    v2p_map = torch.arange(num_points, device='cuda')
    input_features = torch.rand(num_points, 1, device='cuda')
    batch_ids = torch.tensor(
        [0] * (num_points // 2) + [1] * (num_points // 2),
        device='cuda')
    coords = torch.randn(num_points, 3, device='cuda')
    sparse_output = SimpleNamespace(features=features)

    with torch.no_grad():
        base_semantic = model.semantic_linear(features)
        base_offsets = model.offset_linear(features)
        output = model.forward_head(
            sparse_output,
            v2p_map,
            coords=coords,
            input_feats=input_features,
            batch_ids=batch_ids)
    if not torch.equal(output['semantic_prediction_logits'], base_semantic):
        raise RuntimeError('Initial semantic output is not bitwise baseline.')
    if not torch.equal(output['offset_predictions'], base_offsets):
        raise RuntimeError('Initial offset output is not bitwise baseline.')

    before_step = clone_original_state(model)
    before_height = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith('height_context_adapter.')
    }
    optimizer = build_optimizer(model, config.optimizer)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if optimizer_ids != expected_ids:
        raise RuntimeError('Optimizer parameters do not match trainable adapter.')

    model.train()
    optimizer.zero_grad()
    output = model.forward_head(
        sparse_output,
        v2p_map,
        coords=coords,
        input_feats=input_features,
        batch_ids=batch_ids)
    semantic_targets = torch.randint(0, 2, (num_points,), device='cuda')
    offset_targets = torch.randn(num_points, 3, device='cuda')
    loss = (
        50 * F.cross_entropy(
            output['semantic_prediction_logits'].float(), semantic_targets) +
        F.smooth_l1_loss(
            output['offset_predictions'].float(), offset_targets))
    loss.backward()
    optimizer.step()

    after_step = clone_original_state(model)
    changed_original = [
        name for name in before_step
        if not torch.equal(before_step[name], after_step[name])
    ]
    if changed_original:
        raise RuntimeError(
            'Frozen original tensors changed after optimizer step: ' +
            ', '.join(changed_original[:20]))
    changed_height = [
        name for name, parameter in model.named_parameters()
        if name in before_height and not torch.equal(
            before_height[name], parameter.detach().cpu())
    ]
    if not changed_height:
        raise RuntimeError('No height-adapter tensor changed after one step.')

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad)
    print('config:', args.config)
    print('adapter type:', config.model.height_context_type)
    print('trainable parameters:', parameter_count)
    print('candidate-only height tensors:', len(candidate_only))
    print('changed shared tensors during load:', len(changed_shared))
    print('changed original tensors after one step:', len(changed_original))
    print('changed height tensors after one step:', len(changed_height))
    print('PASS: exact initialization and frozen optimizer scope verified.')


if __name__ == '__main__':
    main()