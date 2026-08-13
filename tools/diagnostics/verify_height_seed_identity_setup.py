"""Verify D2 seed-identity loss wiring before a full adapter training run."""

import argparse
import logging

import torch

from tree_learn.model import TreeLearn
from tree_learn.util import get_config, load_checkpoint


LOGGER = logging.getLogger('verify_height_seed_identity_setup')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--reference', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the integration diagnostic.')

    config = get_config(args.config)
    weight = float(config.model.height_seed_identity_weight)
    if not config.model.use_height_context_adapter or weight <= 0:
        raise ValueError(
            'Config must enable a height adapter and a positive identity weight.')

    model = TreeLearn(**config.model).cuda()
    load_checkpoint(args.reference, LOGGER, model)
    model.eval()

    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith('height_context_adapter.')
    }

    # Point 0 is an official base seed whose adapted tree probability is
    # deliberately damaged. The other points exercise non-seed exclusions.
    base_logits = torch.tensor([
        [2.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.0, 0.0]],
        device='cuda')
    adapted_logits = base_logits.clone()
    adapted_logits[0] = torch.tensor([-1.0, 1.0], device='cuda')
    adapted_logits.requires_grad_(True)
    base_offsets = torch.tensor([
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
        [0.0, 0.0, -5.0], [0.0, 0.0, -1.0]], device='cuda')
    adapted_offsets = base_offsets.clone().requires_grad_(True)
    input_feats = torch.zeros(4, int(config.model.dim_feat), device='cuda')
    input_feats[:, -1] = torch.tensor([0.8, 0.8, 0.8, 0.4], device='cuda')
    semantic_labels = torch.zeros(4, dtype=torch.long, device='cuda')
    offset_labels = torch.zeros(4, 3, device='cuda')
    valid = torch.ones(4, dtype=torch.bool, device='cuda')

    model_output = {
        'base_semantic_prediction_logits': base_logits,
        'semantic_prediction_logits': adapted_logits,
        'base_offset_predictions': base_offsets,
        'offset_predictions': adapted_offsets,
    }
    loss, loss_dict = model.get_loss(
        model_output=model_output,
        semantic_labels=semantic_labels,
        offset_labels=offset_labels,
        masks_off=valid,
        masks_sem=valid,
        input_feats=input_feats)
    if 'height_seed_identity_loss' not in loss_dict:
        raise RuntimeError('D2 identity loss is absent from model.get_loss().')
    identity_loss = loss_dict['height_seed_identity_loss']
    if not torch.isfinite(identity_loss) or float(identity_loss) <= 0:
        raise RuntimeError('D2 identity loss must be finite and positive.')

    identity_loss.backward()
    if adapted_logits.grad is None or float(
            adapted_logits.grad[0].abs().sum()) <= 0:
        raise RuntimeError('Damaged base seed received no semantic gradient.')
    if float(adapted_logits.grad[2:].abs().sum()) != 0:
        raise RuntimeError('Excluded non-seeds received an identity gradient.')

    changed_original = [
        name for name, tensor in model.state_dict().items()
        if name in original_state and not torch.equal(
            original_state[name], tensor.detach().cpu())
    ]
    if changed_original:
        raise RuntimeError(
            'Original TreeLearn tensors changed during diagnostic: ' +
            ', '.join(changed_original[:20]))

    print('config:', args.config)
    print('identity weight:', weight)
    print('scaled identity loss:', float(identity_loss))
    print('damaged seed gradient L1:', float(
        adapted_logits.grad[0].abs().sum()))
    print('changed original tensors:', len(changed_original))
    print('PASS: D2 seed-identity loss wiring verified.')


if __name__ == '__main__':
    main()
