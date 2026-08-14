import unittest

import torch
import torch.nn as nn

from tree_learn.util.train import build_optimizer


class ToyTournamentModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.unet = nn.Module()
        self.unet.skip_enhancement = nn.Linear(4, 4)
        self.unet.blocks_tail = nn.Linear(4, 4)
        self.semantic_linear = nn.Linear(4, 2)
        self.offset_linear = nn.Linear(4, 3)
        self.encoder = nn.Linear(4, 4)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False


class UNetFinetuneOptimizerTests(unittest.TestCase):

    def test_parameter_groups_use_controlled_learning_rates(self):
        model = ToyTournamentModel()
        optimizer = build_optimizer(model, {
            'type': 'AdamW',
            'lr': 0.001,
            'weight_decay': 0.001,
            'paramwise': {
                'attention_lr': 0.001,
                'head_lr': 0.002,
                'decoder_lr': 0.0001,
            },
        })
        groups = {
            group['group_name']: group['lr']
            for group in optimizer.param_groups
        }
        self.assertEqual(groups, {
            'enhancement': 0.001,
            'head': 0.002,
            'decoder': 0.0001,
        })
        names = optimizer.parameter_names_by_group
        self.assertTrue(all(
            'skip_enhancement' in name
            for name in names['enhancement']))
        self.assertFalse(names['other'])

    def test_legacy_optimizer_config_remains_supported(self):
        model = nn.Linear(3, 2)
        optimizer = build_optimizer(model, {
            'type': 'AdamW',
            'lr': 0.003,
            'weight_decay': 0.001,
        })
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.003)


if __name__ == '__main__':
    unittest.main()
