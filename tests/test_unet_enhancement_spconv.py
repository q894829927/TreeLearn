import copy
import functools
import unittest

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None

from tree_learn.model.blocks import ResidualBlock, UBlock
from tree_learn.model.tree_learn import TreeLearn


@unittest.skipIf(spconv is None, 'spconv is required')
class UNetEnhancementSpconvTests(unittest.TestCase):

    def _unet(self, enhancement):
        norm = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        return UBlock(
            [8, 16, 24], norm, 2, ResidualBlock, 3,
            enhancement_config=enhancement, total_levels=3).eval()

    def _input(self):
        indices = torch.tensor([
            [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2],
            [0, 4, 4, 4], [0, 8, 8, 8], [0, 12, 12, 12],
            [1, 0, 0, 0], [1, 1, 1, 1], [1, 3, 3, 3],
            [1, 6, 6, 6], [1, 9, 9, 9], [1, 13, 13, 13],
        ], dtype=torch.int32)
        return spconv.SparseConvTensor(
            torch.randn(len(indices), 8), indices, [16, 16, 16], 2)

    def test_identity_config_is_bitwise_equal(self):
        original = self._unet({})
        identity = self._unet({
            'enabled': False, 'type': 'identity', 'levels': []})
        identity.load_state_dict(original.state_dict(), strict=True)
        with torch.no_grad():
            expected = original(self._input())
            actual = identity(self._input())
        self.assertTrue(torch.equal(expected.indices, actual.indices))
        # Fresh random inputs differ, so compare a cloned feature source.
        source = self._input()
        clone = spconv.SparseConvTensor(
            source.features.clone(), source.indices.clone(),
            source.spatial_shape, source.batch_size)
        with torch.no_grad():
            expected = original(source)
            actual = identity(clone)
        self.assertTrue(torch.equal(expected.features, actual.features))

    def test_gamma_zero_only_adds_enhancement_checkpoint_keys(self):
        original = self._unet({})
        enhanced = self._unet({
            'enabled': True,
            'type': 'hcag',
            'levels': [0, 1],
            'residual_gamma_init': 0.0,
        })
        result = enhanced.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(
            'enhancement' in key for key in result.missing_keys))
        source = self._input()
        clone = spconv.SparseConvTensor(
            source.features.clone(), source.indices.clone(),
            source.spatial_shape, source.batch_size)
        with torch.no_grad():
            expected = original(source)
            actual = enhanced(clone)
        torch.testing.assert_close(actual.features, expected.features)
        self.assertTrue(torch.equal(actual.indices, expected.indices))

    def test_finetune_freezes_encoder_and_batchnorm(self):
        model = TreeLearn(
            channels=8,
            num_blocks=3,
            use_upper_anchor=False,
            spatial_shape=[16, 16, 16],
            unet_enhancement={
                'enabled': True,
                'type': 'sparse_se',
                'levels': [0, 1],
                'residual_gamma_init': 0.0,
            },
            unet_finetune={
                'enabled': True,
                'freeze_input_conv': True,
                'freeze_encoder': True,
                'train_decoder_levels': [0, 1],
                'train_heads': True,
                'keep_frozen_batchnorm_eval': True,
            })
        frozen = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                parameters = list(module.parameters(recurse=False))
                if parameters and all(
                        not parameter.requires_grad
                        for parameter in parameters):
                    self.assertFalse(module.training)
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1)
        optimizer.zero_grad()
        loss = sum(parameter.sum() for parameter in model.parameters()
                   if parameter.requires_grad)
        loss.backward()
        optimizer.step()
        for name, expected in frozen.items():
            self.assertTrue(torch.equal(
                model.state_dict()[name], expected), name)


if __name__ == '__main__':
    unittest.main()
