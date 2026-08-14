from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn

from .unet_enhancement import (
    HCAGSkipFusion,
    ResidualSkipAdapter,
    SparseResidualAdapterEnhancement,
    SUPPORTED_UNET_ENHANCEMENTS,
    SelectiveKernelEnhancement,
    SparseSEEnhancement,
    SparseWindowAttentionEnhancement,
    enhancement_levels,
    hidden_channels,
)


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(
            input.features,
            self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(
            features, input.indices, input.spatial_shape, input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, kernel_size,
                 indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(
                    in_channels, out_channels, kernel_size=1, bias=False))
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=int(kernel_size),
                padding=int((kernel_size - 1) / 2), bias=False,
                indice_key=indice_key),
            norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels, out_channels, kernel_size=int(kernel_size),
                padding=int((kernel_size - 1) / 2), bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape,
            input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        return output.replace_feature(out_feats)


class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, kernel_size,
                 indice_key_id=1, enhancement_config=None, level=0,
                 total_levels=None):
        super().__init__()
        self.nPlanes = nPlanes
        self.level = int(level)
        self.total_levels = int(total_levels or len(nPlanes))
        self.enhancement_config = dict(enhancement_config or {})
        self.enhancement_type = self.enhancement_config.get(
            'type', 'identity')
        if self.enhancement_type not in SUPPORTED_UNET_ENHANCEMENTS:
            raise ValueError(
                f'Unsupported U-Net enhancement: {self.enhancement_type}.')
        self.enhancement_enabled = bool(
            self.enhancement_config.get('enabled', False)) and (
                self.enhancement_type != 'identity')
        defaults = {
            'residual_adapter': (3, 4, 5),
            'sparse_se': (0, 1, 2),
            'selective_kernel': (4, 5),
            'hcag': (3, 4, 5),
            'window_attention': (self.total_levels - 1,),
        }
        self.enhancement_levels = enhancement_levels(
            self.enhancement_config,
            defaults.get(self.enhancement_type, ()))
        self.skip_enhancement = None
        self.post_enhancement = None

        if self.enhancement_enabled and self.level in self.enhancement_levels:
            channels = nPlanes[0]
            hidden = hidden_channels(channels, self.enhancement_config)
            common = {
                'gamma_init': float(self.enhancement_config.get(
                    'residual_gamma_init', 0.0)),
                'return_stats': bool(self.enhancement_config.get(
                    'return_attention_stats', False)),
            }
            if self.enhancement_type == 'residual_adapter':
                if len(nPlanes) > 1:
                    self.skip_enhancement = ResidualSkipAdapter(
                        channels, hidden, **common)
                else:
                    self.post_enhancement = (
                        SparseResidualAdapterEnhancement(
                            channels, hidden, **common))
            elif self.enhancement_type == 'hcag':
                self.skip_enhancement = HCAGSkipFusion(
                    channels, hidden, **common)
            elif self.enhancement_type == 'sparse_se':
                self.post_enhancement = SparseSEEnhancement(
                    channels, hidden, **common)
            elif self.enhancement_type == 'selective_kernel':
                self.post_enhancement = SelectiveKernelEnhancement(
                    channels, hidden,
                    indice_key=f'unet_level{self.level}', **common)
            elif self.enhancement_type == 'window_attention':
                self.post_enhancement = SparseWindowAttentionEnhancement(
                    channels,
                    window_size=tuple(self.enhancement_config.get(
                        'window_size', (4, 4, 4))),
                    num_heads=int(self.enhancement_config.get(
                        'num_heads', 4)),
                    ffn_ratio=float(self.enhancement_config.get(
                        'ffn_ratio', 2)),
                    max_windows_per_chunk=int(self.enhancement_config.get(
                        'max_windows_per_chunk', 256)),
                    **common)

        blocks = OrderedDict({
            f'block{i}': block(
                nPlanes[0], nPlanes[0], norm_fn, kernel_size,
                indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        })
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0], nPlanes[1], kernel_size=2, stride=2,
                    bias=False, indice_key=f'spconv{indice_key_id}'))
            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, kernel_size,
                indice_key_id=indice_key_id + 1,
                enhancement_config=self.enhancement_config,
                level=self.level + 1,
                total_levels=self.total_levels)
            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1], nPlanes[0], kernel_size=2, bias=False,
                    indice_key=f'spconv{indice_key_id}'))
            blocks_tail = OrderedDict()
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(
                    nPlanes[0] * (2 - i), nPlanes[0], norm_fn,
                    kernel_size, indice_key=f'subm{indice_key_id}')
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(
            output.features, output.indices, output.spatial_shape,
            output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            skip_features = identity.features
            if self.skip_enhancement is not None:
                skip_features = self.skip_enhancement(
                    skip_features, output_decoder.features,
                    identity.indices)
            out_feats = torch.cat(
                (skip_features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        if self.post_enhancement is not None:
            output = self.post_enhancement(output)
        return output

    def enhancement_statistics(self):
        statistics = {}
        module = self.skip_enhancement or self.post_enhancement
        if module is not None:
            for name, value in module.last_stats.items():
                statistics[f'level{self.level}_{name}'] = value
            statistics[f'level{self.level}_gamma'] = float(
                module.gamma.detach().float().cpu())
        if hasattr(self, 'u'):
            statistics.update(self.u.enhancement_statistics())
        return statistics

    def set_decoder_trainable(self, levels):
        levels = {int(level) for level in levels}
        if self.level in levels and hasattr(self, 'deconv'):
            for module in (self.deconv, self.blocks_tail):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        if hasattr(self, 'u'):
            self.u.set_decoder_trainable(levels)

    def set_enhancement_trainable(self):
        for module in (self.skip_enhancement, self.post_enhancement):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        if hasattr(self, 'u'):
            self.u.set_enhancement_trainable()
