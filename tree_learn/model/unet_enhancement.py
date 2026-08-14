"""Lightweight sparse U-Net enhancement modules for controlled ablations.

The modules in this file operate on ``SparseConvTensor.features`` while
preserving coordinates and sparse-convolution metadata.  Every enhancement
uses a learnable residual scale.  With the default zero initialization the
forward pass is an exact identity, which makes old TreeLearn checkpoints a
safe initialization for the tournament experiments.
"""

from collections import defaultdict

import torch
from torch import nn

try:
    import spconv.pytorch as spconv
except ImportError:  # Dense-only unit tests do not require spconv.
    spconv = None


SUPPORTED_UNET_ENHANCEMENTS = (
    'identity',
    'residual_adapter',
    'sparse_se',
    'selective_kernel',
    'hcag',
    'window_attention',
)


def replace_sparse_features(tensor, features):
    """Replace features without changing sparse indices or metadata."""
    if hasattr(tensor, 'replace_feature'):
        return tensor.replace_feature(features)
    tensor.features = features
    return tensor


def batch_global_mean(features, batch_ids, batch_size=None):
    """Mean-pool irregular sparse features independently for every batch."""
    if features.ndim != 2:
        raise ValueError('features must have shape [N, C].')
    if len(features) != len(batch_ids):
        raise ValueError('features and batch_ids must have matching length.')
    if batch_size is None:
        batch_size = (
            int(batch_ids.max().item()) + 1 if len(batch_ids) else 0)
    pooled = features.new_zeros((batch_size, features.shape[1]))
    counts = features.new_zeros((batch_size, 1))
    if len(features):
        pooled.index_add_(0, batch_ids, features)
        counts.index_add_(
            0, batch_ids,
            features.new_ones((len(features), 1)))
    return pooled / counts.clamp_min(1)


def normalize_sparse_heights(indices, eps=1e-6):
    """Normalize sparse Z coordinates to [0, 1] independently per sample.

    TreeLearn/spconv coordinates are stored as ``[batch, z, y, x]``.
    Degenerate single-height samples receive zero height.
    """
    if indices.ndim != 2 or indices.shape[1] < 2:
        raise ValueError('indices must have shape [N, >=2].')
    if len(indices) == 0:
        return torch.empty(
            0, device=indices.device, dtype=torch.float32)
    batch_ids = indices[:, 0].long()
    heights = indices[:, 1].float()
    normalized = torch.zeros_like(heights)
    for batch_id in torch.unique(batch_ids, sorted=True):
        mask = batch_ids == batch_id
        values = heights[mask]
        low = values.min()
        span = values.max() - low
        if float(span) > eps:
            normalized[mask] = (values - low) / span
    return normalized.clamp_(0, 1)


def _stats(values, prefix):
    values = values.detach().float()
    if values.numel() == 0:
        return {f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0}
    return {
        f'{prefix}_mean': float(values.mean().cpu()),
        f'{prefix}_std': float(values.std(unbiased=False).cpu()),
    }


class ResidualScale(nn.Module):
    """Shared zero-initialized residual and diagnostics behavior."""

    def __init__(self, gamma_init=0.0, return_stats=False):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.return_stats = bool(return_stats)
        self.last_stats = {}

    def apply_delta(self, base, delta):
        return base + self.gamma.to(base.dtype) * delta


class ResidualSkipAdapter(ResidualScale):
    """Parameter-matched non-attention skip-fusion control."""

    def __init__(self, channels, hidden_channels, gamma_init=0.0,
                 return_stats=False):
        super().__init__(gamma_init, return_stats)
        self.encoder_projection = nn.Linear(channels, hidden_channels)
        self.decoder_projection = nn.Linear(channels, hidden_channels)
        self.output_projection = nn.Linear(hidden_channels, channels)
        self.activation = nn.ReLU()

    def forward(self, skip_features, decoder_features, indices=None):
        hidden = self.activation(
            self.encoder_projection(skip_features) +
            self.decoder_projection(decoder_features))
        delta = self.output_projection(hidden)
        if self.return_stats:
            self.last_stats = _stats(delta, 'adapter_delta')
        return self.apply_delta(skip_features, delta)


class SparseResidualAdapterEnhancement(ResidualScale):
    """Non-attention bottleneck control used when no decoder pair exists."""

    def __init__(self, channels, hidden_channels, gamma_init=0.0,
                 return_stats=False):
        super().__init__(gamma_init, return_stats)
        self.input_projection = nn.Linear(channels, hidden_channels)
        self.output_projection = nn.Linear(hidden_channels, channels)
        self.activation = nn.ReLU()

    def forward(self, tensor):
        delta = self.output_projection(
            self.activation(self.input_projection(tensor.features)))
        output = self.apply_delta(tensor.features, delta)
        if self.return_stats:
            self.last_stats = _stats(delta, 'adapter_delta')
        return replace_sparse_features(tensor, output)


class HCAGSkipFusion(ResidualScale):
    """Height-conditioned cross-scale multiplicative skip attention."""

    def __init__(self, channels, hidden_channels, gamma_init=0.0,
                 return_stats=False):
        super().__init__(gamma_init, return_stats)
        self.encoder_projection = nn.Linear(channels, hidden_channels)
        self.decoder_projection = nn.Linear(channels, hidden_channels)
        self.height_projection = nn.Linear(1, hidden_channels)
        self.disagreement_projection = nn.Linear(
            hidden_channels, hidden_channels)
        self.attention_projection = nn.Linear(hidden_channels, channels)
        self.activation = nn.ReLU()

    def forward(self, skip_features, decoder_features, indices):
        encoder = self.encoder_projection(skip_features)
        decoder = self.decoder_projection(decoder_features)
        heights = normalize_sparse_heights(indices).to(skip_features.dtype)
        hidden = (
            encoder + decoder +
            self.height_projection(heights[:, None]) +
            self.disagreement_projection((encoder - decoder).abs()))
        attention = torch.sigmoid(
            self.attention_projection(self.activation(hidden)))
        delta = skip_features * (2 * attention - 1)
        if self.return_stats:
            self.last_stats = _stats(attention, 'attention')
        return self.apply_delta(skip_features, delta)


class SparseSEEnhancement(ResidualScale):
    """Batch-isolated squeeze-and-excitation for sparse voxel features."""

    def __init__(self, channels, hidden_channels, gamma_init=0.0,
                 return_stats=False):
        super().__init__(gamma_init, return_stats)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid())

    def forward(self, tensor):
        batch_ids = tensor.indices[:, 0].long()
        pooled = batch_global_mean(
            tensor.features, batch_ids, tensor.batch_size)
        weights = self.excitation(pooled)[batch_ids]
        delta = tensor.features * (weights - 1)
        output = self.apply_delta(tensor.features, delta)
        if self.return_stats:
            self.last_stats = _stats(weights, 'attention')
        return replace_sparse_features(tensor, output)


def selective_kernel_weights(logits):
    """Normalize two selective-kernel branches channel-wise."""
    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError('selective-kernel logits must have shape [B, 2, C].')
    return torch.softmax(logits, dim=1)


class SelectiveKernelEnhancement(ResidualScale):
    """Sparse selective-kernel block with local and dilated branches."""

    def __init__(self, channels, hidden_channels, indice_key,
                 gamma_init=0.0, return_stats=False):
        if spconv is None:
            raise ImportError('spconv is required for selective_kernel.')
        super().__init__(gamma_init, return_stats)
        self.local_branch = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=1, dilation=1,
            bias=False, indice_key=f'{indice_key}_sk_local')
        self.context_branch = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=2, dilation=2,
            bias=False, indice_key=f'{indice_key}_sk_context')
        self.selector = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels * 2))

    def forward(self, tensor):
        local = self.local_branch(tensor)
        context = self.context_branch(tensor)
        batch_ids = tensor.indices[:, 0].long()
        pooled = batch_global_mean(
            local.features + context.features,
            batch_ids, tensor.batch_size)
        logits = self.selector(pooled).view(
            tensor.batch_size, 2, tensor.features.shape[1])
        weights = selective_kernel_weights(logits)[batch_ids]
        fused = (
            weights[:, 0] * local.features +
            weights[:, 1] * context.features)
        output = self.apply_delta(tensor.features, fused - tensor.features)
        if self.return_stats:
            self.last_stats = _stats(weights[:, 0], 'local_attention')
            self.last_stats.update(_stats(
                weights[:, 1], 'context_attention'))
            self.last_stats['attention_sum_error'] = float(
                (weights.sum(dim=1) - 1).abs().max().detach().cpu())
        return replace_sparse_features(tensor, output)


class SparseWindowAttentionEnhancement(ResidualScale):
    """Local self-attention over fixed sparse voxel windows."""

    def __init__(self, channels, window_size=(4, 4, 4), num_heads=4,
                 ffn_ratio=2, max_windows_per_chunk=256, gamma_init=0.0,
                 return_stats=False):
        if channels % num_heads != 0:
            raise ValueError('channels must be divisible by num_heads.')
        super().__init__(gamma_init, return_stats)
        if len(window_size) != 3 or min(window_size) <= 0:
            raise ValueError('window_size must contain three positive ints.')
        self.window_size = tuple(int(value) for value in window_size)
        self.max_windows_per_chunk = int(max_windows_per_chunk)
        if self.max_windows_per_chunk <= 0:
            raise ValueError('max_windows_per_chunk must be positive.')
        self.position_projection = nn.Linear(3, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels))

    def _window_groups(self, indices):
        spatial = indices[:, 1:].long()
        size = torch.tensor(
            self.window_size, device=indices.device, dtype=torch.long)
        window_xyz = torch.div(spatial, size, rounding_mode='floor')
        keys = torch.cat((indices[:, :1].long(), window_xyz), dim=1)
        _, inverse = torch.unique(keys, dim=0, sorted=True,
                                  return_inverse=True)
        groups = defaultdict(list)
        for point_idx, window_idx in enumerate(inverse.detach().cpu().tolist()):
            groups[window_idx].append(point_idx)
        return [groups[key] for key in sorted(groups)]

    def forward(self, tensor):
        if len(tensor.features) == 0:
            return tensor
        groups = self._window_groups(tensor.indices)
        output = tensor.features.clone()
        attention_values = []
        size = tensor.features.new_tensor(self.window_size).clamp_min(1)
        for start in range(0, len(groups), self.max_windows_per_chunk):
            chunk = groups[start:start + self.max_windows_per_chunk]
            max_tokens = max(len(group) for group in chunk)
            window_capacity = (
                self.window_size[0] *
                self.window_size[1] *
                self.window_size[2])
            if max_tokens > window_capacity:
                raise RuntimeError(
                    'Sparse window contains more active voxels than its '
                    'geometric capacity; sparse indices are not unique.')
            padded = tensor.features.new_zeros(
                (len(chunk), max_tokens, tensor.features.shape[1]))
            positions = tensor.features.new_zeros(
                (len(chunk), max_tokens, 3))
            padding_mask = torch.ones(
                (len(chunk), max_tokens), dtype=torch.bool,
                device=tensor.features.device)
            for row, group in enumerate(chunk):
                point_indices = torch.tensor(
                    group, device=tensor.features.device, dtype=torch.long)
                count = len(group)
                padded[row, :count] = tensor.features[point_indices]
                local = tensor.indices[point_indices, 1:].to(
                    tensor.features.dtype)
                local = torch.remainder(local, size) / size
                positions[row, :count] = (
                    local - local.mean(dim=0, keepdim=True))
                padding_mask[row, :count] = False
            tokens = padded + self.position_projection(positions)
            normalized = self.norm1(tokens)
            attended, weights = self.attention(
                normalized, normalized, normalized,
                key_padding_mask=padding_mask,
                need_weights=self.return_stats,
                average_attn_weights=False)
            tokens = tokens + attended
            tokens = tokens + self.ffn(self.norm2(tokens))
            for row, group in enumerate(chunk):
                point_indices = torch.tensor(
                    group, device=tensor.features.device, dtype=torch.long)
                output[point_indices] = tokens[row, :len(group)]
            if self.return_stats and weights is not None:
                valid_queries = ~padding_mask[:, None, :, None]
                valid_keys = ~padding_mask[:, None, None, :]
                attention_values.append(
                    weights.masked_select(valid_queries & valid_keys))
        delta = output - tensor.features
        output = self.apply_delta(tensor.features, delta)
        if self.return_stats:
            values = (
                torch.cat(attention_values)
                if attention_values else tensor.features.new_empty(0))
            self.last_stats = _stats(values, 'attention')
            self.last_stats['num_windows'] = len(groups)
            self.last_stats['max_window_tokens'] = max(map(len, groups))
        return replace_sparse_features(tensor, output)


def hidden_channels(channels, config):
    ratio = float(config.get('hidden_ratio', 0.5))
    minimum = int(config.get('min_hidden_channels', 8))
    return max(minimum, int(round(channels * ratio)))


def enhancement_levels(config, default_levels):
    levels = config.get('levels', default_levels)
    return {int(level) for level in levels}
