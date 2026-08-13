"""Height-stratified context adapters for frozen TreeLearn features."""

import torch
import torch.nn as nn


def normalize_heights_from_offsets(
        offset_predictions, batch_ids, quantile=0.95, minimum_scale=1.0):
    """Estimate normalized above-ground height from frozen base offsets.

    TreeLearn's base-offset z component points from each point towards the
    estimated tree base. Therefore ``relu(-offset_z)`` is a useful, fully
    prediction-derived height proxy. Normalization is kept batch-local so one
    tile cannot influence another tile in the same mini-batch.
    """
    if offset_predictions.ndim != 2 or offset_predictions.shape[1] < 3:
        raise ValueError('offset_predictions must have shape [N, >=3].')
    if batch_ids.ndim != 1 or len(batch_ids) != len(offset_predictions):
        raise ValueError('batch_ids must have shape [N].')
    if not 0 < quantile <= 1:
        raise ValueError('quantile must be in (0, 1].')

    heights = torch.relu(-offset_predictions[:, 2].detach().float())
    normalized = torch.zeros_like(heights)
    for batch_id in torch.unique(batch_ids, sorted=True):
        mask = batch_ids == batch_id
        if not torch.any(mask):
            continue
        scale = torch.quantile(heights[mask], quantile)
        scale = scale.clamp_min(float(minimum_scale))
        normalized[mask] = (heights[mask] / scale).clamp(0, 1)
    return normalized


def pool_height_tokens(point_features, normalized_heights, batch_ids,
                       num_height_bins):
    """Pool mean/max point features into fixed height tokens per sample."""
    if point_features.ndim != 2:
        raise ValueError('point_features must have shape [N, C].')
    if normalized_heights.shape != (len(point_features),):
        raise ValueError('normalized_heights must have shape [N].')
    if batch_ids.shape != (len(point_features),):
        raise ValueError('batch_ids must have shape [N].')
    if num_height_bins < 2:
        raise ValueError('num_height_bins must be at least 2.')

    if len(point_features) == 0:
        empty_tokens = point_features.new_empty(
            (0, num_height_bins, point_features.shape[1] * 2))
        empty_mask = torch.empty(
            (0, num_height_bins), dtype=torch.bool,
            device=point_features.device)
        empty_indices = torch.empty(
            0, dtype=torch.long, device=point_features.device)
        return empty_tokens, empty_mask, empty_indices, empty_indices

    batch_values, batch_inverse = torch.unique(
        batch_ids.long(), sorted=True, return_inverse=True)
    del batch_values
    bin_indices = torch.floor(
        normalized_heights.float() * num_height_bins).long()
    bin_indices = bin_indices.clamp(0, num_height_bins - 1)

    num_batches = int(batch_inverse.max().item()) + 1
    channels = point_features.shape[1]
    tokens = point_features.new_zeros(
        (num_batches, num_height_bins, channels * 2))
    valid_mask = torch.zeros(
        (num_batches, num_height_bins), dtype=torch.bool,
        device=point_features.device)

    for batch_index in range(num_batches):
        for bin_index in range(num_height_bins):
            mask = (
                (batch_inverse == batch_index) &
                (bin_indices == bin_index))
            if not torch.any(mask):
                continue
            values = point_features[mask]
            tokens[batch_index, bin_index] = torch.cat([
                values.mean(dim=0),
                values.max(dim=0).values,
            ], dim=0)
            valid_mask[batch_index, bin_index] = True

    return tokens, valid_mask, batch_inverse, bin_indices


class HeightTokenMLPMixer(nn.Module):
    """Parameter-matched token-wise MLP control without token attention."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, valid_mask):
        mixed = tokens + self.dropout(self.mlp(self.norm(tokens)))
        return mixed.masked_fill(~valid_mask[..., None], 0)


class HeightTokenAttentionMixer(nn.Module):
    """Single-layer multi-head self-attention across vertical tokens."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads.')
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, valid_mask):
        normalized = self.norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_mask,
            need_weights=False)
        mixed = tokens + self.dropout(attended)
        return mixed.masked_fill(~valid_mask[..., None], 0)


class HeightStratifiedContextAdapter(nn.Module):
    """Predict semantic and base-offset residuals from height context.

    The MLP and attention variants share every component except the token
    mixer. Their mixers differ by only ``hidden_dim`` parameters when biases
    are enabled, providing a close parameter-matched attention control.
    """

    def __init__(self, in_channels=32, hidden_dim=32, num_height_bins=8,
                 adapter_type='attention', num_heads=4, dropout=0.1,
                 height_quantile=0.95, minimum_height_scale=1.0):
        super().__init__()
        if adapter_type not in ('mlp', 'attention'):
            raise ValueError("adapter_type must be 'mlp' or 'attention'.")
        self.adapter_type = adapter_type
        self.num_height_bins = int(num_height_bins)
        self.height_quantile = float(height_quantile)
        self.minimum_height_scale = float(minimum_height_scale)

        self.point_projection = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if adapter_type == 'attention':
            self.token_mixer = HeightTokenAttentionMixer(
                hidden_dim, num_heads, dropout)
        else:
            self.token_mixer = HeightTokenMLPMixer(hidden_dim, dropout)

        self.context_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.semantic_residual_head = nn.Linear(hidden_dim, 2)
        self.offset_residual_head = nn.Linear(hidden_dim, 3)
        self.semantic_residual_scale = nn.Parameter(torch.ones(()))
        self.offset_residual_scale = nn.Parameter(torch.ones(()))

        # Zero heads make the initial predictions exactly equal to the frozen
        # TreeLearn outputs while preserving non-zero head gradients.
        nn.init.zeros_(self.semantic_residual_head.weight)
        nn.init.zeros_(self.semantic_residual_head.bias)
        nn.init.zeros_(self.offset_residual_head.weight)
        nn.init.zeros_(self.offset_residual_head.bias)

    def forward(self, backbone_features, offset_predictions, input_features,
                batch_ids):
        if backbone_features.ndim != 2:
            raise ValueError('backbone_features must have shape [N, C].')
        if input_features.ndim != 2 or len(input_features) != len(
                backbone_features):
            raise ValueError('input_features must have shape [N, F].')
        if input_features.shape[1] < 1:
            raise ValueError('input_features must contain verticality.')

        normalized_heights = normalize_heights_from_offsets(
            offset_predictions,
            batch_ids,
            quantile=self.height_quantile,
            minimum_scale=self.minimum_height_scale)
        verticality = input_features[:, -1].float().clamp(0, 1)
        point_inputs = torch.cat([
            backbone_features,
            verticality[:, None].to(backbone_features.dtype),
            normalized_heights[:, None].to(backbone_features.dtype),
        ], dim=1)
        point_features = self.point_projection(point_inputs)

        tokens, valid_mask, batch_inverse, bin_indices = pool_height_tokens(
            point_features,
            normalized_heights,
            batch_ids,
            self.num_height_bins)
        tokens = self.token_projection(tokens)
        mixed_tokens = self.token_mixer(tokens, valid_mask)
        point_context = mixed_tokens[batch_inverse, bin_indices]
        gate = torch.sigmoid(self.context_gate(torch.cat([
            point_features, point_context], dim=1)))
        adapted_features = point_features + gate * point_context

        semantic_residual = (
            self.semantic_residual_scale *
            self.semantic_residual_head(adapted_features))
        offset_residual = (
            self.offset_residual_scale *
            self.offset_residual_head(adapted_features))
        return {
            'semantic_residual': semantic_residual,
            'offset_residual': offset_residual,
            'normalized_heights': normalized_heights,
            'height_bin_indices': bin_indices,
            'height_token_valid_mask': valid_mask,
        }


def count_trainable_parameters(module):
    """Return the number of trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad)
