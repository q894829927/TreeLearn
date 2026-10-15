"""Candidate-conditioned MLP; no attention, quality filtering or backbone."""

import torch
from torch import nn
import torch.nn.functional as F


class CandidateMaskMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.register_buffer('feature_mean', torch.zeros(32))
        self.register_buffer('feature_scale', torch.ones(32))
        self.point_encoder = nn.Sequential(
            nn.Linear(40, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.candidate_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.mask_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, points, descriptors):
        # Accept [B,N,40]/[B,64] or a single [N,40]/[64] region.
        values = torch.cat([
            (points[..., :32] - self.feature_mean) / self.feature_scale,
            points[..., 32:]], dim=-1)
        desc = (descriptors.reshape(*descriptors.shape[:-1], 2, 32) -
                self.feature_mean) / self.feature_scale
        query = self.candidate_encoder(desc.flatten(start_dim=-2))
        encoded = self.point_encoder(values)
        query = query.unsqueeze(-2).expand_as(encoded)
        return self.mask_head(torch.cat([encoded, query], dim=-1)).squeeze(-1)


def mask_loss(logits, targets, weights):
    """Per-candidate BCE + Dice, then equal weighting of candidates."""
    denom = weights.sum(dim=-1).clamp_min(1e-8)
    bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * weights).sum(dim=-1) / denom
    probability = logits.sigmoid()
    intersection = (weights * probability * targets).sum(dim=-1) / denom
    mass = (weights * (probability + targets)).sum(dim=-1) / denom
    dice = 1 - (2 * intersection + 1e-6) / (mass + 1e-6)
    # An empty target is handled by BCE; Dice has no useful positive gradient.
    dice = torch.where((weights * targets).sum(dim=-1) > 0, dice, 0.0)
    return (bce + dice).mean(), bce.mean(), dice.mean()
