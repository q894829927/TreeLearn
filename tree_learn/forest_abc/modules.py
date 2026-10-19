"""Small adapters around the attributed geometry and region-gated query modules."""

import torch
from torch import nn

from .vendor.geometry import GeoMixer


class GeometryAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mixer = GeoMixer(channels, channels)
        self.strength = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features, neighbors):
        local = self.mixer(features.T.unsqueeze(0), neighbors.T.unsqueeze(0))
        return features + self.strength.sigmoid() * local.squeeze(0).T


class RegionGatedQueryUpdate(nn.Module):
    """P3Former-inspired region summation and dynamic query gates.

    Inputs are QxD queries, NxD features, QxN raw mask logits. The hard mask
    intentionally has no mask gradient. Unlike copying an entire P3Former head,
    this adapter retains the surrounding forest decoder and handles empty masks.
    """
    def __init__(self, dim):
        super().__init__()
        self.query_parameters = nn.Linear(dim, 2 * dim)
        self.region_parameters = nn.Linear(dim, 2 * dim)
        self.input_gate = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.update_gate = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.query_norm = nn.LayerNorm(dim)
        self.region_norm = nn.LayerNorm(dim)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.strength = nn.Parameter(torch.tensor(-2.0))

    def forward(self, queries, features, mask_logits):
        region = (mask_logits.detach() > 0).to(features.dtype)
        populated = region.any(-1, keepdim=True)
        pooled = region @ features  # Sum, as in MFA; do not silently change to mean.
        query_in, query_out = self.query_parameters(queries).chunk(2, -1)
        region_in, region_out = self.region_parameters(pooled).chunk(2, -1)
        interaction = query_in * region_in
        updated = (self.update_gate(interaction).sigmoid() * self.query_norm(query_out)
                   + self.input_gate(interaction).sigmoid() * self.region_norm(region_out))
        updated = self.output(updated)
        blended = queries + self.strength.sigmoid() * (updated - queries)
        return torch.where(populated, blended, queries)


@torch.no_grad()
def farthest_points(values, count):
    """No custom FPS extension; deterministic start and unique selected indices."""
    count = min(int(count), len(values))
    result = torch.empty(count, dtype=torch.long, device=values.device)
    distance = torch.full((len(values),), float('inf'), device=values.device)
    current = ((values - values.mean(0)) ** 2).sum(-1).argmax()
    for index in range(count):
        result[index] = current
        distance = torch.minimum(distance, ((values - values[current]) ** 2).sum(-1))
        distance[result[:index + 1]] = -1
        current = distance.argmax()
    return result
