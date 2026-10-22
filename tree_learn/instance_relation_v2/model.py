"""Symmetric point/connection encoder; official features are immutable inputs."""

import torch
from torch import nn
from torch.nn import functional as F


def masked_pool(features, mask, weights=None):
    weights = mask.float() if weights is None else weights.float() * mask
    mean = (features * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    maximum = features.masked_fill(~mask[..., None], -1e4).max(-2).values
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    return mean, maximum


def gather(values, indices):
    b, n, k = indices.shape
    return values.gather(1, indices.reshape(b, n * k)[..., None].expand(-1, -1, values.shape[-1])).reshape(b, n, k, -1)


class ConnectionRelationHead(nn.Module):
    def __init__(self, hidden_dim=64, geometric_neighbors=4, vote_neighbors=2):
        super().__init__()
        self.settings = dict(hidden_dim=hidden_dim, geometric_neighbors=geometric_neighbors, vote_neighbors=vote_neighbors)
        self.register_buffer('feature_mean', torch.zeros(32))
        self.register_buffer('feature_scale', torch.ones(32))
        h = hidden_dim
        self.point_encoder = nn.Sequential(nn.Linear(42, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.connection_encoder = nn.Sequential(nn.Linear(2 * h + 10, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.connection_classifier = nn.Linear(h, 1)
        # Sum/absolute difference of mean/max and lower point pools: 6h.
        # Four connection means plus their global maximum: 5h.
        self.relation = nn.Sequential(nn.Linear(11 * h, 2 * h), nn.LayerNorm(2 * h), nn.GELU(),
                                      nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 8))

    def features(self, points, valid, center, bottom, height):
        xyz, votes = points[..., :3], points[..., 3:5]
        own_low = xyz[..., 2].masked_fill(~valid, float('inf')).min(1).values
        own_high = xyz[..., 2].masked_fill(~valid, -float('inf')).max(1).values
        own_z = (xyz[..., 2] - own_low[:, None]) / (own_high - own_low).clamp_min(.1)[:, None]
        shared_z = (xyz[..., 2] - bottom[:, None]) / height[:, None]
        geometry = torch.cat(((xyz - center[:, None]) / points.new_tensor([5., 5., 10.]),
            (votes - xyz[..., :2]) / 4., (votes - center[:, None, :2]) / 4.,
            points[..., 5:6], own_z[..., None], shared_z[..., None]), -1)
        frozen = ((points[..., 6:] - self.feature_mean) / self.feature_scale.clamp_min(.05)).clamp(-10, 10)
        encoded = self.point_encoder(torch.cat((geometry.clamp(-10, 10), frozen), -1))
        return encoded, own_z, shared_z

    def neighbors(self, left, right, right_valid):
        scale = left.new_tensor([1., 1., 2.])
        geometry = torch.cdist(left[..., :3] / scale, right[..., :3] / scale,
                               compute_mode='donot_use_mm_for_euclid_dist')
        votes = torch.cdist(left[..., 3:5], right[..., 3:5], compute_mode='donot_use_mm_for_euclid_dist')
        kg = min(self.settings['geometric_neighbors'], right.shape[1])
        kv = min(self.settings['vote_neighbors'], right.shape[1])
        first = geometry.masked_fill(~right_valid[:, None], 1e6).topk(kg, largest=False).indices
        second = votes.masked_fill(~right_valid[:, None], 1e6).topk(kv, largest=False).indices
        indices = torch.cat((first, second), -1)
        duplicate = indices[..., :, None] == indices[..., None, :]
        earlier = torch.ones(indices.shape[-1], indices.shape[-1], dtype=torch.bool, device=left.device).tril(-1)
        unique = ~(duplicate & earlier).any(-1)
        return indices, unique

    def connections(self, left, right, a, b, left_valid, right_valid, own_a, own_b, shared_a, shared_b, weights):
        indices, unique = self.neighbors(left, right, right_valid)
        neighbor_points, neighbor_encoded = gather(right, indices), gather(b, indices)
        neighbor_valid = gather(right_valid[..., None].float(), indices).squeeze(-1).bool()
        mask = left_valid[..., None] & neighbor_valid & unique
        lower_a = own_a[..., None] <= .25
        lower_b = gather(own_b[..., None], indices).squeeze(-1) <= .25
        delta = left[:, :, None, :3] - neighbor_points[..., :3]
        vote_delta = left[:, :, None, 3:5] - neighbor_points[..., 3:5]
        distance, vote_distance = delta.norm(dim=-1), vote_delta.norm(dim=-1)
        height = (shared_a[..., None] + gather(shared_b[..., None], indices).squeeze(-1)) / 2
        geometric = torch.cat((delta.abs() / left.new_tensor([1., 1., 2.]), vote_delta.abs() / 4.,
            distance.log1p()[..., None], vote_distance.log1p()[..., None],
            (lower_a & lower_b).float()[..., None], (lower_a | lower_b).float()[..., None], height[..., None]), -1)
        left_encoded = a[:, :, None].expand_as(neighbor_encoded)
        encoded = self.connection_encoder(torch.cat((left_encoded + neighbor_encoded,
            (left_encoded - neighbor_encoded).abs(), geometric.clamp(-10, 10)), -1))
        logits = self.connection_classifier(encoded).squeeze(-1)
        edge_weights = weights[..., None].expand_as(logits) * mask
        flat = encoded.flatten(1, 2)
        flat_mask, flat_weights = mask.flatten(1), edge_weights.flatten(1)
        masks = (flat_mask, (mask & (distance <= 1.)).flatten(1),
            (mask & (delta[..., 2].abs() >= .5) & (delta[..., :2].norm(dim=-1) <= .75)).flatten(1),
            (mask & lower_a & lower_b).flatten(1))
        pools = [masked_pool(flat, chosen, flat_weights)[0] for chosen in masks]
        maximum = masked_pool(flat, flat_mask)[1]
        return torch.cat((*pools, maximum), -1), logits, indices, mask

    def forward(self, points, valid, weights):
        # points B x 2 x L x 38; GT never enters the forward interface.
        points, weights = points.float(), weights.float()
        if points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 38:
            raise ValueError('Expected two aligned local point sets with 32 frozen feature channels.')
        if not valid.any(-1).all():
            raise ValueError('Each original instance needs at least one valid point.')
        xyz, mask = points[..., :3].flatten(1, 2), valid.flatten(1, 2)
        low = xyz.masked_fill(~mask[..., None], float('inf')).min(1).values
        high = xyz.masked_fill(~mask[..., None], -float('inf')).max(1).values
        center, height = (low + high) / 2, (high[:, 2] - low[:, 2]).clamp_min(.1)
        a, za, sa = self.features(points[:, 0], valid[:, 0], center, low[:, 2], height)
        b, zb, sb = self.features(points[:, 1], valid[:, 1], center, low[:, 2], height)
        pa = torch.cat(masked_pool(a, valid[:, 0], weights[:, 0]), -1)
        pb = torch.cat(masked_pool(b, valid[:, 1], weights[:, 1]), -1)
        la = masked_pool(a, valid[:, 0] & (za <= .25), weights[:, 0])[0]
        lb = masked_pool(b, valid[:, 1] & (zb <= .25), weights[:, 1])[0]
        ca, logits_a, indices_a, mask_a = self.connections(points[:, 0], points[:, 1], a, b,
            valid[:, 0], valid[:, 1], za, zb, sa, sb, weights[:, 0])
        cb, logits_b, indices_b, mask_b = self.connections(points[:, 1], points[:, 0], b, a,
            valid[:, 1], valid[:, 0], zb, za, sb, sa, weights[:, 1])
        fused = torch.cat((pa + pb, (pa - pb).abs(), la + lb, (la - lb).abs(), (ca + cb) / 2), -1)
        result = self.relation(fused)
        return dict(action_logits=result[:, :3], risk_logit=result[:, 3], utility=result[:, 4].tanh(),
            quality=result[:, 5:8].sigmoid(), edge_logits=(logits_a, logits_b),
            edge_indices=(indices_a, indices_b), edge_valid=(mask_a, mask_b))


def loss(result, targets):
    action = F.cross_entropy(result['action_logits'], targets['action'].long())
    risk = F.binary_cross_entropy_with_logits(result['risk_logit'], targets['risk'].float())
    value = F.smooth_l1_loss(result['utility'], targets['utility'].float())
    quality = F.smooth_l1_loss(result['quality'], targets['quality'].float())
    connection_sum = result['utility'].sum() * 0
    connection_count = 0
    for direction in range(2):
        indices = result['edge_indices'][direction]
        source = targets['point_labels'][:, direction]
        destination = targets['point_labels'][:, 1 - direction]
        other = gather(destination[..., None], indices).squeeze(-1)
        known = result['edge_valid'][direction] & (source[..., None] >= 0) & (other >= 0)
        same = (source[..., None] > 0) & (source[..., None] == other)
        # Balance positive/negative connections when both exist; unknown never becomes a negative.
        for category in (False, True):
            chosen = known & (same == category)
            if chosen.any():
                connection_sum = connection_sum + F.binary_cross_entropy_with_logits(
                    result['edge_logits'][direction][chosen], same[chosen].float())
                connection_count += 1
    connection = connection_sum / max(connection_count, 1)
    total = action + risk + 2 * value + .25 * quality + .5 * connection
    return total, dict(action_loss=float(action.detach()), risk_loss=float(risk.detach()),
        utility_loss=float(value.detach()), quality_loss=float(quality.detach()), connection_loss=float(connection.detach()))
