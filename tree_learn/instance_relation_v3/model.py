"""Point relations plus symmetric ordered height tracks; one outcome classifier."""

import torch
from torch import nn
from torch.nn import functional as F

from tree_learn.instance_relation_v2.model import ConnectionRelationHead, masked_pool, gather


def next_layer(value):
    return torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), dim=1)


def height_tokens(points, valid, weights, encoded, center, bottom, height, bands, minimum_points):
    """Twenty observable geometric cues and frozen-feature-conditioned layer tokens.

    Height is relative to this pair's vertical span, not terrain-normalized height.
    Dual tracks are geometric evidence; they are not a hard two-tree decision.
    """
    level = ((points[..., 2] - bottom[:, None, None]) / height[:, None, None] * bands).long().clamp(0, bands - 1)
    membership = F.one_hot(level, bands).permute(0, 1, 3, 2).bool() & valid[:, :, None]
    sample_count = membership.sum(-1)
    present = sample_count >= minimum_points
    mass_weights = membership.float() * weights[:, :, None]
    mass = mass_weights.sum(-1)
    normalized = mass_weights / mass[..., None].clamp_min(1e-8)
    relative = points[..., :3] - center[:, None, None]
    relative_votes = points[..., 3:5] - center[:, None, None, :2]
    centers = torch.einsum('bksn,bknd->bksd', normalized, relative)
    votes = torch.einsum('bksn,bknd->bksd', normalized, relative_votes)
    means = torch.einsum('bksn,bknd->bksd', normalized, encoded) * present[..., None]
    second = torch.einsum('bksn,bkn->bks', normalized, relative[..., :2].square().sum(-1))
    radius = (second - centers[..., :2].square().sum(-1)).clamp_min(1e-8).sqrt()
    fraction = mass / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    ca, cb = centers[:, 0], centers[:, 1]
    va, vb = votes[:, 0], votes[:, 1]
    pa, pb = present[:, 0], present[:, 1]
    both = pa & pb
    separation = (ca[..., :2] - cb[..., :2]).norm(dim=-1) * both
    vote_separation = (va - vb).norm(dim=-1) * both
    ra, rb = radius[:, 0] * pa, radius[:, 1] * pb
    gap = (separation - ra - rb).clamp_min(0) * both
    track_a, track_b = pa & next_layer(pa), pb & next_layer(pb)
    drift_a = (ca[..., :2] - next_layer(ca)[..., :2]).norm(dim=-1) * track_a
    drift_b = (cb[..., :2] - next_layer(cb)[..., :2]).norm(dim=-1) * track_b
    dual = track_a & track_b
    dual_separation = torch.minimum(separation, next_layer(separation)) * dual
    cross_valid = torch.stack((pa & next_layer(pb), pb & next_layer(pa)), -1)
    cross_distance = torch.stack(((ca[..., :2] - next_layer(cb)[..., :2]).norm(dim=-1),
                                   (cb[..., :2] - next_layer(ca)[..., :2]).norm(dim=-1)), -1)
    continuation = cross_distance.masked_fill(~cross_valid, 1e4).min(-1).values
    continuation = torch.where(cross_valid.any(-1), continuation, torch.zeros_like(continuation))
    position = ((torch.arange(bands, device=points.device, dtype=points.dtype) + .5) / bands)[None].expand(len(points), -1)
    lower_dual = dual_separation * (position <= .5) * torch.exp(-(drift_a + drift_b) / 2)
    geometry = torch.stack((position, pa.float() + pb.float(), both.float(),
        fraction.sum(1), (fraction[:, 0] - fraction[:, 1]).abs(), separation / 4, vote_separation / 4,
        (ra + rb) / 4, (ra - rb).abs() / 4, (ca[..., 2] - cb[..., 2]).abs() / height[:, None] * both,
        gap / 4, cross_valid.any(-1).float(), continuation / 4,
        track_a.float() + track_b.float(), (drift_a + drift_b) / 4, (drift_a - drift_b).abs() / 4,
        dual.float(), dual_separation / 4, lower_dual / 4, both / (1 + ra + rb)), -1)
    tokens = torch.cat((means.sum(1), (means[:, 0] - means[:, 1]).abs(), geometry.clamp(-10, 10)), -1)
    return tokens, present.any(1), geometry


class StructureRelationHead(ConnectionRelationHead):
    def __init__(self, hidden_dim=64, geometric_neighbors=4, vote_neighbors=2, height_bands=8, minimum_layer_points=2):
        super().__init__(hidden_dim, geometric_neighbors, vote_neighbors)
        self.height_bands, self.minimum_layer_points = height_bands, minimum_layer_points
        h = hidden_dim
        self.structure_encoder = nn.Sequential(nn.Linear(2 * h + 20, h), nn.LayerNorm(h), nn.GELU())
        self.track_encoder = nn.Sequential(nn.Conv1d(h, h, 3, padding=1), nn.GELU(),
                                           nn.Conv1d(h, h, 3, padding=1), nn.GELU())
        self.relation = nn.Sequential(nn.Linear(13 * h, 2 * h), nn.LayerNorm(2 * h), nn.GELU(),
                                      nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 6))

    def forward(self, points, valid, weights):
        points, weights = points.float(), weights.float()
        if (points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 38
                or valid.shape != points.shape[:-1] or weights.shape != valid.shape):
            raise ValueError('Require aligned B x 2 x L frozen point samples and masks.')
        if not valid.any(-1).all() or not torch.isfinite(points).all() or not torch.isfinite(weights).all():
            raise ValueError('Each original instance needs finite valid observations.')
        if (weights < 0).any() or not ((weights * valid).sum(-1) > 0).all():
            raise ValueError('Require positive represented mass in both original instances.')
        # Padded observations cannot affect either geometric bounds or encoders.
        points = points.masked_fill(~valid[..., None], 0)
        weights = weights * valid
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
        ca, ea, ia, ma = self.connections(points[:, 0], points[:, 1], a, b, valid[:, 0], valid[:, 1],
                                        za, zb, sa, sb, weights[:, 0])
        cb, eb, ib, mb = self.connections(points[:, 1], points[:, 0], b, a, valid[:, 1], valid[:, 0],
                                        zb, za, sb, sa, weights[:, 1])
        tokens, layer_valid, geometry = height_tokens(points, valid, weights, torch.stack((a, b), 1),
            center, low[:, 2], height, self.height_bands, self.minimum_layer_points)
        encoded = self.structure_encoder(tokens).masked_fill(~layer_valid[..., None], 0)
        tracks = self.track_encoder(encoded.transpose(1, 2)).transpose(1, 2)
        structural = torch.cat(masked_pool(tracks, layer_valid), -1)
        fused = torch.cat((pa + pb, (pa - pb).abs(), la + lb, (la - lb).abs(), (ca + cb) / 2, structural), -1)
        output = self.relation(fused)
        logits = output[:, :3]
        probabilities = logits.softmax(-1)
        # Compatibility fields are derived from this same categorical distribution.
        risk_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], -1)
        return dict(action_logits=logits, risk_logit=risk_logit, utility=probabilities[:, 1] - probabilities[:, 2],
            quality=output[:, 3:6].sigmoid(), edge_logits=(ea, eb), edge_indices=(ia, ib),
            edge_valid=(ma, mb), height_evidence=geometry, height_valid=layer_valid)


def loss(result, targets, settings):
    action = F.cross_entropy(result['action_logits'], targets['action'].long())
    probabilities = result['action_logits'].softmax(-1)
    score = probabilities[:, 1] - settings['destruction_cost'] * probabilities[:, 2]
    real = targets['is_real'].bool()
    positive = score[real & (targets['action'] == 1)]
    negative = score[real & (targets['action'] == 2)]
    ranking = F.softplus(settings['ranking_margin'] + negative[:, None] - positive[None]).mean() \
        if len(positive) and len(negative) else score.sum() * 0
    quality = F.smooth_l1_loss(result['quality'], targets['quality'].float())
    connection_sum, connection_count = score.sum() * 0, 0
    for direction in range(2):
        source = targets['point_labels'][:, direction]
        other = gather(targets['point_labels'][:, 1 - direction, :, None], result['edge_indices'][direction]).squeeze(-1)
        known = result['edge_valid'][direction] & (source[..., None] >= 0) & (other >= 0)
        same = (source[..., None] > 0) & (source[..., None] == other)
        for category in (False, True):
            chosen = known & (same == category)
            if chosen.any():
                connection_sum += F.binary_cross_entropy_with_logits(result['edge_logits'][direction][chosen], same[chosen].float())
                connection_count += 1
    connection = connection_sum / max(connection_count, 1)
    total = action + settings['ranking_weight'] * ranking + settings['quality_weight'] * quality + settings['connection_weight'] * connection
    return total, dict(action_loss=float(action.detach()), ranking_loss=float(ranking.detach()),
        quality_loss=float(quality.detach()), connection_loss=float(connection.detach()))
