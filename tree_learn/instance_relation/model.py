"""Symmetric relation, union quality, and original-instance quality heads."""

import torch
from torch import nn
from torch.nn import functional as F


class InstanceRelationHead(nn.Module):
    def __init__(self, node_dim, pair_dim, hidden_dim=64):
        super().__init__()
        self.dimensions = dict(node_dim=node_dim, pair_dim=pair_dim, hidden_dim=hidden_dim)
        for name, size in (('node', node_dim), ('pair', pair_dim)):
            self.register_buffer(name + '_mean', torch.zeros(size))
            self.register_buffer(name + '_scale', torch.ones(size))
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.node_quality = nn.Linear(hidden_dim, 1)
        self.relation = nn.Sequential(nn.Linear(2 * hidden_dim + pair_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))

    def encode(self, features):
        return self.node_encoder((features.float() - self.node_mean) / self.node_scale)

    def forward(self, left, right, pair_features):
        a, b = self.encode(left), self.encode(right)
        pair = (pair_features.float() - self.pair_mean) / self.pair_scale
        result = self.relation(torch.cat((a + b, (a - b).abs(), pair), -1))
        return dict(merge_logit=result[:, 0], same_logit=result[:, 1],
                    quality=result[:, 2:5].sigmoid(), gain=result[:, 5].tanh())


def relation_loss(result, merge, same, quality, gain):
    action = F.binary_cross_entropy_with_logits(result['merge_logit'], merge.float())
    connection = F.binary_cross_entropy_with_logits(result['same_logit'], same.float())
    errors = F.smooth_l1_loss(result['quality'], quality.float(), reduction='none')
    union = (errors * errors.new_tensor([1., 1., 2.])).mean()
    improvement = F.smooth_l1_loss(result['gain'], gain.float())
    return action + .5 * connection + 2. * union + improvement, dict(merge=float(action.detach()),
        same=float(connection.detach()), union_quality=float(union.detach()), gain_loss=float(improvement.detach()))
