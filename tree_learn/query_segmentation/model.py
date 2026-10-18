"""Joint point embedding, point queries, instance decoder and quality head."""

import math

import torch
from torch import nn


@torch.no_grad()
def farthest(values, available, count, first_scores=None, selected=None):
    available = available.clone()
    minimum = torch.full((len(values),), float('inf'), device=values.device)
    chosen = [] if selected is None else list(selected)
    for row in chosen:
        minimum = torch.minimum(minimum, (values - values[row]).square().sum(-1))
        available[row] = False
    added = []
    for _ in range(min(int(count), int(available.sum()))):
        if not chosen and first_scores is not None:
            row = int(first_scores.masked_fill(~available, -float('inf')).argmax())
        else:
            row = int(minimum.masked_fill(~available, -1.).argmax())
        chosen.append(row)
        added.append(row)
        available[row] = False
        minimum = torch.minimum(minimum, (values - values[row]).square().sum(-1))
    return added


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, feedforward):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(dim, feedforward), nn.GELU(), nn.Linear(feedforward, dim))

    def forward(self, queries, memory):
        update = self.cross(queries, memory, memory, need_weights=False)[0]
        queries = self.norms[0](queries + update)
        update = self.self_attention(queries, queries, queries, need_weights=False)[0]
        queries = self.norms[1](queries + update)
        return self.norms[2](queries + self.ffn(queries))


class QueryInstanceDecoder(nn.Module):
    def __init__(self, dim=128, layers=3, heads=4, feedforward=512,
                 num_queries=256, coverage_queries=32, coordinate_scale=16., vote_sigma=1.):
        super().__init__()
        if (heads <= 0 or dim <= 0 or layers < 1 or feedforward <= 0 or
                not 0 < coverage_queries <= num_queries or dim % heads or
                coordinate_scale <= 0 or vote_sigma <= 0):
            raise ValueError('Invalid query budget or attention dimensions.')
        self.dim, self.num_queries, self.coverage_queries = dim, num_queries, coverage_queries
        self.coordinate_scale, self.vote_sigma = float(coordinate_scale), float(vote_sigma)
        self.register_buffer('feature_mean', torch.zeros(32))
        self.register_buffer('feature_scale', torch.ones(32))
        self.point_encoder = nn.Sequential(nn.Linear(38, dim), nn.LayerNorm(dim), nn.GELU(),
                                           nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.embedding = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 5))
        self.foreground = nn.Linear(dim, 1)
        self.memory_projection = nn.Linear(dim, dim)
        self.query_projection = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([DecoderLayer(dim, heads, feedforward) for _ in range(layers)])
        self.query_mask = nn.Linear(dim, dim)
        self.point_mask = nn.Linear(dim, dim)
        # A bounded base-vote prior makes the initial masks spatially meaningful.
        # Queries can shift its center, change its spread, or suppress its weight.
        self.vote_geometry = nn.Linear(dim, 4)
        nn.init.zeros_(self.vote_geometry.weight)
        nn.init.zeros_(self.vote_geometry.bias)
        nn.init.normal_(self.query_mask.weight, std=.01)
        nn.init.zeros_(self.query_mask.bias)
        # The score can inspect both the query and its predicted soft-mask pool.
        self.quality = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def encode(self, inputs):
        normalized = torch.cat([(inputs[..., :32] - self.feature_mean) / self.feature_scale,
                                 inputs[..., 32:]], dim=-1)
        return self.point_encoder(normalized)

    @torch.no_grad()
    def select_queries(self, inputs, encoded, core, progress):
        if not torch.any(core):
            raise ValueError('The query core is empty.')
        alpha = float(max(0., min(1., progress)))
        embedded = self.embedding(encoded)
        xyz = inputs[:, 32:35]
        geometry = xyz / xyz.std(dim=0, unbiased=False).clamp_min(.1)
        embedding = embedded / embedded.std(dim=0, unbiased=False).clamp_min(.1)
        metric = torch.cat([math.sqrt(1 - alpha) * geometry, math.sqrt(alpha) * embedding], dim=-1)
        probability = ((1 - alpha) * inputs[:, -1] +
                       alpha * self.foreground(encoded).squeeze(-1).sigmoid())
        foreground = core & (probability >= .5)
        count = min(self.num_queries, int(core.sum()))
        first = farthest(metric, foreground, min(count, self.num_queries - self.coverage_queries),
                         first_scores=probability)
        # Fill unused foreground slots with spatial coverage queries.
        second = farthest(geometry, core, count - len(first), first_scores=probability, selected=first)
        return torch.tensor(first + second, dtype=torch.long, device=inputs.device)

    def decode(self, token_inputs, core, progress=1.):
        encoded = self.encode(token_inputs)
        selected = self.select_queries(token_inputs, encoded, core, progress)
        queries = self.query_projection(encoded[selected]).unsqueeze(0)
        memory = self.memory_projection(encoded).unsqueeze(0)
        projected, geometry = [], []
        for layer in self.layers:
            queries = layer(queries, memory)
            features = queries.squeeze(0)
            projected.append(self.query_mask(features))
            parameters = self.vote_geometry(features)
            geometry.append(dict(center=token_inputs[selected, 35:37] +
                                 1.5 * parameters[:, :2].tanh() / self.coordinate_scale,
                                 sigma=self.vote_sigma * (.7 * parameters[:, 2].tanh()).exp(),
                                 strength=2 * parameters[:, 3].sigmoid()))
        token_masks = self.mask_logits(projected[-1], self.point_mask(encoded), token_inputs, geometry[-1])
        soft = token_masks.sigmoid()
        pooled = soft @ encoded / soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
        quality_logits = self.quality(torch.cat([queries.squeeze(0), pooled], dim=-1)).squeeze(-1)
        return dict(queries=projected, geometry=geometry, query_indices=selected, quality_logits=quality_logits)

    def mask_logits(self, query, features, inputs, geometry=None):
        logits = query @ features.T / math.sqrt(self.dim)
        if geometry is not None:
            delta = (inputs[None, :, 35:37] - geometry['center'][:, None, :]) * self.coordinate_scale
            squared = delta.square().sum(-1) / geometry['sigma'][:, None].square()
            prior = (2 - .5 * squared).clamp(-4, 2)
            logits = logits + geometry['strength'][:, None] * prior
        return logits

    def point_predictions(self, inputs, queries, geometry=None):
        encoded = self.encode(inputs)
        features = self.point_mask(encoded)
        if geometry is None:
            geometry = [None] * len(queries)
        return dict(masks=[self.mask_logits(query, features, inputs, prior) for query, prior in zip(queries, geometry)],
                    embedding=self.embedding(encoded), foreground=self.foreground(encoded).squeeze(-1))

    def forward(self, token_inputs, core, point_inputs, progress=1.):
        result = self.decode(token_inputs, core, progress)
        result.update(self.point_predictions(point_inputs, result['queries'], result['geometry']))
        return result
