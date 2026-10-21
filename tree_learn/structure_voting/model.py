"""Small graph branch. The official feature, semantic and offset paths stay frozen."""

import numpy as np
import torch
from torch import nn

from .graph import LocalSupportGraph


class StructureVoteBranch(nn.Module):
    def __init__(self, channels=32, hidden_dim=32, neighbors=8, support_voxel_size=.4,
                 radius=1.2, query_chunk_size=8192, max_delta_xy=1.5,
                 tree_conf_thresh=.5, vote_scale=4., enabled=True):
        super().__init__()
        if (min(channels, hidden_dim, neighbors, query_chunk_size) < 1 or
                min(support_voxel_size, radius, max_delta_xy, vote_scale) <= 0 or
                not 0 < tree_conf_thresh < 1):
            raise ValueError('Invalid structure-vote dimensions or geometry.')
        self.neighbors, self.radius = neighbors, radius
        self.support_voxel_size, self.query_chunk_size = support_voxel_size, query_chunk_size
        self.max_delta_xy, self.tree_conf_thresh, self.vote_scale = max_delta_xy, tree_conf_thresh, vote_scale
        self.projection = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, hidden_dim), nn.GELU())
        self.connection = nn.Sequential(nn.Linear(2 * hidden_dim + 8, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 1))
        self.correction = nn.Sequential(nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 2))
        # Zero only the final projection: exact identity, live gradients from step one.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def connected_correction(self, features, xyz, offsets, probabilities, rows, neighbors, valid):
        query = self.projection(features[rows].float())
        support = self.projection(features[neighbors].float())
        relative = (xyz[neighbors] - xyz[rows, None]) / self.radius
        votes = xyz[:, :2] + offsets[:, :2]
        vote_difference = (votes[neighbors] - votes[rows, None]) / self.vote_scale
        edge_input = torch.cat((query[:, None].expand_as(support), support, relative,
            torch.linalg.vector_norm(relative, dim=-1, keepdim=True), vote_difference,
            probabilities[rows, None, None].expand(-1, self.neighbors, 1),
            probabilities[neighbors].unsqueeze(-1)), dim=-1)
        logits = self.connection(edge_input).squeeze(-1)
        weight = logits.sigmoid() * valid.float()
        # An implicit self edge of weight 1 keeps isolated points finite.
        denominator = 1. + weight.sum(1, keepdim=True)
        message = (weight[..., None] * (support - query[:, None])).sum(1) / denominator
        vote_message = (weight[..., None] * vote_difference).sum(1) / denominator
        raw = self.correction(torch.cat((query, message, vote_message), dim=-1))
        delta = self.max_delta_xy * raw.tanh()
        return delta, logits

    def forward(self, features, xyz, offsets, semantic_logits, batches, query_indices=None):
        # All official tensors are inputs only. No backward graph can enter them.
        features, xyz, offsets = features.detach().float(), xyz.detach().float(), offsets.detach().float()
        probabilities = semantic_logits.detach().float().softmax(-1)[:, 0]
        batches = batches.detach()
        graph = LocalSupportGraph(xyz.cpu().numpy(), batches.cpu().numpy(),
            (probabilities >= self.tree_conf_thresh).cpu().numpy(),
            self.support_voxel_size, self.neighbors, self.radius)
        supervised = query_indices is not None
        queries = (query_indices.detach().cpu().numpy().astype(np.int64) if supervised
                   else np.arange(len(xyz), dtype=np.int64))
        device = features.device
        deltas, edge_logits, edge_indices, edge_valid = [], [], [], []
        # In inference only Nx2 corrections survive, not all edge features/logits.
        for begin in range(0, len(queries), self.query_chunk_size):
            local_rows = queries[begin:begin + self.query_chunk_size]
            index, valid = graph.query(local_rows)
            rows = torch.as_tensor(local_rows, device=device, dtype=torch.long)
            index = torch.as_tensor(index, device=device, dtype=torch.long)
            valid = torch.as_tensor(valid, device=device)
            # Small branch runs in FP32 even if the frozen backbone uses AMP.
            with torch.cuda.amp.autocast(enabled=False):
                delta, logits = self.connected_correction(features, xyz, offsets,
                                                          probabilities, rows, index, valid)
            deltas.append(delta)
            if supervised:
                edge_logits.append(logits)
                edge_indices.append(index)
                edge_valid.append(valid)
        delta = torch.cat(deltas) if deltas else features.new_empty((0, 2))
        result = dict(rows=torch.as_tensor(queries, device=device), delta_xy=delta)
        if supervised:
            result.update(edge_logits=torch.cat(edge_logits) if edge_logits else features.new_empty((0, self.neighbors)),
                neighbors=torch.cat(edge_indices) if edge_indices else torch.empty((0, self.neighbors), device=device, dtype=torch.long),
                edge_valid=torch.cat(edge_valid) if edge_valid else torch.empty((0, self.neighbors), device=device, dtype=torch.bool))
        return result


def apply_xy_correction(offsets, result):
    """Keep the original Z tensor values, including mixed-precision dtype."""
    corrected = offsets.clone()
    rows = result['rows']
    corrected[rows, :2] = offsets[rows, :2] + result['delta_xy'].to(offsets.dtype)
    return corrected
