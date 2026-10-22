"""5,265-parameter symmetric edge head; official model is never imported."""

import numpy as np
import torch
from torch import nn


class EdgeHead(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.projection = nn.Linear(32, 16)
        self.edge = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, raw):
        a = self.projection((raw[:, :32] - self.mean) / self.scale)
        b = self.projection((raw[:, 32:64] - self.mean) / self.scale)
        values = torch.cat((torch.abs(a - b), (a + b) / 2, raw[:, 64:]), dim=1)
        return self.edge(values).squeeze(-1)


def raw_edges(arrays, indices, edges):
    a, b = indices[edges[:, 0]], indices[edges[:, 1]]
    delta = np.abs(arrays['coords'][a] - arrays['coords'][b])
    votes = np.abs(arrays['votes_xy'][a] - arrays['votes_xy'][b])
    pa, pb = arrays['tree_probability'][a], arrays['tree_probability'][b]
    result = np.concatenate((arrays['features'][a], arrays['features'][b], delta,
        np.linalg.norm(delta, axis=1)[:, None], votes, np.minimum(pa, pb)[:, None], np.maximum(pa, pb)[:, None]), axis=1)
    if result.shape != (len(edges), 72) or not np.isfinite(result).all():
        raise ValueError('Invalid raw observable edge inputs.')
    return result.astype(np.float32)


def predict_edges(model, arrays, indices, edges, device, chunk=65536):
    result = np.empty(len(edges), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(edges), chunk):
            raw = raw_edges(arrays, indices, edges[start:start + chunk])
            result[start:start + len(raw)] = model(torch.as_tensor(raw, device=device)).sigmoid().cpu().numpy()
    return result
