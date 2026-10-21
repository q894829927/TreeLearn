"""Small residual feature fusion and bounded, training-only instance supervision.

No sparse-convolution or custom point operators are required in this module.
The boundary term is an instance-label adaptation, not a reproduction of CBL.
"""

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree


def align_parent_features(fine_indices, coarse_indices, coarse_features, level):
    """Gather strided-convolution parents by coordinates, never by row order.

    TreeLearn contracts with kernel=2, stride=2, padding=0. Boundary voxels
    without a parent (possible for odd dynamic shapes) receive zero features.
    The batch column is part of every key. Indices remain in their native XYZ
    order; no axis convention is assumed here.
    """
    if level < 1 or fine_indices.shape[1] != 4 or coarse_indices.shape[1] != 4:
        raise ValueError('Expected Nx4 sparse indices and level >= 1.')
    if len(coarse_indices) != len(coarse_features) or not len(coarse_indices):
        raise ValueError('Coarse indices/features must be aligned and nonempty.')
    parent = fine_indices.long().clone()
    parent[:, 1:] = torch.div(parent[:, 1:], 2 ** level, rounding_mode='floor')
    coarse = coarse_indices.long()
    dimensions = torch.maximum(parent.max(0).values, coarse.max(0).values) + 1
    # Python integer arithmetic also detects overflow before forming int64 keys.
    dims = dimensions.tolist()
    if np.prod(dims, dtype=object) >= 2 ** 63:
        raise ValueError('Sparse coordinate keys would overflow int64.')

    def key(indices):
        return ((indices[:, 0] * dims[1] + indices[:, 1]) * dims[2]
                + indices[:, 2]) * dims[3] + indices[:, 3]

    keys, order = key(coarse).sort()
    if bool((keys[1:] == keys[:-1]).any()):
        raise ValueError('Duplicate coarse sparse coordinates.')
    wanted = key(parent)
    positions = torch.searchsorted(keys, wanted)
    bounded = positions.clamp_max(len(keys) - 1)
    valid = (positions < len(keys)) & (keys[bounded] == wanted)
    aligned = coarse_features[order[bounded]] * valid[:, None].to(coarse_features.dtype)
    return aligned, valid


class ResidualInstanceAdapter(nn.Module):
    """Reuse two decoder scales, then feed the original semantic/offset heads."""

    def __init__(self, channels=32, levels=(1, 2), projection_dim=16,
                 hidden_dim=64, embedding_dim=8, **unused):
        super().__init__()
        self.levels = tuple(levels)
        if len(self.levels) != 2 or len(set(self.levels)) != 2 or min(self.levels) < 1:
            raise ValueError('Select exactly two distinct positive decoder levels.')
        self.projections = nn.ModuleDict({str(level): nn.Sequential(
            nn.LayerNorm(channels * (level + 1)),
            nn.Linear(channels * (level + 1), projection_dim), nn.GELU())
            for level in self.levels})
        self.fusion = nn.Sequential(nn.LayerNorm(channels + 2 * projection_dim),
            nn.Linear(channels + 2 * projection_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, channels))
        # Only the scale is zero. Zeroing both scale and branch would kill learning.
        self.gamma = nn.Parameter(torch.zeros(()))
        self.embedding = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                       nn.Linear(channels, embedding_dim))
        self.collect_alignment = False
        self.alignment_missing = {}

    def forward(self, fine, taps):
        features = [fine.features]
        for level in self.levels:
            coarse = taps[level]
            projected = self.projections[str(level)](coarse.features)
            aligned, valid = align_parent_features(fine.indices, coarse.indices, projected, level)
            if self.collect_alignment:
                self.alignment_missing[level] = int((~valid).sum())
            features.append(aligned)
        delta = self.fusion(torch.cat(features, dim=-1))
        fused = fine.features + self.gamma.tanh().to(delta.dtype) * delta
        return fine.replace_feature(fused)


def sample_instance_points(coords, labels, batches, valid, settings, seed):
    """Stratify by tree and height before a small spatial-neighbour search.

    Return original row indices and hard-negative pairs in sampled-row space.
    All selection is training-only; IDs are isolated by batch. Unknown/background
    rows never become negatives. No NxN matrix over the full crop is built.
    """
    rng = np.random.default_rng(seed)
    xyz = coords.detach().float().cpu().numpy()
    ids = labels.detach().cpu().numpy()
    batch = batches.detach().cpu().numpy()
    mask = valid.detach().cpu().numpy().astype(bool) & (ids > 0)
    selected, groups, negative_pairs = [], [], []
    next_group = 0
    for batch_id in np.unique(batch[mask]):
        tree_ids = np.unique(ids[mask & (batch == batch_id)])
        if len(tree_ids) > settings['max_trees']:
            tree_ids = np.sort(rng.choice(tree_ids, settings['max_trees'], replace=False))
        begin = len(selected)
        for tree_id in tree_ids:
            pool = np.flatnonzero(mask & (batch == batch_id) & (ids == tree_id))
            pool = pool[np.argsort(xyz[pool, 2], kind='stable')]
            # One random draw from each height stratum, including small trees.
            for stratum in np.array_split(pool, min(len(pool), settings['points_per_tree'])):
                selected.append(int(rng.choice(stratum)))
                groups.append(next_group)
            next_group += 1
        stop = len(selected)
        if stop - begin < 2:
            continue
        local_indices = np.asarray(selected[begin:stop], dtype=np.int64)
        local_groups = np.asarray(groups[begin:stop])
        k = min(settings['boundary_neighbors'] + 1, len(local_indices))
        distances, neighbors = cKDTree(xyz[local_indices]).query(xyz[local_indices], k=k, workers=1)
        source = np.broadcast_to(np.arange(len(local_indices))[:, None], neighbors.shape)
        keep = ((distances <= settings['boundary_radius']) & (source < neighbors)
                & (local_groups[:, None] != local_groups[neighbors]))
        pairs = np.column_stack((source[keep], neighbors[keep])) + begin
        if len(pairs) > settings['max_boundary_pairs']:
            pairs = pairs[rng.choice(len(pairs), settings['max_boundary_pairs'], replace=False)]
        negative_pairs.extend(pairs.tolist())
    return (np.asarray(selected, dtype=np.int64), np.asarray(groups, dtype=np.int64),
            np.asarray(negative_pairs, dtype=np.int64).reshape(-1, 2))


def discriminative_terms(embeddings, groups, pairs, settings):
    """Per-tree pull, per-crop centroid separation, and nearby-tree separation."""
    features = embeddings.float()
    zero = features.sum() * 0
    if not len(features):
        return zero, dict(pull=zero, push=zero, boundary=zero)
    group_ids, inverse, counts = groups.unique(sorted=True, return_inverse=True, return_counts=True)
    means = features.new_zeros((len(group_ids), features.shape[1])).index_add(0, inverse, features)
    means = means / counts[:, None]
    distance = torch.linalg.vector_norm(features - means[inverse], dim=-1)
    point_pull = (distance - settings['pull_margin']).clamp_min(0).square()
    per_tree = features.new_zeros(len(group_ids)).index_add(0, inverse, point_pull) / counts
    pull = per_tree.mean()
    if len(means) > 1:
        separation = torch.cdist(means[None], means[None]).squeeze(0)
        off_diagonal = ~torch.eye(len(means), device=features.device, dtype=torch.bool)
        push = (2 * settings['push_margin'] - separation[off_diagonal]).clamp_min(0).square().mean()
    else:
        push = zero
    if len(pairs):
        pair_distance = torch.linalg.vector_norm(features[pairs[:, 0]] - features[pairs[:, 1]], dim=-1)
        boundary = (settings['push_margin'] - pair_distance).clamp_min(0).square().mean()
    else:
        boundary = zero
    regularization = .001 * torch.linalg.vector_norm(means, dim=-1).mean()
    return pull + push + settings['boundary_weight'] * boundary + regularization, dict(
        pull=pull, push=push, boundary=boundary)


def instance_auxiliary_loss(adapter, point_features, batch, settings, seed):
    indices, groups, pairs = sample_instance_points(batch['coords'], batch['instance_labels'],
        batch['batch_ids'], batch['masks_sem'], settings, seed)
    device = point_features.device
    # The expensive head only sees at most max_trees*points_per_tree per crop.
    index = torch.as_tensor(indices, device=device)
    embeddings = adapter.embedding(point_features[index])
    total = embeddings.float().sum() * 0
    # Keep centroid push batch-isolated as well as neighbour-pair sampling.
    sampled_batches = batch['batch_ids'].to(device)[index]
    group_tensor = torch.as_tensor(groups, device=device)
    pair_tensor = torch.as_tensor(pairs, device=device)
    terms = {name: total for name in ('pull', 'push', 'boundary')}
    batch_count = 0
    for batch_id in sampled_batches.unique():
        keep = sampled_batches == batch_id
        local_rows = torch.where(keep)[0]
        remap = torch.full((len(index),), -1, device=device, dtype=torch.long)
        remap[local_rows] = torch.arange(len(local_rows), device=device)
        local_pairs = pair_tensor[keep[pair_tensor[:, 0]]] if len(pair_tensor) else pair_tensor
        value, components = discriminative_terms(embeddings[keep], group_tensor[keep], remap[local_pairs], settings)
        total = total + value
        terms = {name: terms[name] + components[name] for name in terms}
        batch_count += 1
    divisor = max(batch_count, 1)
    return total / divisor, {**{name: value / divisor for name, value in terms.items()},
                            'sampled_points': len(indices), 'boundary_pairs': len(pairs)}
