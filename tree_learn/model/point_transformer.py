import warnings

import torch
import torch.nn as nn


class LocalPointTransformerLayer(nn.Module):
    """A single Point Transformer layer over voxel-sampled local supports.

    Query points attend to the closest occupied cells in their 3x3x3 voxel
    neighbourhood. Support coordinates and features are mean pooled per cell,
    which bounds the neighbourhood construction cost without compiled KNN
    extensions.
    """

    def __init__(
            self,
            channels,
            num_neighbors=8,
            support_voxel_size=0.4,
            max_support_points=32768):
        super().__init__()
        if num_neighbors < 1 or num_neighbors > 27:
            raise ValueError('num_neighbors must be between 1 and 27.')
        if support_voxel_size <= 0:
            raise ValueError('support_voxel_size must be positive.')

        self.channels = channels
        self.num_neighbors = num_neighbors
        self.support_voxel_size = support_voxel_size
        self.max_support_points = max_support_points
        self._warned_support_cap = False

        self.query_projection = nn.Linear(channels, channels, bias=False)
        self.key_projection = nn.Linear(channels, channels, bias=False)
        self.value_projection = nn.Linear(channels, channels, bias=False)
        self.position_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels))
        self.attention_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels))
        self.output_projection = nn.Linear(channels, channels, bias=False)
        self.output_norm = nn.LayerNorm(channels)

        offsets = torch.stack(torch.meshgrid(
            torch.arange(-1, 2),
            torch.arange(-1, 2),
            torch.arange(-1, 2),
            indexing='ij'), dim=-1).reshape(-1, 3)
        self.register_buffer(
            'neighbour_cell_offsets', offsets.long(), persistent=False)

    @staticmethod
    def _pool_supports(coords, features, cell_coords):
        unique_cells, inverse = torch.unique(
            cell_coords, dim=0, sorted=True, return_inverse=True)
        num_supports = len(unique_cells)
        counts = torch.bincount(
            inverse, minlength=num_supports).to(features.dtype).unsqueeze(1)

        support_coords = coords.new_zeros((num_supports, 3))
        support_coords.index_add_(0, inverse, coords)
        support_coords = support_coords / counts.to(coords.dtype)

        support_features = features.new_zeros(
            (num_supports, features.shape[1]))
        support_features.index_add_(0, inverse, features)
        support_features = support_features / counts
        return unique_cells, support_coords, support_features

    def _cap_supports(self, cells, coords, features):
        if (
            self.max_support_points is None or
            self.max_support_points <= 0 or
            len(cells) <= self.max_support_points
        ):
            return cells, coords, features

        if not self._warned_support_cap:
            warnings.warn(
                f'Point Transformer support count {len(cells):,} exceeds '
                f'axis_max_support_points={self.max_support_points:,}; '
                'using deterministic spatially ordered subsampling.',
                RuntimeWarning)
            self._warned_support_cap = True

        selection = torch.linspace(
            0,
            len(cells) - 1,
            steps=self.max_support_points,
            device=cells.device).round().long()
        return cells[selection], coords[selection], features[selection]

    @staticmethod
    def _linear_cell_keys(cell_coords, cell_min, cell_extent):
        normalized = cell_coords - cell_min
        return (
            normalized[..., 0] +
            cell_extent[0] * (
                normalized[..., 1] +
                cell_extent[1] * normalized[..., 2]))

    def _get_neighbours(self, query_coords, query_cells, support_cells,
                        support_coords):
        """Return local support indices and validity without tracking gradients."""
        with torch.no_grad():
            cell_min = support_cells.min(dim=0).values
            cell_max = support_cells.max(dim=0).values
            cell_extent = cell_max - cell_min + 1

            support_keys = self._linear_cell_keys(
                support_cells, cell_min, cell_extent)
            sorted_keys, sort_order = torch.sort(support_keys)

            neighbour_cells = (
                query_cells[:, None, :] +
                self.neighbour_cell_offsets[None, :, :])
            within_bounds = (
                (neighbour_cells >= cell_min).all(dim=-1) &
                (neighbour_cells <= cell_max).all(dim=-1))
            neighbour_keys = self._linear_cell_keys(
                neighbour_cells, cell_min, cell_extent)

            locations = torch.searchsorted(
                sorted_keys, neighbour_keys.reshape(-1))
            locations = locations.reshape(neighbour_keys.shape)
            safe_locations = locations.clamp(max=len(sorted_keys) - 1)
            occupied = (
                within_bounds &
                (locations < len(sorted_keys)) &
                (sorted_keys[safe_locations] == neighbour_keys))

            support_indices = sort_order[safe_locations]
            candidate_coords = support_coords[support_indices]
            squared_distances = (
                candidate_coords.float() -
                query_coords[:, None, :].float()).pow(2).sum(dim=-1)
            squared_distances.masked_fill_(~occupied, float('inf'))

            k = min(self.num_neighbors, squared_distances.shape[1])
            _, closest = torch.topk(
                squared_distances, k=k, dim=1, largest=False, sorted=False)
            neighbour_indices = torch.gather(
                support_indices, 1, closest)
            neighbour_valid = torch.gather(occupied, 1, closest)
        return neighbour_indices, neighbour_valid

    def forward(self, features, coords, batch_ids):
        if len(features) == 0:
            return features

        output = torch.empty_like(features)
        for batch_id in torch.unique(batch_ids, sorted=True):
            batch_mask = batch_ids == batch_id
            batch_indices = torch.where(batch_mask)[0]
            batch_features = features[batch_mask]
            batch_coords = coords[batch_mask]

            cell_coords = torch.floor(
                batch_coords.float() / self.support_voxel_size).long()
            support_cells, support_coords, support_features = \
                self._pool_supports(
                    batch_coords, batch_features, cell_coords)
            support_cells, support_coords, support_features = \
                self._cap_supports(
                    support_cells, support_coords, support_features)

            neighbour_indices, neighbour_valid = self._get_neighbours(
                batch_coords, cell_coords, support_cells, support_coords)
            neighbour_features = support_features[neighbour_indices]
            neighbour_coords = support_coords[neighbour_indices]

            query = self.query_projection(batch_features)[:, None, :]
            key = self.key_projection(neighbour_features)
            value = self.value_projection(neighbour_features)
            relative_xyz = (
                batch_coords[:, None, :] - neighbour_coords).to(
                    batch_features.dtype)
            position = self.position_mlp(relative_xyz)
            attention_logits = self.attention_mlp(
                query - key + position)
            attention_logits = attention_logits.masked_fill(
                ~neighbour_valid[..., None], -1e4)
            attention = torch.softmax(attention_logits, dim=1)
            attention = attention * neighbour_valid[..., None].to(
                attention.dtype)
            attention = attention / attention.sum(
                dim=1, keepdim=True).clamp_min(1e-6)

            transformed = (
                attention * (value + position)).sum(dim=1)
            transformed = self.output_projection(transformed)
            output[batch_indices] = self.output_norm(
                batch_features + transformed)
        return output
