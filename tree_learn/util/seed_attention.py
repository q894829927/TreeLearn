"""Models and tensor utilities for complexity-aware seed relations."""


def masked_mean_std(values, valid_mask):
    """Return masked neighbour mean/std without leaking invalid values."""
    import torch

    if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
        raise ValueError('Neighbour values and validity mask are inconsistent.')
    valid = valid_mask.to(values.dtype).unsqueeze(-1)
    count = valid.sum(dim=1).clamp_min(1.0)
    mean = (values * valid).sum(dim=1) / count
    variance = (
        (values - mean[:, None, :]).square() * valid
    ).sum(dim=1) / count
    return mean, variance.clamp_min(0.0).sqrt()


def build_seed_relation_model(input_dim, geometry_dim, model_type, config):
    """Build the parameter-matched neighbourhood MLP or relation attention."""
    import torch
    from torch import nn

    input_dim = int(input_dim)
    geometry_dim = int(geometry_dim)
    dropout = float(config.get('dropout', 0.1))
    if input_dim < 1 or geometry_dim < 1:
        raise ValueError('Seed relation dimensions must be positive.')

    class DualHeads(nn.Module):
        def _finish(self, encoded):
            return (
                self.reliability_head(encoded).squeeze(-1),
                self.coverage_head(encoded).squeeze(-1),
            )

    class NeighbourhoodMLP(DualHeads):
        def __init__(self):
            super().__init__()
            summary_dim = 3 * input_dim + 2 * geometry_dim
            layers = []
            current_dim = summary_dim
            for hidden_dim in config['control_hidden_dims']:
                hidden_dim = int(hidden_dim)
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                current_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)
            self.reliability_head = nn.Linear(current_dim, 1)
            self.coverage_head = nn.Linear(current_dim, 1)

        def forward(
                self, query_features, neighbour_features,
                relative_geometry, neighbour_valid):
            feature_mean, feature_std = masked_mean_std(
                neighbour_features, neighbour_valid)
            geometry_mean, geometry_std = masked_mean_std(
                relative_geometry, neighbour_valid)
            summary = torch.cat([
                query_features,
                feature_mean,
                feature_std,
                geometry_mean,
                geometry_std,
            ], dim=-1)
            return self._finish(self.encoder(summary))

    class RelationAttention(DualHeads):
        def __init__(self):
            super().__init__()
            hidden_dim = int(config['attention_hidden_dim'])
            feedforward_dim = int(config['attention_feedforward_dim'])
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.query_projection = nn.Linear(
                hidden_dim, hidden_dim, bias=False)
            self.key_projection = nn.Linear(
                hidden_dim, hidden_dim, bias=False)
            self.value_projection = nn.Linear(
                hidden_dim, hidden_dim, bias=False)
            self.position_mlp = nn.Sequential(
                nn.Linear(geometry_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.output_projection = nn.Linear(
                hidden_dim, hidden_dim, bias=False)
            self.attention_norm = nn.LayerNorm(hidden_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden_dim, feedforward_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, hidden_dim),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.reliability_head = nn.Linear(hidden_dim, 1)
            self.coverage_head = nn.Linear(hidden_dim, 1)

        def forward(
                self, query_features, neighbour_features,
                relative_geometry, neighbour_valid):
            if (
                    neighbour_features.ndim != 3 or
                    neighbour_valid.shape != neighbour_features.shape[:2] or
                    relative_geometry.shape[:2] !=
                    neighbour_features.shape[:2]):
                raise ValueError('Seed attention neighbourhoods are invalid.')
            if not torch.all(neighbour_valid.any(dim=1)):
                raise ValueError('Every query must have a valid neighbour.')
            query_encoded = self.input_projection(query_features)
            neighbour_encoded = self.input_projection(neighbour_features)
            query = self.query_projection(query_encoded)[:, None, :]
            key = self.key_projection(neighbour_encoded)
            value = self.value_projection(neighbour_encoded)
            position = self.position_mlp(relative_geometry)
            logits = self.attention_mlp(query - key + position)
            logits = logits.masked_fill(
                ~neighbour_valid[..., None], -1e4)
            weights = torch.softmax(logits, dim=1)
            weights = weights * neighbour_valid[..., None].to(weights.dtype)
            weights = weights / weights.sum(
                dim=1, keepdim=True).clamp_min(1e-6)
            attended = (weights * (value + position)).sum(dim=1)
            encoded = self.attention_norm(
                query_encoded + self.output_projection(attended))
            encoded = self.output_norm(encoded + self.feedforward(encoded))
            return self._finish(encoded)

    if model_type == 'neighborhood_mlp':
        return NeighbourhoodMLP()
    if model_type == 'relation_attention':
        return RelationAttention()
    raise ValueError(
        "model_type must be 'neighborhood_mlp' or 'relation_attention'.")


def parameter_count(model):
    return int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad))
