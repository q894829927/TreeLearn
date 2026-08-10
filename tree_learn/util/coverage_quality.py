"""Small reusable components for coverage-preserving instance scoring."""


def build_two_head_instance_mlp(input_dim, hidden_dims, dropout=0.1):
    """Build the parameter-matched two-output MLP used by Q2.

    PyTorch is imported lazily so data audits can run without importing the
    training stack.
    """
    import torch
    from torch import nn

    class TwoHeadInstanceMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            current_dim = int(input_dim)
            for hidden_dim in hidden_dims:
                hidden_dim = int(hidden_dim)
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ])
                current_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)
            self.head_a = nn.Linear(current_dim, 1)
            self.head_b = nn.Linear(current_dim, 1)

        def forward(self, values):
            encoded = self.encoder(values)
            return (
                self.head_a(encoded).squeeze(1),
                self.head_b(encoded).squeeze(1),
            )

    return TwoHeadInstanceMLP()


def coverage_rejection_risk(first_logit, second_logit, mode):
    """Convert the two parameter-matched outputs to rejection risk."""
    import torch

    if mode == 'quality_iou_control':
        predicted_quality = torch.sigmoid(first_logit)
        predicted_iou = torch.sigmoid(second_logit)
        return 1.0 - predicted_quality * predicted_iou
    if mode == 'safe_iou_control':
        return torch.sigmoid(first_logit)
    if mode == 'coverage_cvar':
        return torch.sigmoid(first_logit)
    raise ValueError(f'Unknown coverage-quality mode: {mode!r}')


def count_trainable_parameters(model):
    return int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad))
