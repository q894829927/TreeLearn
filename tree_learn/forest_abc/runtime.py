"""Device selection and gradient checks without importing the sparse backbone."""

import math

import torch


def resolve_cuda_device(specification):
    device = torch.device(specification)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Full sparse backbone requires CUDA. Run in the server TreeLearn environment.')
    # Older torch CUDA memory APIs reject torch.device('cuda') without an
    # index. Resolve the logical current device, respecting CUDA_VISIBLE_DEVICES.
    index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_device(index)
    return torch.device('cuda', index)


class MaskGradientAudit:
    """Require finite graph connections and positive group evidence over a window.

    A hard-mask fallback can produce a connected, exactly-zero gradient for one
    block. That is distinct from an unused parameter (None) or NaN/Inf. Prefixes
    before '.' identify module groups; per-layer evidence remains in the report.
    """
    def __init__(self, probes):
        self.probes = tuple(probes)
        self.probe_totals = dict.fromkeys(self.probes, 0.)
        self.group_totals = dict.fromkeys((name.split('.', 1)[0] for name in self.probes), 0.)
        self.steps = 0

    def observe(self, gradients):
        if len(gradients) != len(self.probes):
            raise ValueError('Gradient audit probe count changed.')
        per_probe = {}
        groups = dict.fromkeys(self.group_totals, 0.)
        for name, gradient in zip(self.probes, gradients):
            if gradient is None:
                raise RuntimeError(f'Mask loss graph is disconnected from {name} (gradient=None).')
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f'Non-finite mask gradient in {name} (NaN/Inf).')
            magnitude = float(gradient.detach().abs().sum())
            if not math.isfinite(magnitude):
                raise FloatingPointError(f'Mask gradient magnitude overflow in {name}.')
            per_probe[name] = magnitude
            groups[name.split('.', 1)[0]] += magnitude
        for name, magnitude in per_probe.items():
            self.probe_totals[name] += magnitude
        for name, magnitude in groups.items():
            self.group_totals[name] += magnitude
        self.steps += 1
        return groups, per_probe

    def finish(self):
        inactive = [name for name, value in self.group_totals.items() if value <= 0]
        if not self.steps or inactive:
            raise RuntimeError(f'No nonzero mask-gradient evidence across {self.steps} check steps '
                               f'for groups {inactive}; inspect per-layer gradients and MFA regions.')
        return dict(steps=self.steps, group_gradient_l1=self.group_totals.copy(),
                    probe_gradient_l1=self.probe_totals.copy(),
                    zero_gradient_probes=[name for name, value in self.probe_totals.items() if value <= 0])
