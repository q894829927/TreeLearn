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


def selection_policy(config):
    """Rules that determine validation F1 and therefore the best checkpoint."""
    keys = ('mask_threshold', 'quality_threshold', 'nms_iou', 'nms_containment',
            'minimum_instance_points', 'minimum_retention', 'allocation_score_threshold')
    return dict(revision=2, **{name: config['inference'][name] for name in keys})


def require_resume_compatible(saved, identity, config):
    if saved.get('signature') != identity or saved.get('check_only', True):
        raise ValueError('Resume configuration/data differ from the full-training checkpoint.')
    if saved.get('selection_policy') != selection_policy(config):
        raise ValueError('Validation inference policy changed; previous best F1 is not comparable. '
                         'Restore the training policy or use a new work_dir.')


def restore_training_state(saved, model, optimizer, rng):
    """Shared by actual resume and the server serialization/optimizer check."""
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state = saved['sampler_rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])


def require_trainable_gradients(model):
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    if missing:
        raise RuntimeError(f'Total loss is disconnected from trainable parameters: {missing}')


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
