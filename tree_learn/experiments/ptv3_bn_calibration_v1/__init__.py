"""Inference-only, pooled BN statistics; never an optimizer or training adapter."""
import torch


class Moments:
    """Merge central moments in float64, including between-crop variance."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def merge(self, count, mean, m2):
        if count < 1 or not torch.isfinite(mean).all() or not torch.isfinite(m2).all():
            raise FloatingPointError("Empty or nonfinite calibration population")
        mean, m2 = mean.detach().double().cpu(), m2.detach().double().cpu()
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean.clone(), m2.clone()
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta.square() * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def add(self, value, chunk_size=65536):
        if value.ndim != 2:
            raise ValueError("Expected a point-by-channel BN input")
        for chunk in value.detach().split(chunk_size):
            variance, mean = torch.var_mean(chunk.double(), dim=0, unbiased=False)
            self.merge(len(chunk), mean, variance * len(chunk))

    def snapshot(self):
        if self.count < 2:
            raise ValueError("At least two activations are required")
        return dict(count=self.count, mean=self.mean.tolist(),
                    population_variance=(self.m2 / self.count).tolist(),
                    running_variance_unbiased=(self.m2 / (self.count - 1)).tolist())

    def install(self, module):
        values = self.snapshot()
        with torch.no_grad():
            module.running_mean.copy_(self.mean)
            module.running_var.copy_(self.m2 / (self.count - 1))
        return values


def validate_trace(groups, trace):
    flat = [name for group in groups for name in group]
    if len(set(flat)) != len(flat) or set(flat) != set(trace) or len(trace) != len(flat):
        raise ValueError("BN calibration groups do not cover the execution trace")
    position = {name: i for i, group in enumerate(groups) for name in group}
    indices = [position[name] for name in trace]
    if indices != sorted(indices):
        raise ValueError("BN calibration order disagrees with data flow")


def verify_state(before, after, selected_bn):
    allowed = {name + suffix for name in selected_bn
               for suffix in (".running_mean", ".running_var")}
    if before.keys() != after.keys():
        raise ValueError("State keys changed")
    changed = []
    for name, value in after.items():
        reference = before[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("State shape/dtype changed: " + name)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite state: " + name)
        if not torch.equal(reference.cpu(), value.detach().cpu()):
            if name not in allowed:
                raise ValueError("Changed non-calibration state: " + name)
            changed.append(name)
    return dict(allowed_changed_keys=sorted(allowed), actual_changed_keys=changed,
                all_other_state_bitwise_preserved=True,
                bn_counters_and_affine_parameters_bitwise_preserved=True)
