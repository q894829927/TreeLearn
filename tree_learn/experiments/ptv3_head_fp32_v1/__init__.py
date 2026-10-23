"""Explicit numerical-safety variant: FP32 TreeLearn heads, unchanged backbone."""
from types import MethodType

import torch
from torch import nn

POLICY = "treelearn-both-heads-fp32-v1"


def enable_head_fp32(model):
    """Preserve state keys, BN update count and RNG; opt in per model instance.

    Precision is intentionally different from the original AMP heads. This is
    a numerical-safety variant, not an equivalent-memory-optimization claim.
    """
    for name in ("semantic_linear", "offset_linear"):
        head = getattr(model, name)
        if getattr(head, "_precision_policy", None) == POLICY:
            continue
        if not isinstance(head, nn.Sequential):
            raise TypeError("Expected the original TreeLearn sequential MLP")
        if any(p.dtype != torch.float32 for p in head.parameters()):
            raise ValueError("FP32 head parameters are required")

        def forward(self, features):
            with torch.cuda.amp.autocast(enabled=False):
                value = features.float()
                for module in self:
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        if not bool(torch.isfinite(value).all()):
                            raise FloatingPointError("Nonfinite FP32 head activation before BatchNorm")
                    value = module(value)
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite FP32 head output")
                return value

        head.forward = MethodType(forward, head)
        head._precision_policy = POLICY
    model.head_precision_policy = POLICY
    return model


def variant_config():
    from tree_learn.experiments.ptv3_memory_v1.formal import formal_config
    cfg = formal_config()
    cfg["experiment"].update(
        variant="full-crop-micro1-accum2-block-nested-head-fp32-v1",
        head_precision=POLICY)
    return cfg
