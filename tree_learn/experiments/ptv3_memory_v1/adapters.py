"""Versioned execution adapters. No parameter, topology, input or precision changes."""
from types import MethodType

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tree_learn.experiments.ptv3_treelearn.vendor.model import Point, Block
from tree_learn.experiments.ptv3_treelearn.vendor.serialization import encode

METADATA = ("coord", "grid_coord", "batch", "offset", "serialized_code", "serialized_order",
            "serialized_inverse", "serialized_depth", "sparse_shape")


def checkpoint_region(point, forwards):
    """Checkpoint only tensor outputs and graph-free sparse/serialization metadata.

    Two feature inputs are necessary: upstream unpooling leaves sparse.features
    equal to projected skip, while point.feat includes the decoder sum. Unifying
    these tensors changes the first decoder CPE and is NOT equivalent.
    """
    feat, sparse_feat = point.feat, point.sparse_conv_feat.features
    if not feat.requires_grad and not sparse_feat.requires_grad:
        # Reentrant checkpoint cannot discover parameter gradients without a
        # differentiable input (e.g. a future frozen embedding). Keep it safe.
        for forward in forwards:
            point = forward(point)
        return point
    metadata = {key: point[key] for key in METADATA if key in point}
    if any(isinstance(v, torch.Tensor) and v.requires_grad for v in metadata.values()):
        raise ValueError("This version supports fixed geometric metadata only")
    # Public spconv replace_feature preserves indices/cache/allocator/timer while
    # dropping the feature graph. Never close over Point or its pooling parents.
    sparse = point.sparse_conv_feat.replace_feature(sparse_feat.new_empty((0, sparse_feat.shape[1])))
    def call(features, sparse_features):
        shadow = Point({**metadata, "feat": features,
                        "sparse_conv_feat": sparse.replace_feature(sparse_features)})
        for forward in forwards:
            shadow = forward(shadow)
        return shadow.feat
    result = checkpoint(call, feat, sparse_feat, use_reentrant=True, preserve_rng_state=True)
    point.feat = result
    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(result)
    return point


def check_blocks(blocks):
    # BN in the real network stays in embedding, pool/unpool and heads, outside
    # all recomputed regions. Refuse configurations for which that is untrue.
    if any(isinstance(child, nn.modules.batchnorm._BatchNorm) for block in blocks for child in block.modules()):
        raise ValueError("BatchNorm is not allowed inside a recomputed region")


def install_blocks(backbone):
    for block in backbone.modules():
        if not isinstance(block, Block):
            continue
        check_blocks((block,))
        original = block.forward
        def forward(self, point, _original=original):
            if not self.training or not torch.is_grad_enabled():
                return _original(point)
            return checkpoint_region(point, (_original,))
        block.forward = MethodType(forward, block)


def install_stages(backbone):
    for sequence in (backbone.enc, backbone.dec):
        for stage in sequence.children():
            modules = tuple(stage.children())
            blocks = tuple(module for module in modules if isinstance(module, Block))
            prefix_length = len(modules)-len(blocks)
            if not blocks or any(isinstance(module, Block) for module in modules[:prefix_length]) or any(
                    not isinstance(module, Block) for module in modules[prefix_length:]):
                raise ValueError("Expected pool/unpool prefix followed by contiguous Blocks")
            check_blocks(blocks)
            def forward(self, point, _modules=modules, _blocks=blocks):
                if not self.training or not torch.is_grad_enabled():
                    for module in _modules:
                        point = module(point)
                    return point
                for module in _modules:
                    if not isinstance(module, Block):
                        point = module(point)  # Pool/unpool and their BN run once.
                return checkpoint_region(point, _blocks)
            stage.forward = MethodType(forward, stage)


class ExpandedInversePoint(Point):
    """Same serialization and sort/RNG order; expand replaces repeat workspace."""
    def serialization(self, order="z", depth=None, shuffle_orders=False):
        if "grid_coord" not in self:
            self.grid_coord = torch.div(self.coord-self.coord.min(0)[0], self.grid_size, rounding_mode="trunc").int()
        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self.serialized_depth = depth
        assert depth*3+len(self.offset).bit_length() <= 63 and depth <= 16
        code = torch.stack([encode(self.grid_coord, self.batch, depth, order=value) for value in order])
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(1, order,
            torch.arange(code.shape[1], device=order.device).expand(code.shape[0], -1))
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]
        self.serialized_code, self.serialized_order, self.serialized_inverse = code, order, inverse


def install_serialization(backbone):
    def forward(self, data):
        point = ExpandedInversePoint(data)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        return self.dec(point) if not self.cls_mode else point
    backbone.forward = MethodType(forward, backbone)


def apply_optimization(model, variant):
    if variant == "serialization":
        install_serialization(model.backbone)
    elif variant in ("block", "block_nested"):
        install_blocks(model.backbone)
    elif variant in ("stage", "stage_nested"):
        install_stages(model.backbone)
    elif variant != "offload":
        raise ValueError(variant)
    if variant == "offload":
        original = model.forward
        def forward(self, *args, **kwargs):
            # Synchronous, byte-preserving offload; no dtype conversion or lossy
            # compression. Backward reentrant recomputation is not offloaded.
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                return original(*args, **kwargs)
        model.forward = MethodType(forward, model)
    return model
