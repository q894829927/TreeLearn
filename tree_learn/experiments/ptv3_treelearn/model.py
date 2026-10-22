"""PTv3 extracts features; main's heads, loss and voxelizer remain shared."""
import functools
from types import SimpleNamespace

import torch
from torch import nn
import spconv.pytorch as spconv

from tree_learn.model.tree_learn import TreeLearn, voxelize
from tree_learn.model.blocks import MLP
from tree_learn.util.train import cuda_cast
from .vendor.model import PointTransformerV3, Point, SerializedAttention, MLP as PTMLP


def canonicalize_voxels(features, indices, point_map, spatial_shape):
    """Remove GPU voxel-ID allocation order without changing the voxel set."""
    shape = spatial_shape.long()
    index = indices.long()
    key = ((index[:, 0] * shape[0] + index[:, 1]) * shape[1] + index[:, 2]) * shape[2] + index[:, 3]
    order = torch.argsort(key)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order), device=order.device)
    return features[order], indices[order], inverse[point_map], spatial_shape


class OrderedVoxelTreeLearn(TreeLearn):
    """Main network/head/loss; optional shared canonical voxel-ID engineering."""
    def __init__(self, cover_input_spatial_shape=False, **kwargs):
        super().__init__(**kwargs)
        self.cover_input_spatial_shape = cover_input_spatial_shape

    @cuda_cast
    def forward_backbone(self, coords, input_feats, batch_ids, batch_size, **kwargs):
        features, indices, point_map, shape = canonicalize_voxels(*voxelize(
            torch.hstack([coords, input_feats]), batch_ids, batch_size, self.voxel_size,
            self.use_coords, self.use_feats, self.max_num_points_per_voxel))
        actual = shape.long()
        outside = 0
        if self.spatial_shape is not None:
            fixed = torch.tensor(self.spatial_shape, device=indices.device).long()
            outside = int((indices[:, 1:] >= fixed).any(1).sum())
            multiple = 2 ** (len(self.unet.nPlanes) - 1)
            aligned = torch.div(actual + multiple - 1, multiple, rounding_mode="floor") * multiple
            shape = torch.where(actual > fixed, aligned, fixed) if self.cover_input_spatial_shape else fixed
        self.last_spatial_support = {"actual_grid_shape": actual.tolist(), "used_shape": shape.long().tolist(),
            "voxels_outside_main_fixed_shape": outside, "cover_all_input": self.cover_input_spatial_shape}
        grouped_indices = torch.cat([torch.where(batch_ids == i)[0] for i in range(batch_size)])
        restored_map = torch.empty_like(point_map)
        restored_map[grouped_indices] = point_map
        sparse = spconv.SparseConvTensor(features, indices.int(), shape, batch_size)
        return self.output_layer(self.unet(self.input_conv(sparse))), restored_map


def enable_tensor_checkpointing(backbone):
    """Checkpoint attention/MLP tensors only; never recompute BatchNorm/spconv.

    A fresh shallow Point for attention isolates its feat/padding dictionary
    mutation. Shared serialized/grid/batch tensors are read-only here.
    Upstream source and all parameter names remain unchanged.
    """
    from torch.utils.checkpoint import checkpoint
    for module in backbone.modules():
        original_forward = module.forward
        if isinstance(module, SerializedAttention):
            def attention_forward(point, _forward=original_forward, _module=module):
                if not _module.training or not torch.is_grad_enabled():
                    return _forward(point)
                # Do not retain feat, sparse_conv_feat or pooling_parent: those
                # carry autograd graphs and form reference cycles through the
                # checkpoint closure, causing memory growth across updates.
                metadata = {key: point[key] for key in
                    ("offset", "batch", "grid_coord", "serialized_order", "serialized_inverse")}
                def tensor_forward(feat):
                    shadow = Point({**metadata, "feat": feat})
                    return _forward(shadow).feat
                point.feat = checkpoint(tensor_forward, point.feat, use_reentrant=True, preserve_rng_state=True)
                return point
            module.forward = attention_forward
        elif isinstance(module, PTMLP):
            def mlp_forward(feat, _forward=original_forward, _module=module):
                if not _module.training or not torch.is_grad_enabled():
                    return _forward(feat)
                return checkpoint(_forward, feat, use_reentrant=True, preserve_rng_state=True)
            module.forward = mlp_forward


class PTv3TreeLearn(TreeLearn):
    def __init__(self, ptv3, voxel_size=0.1, max_num_points_per_voxel=3,
                 use_feats=False, use_coords=False, dim_coord=3, dim_feat=1,
                 activation_checkpointing=False, deterministic_flash=False,
                 deterministic_voxels=False, cover_input_spatial_shape=False, **kwargs):
        nn.Module.__init__(self)
        if use_feats or use_coords or dim_coord != 3 or dim_feat != 1:
            raise ValueError("The frozen first comparison requires four constant input channels.")
        self.voxel_size = voxel_size
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.use_feats = self.use_coords = False
        self.fixed_modules = []
        self.deterministic_voxels = deterministic_voxels
        self.cover_input_spatial_shape = cover_input_spatial_shape
        self.main_spatial_shape = kwargs.get("spatial_shape")
        if deterministic_flash:
            import inspect
            from .vendor import model as upstream
            flash = upstream.flash_attn
            if "deterministic" not in inspect.signature(flash.flash_attn_varlen_qkvpacked_func).parameters:
                raise RuntimeError("deterministic_flash requires the isolated FlashAttention >=2.4 runtime")
            # Preserve the upstream model file; only select its supported
            # deterministic backward kernel in the experiment adapter.
            upstream.flash_attn = SimpleNamespace(flash_attn_varlen_qkvpacked_func=functools.partial(
                flash.flash_attn_varlen_qkvpacked_func, deterministic=True))
            # torch 2.0 rejects CUDA bincount under its deterministic guard even
            # for exact integer counts. Integer sums are the identical mapping.
            upstream.batch2offset = lambda batch: torch.stack([
                (batch == i).sum() for i in range(int(batch.max()) + 1)]).cumsum(0).long()
        self.backbone = PointTransformerV3(in_channels=4, **ptv3)
        self.activation_checkpointing = activation_checkpointing
        if activation_checkpointing:
            enable_tensor_checkpointing(self.backbone)
        channels = ptv3.get("dec_channels", [64])[0]
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.semantic_linear = MLP(channels, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)
        # Initialize the two TreeLearn heads exactly as main. Keep PTv3's own init.
        for head in (self.semantic_linear, self.offset_linear):
            head.init_weights()

    @cuda_cast
    def forward_backbone(self, coords, input_feats, batch_ids, batch_size, **kwargs):
        features, indices, grouped_map, shape = voxelize(
            torch.hstack([coords, input_feats]), batch_ids, batch_size,
            self.voxel_size, False, False, self.max_num_points_per_voxel)
        if self.deterministic_voxels:
            features, indices, grouped_map, _ = canonicalize_voxels(features, indices, grouped_map, shape)
        self.last_spatial_support = {"actual_grid_shape": shape.long().tolist(), "cover_all_input": True,
                                     "policy": "PTv3 native sparse shape covers grid with upstream padding"}
        if self.main_spatial_shape is not None:
            fixed = torch.tensor(self.main_spatial_shape, device=indices.device)
            self.last_spatial_support["voxels_outside_main_fixed_shape"] = int((indices[:, 1:] >= fixed).any(1).sum())
        # main's voxelizer concatenates samples; explicitly undo that grouping
        # for pointwise output, including callers with interleaved batch IDs.
        grouped_indices = torch.cat([torch.where(batch_ids == i)[0] for i in range(batch_size)])
        point_map = torch.empty_like(grouped_map)
        point_map[grouped_indices] = grouped_map
        grid = indices[:, 1:].int()
        batch = indices[:, 0].long()
        # XYZ voxel centers in each sample's original centered coordinate frame.
        origins = torch.stack([coords[batch_ids == i].min(0).values for i in range(batch_size)])
        coord = (grid.float() + 0.5) * self.voxel_size + origins[batch]
        data = {"feat": features, "grid_coord": grid, "coord": coord, "batch": batch,
                "offset": torch.stack([(batch == i).sum() for i in range(batch_size)]).cumsum(0)}
        output = self.backbone(data)
        if not torch.equal(output.grid_coord, grid) or not torch.equal(output.batch, batch):
            raise RuntimeError("PTv3 unpooling did not restore the input voxel order")
        return SimpleNamespace(features=output.feat), point_map


def build_model(config):
    options = dict(config["model"])
    backbone = options.pop("backbone", "unet")
    fp32_accum = options.pop("spconv_fp32_accum", None)
    ordered = options.pop("deterministic_voxels", False)
    cover = options.pop("cover_input_spatial_shape", False)
    if cover and not ordered:
        raise ValueError("The covered-grid interface requires the shared canonical voxel policy")
    if backbone == "unet":
        model = OrderedVoxelTreeLearn(cover_input_spatial_shape=cover, **options) if ordered else TreeLearn(**options)
    elif backbone == "ptv3":
        model = PTv3TreeLearn(deterministic_voxels=ordered, cover_input_spatial_shape=cover, **options)
    else:
        raise ValueError(backbone)
    if fp32_accum is not None:
        from spconv.pytorch.conv import SparseConvolutionBase
        for module in model.modules():
            if isinstance(module, SparseConvolutionBase):
                module.fp32_accum = bool(fp32_accum)
    return model
