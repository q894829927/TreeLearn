import functools
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch.utils import PointToVoxel
from .blocks import MLP, ResidualBlock, UBlock
from .point_transformer import LocalPointTransformerLayer
from tree_learn.util.train import cuda_cast, masked_offset_loss, point_wise_loss

LOSS_MULTIPLIER_SEMANTIC = 50 # multiply semantic loss for similar magnitude with offset loss

class TreeLearn(nn.Module):
    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 kernel_size=3,
                 dim_coord=3,
                 dim_feat=1,
                 fixed_modules=[],
                 use_feats=True,
                 use_coords=False,
                 spatial_shape=None,
                 max_num_points_per_voxel=3,
                 voxel_size=0.1,
                 use_upper_anchor=True,
                 upper_offset_loss_weight=1.0,
                 axis_loss_weight=0.1,
                 offset_loss_type='smooth_l1',
                 smooth_l1_beta=1.0,
                 use_axis_branch=False,
                 axis_branch_type='point_transformer',
                 axis_hidden_dim=32,
                 axis_num_neighbors=8,
                 axis_support_voxel_size=0.4,
                 axis_max_support_points=32768,
                 axis_query_chunk_size=8192,
                 axis_gradient_checkpointing=True,
                 axis_tree_conf_thresh=0.5,
                 axis_log_variance_min=-4.0,
                 axis_log_variance_max=4.0,
                 axis_branch_only=True,
                 **kwargs):

        super().__init__()
        self.voxel_size = voxel_size
        self.fixed_modules = list(fixed_modules)
        self.use_feats = use_feats
        self.use_coords = use_coords
        self.spatial_shape = spatial_shape
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.use_upper_anchor = use_upper_anchor
        self.upper_offset_loss_weight = upper_offset_loss_weight
        self.axis_loss_weight = axis_loss_weight
        self.offset_loss_type = offset_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.use_axis_branch = use_axis_branch
        self.axis_branch_type = axis_branch_type
        self.axis_tree_conf_thresh = axis_tree_conf_thresh
        self.axis_log_variance_min = axis_log_variance_min
        self.axis_log_variance_max = axis_log_variance_max
        self.axis_branch_only = axis_branch_only

        if axis_log_variance_min >= axis_log_variance_max:
            raise ValueError(
                'axis_log_variance_min must be smaller than '
                'axis_log_variance_max.')
        if axis_branch_type not in ('mlp', 'point_transformer'):
            raise ValueError(
                "axis_branch_type must be 'mlp' or 'point_transformer'.")

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                dim_coord + dim_feat, channels, kernel_size=kernel_size, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, ResidualBlock, kernel_size, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
        
        # head
        self.semantic_linear = MLP(channels, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)
        if use_upper_anchor:
            self.upper_offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)
        if use_axis_branch:
            self.axis_input_projection = nn.Sequential(
                nn.Linear(channels + 2, axis_hidden_dim),
                nn.LayerNorm(axis_hidden_dim),
                nn.ReLU())
            if axis_branch_type == 'point_transformer':
                self.axis_point_transformer = LocalPointTransformerLayer(
                    axis_hidden_dim,
                    num_neighbors=axis_num_neighbors,
                    support_voxel_size=axis_support_voxel_size,
                    max_support_points=axis_max_support_points,
                    query_chunk_size=axis_query_chunk_size,
                    gradient_checkpointing=axis_gradient_checkpointing)
            else:
                self.axis_point_transformer = nn.Identity()
            self.axis_xy_head = nn.Linear(axis_hidden_dim, 2)
            self.axis_uncertainty_head = nn.Linear(axis_hidden_dim, 1)
        self.init_weights()
        if use_axis_branch:
            nn.init.zeros_(self.axis_xy_head.weight)
            nn.init.zeros_(self.axis_xy_head.bias)
            nn.init.zeros_(self.axis_uncertainty_head.weight)
            nn.init.constant_(self.axis_uncertainty_head.bias, 2.0)

        if use_axis_branch and axis_branch_only:
            frozen_axis_base = [
                'input_conv',
                'unet',
                'output_layer',
                'semantic_linear',
                'offset_linear',
            ]
            self.fixed_modules = list(dict.fromkeys(
                self.fixed_modules + frozen_axis_base))

        # weight init
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()


    # manually set batchnorms in fixed modules to eval mode
    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()


    def forward(self, batch, return_loss):
        backbone_output, v2p_map = self.forward_backbone(**batch)
        output = self.forward_head(
            backbone_output,
            v2p_map,
            coords=batch['coords'],
            input_feats=batch['input_feats'],
            batch_ids=batch['batch_ids'])
        if return_loss:
            output = self.get_loss(model_output=output, **batch)
        
        return output

    @cuda_cast
    def forward_backbone(self, coords, input_feats, batch_ids, batch_size, **kwargs):
        voxel_feats, voxel_coords, v2p_map, spatial_shape = voxelize(torch.hstack([coords, input_feats]), batch_ids, batch_size, self.voxel_size, self.use_coords, self.use_feats, max_num_points_per_voxel=self.max_num_points_per_voxel)
        if self.spatial_shape is not None:
            spatial_shape = torch.tensor(self.spatial_shape, device=voxel_coords.device)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        output = self.input_conv(input)

        output = self.unet(output)
        output = self.output_layer(output)
        return output, v2p_map
    

    def forward_head(
            self, backbone_output, v2p_map, coords=None, input_feats=None,
            batch_ids=None):
        output = dict()
        backbone_feats = backbone_output.features[v2p_map]
        output['backbone_feats'] = backbone_feats
        output['semantic_prediction_logits'] = self.semantic_linear(backbone_feats)
        output['offset_predictions'] = self.offset_linear(backbone_feats)
        if self.use_upper_anchor:
            output['upper_offset_predictions'] = self.upper_offset_linear(backbone_feats)
        if self.use_axis_branch:
            if coords is None or input_feats is None or batch_ids is None:
                raise ValueError(
                    'coords, input_feats and batch_ids are required when '
                    'use_axis_branch=True.')
            output.update(self.forward_axis_branch(
                backbone_feats,
                output['semantic_prediction_logits'],
                output['offset_predictions'],
                coords.to(backbone_feats.device),
                input_feats.to(backbone_feats.device),
                batch_ids.to(backbone_feats.device)))
        return output


    def forward_axis_branch(
            self, backbone_feats, semantic_logits, offset_predictions,
            coords, input_feats, batch_ids):
        semantic_probs = semantic_logits.detach().float().softmax(dim=-1)
        candidate_mask = (
            semantic_probs[:, 0] >= self.axis_tree_conf_thresh)
        candidate_indices = torch.where(candidate_mask)[0]

        num_points = len(backbone_feats)
        axis_xy = backbone_feats.new_zeros((num_points, 2))
        raw_log_variance = backbone_feats.new_full(
            (num_points, 1), self.axis_log_variance_max)

        if len(candidate_indices) > 0:
            candidate_batches = batch_ids[candidate_indices]
            heights = torch.relu(
                -offset_predictions[candidate_indices, 2].detach().float())
            normalized_heights = torch.zeros_like(heights)
            for batch_id in torch.unique(candidate_batches, sorted=True):
                batch_mask = candidate_batches == batch_id
                scale = torch.quantile(
                    heights[batch_mask], 0.95).clamp_min(1.0)
                normalized_heights[batch_mask] = (
                    heights[batch_mask] / scale).clamp(0, 1)

            verticality = input_feats[candidate_indices, -1].float()
            branch_inputs = torch.cat([
                backbone_feats[candidate_indices],
                verticality[:, None].to(backbone_feats.dtype),
                normalized_heights[:, None].to(backbone_feats.dtype),
            ], dim=1)
            branch_features = self.axis_input_projection(branch_inputs)
            if self.axis_branch_type == 'point_transformer':
                branch_features = self.axis_point_transformer(
                    branch_features,
                    coords[candidate_indices],
                    candidate_batches)
            else:
                branch_features = self.axis_point_transformer(branch_features)

            axis_xy[candidate_indices] = self.axis_xy_head(branch_features)
            raw_log_variance[candidate_indices] = \
                self.axis_uncertainty_head(branch_features)

        log_variance = raw_log_variance.clamp(
            self.axis_log_variance_min,
            self.axis_log_variance_max)
        confidence = torch.sigmoid(-log_variance)
        return {
            'axis_xy_predictions': axis_xy,
            'axis_log_variance': log_variance,
            'axis_confidence': confidence,
            'axis_candidate_mask': candidate_mask,
        }


    @cuda_cast
    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, masks_sem,
                 upper_offset_labels=None, masks_upper=None, **kwargs):
        loss_dict = dict()
        
        # Define variables
        semantic_prediction_logits = model_output['semantic_prediction_logits'].float()
        offset_predictions = model_output['offset_predictions'].float()
        
        # semantic and offset losses
        semantic_loss, offset_loss = point_wise_loss(
            semantic_prediction_logits,
            offset_predictions, 
            masks_sem, masks_off,
            semantic_labels, offset_labels,
            offset_loss_type=self.offset_loss_type,
            smooth_l1_beta=self.smooth_l1_beta
        )
        loss_dict['semantic_loss'] = semantic_loss * LOSS_MULTIPLIER_SEMANTIC
        loss_dict['offset_loss'] = offset_loss

        if self.use_upper_anchor:
            upper_offset_predictions = model_output['upper_offset_predictions'].float()
            if upper_offset_labels is None or masks_upper is None:
                upper_offset_loss = 0 * upper_offset_predictions.sum()
            else:
                upper_offset_loss = masked_offset_loss(
                    upper_offset_predictions,
                    upper_offset_labels,
                    masks_upper,
                    loss_type=self.offset_loss_type,
                    smooth_l1_beta=self.smooth_l1_beta)
            loss_dict['upper_offset_loss'] = (
                upper_offset_loss * self.upper_offset_loss_weight)

            masks_axis = masks_off if masks_upper is None else (masks_off & masks_upper)
            if upper_offset_labels is None or masks_axis.sum() == 0:
                axis_loss = 0 * upper_offset_predictions.sum()
            else:
                predicted_axis = (
                    upper_offset_predictions[masks_axis] - offset_predictions[masks_axis])
                target_axis = upper_offset_labels[masks_axis] - offset_labels[masks_axis]
                axis_loss = (
                    1 - F.cosine_similarity(
                        predicted_axis, target_axis, dim=1, eps=1e-6)
                ).mean()
            loss_dict['axis_loss'] = axis_loss * self.axis_loss_weight

        if self.use_axis_branch:
            axis_predictions = model_output['axis_xy_predictions'].float()
            axis_log_variance = model_output['axis_log_variance'].float()
            candidate_mask = model_output['axis_candidate_mask']
            masks_axis = masks_off if masks_upper is None else (
                masks_off & masks_upper)
            masks_axis = masks_axis.to(candidate_mask.device) & candidate_mask
            branch_zero = sum(
                parameter.sum() * 0
                for name, parameter in self.named_parameters()
                if name.startswith('axis_'))
            if (
                upper_offset_labels is None or
                offset_labels is None or
                masks_axis.sum() == 0
            ):
                axis_xy_loss = branch_zero
            else:
                target_axis_xy = (
                    upper_offset_labels[masks_axis, :2].to(
                        axis_predictions.device) -
                    offset_labels[masks_axis, :2].to(
                        axis_predictions.device))
                point_error = F.smooth_l1_loss(
                    axis_predictions[masks_axis],
                    target_axis_xy.float(),
                    reduction='none',
                    beta=self.smooth_l1_beta).sum(dim=1)
                log_variance = axis_log_variance[
                    masks_axis, 0]
                axis_xy_loss = (
                    torch.exp(-log_variance) * point_error +
                    log_variance).mean()
            loss_dict['axis_xy_loss'] = (
                axis_xy_loss * self.axis_loss_weight)

        # Sum all losses
        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict


def voxelize(feats, batch_ids, batch_size, voxel_size, use_coords, use_feats, max_num_points_per_voxel, epsilon=1):
    voxel_coords, voxel_feats, v2p_maps = [], [], []
    total_len_voxels = 0
    for i in range(batch_size):
        feats_one_element = feats[batch_ids == i]
        min_range = torch.min(feats_one_element[:, :3], dim=0).values
        max_range = torch.max(feats_one_element[:, :3], dim=0).values + epsilon
        voxelizer = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, voxel_size], 
            coors_range_xyz=min_range.tolist() + max_range.tolist(),
            num_point_features=feats.shape[1], 
            max_num_voxels=len(feats), 
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=feats.device)
        voxel_feat, voxel_coord, _, v2p_map = voxelizer.generate_voxel_with_id(feats_one_element)
        assert torch.sum(v2p_map == -1) == 0
        voxel_coord[:, [0, 2]] = voxel_coord[:, [2, 0]]
        voxel_coord = torch.cat((torch.ones((len(voxel_coord), 1), device=feats.device)*i, voxel_coord), dim=1)

        # get mean feature of voxel
        zero_rows = torch.sum(voxel_feat == 0, dim=2) == voxel_feat.shape[2]
        voxel_feat[zero_rows] = float("nan")
        voxel_feat = torch.nanmean(voxel_feat, dim=1)
        if not use_coords:
            voxel_feat[:, :3] = torch.ones_like(voxel_feat[:, :3])
        if not use_feats:
            voxel_feat[:, 3:] = torch.ones_like(voxel_feat[:, 3:])
        voxel_feat = torch.hstack([voxel_feat[:, 3:], voxel_feat[:, :3]])

        voxel_coords.append(voxel_coord)
        voxel_feats.append(voxel_feat)
        v2p_maps.append(v2p_map + total_len_voxels)
        total_len_voxels += len(voxel_coord) 
    voxel_coords = torch.cat(voxel_coords, dim=0)
    voxel_feats = torch.cat(voxel_feats, dim=0)
    v2p_maps = torch.cat(v2p_maps, dim=0)
    spatial_shape = voxel_coords.max(dim=0).values + 1

    return voxel_feats, voxel_coords, v2p_maps, spatial_shape[1:]
