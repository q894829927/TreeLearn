"""Full sparse backbone + ISA + six-layer forest decoder + B/C adapters."""

import torch
from torch import nn
import spconv.pytorch as spconv

from .modules import GeometryAdapter, RegionGatedQueryUpdate, farthest_points
from .vendor.spconv_unet import SpConvUNet
from .vendor.decoder import ForAINetv2QueryDecoder_XAwarequery


class CombinedForestDecoder(ForAINetv2QueryDecoder_XAwarequery):
    def __init__(self, use_mfa=True, **kwargs):
        super().__init__(**kwargs)
        self.region_updates = nn.ModuleDict({
            str(i): RegionGatedQueryUpdate(kwargs['d_model'])
            for i in range(1, kwargs['num_layers']) if use_mfa})
        for index in self.region_updates:
            # Replaced cross-attention weights must not remain as unused
            # trainable parameters in an otherwise joint-training model.
            self.cross_attn_layers[int(index)] = nn.Identity()

    def forward_iter_pred(self, x, queries):
        sources = [self.input_proj(y) for y in x]
        mask_features = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        scores, masks, attention = self._forward_head(queries, mask_features)
        outputs = [dict(scores=scores, masks=masks)]
        for layer in range(len(self.cross_attn_layers)):
            if str(layer) in self.region_updates:
                queries = [self.region_updates[str(layer)](q, f, m)
                           for q, f, m in zip(queries, sources, masks)]
            else:
                queries = self.cross_attn_layers[layer](sources, queries, attention)
            queries = self.self_attn_layers[layer](queries)
            queries = self.ffn_layers[layer](queries)
            scores, masks, attention = self._forward_head(queries, mask_features)
            if self.training:
                outputs.append(dict(scores=scores, masks=masks))
            else:
                outputs = [dict(scores=scores, masks=masks)]
        return dict(outputs[-1], aux_outputs=outputs[:-1])


class ForestABC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        channels = config['channels']
        self.input_conv = spconv.SparseSequential(spconv.SubMConv3d(
            3, channels, 3, padding=1, bias=False, indice_key='subm1'))
        self.backbone = SpConvUNet([channels * (i + 1) for i in range(5)],
                                  return_blocks=False)
        self.output_norm = nn.Sequential(nn.BatchNorm1d(channels, eps=1e-4), nn.ReLU())
        self.geometry = GeometryAdapter(channels) if config['use_geometry'] else None
        self.semantic = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                      nn.Linear(channels, 2))
        self.embedding = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                       nn.Linear(channels, 5))
        self.decoder = CombinedForestDecoder(
            use_mfa=config['use_mfa'], num_layers=config['layers'],
            num_instance_queries=0, num_semantic_queries=2, num_classes=1,
            in_channels=channels, d_model=config['dim'], num_heads=config['heads'],
            hidden_dim=config['feedforward'], dropout=0., activation_fn='gelu',
            iter_pred=True, attn_mask=True, fix_attention=True, objectness_flag=True)

    @torch.no_grad()
    def select_queries(self, semantic, embeddings, core, labels=None, teacher_fraction=0.):
        candidates = torch.where(core)[0]
        if not len(candidates):
            candidates = torch.arange(len(semantic), device=semantic.device)
        probability = semantic.softmax(-1)[:, 1]
        if labels is not None and teacher_fraction > 0:
            known = labels >= 0
            probability = torch.where(known, teacher_fraction * (labels > 0).float()
                                      + (1 - teacher_fraction) * probability, probability)
        foreground = candidates[probability[candidates] >= .5]
        # Empty/weak semantic predictions must not make the decoder disappear.
        minimum = min(self.config['num_queries'], len(candidates))
        if len(foreground) < minimum:
            foreground = candidates[probability[candidates].argsort(descending=True)[:minimum]]
        if len(foreground) > self.config['isa_candidates']:
            positions = torch.linspace(0, len(foreground) - 1, self.config['isa_candidates'],
                                       device=foreground.device).long()
            foreground = foreground[positions]
        return foreground[farthest_points(embeddings[foreground], self.config['num_queries'])]

    def forward(self, block, teacher_fraction=0.):
        xyz, grid = block['xyz'], block['grid']
        indices = torch.cat((torch.zeros_like(grid[:, :1]), grid), 1).int()
        shape = torch.clamp(grid.max(0).values + 1, min=128).tolist()
        sparse = spconv.SparseConvTensor(xyz - xyz.mean(0), indices, shape, 1)
        sparse = self.backbone(self.input_conv(sparse))
        # Inverse convolutions should restore input order. Check rather than
        # silently associating features with the wrong labels or coordinates.
        if not torch.equal(sparse.indices, indices):
            raise RuntimeError('Sparse U-Net changed input ordering; cannot align masks.')
        features = self.output_norm(sparse.features)
        if self.geometry is not None:
            features = self.geometry(features, block['neighbors'])
        semantic, embeddings = self.semantic(features), self.embedding(features)
        anchors = self.select_queries(semantic, embeddings, block['core'],
                                      block.get('labels') if self.training else None,
                                      teacher_fraction if self.training else 0.)
        decoded = self.decoder([features], [features[anchors]])
        outputs = [dict(masks=x['masks'][0], scores=x['scores'][0].squeeze(-1))
                   for x in decoded['aux_outputs'] + [decoded]]
        return dict(layers=outputs, semantic=semantic, embedding=embeddings,
                    anchors=anchors, num_voxels=len(xyz))


def tensor_block(block, device, supervision=False):
    dtypes = dict(xyz=torch.float32, grid=torch.int32, neighbors=torch.int64, core=torch.bool)
    if supervision:
        dtypes.update(labels=torch.int64, weights=torch.float32, whole_sizes=torch.float32)
    return {name: torch.from_numpy(block[name]).to(device=device, dtype=dtype)
            for name, dtype in dtypes.items()}
