"""V2 keeps every cross-attention layer and adds center-guided masks."""

from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import spconv.pytorch as spconv

from tree_learn.forest_abc.model import ForestABC, tensor_block as original_tensor_block
from tree_learn.forest_abc.modules import RegionGatedQueryUpdate
from tree_learn.forest_abc.vendor.decoder import ForAINetv2QueryDecoder_XAwarequery


class CenterDecoder(ForAINetv2QueryDecoder_XAwarequery):
    def __init__(self, config):
        super().__init__(num_layers=config['layers'], num_instance_queries=0,
                         num_semantic_queries=2, num_classes=1, in_channels=config['channels'],
                         d_model=config['dim'], num_heads=config['heads'],
                         hidden_dim=config['feedforward'], dropout=0., activation_fn='gelu',
                         iter_pred=True, attn_mask=True, fix_attention=True, objectness_flag=True)
        dim = config['dim']
        self.region_updates = nn.ModuleDict({str(i): RegionGatedQueryUpdate(dim)
                                             for i in range(1, config['layers'])})
        self.source_position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query_position = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.center_strength = nn.Parameter(torch.tensor(-2.))
        self.bandwidth = float(config['center_bandwidth'])
        self.position_scale = float(config['position_scale'])
        self.checkpoint_layers = config['checkpoint_decoder']

    def head(self, queries, mask_features, center_cost):
        normalized = self.out_norm(queries)
        scores = self.out_score(normalized).squeeze(-1)
        masks = normalized @ mask_features.T - self.center_strength.sigmoid() * center_cost
        attention = masks.detach() < 0
        attention[attention.all(-1)] = False
        return dict(masks=masks, scores=scores), attention

    def update(self, queries, sources, masks, attention, layer):
        # All six cross-attention paths remain active. MFA supplements them.
        queries = self.cross_attn_layers[layer]([sources], [queries], [attention])[0]
        if str(layer) in self.region_updates:
            queries = self.region_updates[str(layer)](queries, sources, masks)
        queries = self.self_attn_layers[layer]([queries])[0]
        return self.ffn_layers[layer]([queries])[0]

    def forward(self, features, xyz, votes, anchors):
        sources = self.input_proj(features) + self.source_position(xyz / self.position_scale)
        mask_features = self.x_mask(features)
        queries = self._get_queries([features[anchors]], 1)[0]
        position = self.query_position(votes[anchors] / self.position_scale)
        queries = queries + torch.cat((position, position.new_zeros((2, position.shape[1]))))
        # QxN, not QxNxD. Semantic queries receive no center penalty.
        delta = (votes[anchors].square().sum(-1)[:, None] + votes.square().sum(-1)[None]
                 - 2 * votes[anchors] @ votes.T).clamp_min(0)
        cost = (delta / self.bandwidth ** 2).clamp_max(16.)
        cost = torch.cat((cost, cost.new_zeros((2, len(xyz)))))
        prediction, attention = self.head(queries, mask_features, cost)
        outputs = [prediction] if self.training else []
        for layer in range(len(self.cross_attn_layers)):
            if self.training and self.checkpoint_layers:
                queries = checkpoint(partial(self.update, layer=layer), queries, sources,
                                     prediction['masks'], attention, use_reentrant=False)
            else:
                queries = self.update(queries, sources, prediction['masks'], attention, layer)
            prediction, attention = self.head(queries, mask_features, cost)
            if self.training:
                outputs.append(prediction)
        return outputs if self.training else [prediction]


class CenterForestModel(ForestABC):
    def __init__(self, config):
        # Reuse the proven sparse backbone, point heads and query selection.
        super().__init__(dict(config, use_mfa=False))
        self.config = dict(config)
        self.decoder = CenterDecoder(config)
        self.offset = nn.Sequential(nn.Linear(config['channels'], config['channels']), nn.ReLU(),
                                    nn.Linear(config['channels'], 2))

    def forward(self, block, teacher_fraction=0.):
        xyz, grid = block['xyz'], block['grid']
        indices = torch.cat((torch.zeros_like(grid[:, :1]), grid), 1).int()
        shape = torch.clamp(grid.max(0).values + 1, min=128).tolist()
        sparse = spconv.SparseConvTensor(xyz - xyz.mean(0), indices, shape, 1)
        sparse = self.backbone(self.input_conv(sparse))
        if not torch.equal(sparse.indices, indices):
            raise RuntimeError('Sparse U-Net changed input ordering.')
        features = self.output_norm(sparse.features)
        if self.geometry is not None:
            features = self.geometry(features, block['neighbors'])
        semantic, embeddings = self.semantic(features), self.embedding(features)
        offsets = self.offset(features)
        votes = xyz[:, :2] + offsets
        anchors = self.select_queries(semantic, embeddings, block['core'],
                                      block.get('labels') if self.training else None,
                                      teacher_fraction if self.training else 0.)
        layers = self.decoder(features, xyz, votes, anchors)
        return dict(layers=layers, semantic=semantic, embedding=embeddings, offsets=offsets,
                    centers=votes[anchors], anchors=anchors, num_voxels=len(xyz))


def tensor_block(block, device, supervision=False):
    result = original_tensor_block(block, device, supervision=supervision)
    if supervision:
        result['center_offsets'] = torch.from_numpy(block['center_offsets']).to(device=device, dtype=torch.float32)
    return result
