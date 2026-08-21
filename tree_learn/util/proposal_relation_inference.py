"""Inference bridge from TreeLearn semantics to relation instances."""

import json
from pathlib import Path
import time
import numpy as np
import torch
from .proposal_relation import (
    VerticalRelationAttention, build_proposal_graph,
    build_vertical_micro_proposals, decode_relation_graph,
    point_instances_from_nodes, save_inference_graph)


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_proposal_relation_instances(
        coords, semantic_logits, verticality, settings, output_dir=None,
        logger=None):
    start = time.time()
    summary = json.loads(Path(str(settings.summary)).read_text(encoding='utf-8'))
    if not bool(summary['gate']['passed']):
        raise RuntimeError('P2 gate did not pass; relation inference is locked.')
    checkpoint = str(summary.get('locked_checkpoint') or
                     summary.get('recommended_checkpoint') or '')
    threshold_value = summary.get('locked_threshold',
                                  summary.get('recommended_threshold'))
    if threshold_value is None:
        raise ValueError('P2 summary does not contain a locked threshold.')
    threshold = float(threshold_value)
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(f'Missing locked relation checkpoint: {checkpoint}')
    with np.load(str(settings.normalization), allow_pickle=False) as payload:
        normalization = {key: payload[key] for key in payload.files}
    proposals = build_vertical_micro_proposals(
        coords, semantic_logits, verticality=verticality,
        tree_probability_threshold=float(settings.tree_probability_threshold),
        xy_cell_size=float(settings.xy_cell_size),
        z_bin_size=float(settings.z_bin_size),
        vertical_split_gap=float(settings.vertical_split_gap),
        tree_class_index=int(settings.tree_class_index))
    graph = build_proposal_graph(
        proposals, max_xy_distance=float(settings.max_xy_distance),
        max_xy_bbox_gap=float(settings.max_xy_bbox_gap),
        max_vertical_gap=float(settings.max_vertical_gap),
        max_neighbors=int(settings.max_neighbors))
    device_name = str(getattr(settings, 'device', 'cuda'))
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    model = VerticalRelationAttention(
        node_dim=int(settings.node_dim), edge_dim=int(settings.edge_dim),
        num_layers=int(settings.num_layers), num_heads=int(settings.num_heads),
        ffn_dim=int(settings.ffn_dim), dropout=float(settings.dropout),
        use_vertical=True).to(device)
    model.load_state_dict(_load_checkpoint(checkpoint, device)['model'])
    model.eval()
    node = ((np.asarray(graph['node_features'], np.float32) -
             normalization['node_mean']) / normalization['node_std'])
    edge = ((np.asarray(graph['edge_features'], np.float32) -
             normalization['edge_mean']) / normalization['edge_std'])
    with torch.inference_mode():
        result = model(
            torch.as_tensor(node, device=device),
            torch.as_tensor(graph['edge_index'], dtype=torch.long, device=device),
            torch.as_tensor(edge, device=device))
        scores = torch.sigmoid(result['edge_logits']).cpu().numpy()
    node_instances = decode_relation_graph(
        graph, scores, threshold,
        reciprocal_top_k=int(settings.reciprocal_top_k),
        max_component_xy_diameter=float(settings.max_component_xy_diameter),
        max_component_height=float(settings.max_component_height))
    predictions = point_instances_from_nodes(
        graph['point_node_id'], node_instances)
    elapsed = time.time()-start
    stats = {
        'num_points': int(len(coords)),
        'num_nodes': int(len(graph['node_features'])),
        'num_edges': int(graph['edge_index'].shape[1]),
        'num_instances': int(len(np.unique(node_instances))),
        'threshold': threshold, 'checkpoint': checkpoint,
        'seconds': float(elapsed), 'offset_free': True, 'hdbscan_free': True}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if bool(getattr(settings, 'save_graph', False)):
            save_inference_graph(output_dir/'graph.npz', graph, {
                'checkpoint': checkpoint, 'split': 'inference',
                'threshold': threshold})
        np.savez_compressed(
            output_dir/'scores.npz', edge_scores=scores,
            node_instances=node_instances)
        (output_dir/'summary.json').write_text(
            json.dumps(stats, indent=2), encoding='utf-8')
    if logger is not None:
        logger.info(
            'Proposal-relation grouping produced %s instances from %s nodes '
            'and %s edges in %.1fs (offset/HDBSCAN bypassed).',
            f'{stats["num_instances"]:,}', f'{stats["num_nodes"]:,}',
            f'{stats["num_edges"]:,}', elapsed)
    return predictions, stats
