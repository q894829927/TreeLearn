"""Training helpers for proposal-relation models."""

import csv
import json
from pathlib import Path
import numpy as np
from .proposal_relation import load_inference_graph


def load_learning_forest(graph_path, learning_path):
    graph, metadata = load_inference_graph(graph_path)
    with np.load(learning_path, allow_pickle=False) as payload:
        targets = {key: payload[key] for key in payload.files}
    required = {
        'node_gt', 'node_purity', 'edge_label', 'edge_valid',
        'point_gt', 'baseline_predictions'}
    if required-set(targets):
        raise ValueError(
            f'Learning artifact misses {sorted(required-set(targets))}.')
    if len(targets['node_gt']) != len(graph['node_features']):
        raise ValueError('Learning node targets do not align with graph nodes.')
    if len(targets['edge_label']) != graph['edge_index'].shape[1]:
        raise ValueError('Learning edge targets do not align with graph edges.')
    if (len(targets['point_gt']) != len(graph['point_node_id']) or
            len(targets['baseline_predictions']) !=
            len(graph['point_node_id'])):
        raise ValueError('Learning point arrays do not align with graph points.')
    return graph, targets, metadata


def read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    required = {'plot', 'split', 'graph_path', 'learning_path'}
    if not rows or required-set(rows[0]):
        raise ValueError('Proposal-relation manifest is empty or incomplete.')
    train = {row['plot'] for row in rows if row['split'] == 'train'}
    validation = {row['plot'] for row in rows if row['split'] == 'validation'}
    if train & validation:
        raise ValueError(f'Plot leakage in manifest: {sorted(train & validation)}')
    return rows


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': int(epoch), 'metrics': dict(metrics), 'config': dict(config),
    }, path)


def load_checkpoint(path, model, map_location='cpu'):
    import torch
    try:
        payload = torch.load(
            path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload['model'])
    return payload


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
