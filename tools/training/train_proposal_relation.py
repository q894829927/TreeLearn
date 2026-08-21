"""Train and compare P2 proposal-relation models on fixed forests."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from tree_learn.util.omission_diagnostics import detection_metrics
from tree_learn.util.proposal_relation import (
    EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, EdgeMLP,
    VerticalRelationAttention, build_parameter_matched_edge_mlp,
    decode_relation_graph, evaluate_instances, focal_edge_loss,
    parameter_count, point_instances_from_nodes, supervised_contrastive_loss)
from tree_learn.util.proposal_relation_learning import (
    load_checkpoint, load_learning_forest, save_checkpoint)


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_split(root, plots, split):
    forests = []
    for plot in plots:
        directory = Path(root) / split / plot
        graph, targets, metadata = load_learning_forest(
            directory / 'graph.npz', directory / 'learning.npz')
        if metadata['source_plot'] != plot or metadata['split'] != split:
            raise ValueError(f'P1 metadata mismatch for {plot}.')
        forests.append((plot, graph, targets))
    return forests


def _normalization(forests):
    nodes = np.concatenate([g['node_features'] for _, g, _ in forests], axis=0)
    edges = np.concatenate([g['edge_features'] for _, g, _ in forests], axis=0)
    return {
        'node_mean': nodes.mean(0).astype(np.float32),
        'node_std': np.maximum(nodes.std(0), 1e-6).astype(np.float32),
        'edge_mean': edges.mean(0).astype(np.float32),
        'edge_std': np.maximum(edges.std(0), 1e-6).astype(np.float32),
    }


def _tensor_forest(graph, targets, norm, device):
    nodes = (graph['node_features']-norm['node_mean'])/norm['node_std']
    edges = (graph['edge_features']-norm['edge_mean'])/norm['edge_std']
    return (
        torch.as_tensor(nodes, dtype=torch.float32, device=device),
        torch.as_tensor(graph['edge_index'], dtype=torch.long, device=device),
        torch.as_tensor(edges, dtype=torch.float32, device=device),
        torch.as_tensor(targets['node_gt'], dtype=torch.long, device=device),
        torch.as_tensor(targets['edge_label'], dtype=torch.float32, device=device),
        torch.as_tensor(targets['edge_valid'], dtype=torch.bool, device=device),
    )


def _balanced_mask(labels, valid, sources, generator, negative_ratio):
    """Balance adjacent hard negatives independently for each source node."""
    mask = torch.zeros_like(valid)
    for source in torch.unique(sources[valid]):
        local = valid & (sources == source)
        positive = torch.flatnonzero(local & (labels > .5))
        negative = torch.flatnonzero(local & (labels <= .5))
        if len(positive):
            keep = min(len(negative), max(int(len(positive)*negative_ratio), 1))
        else:
            keep = min(len(negative), 1)
        if keep and keep < len(negative):
            order = torch.randperm(
                len(negative), generator=generator,
                device=negative.device)[:keep]
            negative = negative[order]
        mask[positive] = True
        mask[negative] = True
    return mask


def _balanced_contrastive_nodes(labels, generator, max_trees=64,
                                nodes_per_tree=8):
    """Bound the quadratic contrastive term while balancing GT trees."""
    trees = torch.unique(labels[labels > 0])
    if len(trees) > max_trees:
        order = torch.randperm(
            len(trees), generator=generator, device=trees.device)[:max_trees]
        trees = trees[order]
    selected = []
    for tree in trees:
        indices = torch.flatnonzero(labels == tree)
        if len(indices) > nodes_per_tree:
            order = torch.randperm(
                len(indices), generator=generator,
                device=indices.device)[:nodes_per_tree]
            indices = indices[order]
        selected.append(indices)
    return (torch.cat(selected) if selected else
            torch.empty(0, dtype=torch.long, device=labels.device))


def _build_model(name, settings, target_parameters=None):
    common = dict(
        node_input_dim=len(NODE_FEATURE_NAMES),
        edge_input_dim=len(EDGE_FEATURE_NAMES))
    if name in ('vertical_relation_attention', 'relation_attention_no_vertical'):
        return VerticalRelationAttention(
            **common, node_dim=int(settings['node_dim']),
            edge_dim=int(settings['edge_dim']),
            num_layers=int(settings['num_layers']),
            num_heads=int(settings['num_heads']),
            ffn_dim=int(settings['ffn_dim']),
            dropout=float(settings['dropout']),
            use_vertical=name == 'vertical_relation_attention')
    if name == 'edge_mlp':
        if target_parameters is None:
            raise ValueError('Edge-MLP requires a target parameter count.')
        return build_parameter_matched_edge_mlp(
            target_parameters, tolerance=float(settings['parameter_tolerance']),
            **common, dropout=float(settings['dropout']))
    raise ValueError(f'Unknown trainable model: {name}')


def _geometry_scores(graph):
    names = {name: i for i, name in enumerate(EDGE_FEATURE_NAMES)}
    edge = graph['edge_features']
    if len(edge) == 0:
        return np.empty(0, np.float32)
    cost = (
        edge[:, names['xy_bbox_gap']]/.6 +
        edge[:, names['vertical_gap']]/1.5 +
        edge[:, names['xy_distance']]/2.5 -
        edge[:, names['vertical_histogram_cosine']])
    return (1/(1+np.exp(np.clip(cost, -20, 20)))).astype(np.float32)


@torch.no_grad()
def _score_model(model, forests, norm, device):
    model.eval()
    results, all_scores, all_targets = {}, [], []
    attention_values = []
    for plot, graph, targets in forests:
        nodes, edge_index, edges, _, labels, valid = _tensor_forest(
            graph, targets, norm, device)
        output = model(nodes, edge_index, edges, return_attention=True)
        scores = torch.sigmoid(output['edge_logits']).cpu().numpy()
        results[plot] = scores
        all_scores.append(scores[valid.cpu().numpy()])
        all_targets.append(labels.cpu().numpy()[valid.cpu().numpy()])
        for attention in output.get('attention', []):
            if attention.numel():
                attention_values.append(attention.detach().cpu().numpy().ravel())
    scores = np.concatenate(all_scores) if all_scores else np.empty(0)
    targets = np.concatenate(all_targets) if all_targets else np.empty(0)
    ap = float(average_precision_score(targets, scores)) if len(np.unique(targets)) > 1 else 0.
    values = np.concatenate(attention_values) if attention_values else np.empty(0)
    stats = {
        'finite': bool(np.all(np.isfinite(values))) if len(values) else True,
        'mean': float(values.mean()) if len(values) else None,
        'std': float(values.std()) if len(values) else None,
        'saturated_rate': float(np.mean((values < 1e-4) | (values > 1-1e-4)))
        if len(values) else None,
    }
    return results, ap, stats


def _aggregate_at_threshold(forests, scores_by_plot, threshold, settings):
    per_plot, totals = [], {'tp': 0, 'fp': 0, 'fn': 0}
    for plot, graph, targets in forests:
        nodes = decode_relation_graph(
            graph, scores_by_plot[plot], threshold,
            reciprocal_top_k=int(settings['reciprocal_top_k']),
            max_component_xy_diameter=float(settings['max_component_xy_diameter']),
            max_component_height=float(settings['max_component_height']))
        predictions = point_instances_from_nodes(graph['point_node_id'], nodes)
        metrics, _, _ = evaluate_instances(
            targets['point_gt'], predictions,
            float(settings['match_iou_threshold']),
            float(settings['min_precision_for_counted_fp']))
        base, _, _ = evaluate_instances(
            targets['point_gt'], targets['baseline_predictions'],
            float(settings['match_iou_threshold']),
            float(settings['min_precision_for_counted_fp']))
        row = {'plot': plot, **metrics, 'baseline_f1': base['f1']}
        row['f1_gain_pp'] = 100*(metrics['f1']-base['f1'])
        per_plot.append(row)
        for key in totals:
            totals[key] += int(metrics[key])
    aggregate = detection_metrics(totals['tp'], totals['fp'], totals['fn'])
    return aggregate, per_plot


def _select_threshold(forests, scores, settings):
    candidates = []
    for threshold in settings['threshold_grid']:
        aggregate, per_plot = _aggregate_at_threshold(
            forests, scores, float(threshold), settings)
        candidates.append({
            'threshold': float(threshold), 'aggregate': aggregate,
            'per_plot': per_plot,
            'nonnegative_plots': sum(r['f1_gain_pp'] >= 0 for r in per_plot),
            'worst_plot_f1_gain_pp': min(r['f1_gain_pp'] for r in per_plot)})
    return max(candidates, key=lambda r: (
        r['aggregate']['f1'], r['aggregate']['completeness'],
        -r['aggregate']['commission'], -r['threshold'])), candidates


def _train(name, seed, train, validation, norm, settings, output, target_parameters):
    _seed(seed)
    device = torch.device(settings['device'] if torch.cuda.is_available() else 'cpu')
    model = _build_model(name, settings, target_parameters).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings['learning_rate']),
        weight_decay=float(settings['weight_decay']))
    generator = torch.Generator(device=device).manual_seed(seed)
    best = None
    checkpoint = output / 'checkpoints' / f'{name}_seed{seed}.pth'
    patience = 0
    for epoch in range(1, int(settings['max_epochs'])+1):
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(len(train))
        losses = []
        for index in order:
            _, graph, targets = train[int(index)]
            nodes, edge_index, edges, node_gt, labels, valid = _tensor_forest(
                graph, targets, norm, device)
            selected = _balanced_mask(
                labels, valid, edge_index[0], generator,
                float(settings['negative_ratio']))
            output_values = model(nodes, edge_index, edges)
            if not torch.any(selected):
                continue
            positives = max(int(torch.sum(labels[selected] > .5)), 1)
            negatives = max(int(torch.sum(labels[selected] <= .5)), 1)
            edge_loss = focal_edge_loss(
                output_values['edge_logits'][selected], labels[selected],
                gamma=float(settings['focal_gamma']),
                positive_weight=negatives/positives)
            contrastive_nodes = _balanced_contrastive_nodes(
                node_gt, generator,
                max_trees=int(settings['contrastive_max_trees']),
                nodes_per_tree=int(settings['contrastive_nodes_per_tree']))
            contrast = supervised_contrastive_loss(
                output_values['node_embeddings'][contrastive_nodes],
                node_gt[contrastive_nodes],
                temperature=float(settings['contrastive_temperature']))
            reliability_target = (node_gt > 0).to(
                output_values['node_reliability_logits'].dtype)
            if name.startswith('relation_attention'):
                reliability = F.binary_cross_entropy_with_logits(
                    output_values['node_reliability_logits'],
                    reliability_target)
            else:
                reliability = edge_loss.new_zeros(())
            loss = (edge_loss +
                    float(settings['contrastive_weight'])*contrast +
                    float(settings['reliability_weight'])*reliability)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite P2 loss for {name} seed {seed}.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scores, ap, attention = _score_model(model, validation, norm, device)
        chosen, _ = _select_threshold(validation, scores, settings)
        key = (ap, chosen['aggregate']['f1'])
        print(f'[{name} seed {seed}] epoch {epoch:03d}: '
              f'loss={np.mean(losses):.5f}, AP={ap:.6f}, '
              f'F1={100*chosen["aggregate"]["f1"]:.3f}%', flush=True)
        if best is None or key > best['key']:
            best = {'key': key, 'epoch': epoch}
            save_checkpoint(checkpoint, model, optimizer, epoch,
                            {'edge_ap': ap, 'f1': chosen['aggregate']['f1']},
                            {'model': name, **settings})
            patience = 0
        else:
            patience += 1
        if patience >= int(settings['patience']):
            break
    payload = load_checkpoint(checkpoint, model, map_location=device)
    scores, ap, attention = _score_model(model, validation, norm, device)
    chosen, curves = _select_threshold(validation, scores, settings)
    return {
        'model': name, 'seed': seed, 'epoch': int(payload['epoch']),
        'parameter_count': parameter_count(model), 'edge_ap': ap,
        'attention': attention, **chosen, 'curves': curves,
        'checkpoint': str(checkpoint)}


def _baseline_metrics(forests, settings):
    totals = {'tp': 0, 'fp': 0, 'fn': 0}
    per_plot = {}
    for plot, _, targets in forests:
        metrics, _, _ = evaluate_instances(
            targets['point_gt'], targets['baseline_predictions'],
            float(settings['match_iou_threshold']),
            float(settings['min_precision_for_counted_fp']))
        per_plot[plot] = metrics
        for key in totals:
            totals[key] += int(metrics[key])
    return detection_metrics(totals['tp'], totals['fp'], totals['fn']), per_plot


def _compact(result):
    return {key: value for key, value in result.items()
            if key not in {'curves'}}


def _summarize(results, baseline, settings, output):
    by_model = {}
    for name in ('fixed_geometry', 'edge_mlp', 'relation_attention_no_vertical',
                 'vertical_relation_attention'):
        runs = [r for r in results if r['model'] == name]
        if not runs:
            continue
        by_model[name] = {
            'edge_ap_mean': float(np.mean([r['edge_ap'] for r in runs])),
            'f1_mean': float(np.mean([r['aggregate']['f1'] for r in runs])),
            'f1_std': float(np.std([r['aggregate']['f1'] for r in runs], ddof=1))
            if len(runs) > 1 else 0.,
            'completeness_mean': float(np.mean(
                [r['aggregate']['completeness'] for r in runs])),
            'commission_mean': float(np.mean(
                [r['aggregate']['commission'] for r in runs])),
            'parameter_count': int(runs[0]['parameter_count']),
        }
    c1, m1 = by_model['edge_mlp'], by_model['vertical_relation_attention']
    m1_runs = [r for r in results if r['model'] == 'vertical_relation_attention']
    c1_runs = {r['seed']: r for r in results if r['model'] == 'edge_mlp'}
    gate_cfg = settings['gate']
    gate = {
        'edge_ap_improved': m1['edge_ap_mean']-c1['edge_ap_mean'] >= float(gate_cfg['minimum_edge_ap_gain']),
        'edge_ap_seed_wins': sum(r['edge_ap'] > c1_runs[r['seed']]['edge_ap'] for r in m1_runs) >= 2,
        'f1_not_below_b1': 100*(m1['f1_mean']-baseline['f1']) >= -float(gate_cfg['maximum_f1_drop_pp']),
        'completeness_preserved': 100*(baseline['completeness']-m1['completeness_mean']) <= float(gate_cfg['maximum_completeness_drop_pp']),
        'commission_preserved': 100*(m1['commission_mean']-baseline['commission']) <= float(gate_cfg['maximum_commission_increase_pp']),
        'plot_consistency': sum(
            np.mean([next(x['f1_gain_pp'] for x in r['per_plot'] if x['plot'] == plot)
                     for r in m1_runs]) >= 0
            for plot in [x['plot'] for x in m1_runs[0]['per_plot']]) >= 3,
        'worst_plot_preserved': min(
            x['f1_gain_pp'] for r in m1_runs for x in r['per_plot']) >= -float(gate_cfg['maximum_plot_f1_drop_pp']),
        'seed_stability': 100*m1['f1_std'] <= float(gate_cfg['maximum_f1_std_pp']),
        'attention_valid': all(
            r['attention']['finite'] and
            r['attention']['std'] is not None and r['attention']['std'] > 1e-6 and
            r['attention']['saturated_rate'] < .99 for r in m1_runs),
        'parameter_count_matched': abs(m1['parameter_count']-c1['parameter_count']) /
            max(m1['parameter_count'], 1) <= float(settings['parameter_tolerance']),
    }
    gate['passed'] = all(gate.values())
    locked = max(m1_runs, key=lambda r: (
        r['aggregate']['f1'], r['aggregate']['completeness'],
        -r['aggregate']['commission'], -r['seed']))
    summary = {
        'baseline': baseline, 'models': by_model, 'gate': gate,
        'locked_model': 'vertical_relation_attention' if gate['passed'] else None,
        'locked_seed': int(locked['seed']) if gate['passed'] else None,
        'locked_threshold': float(locked['threshold']) if gate['passed'] else None,
        'locked_checkpoint': locked['checkpoint'] if gate['passed'] else None,
        'runs': [_compact(r) for r in results],
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# P2 垂直关系注意力模型比较', '',
             f'- B1 F1: {100*baseline["f1"]:.3f}%', '',
             '| Model | Edge AP | F1 | Completeness | Commission | Params |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in by_model.items():
        lines.append(f'| {name} | {row["edge_ap_mean"]:.6f} | '
                     f'{100*row["f1_mean"]:.3f}% | '
                     f'{100*row["completeness_mean"]:.3f}% | '
                     f'{100*row["commission_mean"]:.3f}% | '
                     f'{row["parameter_count"]:,} |')
    lines += ['', '## Gate', ''] + [
        f'- {key}: **{value}**' for key, value in gate.items()]
    lines += ['', 'PASS：锁定 P3。' if gate['passed']
              else 'STOP：不得运行 Wytham。', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return summary


def run(config_path):
    raw = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    settings = raw['proposal_relation_training']
    gate_path = Path(settings['data_root']) / 'generation_summary.json'
    data_gate = json.loads(gate_path.read_text(encoding='utf-8'))['gate']['passed']
    if not data_gate:
        raise RuntimeError('P1 data gate did not pass.')
    train = _load_split(settings['data_root'], settings['train_plots'], 'train')
    validation = _load_split(
        settings['data_root'], settings['validation_plots'], 'validation')
    norm = _normalization(train)
    output = Path(settings['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'normalization.npz', **norm)
    baseline, _ = _baseline_metrics(validation, settings)
    results = []
    geometry_scores = {plot: _geometry_scores(graph)
                       for plot, graph, _ in validation}
    selected, curves = _select_threshold(
        validation, geometry_scores, settings)
    results.append({
        'model': 'fixed_geometry', 'seed': int(settings['seeds'][0]),
        'epoch': 0, 'parameter_count': 0, 'edge_ap': 0.,
        'attention': {'finite': True, 'mean': None, 'std': None,
                      'saturated_rate': None},
        **selected, 'curves': curves, 'checkpoint': None})
    target = parameter_count(_build_model(
        'vertical_relation_attention', settings))
    for name in ('edge_mlp', 'relation_attention_no_vertical',
                 'vertical_relation_attention'):
        for seed in settings['seeds']:
            results.append(_train(
                name, int(seed), train, validation, norm, settings,
                output, target))
    summary = _summarize(results, baseline, settings, output)
    with (output / 'per_seed_metrics.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        fields = ['model', 'seed', 'epoch', 'parameter_count', 'edge_ap',
                  'threshold', 'f1', 'completeness', 'commission',
                  'nonnegative_plots', 'worst_plot_f1_gain_pp']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'model': r['model'], 'seed': r['seed'], 'epoch': r['epoch'],
                'parameter_count': r['parameter_count'], 'edge_ap': r['edge_ap'],
                'threshold': r['threshold'], 'f1': r['aggregate']['f1'],
                'completeness': r['aggregate']['completeness'],
                'commission': r['aggregate']['commission'],
                'nonnegative_plots': r['nonnegative_plots'],
                'worst_plot_f1_gain_pp': r['worst_plot_f1_gain_pp']})
    if not summary['gate']['passed']:
        raise RuntimeError('P2 gate failed; do not run Wytham.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
