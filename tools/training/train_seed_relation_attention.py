"""Train E1c parameter-matched neighbourhood MLP and relation attention."""

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
E1B_PATH = ROOT / 'tools' / 'training' / 'train_seed_quality_mlp.py'
E1B_SPEC = importlib.util.spec_from_file_location(
    'seed_quality_mlp_for_e1c', E1B_PATH)
E1B = importlib.util.module_from_spec(E1B_SPEC)
E1B_SPEC.loader.exec_module(E1B)
MODEL_PATH = ROOT / 'tree_learn' / 'util' / 'seed_attention.py'
MODEL_SPEC = importlib.util.spec_from_file_location(
    'seed_attention_for_e1c', MODEL_PATH)
SEED_ATTENTION = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(SEED_ATTENTION)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def load_relation_dataset(settings):
    dataset = E1B.load_seed_quality_dataset(
        settings['data_root'], settings['manifest_path'],
        int(settings['expected_backbone_dim']),
        int(settings['expected_scalar_dim']))
    source_rows = read_csv(settings['manifest_path'])
    neighbour_rows = read_csv(settings['neighbor_manifest_path'])
    neighbour_map = {row['source_plot']: row for row in neighbour_rows}
    if set(neighbour_map) != {row['source_plot'] for row in source_rows}:
        raise ValueError('E1a and E1c-a manifests contain different forests.')

    total = len(dataset['features'])
    k = int(settings['neighborhood']['num_neighbors'])
    coords = np.empty((total, 3), dtype=np.float32)
    votes = np.empty((total, 2), dtype=np.float32)
    neighbour_index = np.empty((total, k), dtype=np.int32)
    neighbour_valid = np.empty((total, k), dtype=bool)
    target_is_tree = np.empty(total, dtype=bool)
    offset = 0
    for row in source_rows:
        plot = row['source_plot']
        count = int(row['num_candidates'])
        end = offset + count
        neighbour_row = neighbour_map[plot]
        if neighbour_row['split'] != row['split']:
            raise ValueError(f'Neighbour split differs for {plot}.')
        with np.load(row['artifact_path'], allow_pickle=False) as source:
            source_candidates = source['candidate_indices']
            coords[offset:end] = source['coords']
            votes[offset:end] = source['base_votes_xy']
            target_is_tree[offset:end] = source['target_is_tree']
        with np.load(
                neighbour_row['neighbor_artifact_path'],
                allow_pickle=False) as cache:
            cache_candidates = cache['candidate_indices']
            local_neighbours = cache['neighbor_indices']
        if not np.array_equal(source_candidates, cache_candidates):
            raise ValueError(f'Neighbour candidates differ for {plot}.')
        if local_neighbours.shape != (count, k):
            raise ValueError(f'Neighbour shape differs for {plot}.')
        valid = local_neighbours >= 0
        if np.any(local_neighbours[valid] >= count):
            raise ValueError(f'Neighbour index is out of range for {plot}.')
        fallback = np.arange(offset, end, dtype=np.int32)[:, None]
        global_neighbours = np.where(
            valid, local_neighbours + offset, fallback)
        neighbour_index[offset:end] = global_neighbours.astype(np.int32)
        neighbour_valid[offset:end] = valid
        if not np.all(valid[:, 0]):
            raise ValueError(f'Self support is missing for {plot}.')
        if not np.array_equal(
                global_neighbours[:, 0],
                np.arange(offset, end, dtype=np.int32)):
            raise ValueError(f'Self support is misaligned for {plot}.')
        offset = end
        print(f'Loaded relation data {plot}: {count:,} candidates', flush=True)
    if offset != total:
        raise ValueError('Relation data do not cover the E1b dataset.')
    dataset.update({
        'coords': coords,
        'base_votes_xy': votes,
        'neighbor_index': neighbour_index,
        'neighbor_valid': neighbour_valid,
        'target_is_tree': target_is_tree,
    })
    return dataset


def prepare_relation_batch(dataset, indices, mean, scale, relation_config):
    indices = np.asarray(indices, dtype=np.int64)
    neighbours = dataset['neighbor_index'][indices].astype(np.int64)
    valid = dataset['neighbor_valid'][indices]
    query_features = E1B.standardize_features(
        dataset['features'][indices], mean, scale)
    neighbour_features = E1B.standardize_features(
        dataset['features'][neighbours], mean, scale)
    radius = float(relation_config['radius'])
    vertical_scale = float(relation_config['vertical_scale'])
    vote_delta = (
        dataset['base_votes_xy'][neighbours] -
        dataset['base_votes_xy'][indices, None, :]
    ) / radius
    point_delta = (
        dataset['coords'][neighbours] -
        dataset['coords'][indices, None, :])
    geometry = np.concatenate([
        vote_delta,
        np.linalg.norm(vote_delta, axis=2, keepdims=True),
        point_delta[:, :, :2] / radius,
        point_delta[:, :, 2:3] / vertical_scale,
    ], axis=2).astype(np.float32)
    geometry[~valid] = 0.0
    if not np.isfinite(geometry).all():
        raise ValueError('Relative seed geometry is non-finite.')
    return query_features, neighbour_features, geometry, valid


def evaluate_relation_scores(
        reliability_scores, coverage_scores, dataset, mask, selector):
    reliable = dataset['target_reliable'][mask]
    critical = dataset['target_coverage_critical'][mask]
    utility = dataset['target_utility'][mask]
    tree_ids = dataset['target_tree_id'][mask]
    plot_ids = dataset['plot_id'][mask]
    is_tree = dataset['target_is_tree'][mask]
    metrics = E1B.evaluate_seed_scores(
        reliability_scores, coverage_scores,
        reliable, critical, utility, tree_ids, plot_ids, selector)
    metrics['tree_detection'] = E1B.binary_metrics(
        is_tree, reliability_scores)
    metrics['tree_only_reliability'] = E1B.binary_metrics(
        reliable[is_tree], reliability_scores[is_tree])
    return metrics


def model_parameter_counts(settings):
    models = {}
    for model_type in settings['model_types']:
        model = SEED_ATTENTION.build_seed_relation_model(
            int(settings['expected_input_dim']),
            int(settings['geometry_dim']), model_type,
            settings['model'])
        models[model_type] = SEED_ATTENTION.parameter_count(model)
    return models


def predict_in_chunks(
        model, dataset, eligible_indices, mean, scale,
        relation_config, batch_size, device, amp):
    import torch

    reliability = np.empty(len(eligible_indices), dtype=np.float32)
    coverage = np.empty(len(eligible_indices), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eligible_indices), int(batch_size)):
            end = min(start + int(batch_size), len(eligible_indices))
            indices = eligible_indices[start:end]
            query, neighbours, geometry, valid = prepare_relation_batch(
                dataset, indices, mean, scale, relation_config)
            tensors = [
                torch.from_numpy(query).to(device),
                torch.from_numpy(neighbours).to(device),
                torch.from_numpy(geometry).to(device),
                torch.from_numpy(valid).to(device),
            ]
            with torch.cuda.amp.autocast(
                    enabled=bool(amp and device.type == 'cuda')):
                reliability_logits, coverage_logits = model(*tensors)
            reliability[start:end] = torch.sigmoid(
                reliability_logits).float().cpu().numpy()
            coverage[start:end] = torch.sigmoid(
                coverage_logits).float().cpu().numpy()
    return reliability, coverage


def train_one_model_seed(dataset, settings, model_type, seed):
    import torch
    from torch import nn

    E1B.set_deterministic_seed(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    training = settings['training']
    train_mask = dataset['split_id'] == 0
    validation_mask = dataset['split_id'] == 1
    validation_indices = np.flatnonzero(validation_mask)
    mean, scale, standardizer_indices = E1B.fit_sampled_standardizer(
        dataset['features'], dataset['plot_id'], train_mask,
        int(training['max_standardizer_points']), int(seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SEED_ATTENTION.build_seed_relation_model(
        dataset['input_dim'], int(settings['geometry_dim']),
        model_type, settings['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training['learning_rate']),
        weight_decay=float(training['weight_decay']))
    amp_enabled = bool(training.get('amp', True) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_reliable = dataset['target_reliable'][train_mask]
    train_critical = dataset['target_coverage_critical'][train_mask]
    max_pos_weight = float(training.get('max_pos_weight', 30.0))
    reliability_pos_weight = min(
        np.count_nonzero(~train_reliable) /
        max(np.count_nonzero(train_reliable), 1), max_pos_weight)
    coverage_pos_weight = min(
        np.count_nonzero(~train_critical) /
        max(np.count_nonzero(train_critical), 1), max_pos_weight)
    reliability_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(reliability_pos_weight, device=device))
    coverage_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(coverage_pos_weight, device=device))
    utility_loss = nn.SmoothL1Loss()

    best_state = None
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    total_epochs = int(training['epochs'])
    warmup_epochs = int(training['warmup_epochs'])
    for epoch in range(1, total_epochs + 1):
        if epoch <= warmup_epochs:
            progress = epoch / max(warmup_epochs, 1)
            learning_rate = (
                float(training['min_learning_rate']) + progress * (
                    float(training['learning_rate']) -
                    float(training['min_learning_rate'])))
        else:
            progress = (
                (epoch - warmup_epochs) /
                max(total_epochs - warmup_epochs, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            learning_rate = (
                float(training['min_learning_rate']) + cosine * (
                    float(training['learning_rate']) -
                    float(training['min_learning_rate'])))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        epoch_indices = E1B.balanced_plot_sample_indices(
            dataset['plot_id'], train_mask,
            int(training['examples_per_plot']),
            int(seed) * 100000 + epoch)
        model.train()
        losses = []
        for start in range(0, len(epoch_indices), int(training['batch_size'])):
            indices = epoch_indices[
                start:start + int(training['batch_size'])]
            query, neighbours, geometry, valid = prepare_relation_batch(
                dataset, indices, mean, scale, settings['neighborhood'])
            tensors = [
                torch.from_numpy(query).to(device),
                torch.from_numpy(neighbours).to(device),
                torch.from_numpy(geometry).to(device),
                torch.from_numpy(valid).to(device),
            ]
            reliable = torch.from_numpy(
                dataset['target_reliable'][indices].astype(np.float32)
            ).to(device)
            critical = torch.from_numpy(
                dataset['target_coverage_critical'][indices].astype(
                    np.float32)).to(device)
            utility = torch.from_numpy(
                dataset['target_utility'][indices].astype(np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                reliability_logits, coverage_logits = model(*tensors)
                loss = (
                    reliability_bce(reliability_logits, reliable) +
                    float(training['coverage_loss_weight']) *
                    coverage_bce(coverage_logits, critical) +
                    float(training['utility_loss_weight']) *
                    utility_loss(
                        torch.sigmoid(reliability_logits), utility))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training['grad_norm_clip']))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        if (
                epoch != 1 and
                epoch % int(training['validation_frequency']) != 0 and
                epoch != total_epochs):
            continue
        reliability_scores, coverage_scores = predict_in_chunks(
            model, dataset, validation_indices, mean, scale,
            settings['neighborhood'],
            int(training['validation_batch_size']), device, amp_enabled)
        metrics = evaluate_relation_scores(
            reliability_scores, coverage_scores,
            dataset, validation_mask, settings['selector'])
        metrics['epoch'] = epoch
        metrics['training_loss'] = float(np.mean(losses))
        metrics['learning_rate'] = learning_rate
        history.append(metrics)
        key = (
            metrics['min_plot_critical_tree_coverage'],
            metrics['dual']['critical_recall'],
            metrics['tree_only_reliability']['roc_auc'],
            metrics['dual']['selected_mean_utility'],
            metrics['coverage']['average_precision'],
        )
        print(
            f'[{model_type} seed {seed}] epoch {epoch:03d}: '
            f'loss={metrics["training_loss"]:.5f}, '
            f'rel_AUC={metrics["reliability"]["roc_auc"]:.5f}, '
            f'tree_AUC='
            f'{metrics["tree_only_reliability"]["roc_auc"]:.5f}, '
            f'critical_recall={metrics["dual"]["critical_recall"]:.5f}',
            flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= int(training['patience']):
            print(
                f'[{model_type} seed {seed}] early stop at epoch {epoch}; '
                f'best={best_epoch}.', flush=True)
            break

    if best_state is None:
        raise RuntimeError('E1c training produced no checkpoint.')
    model.load_state_dict(best_state)
    reliability_scores, coverage_scores = predict_in_chunks(
        model, dataset, validation_indices, mean, scale,
        settings['neighborhood'],
        int(training['validation_batch_size']), device, amp_enabled)
    metrics = evaluate_relation_scores(
        reliability_scores, coverage_scores,
        dataset, validation_mask, settings['selector'])
    return {
        'model_type': model_type,
        'seed': int(seed),
        'best_epoch': int(best_epoch),
        'state_dict': best_state,
        'input_mean': mean,
        'input_scale': scale,
        'metrics': metrics,
        'history': history,
        'reliability_scores': reliability_scores,
        'coverage_scores': coverage_scores,
        'device': str(device),
        'parameter_count': SEED_ATTENTION.parameter_count(model),
        'standardizer_sample_size': int(len(standardizer_indices)),
    }


def flatten_result(result):
    metrics = result['metrics']
    return {
        'model': result['model_type'],
        'seed': result['seed'],
        'best_epoch': result['best_epoch'],
        'device': result['device'],
        'parameter_count': result['parameter_count'],
        'reliability_roc_auc': metrics['reliability']['roc_auc'],
        'reliability_ap': metrics['reliability']['average_precision'],
        'tree_detection_roc_auc': metrics['tree_detection']['roc_auc'],
        'tree_only_roc_auc': metrics['tree_only_reliability']['roc_auc'],
        'tree_only_ap': metrics['tree_only_reliability']['average_precision'],
        'coverage_ap': metrics['coverage']['average_precision'],
        'dual_critical_recall': metrics['dual']['critical_recall'],
        'dual_critical_tree_coverage': (
            metrics['dual']['critical_tree_coverage']),
        'dual_selected_mean_utility': (
            metrics['dual']['selected_mean_utility']),
        'dual_selected_non_tree_rate': (
            metrics['dual']['selected_non_tree_rate']),
        'min_plot_critical_tree_coverage': (
            metrics['min_plot_critical_tree_coverage']),
    }


def aggregate_rows(rows):
    excluded = {'model', 'seed', 'best_epoch', 'device'}
    output = {}
    for name in rows[0]:
        if name in excluded:
            continue
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        output[name] = {
            'mean': float(values.mean()),
            'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return output


def summarize(results, settings, neighbour_summary):
    rows = [flatten_result(result) for result in results]
    by_model = {}
    for model_type in settings['model_types']:
        model_rows = [row for row in rows if row['model'] == model_type]
        by_model[model_type] = aggregate_rows(model_rows)
    control = by_model['neighborhood_mlp']
    attention = by_model['relation_attention']
    parameter_gap = abs(
        attention['parameter_count']['mean'] -
        control['parameter_count']['mean']) / max(
            attention['parameter_count']['mean'],
            control['parameter_count']['mean'], 1)
    deltas = {
        'reliability_roc_auc': (
            attention['reliability_roc_auc']['mean'] -
            control['reliability_roc_auc']['mean']),
        'tree_only_roc_auc': (
            attention['tree_only_roc_auc']['mean'] -
            control['tree_only_roc_auc']['mean']),
        'coverage_ap': (
            attention['coverage_ap']['mean'] -
            control['coverage_ap']['mean']),
        'critical_recall': (
            attention['dual_critical_recall']['mean'] -
            control['dual_critical_recall']['mean']),
        'selected_utility_relative': (
            (attention['dual_selected_mean_utility']['mean'] -
             control['dual_selected_mean_utility']['mean']) /
            max(abs(control['dual_selected_mean_utility']['mean']), 1e-12)),
    }
    control_by_seed = {
        row['seed']: row for row in rows
        if row['model'] == 'neighborhood_mlp'}
    attention_by_seed = {
        row['seed']: row for row in rows
        if row['model'] == 'relation_attention'}
    tree_auc_wins = sum(
        attention_by_seed[seed]['tree_only_roc_auc'] >
        control_by_seed[seed]['tree_only_roc_auc']
        for seed in settings['seeds'])
    gate_cfg = settings['gate']
    gate = {
        'neighbor_data_gate_passed': bool(neighbour_summary['gate']['passed']),
        'parameter_count_matched': (
            parameter_gap <= float(gate_cfg['max_parameter_gap_ratio'])),
        'absolute_reliability_auc_passed': (
            attention['reliability_roc_auc']['mean'] >=
            float(gate_cfg['min_attention_reliability_auc'])),
        'absolute_tree_only_auc_passed': (
            attention['tree_only_roc_auc']['mean'] >=
            float(gate_cfg['min_attention_tree_only_auc'])),
        'reliability_gain_passed': (
            deltas['reliability_roc_auc'] >=
            float(gate_cfg['min_reliability_auc_gain'])),
        'tree_only_gain_passed': (
            deltas['tree_only_roc_auc'] >=
            float(gate_cfg['min_tree_only_auc_gain'])),
        'coverage_not_worse': (
            deltas['coverage_ap'] >=
            -float(gate_cfg['max_coverage_ap_drop'])),
        'critical_recall_not_worse': (
            deltas['critical_recall'] >=
            -float(gate_cfg['max_critical_recall_drop'])),
        'selected_utility_gain_passed': (
            deltas['selected_utility_relative'] >=
            float(gate_cfg['min_selected_utility_relative_gain'])),
        'seed_wins_passed': (
            tree_auc_wins >= int(gate_cfg['min_tree_auc_seed_wins'])),
        'attention_stability_passed': (
            attention['tree_only_roc_auc']['sample_std'] <=
            float(gate_cfg['max_tree_auc_sample_std'])),
        'locked_seed_present': any(
            result['model_type'] == 'relation_attention' and
            result['seed'] == int(settings['locked_seed'])
            for result in results),
    }
    gate['passed'] = bool(all(gate.values()))
    return {
        'num_validation_candidates': int(
            len(results[0]['reliability_scores'])),
        'seeds': [int(seed) for seed in settings['seeds']],
        'locked_seed': int(settings['locked_seed']),
        'neighborhood': settings['neighborhood'],
        'selector': settings['selector'],
        'per_seed': rows,
        'aggregate': by_model,
        'parameter_gap_ratio': float(parameter_gap),
        'tree_auc_seed_wins': int(tree_auc_wins),
        'deltas': deltas,
        'gate': gate,
    }


def format_markdown(summary):
    lines = [
        '# E1c 参数匹配的 Complexity Seed Relation Attention', '',
        f'- Seeds：{summary["seeds"]}',
        f'- 锁定 seed：{summary["locked_seed"]}',
        f'- K / radius：{summary["neighborhood"]["num_neighbors"]} / '
        f'{summary["neighborhood"]["radius"]:.3f} m',
        f'- 参数差比例：{100 * summary["parameter_gap_ratio"]:.3f}%',
        f'- Attention tree-AUC seed wins：'
        f'{summary["tree_auc_seed_wins"]}/{len(summary["seeds"])}', '',
        '| Model | Params | Reliability AUC | Tree-only AUC | Coverage AP | '
        'Critical recall | Utility |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for model_type in ('neighborhood_mlp', 'relation_attention'):
        row = summary['aggregate'][model_type]
        lines.append(
            f'| {model_type} | {row["parameter_count"]["mean"]:.0f} | '
            f'{row["reliability_roc_auc"]["mean"]:.6f} | '
            f'{row["tree_only_roc_auc"]["mean"]:.6f} | '
            f'{row["coverage_ap"]["mean"]:.6f} | '
            f'{row["dual_critical_recall"]["mean"]:.6f} | '
            f'{row["dual_selected_mean_utility"]["mean"]:.6f} |')
    lines.extend(['', '## Attention − Neighborhood-MLP', ''])
    for name, value in summary['deltas'].items():
        lines.append(f'- {name}: {value:+.6f}')
    lines.extend(['', '## Gate', ''])
    for name, passed in summary['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend([
        '',
        ('PASS：可以进入固定 L1W 的 seed-attention pipeline 对照。'
         if summary['gate']['passed'] else
         'STOP：注意力未稳定优于参数匹配的邻域 MLP，不进入 pipeline。'),
    ])
    return '\n'.join(lines) + '\n'


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    import torch
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    with Path(settings['neighbor_summary_path']).open(
            encoding='utf-8') as file:
        neighbour_summary = json.load(file)
    if not neighbour_summary['gate']['passed']:
        raise RuntimeError('E1c-a neighbour data gate did not pass.')
    if (
            int(neighbour_summary['num_neighbors']) !=
            int(settings['neighborhood']['num_neighbors']) or
            not math.isclose(
                float(neighbour_summary['radius']),
                float(settings['neighborhood']['radius']))):
        raise ValueError('Neighbour cache and training configuration differ.')

    counts = model_parameter_counts(settings)
    parameter_gap = abs(
        counts['relation_attention'] - counts['neighborhood_mlp']) / max(
            counts.values())
    if parameter_gap > float(settings['gate']['max_parameter_gap_ratio']):
        raise RuntimeError(
            f'Model parameter gap {parameter_gap:.3%} exceeds the gate.')
    print(f'Parameter counts: {counts}; gap={parameter_gap:.3%}', flush=True)
    dataset = load_relation_dataset(settings)
    if dataset['input_dim'] != int(settings['expected_input_dim']):
        raise ValueError('E1c input dimension differs from E1b.')
    output_dir = Path(settings['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model_type in settings['model_types']:
        for seed in settings['seeds']:
            print(
                f'===== START {model_type} seed {seed} =====', flush=True)
            result = train_one_model_seed(
                dataset, settings, model_type, int(seed))
            checkpoint = {
                'model_type': model_type,
                'state_dict': result['state_dict'],
                'input_mean': result['input_mean'],
                'input_scale': result['input_scale'],
                'feature_names': dataset['feature_names'],
                'model_config': settings['model'],
                'neighborhood_config': settings['neighborhood'],
                'selector_config': settings['selector'],
                'seed': result['seed'],
                'best_epoch': result['best_epoch'],
                'metrics': result['metrics'],
                'parameter_count': result['parameter_count'],
            }
            torch.save(
                checkpoint,
                checkpoint_dir /
                f'{model_type}_seed{int(seed)}.pth')
            if int(seed) == int(settings['locked_seed']):
                np.savez_compressed(
                    output_dir / f'locked_{model_type}_scores.npz',
                    reliability_scores=result['reliability_scores'],
                    coverage_scores=result['coverage_scores'])
            results.append(result)
            print(
                f'DONE {model_type} seed {seed}: '
                f'best epoch={result["best_epoch"]}', flush=True)

    summary = summarize(results, settings, neighbour_summary)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    write_csv(output_dir / 'per_seed_metrics.csv', summary['per_seed'])
    print(markdown, flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError(
            'E1c relation-attention gate failed; do not integrate pipeline.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train E1c parameter-matched seed relation models.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
