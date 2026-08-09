"""Train the E1b frozen-backbone reliability/coverage Seed-MLP control."""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEED_QUALITY_PATH = ROOT / 'tree_learn' / 'util' / 'seed_quality.py'
SEED_QUALITY_SPEC = importlib.util.spec_from_file_location(
    'seed_quality_for_e1b', SEED_QUALITY_PATH)
SEED_QUALITY = importlib.util.module_from_spec(SEED_QUALITY_SPEC)
SEED_QUALITY_SPEC.loader.exec_module(SEED_QUALITY)


def load_seed_quality_dataset(
        data_root, manifest_path, expected_backbone_dim=32,
        expected_scalar_dim=29):
    root = Path(data_root).resolve()
    with Path(manifest_path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('The E1a seed manifest is empty.')
    if {row['split'] for row in rows} != {'train', 'validation'}:
        raise ValueError('Both train and validation seed splits are required.')
    if len({row['source_plot'] for row in rows}) != len(rows):
        raise ValueError('Every manifest row must describe a unique forest.')

    counts = [int(row['num_candidates']) for row in rows]
    total = sum(counts)
    backbone_dim = int(expected_backbone_dim)
    scalar_dim = int(expected_scalar_dim)
    input_dim = backbone_dim + scalar_dim
    output = {
        'features': np.empty((total, input_dim), dtype=np.float32),
        'target_valid': np.empty(total, dtype=bool),
        'target_is_tree': np.empty(total, dtype=bool),
        'target_reliable': np.empty(total, dtype=bool),
        'target_coverage_critical': np.empty(total, dtype=bool),
        'target_utility': np.empty(total, dtype=np.float32),
        'target_tree_id': np.empty(total, dtype=np.int64),
        'candidate_index': np.empty(total, dtype=np.int64),
        'plot_id': np.empty(total, dtype=np.int16),
        'split_id': np.empty(total, dtype=np.int8),
    }
    scalar_feature_names = None
    plot_names = []
    plot_splits = []
    offset = 0
    for plot_id, (row, count) in enumerate(zip(rows, counts)):
        path = Path(row['artifact_path']).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid E1a artifact path: {path}')
        metadata_path = path.with_name(f'{path.stem}_metadata.json')
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f'Missing E1a metadata beside artifact: {metadata_path}')
        with metadata_path.open(encoding='utf-8') as file:
            metadata = json.load(file)
        names = [str(name) for name in metadata['scalar_feature_names']]
        if scalar_feature_names is None:
            scalar_feature_names = names
        elif names != scalar_feature_names:
            raise ValueError(f'Scalar feature schema differs in {path}.')
        if metadata['source_plot'] != row['source_plot']:
            raise ValueError(f'Manifest plot differs from {metadata_path}.')
        if metadata['split'] != row['split']:
            raise ValueError(f'Manifest split differs from {metadata_path}.')

        end = offset + count
        with np.load(path, allow_pickle=False) as data:
            if len(data['candidate_indices']) != count:
                raise ValueError(
                    f'Manifest count differs from artifact {path}.')
            backbone = data['backbone_features']
            scalar = data['scalar_features']
            if backbone.shape != (count, backbone_dim):
                raise ValueError(
                    f'Unexpected backbone shape in {path}: {backbone.shape}.')
            if scalar.shape != (count, scalar_dim):
                raise ValueError(
                    f'Unexpected scalar shape in {path}: {scalar.shape}.')
            output['features'][offset:end, :backbone_dim] = backbone
            output['features'][offset:end, backbone_dim:] = scalar
            for source_name, output_name in (
                    ('target_valid', 'target_valid'),
                    ('target_is_tree', 'target_is_tree'),
                    ('target_reliable', 'target_reliable'),
                    ('target_coverage_critical',
                     'target_coverage_critical'),
                    ('target_utility', 'target_utility'),
                    ('target_tree_id', 'target_tree_id'),
                    ('candidate_indices', 'candidate_index')):
                output[output_name][offset:end] = data[source_name]
        output['plot_id'][offset:end] = plot_id
        output['split_id'][offset:end] = (
            0 if row['split'] == 'train' else 1)
        plot_names.append(row['source_plot'])
        plot_splits.append(row['split'])
        offset = end

    if offset != total or not np.isfinite(output['features']).all():
        raise ValueError('Loaded E1a features are incomplete or non-finite.')
    if not np.isfinite(output['target_utility']).all():
        raise ValueError('Loaded E1a utility targets are non-finite.')
    if np.any(
            output['target_coverage_critical'] &
            ~output['target_is_tree']):
        raise ValueError('Coverage-critical targets must be tree candidates.')
    if not np.all(output['target_valid']):
        raise ValueError(
            'E1b requires the fully labeled forests that passed E1a.')
    output.update({
        'plot_names': plot_names,
        'plot_splits': plot_splits,
        'feature_names': [
            f'backbone_{index}' for index in range(backbone_dim)
        ] + scalar_feature_names,
        'input_dim': input_dim,
    })
    return output


def balanced_plot_sample_indices(
        plot_ids, eligible_mask, examples_per_plot, seed):
    plot_ids = np.asarray(plot_ids, dtype=np.int64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if len(plot_ids) != len(eligible):
        raise ValueError('plot_ids and eligible_mask are not aligned.')
    count = int(examples_per_plot)
    if count < 1:
        raise ValueError('examples_per_plot must be positive.')
    generator = np.random.default_rng(int(seed))
    samples = []
    for plot_id in sorted(np.unique(plot_ids[eligible]).tolist()):
        indices = np.flatnonzero(eligible & (plot_ids == plot_id))
        if len(indices) == 0:
            continue
        chosen = generator.choice(
            indices, size=count, replace=len(indices) < count)
        samples.append(chosen)
    if not samples:
        raise ValueError('No eligible plot examples are available.')
    result = np.concatenate(samples).astype(np.int64)
    generator.shuffle(result)
    return result


def fit_sampled_standardizer(
        features, plot_ids, train_mask, max_points, seed):
    plot_ids = np.asarray(plot_ids)
    train_mask = np.asarray(train_mask, dtype=bool)
    train_plots = np.unique(plot_ids[train_mask])
    per_plot = max(1, int(max_points) // max(len(train_plots), 1))
    indices = balanced_plot_sample_indices(
        plot_ids, train_mask, per_plot, seed)
    if len(indices) > int(max_points):
        indices = indices[:int(max_points)]
    values = np.asarray(features[indices], dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (
        mean.astype(np.float32),
        scale.astype(np.float32),
        indices.astype(np.int64),
    )


def standardize_features(values, mean, scale):
    output = (
        (np.asarray(values, dtype=np.float32) - mean) / scale
    ).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError('Standardized seed features are non-finite.')
    return output


def binary_metrics(targets, scores):
    """Compute tie-aware ROC-AUC and average precision in NumPy."""
    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if len(targets) != len(scores) or len(np.unique(targets)) != 2:
        raise ValueError('Binary metrics require aligned examples of both classes.')
    positives = int(targets.sum())
    negatives = len(targets) - positives

    ascending = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[ascending]
    _, group_starts, group_counts = np.unique(
        sorted_scores, return_index=True, return_counts=True)
    sorted_ranks = np.repeat(
        group_starts + 0.5 * (group_counts + 1), group_counts)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[ascending] = sorted_ranks
    roc_auc = (
        (ranks[targets].sum() - positives * (positives + 1) / 2.0) /
        (positives * negatives))

    descending = np.argsort(-scores, kind='mergesort')
    sorted_targets = targets[descending].astype(np.int64)
    sorted_scores = scores[descending]
    true_positives = np.cumsum(sorted_targets)
    false_positives = np.cumsum(1 - sorted_targets)
    group_ends = np.flatnonzero(np.r_[
        sorted_scores[1:] != sorted_scores[:-1], True])
    true_positives = true_positives[group_ends]
    false_positives = false_positives[group_ends]
    recall = true_positives / positives
    precision = true_positives / (true_positives + false_positives)
    average_precision = float(np.sum(
        np.diff(np.r_[0.0, recall]) * precision))
    return {
        'roc_auc': float(roc_auc),
        'average_precision': average_precision,
        'prevalence': float(targets.mean()),
    }

def selector_metrics(
        selected, target_reliable, target_critical, target_utility,
        target_tree_id):
    selected = np.asarray(selected, dtype=bool)
    reliable = np.asarray(target_reliable, dtype=bool)
    critical = np.asarray(target_critical, dtype=bool)
    utility = np.asarray(target_utility, dtype=np.float64)
    tree_ids = np.asarray(target_tree_id, dtype=np.int64)
    lengths = {
        len(selected), len(reliable), len(critical), len(utility),
        len(tree_ids)}
    if len(lengths) != 1 or not np.any(selected):
        raise ValueError('Selector arrays are invalid or selection is empty.')
    tree = tree_ids > 0
    all_tree_ids = np.unique(tree_ids[tree])
    selected_tree_ids = np.unique(tree_ids[selected & tree])
    critical_tree_ids = np.unique(tree_ids[critical])
    selected_critical_tree_ids = np.unique(
        tree_ids[selected & critical])
    return {
        'keep_rate': float(selected.mean()),
        'reliable_precision': float(
            np.count_nonzero(selected & reliable) /
            max(np.count_nonzero(selected), 1)),
        'reliable_recall': float(
            np.count_nonzero(selected & reliable) /
            max(np.count_nonzero(reliable), 1)),
        'critical_recall': float(
            np.count_nonzero(selected & critical) /
            max(np.count_nonzero(critical), 1)),
        'tree_coverage': float(
            len(selected_tree_ids) / max(len(all_tree_ids), 1)),
        'critical_tree_coverage': float(
            len(selected_critical_tree_ids) /
            max(len(critical_tree_ids), 1)),
        'selected_mean_utility': float(utility[selected].mean()),
        'selected_non_tree_rate': float(
            np.count_nonzero(selected & ~tree) /
            max(np.count_nonzero(selected), 1)),
    }


def exact_random_mask(num_candidates, keep_ratio, seed):
    keep_count = min(
        int(num_candidates),
        int(math.ceil(int(num_candidates) * float(keep_ratio))))
    generator = np.random.default_rng(int(seed))
    selected = np.zeros(int(num_candidates), dtype=bool)
    selected[generator.permutation(int(num_candidates))[:keep_count]] = True
    return selected


def evaluate_seed_scores(
        reliability_scores, coverage_scores, reliable, critical, utility,
        tree_ids, plot_ids, selector_config):
    reliability_scores = np.asarray(reliability_scores, dtype=np.float32)
    coverage_scores = np.asarray(coverage_scores, dtype=np.float32)
    reliable = np.asarray(reliable, dtype=bool)
    critical = np.asarray(critical, dtype=bool)
    utility = np.asarray(utility, dtype=np.float32)
    tree_ids = np.asarray(tree_ids, dtype=np.int64)
    plot_ids = np.asarray(plot_ids, dtype=np.int64)
    reliability = binary_metrics(reliable, reliability_scores)
    coverage = binary_metrics(critical, coverage_scores)
    keep_ratio = float(selector_config['keep_ratio'])
    protect_ratio = float(selector_config['coverage_protect_ratio'])
    dual_mask = np.zeros(len(reliable), dtype=bool)
    reliability_mask = np.zeros(len(reliable), dtype=bool)
    oracle_mask = np.zeros(len(reliable), dtype=bool)
    random_masks = {
        int(seed): np.zeros(len(reliable), dtype=bool)
        for seed in selector_config['random_control_seeds']}
    selection_info = {
        'num_candidates': int(len(reliable)),
        'num_kept': 0,
        'num_coverage_protected': 0,
        'num_forests': int(len(np.unique(plot_ids))),
        'keep_ratio': keep_ratio,
        'coverage_protect_ratio': protect_ratio,
    }
    per_plot = {}
    for plot_id in sorted(np.unique(plot_ids).tolist()):
        plot_mask = plot_ids == plot_id
        local_dual, local_info = SEED_QUALITY.select_seed_candidates(
            reliability_scores[plot_mask], coverage_scores[plot_mask],
            keep_ratio, protect_ratio)
        local_reliability = SEED_QUALITY.exact_top_k_mask(
            reliability_scores[plot_mask],
            int(math.ceil(np.count_nonzero(plot_mask) * keep_ratio)))
        local_oracle, _ = SEED_QUALITY.select_seed_candidates(
            utility[plot_mask], critical[plot_mask].astype(np.float32),
            keep_ratio, protect_ratio)
        dual_mask[plot_mask] = local_dual
        reliability_mask[plot_mask] = local_reliability
        oracle_mask[plot_mask] = local_oracle
        selection_info['num_kept'] += local_info['num_kept']
        selection_info['num_coverage_protected'] += local_info[
            'num_coverage_protected']
        for seed, random_mask in random_masks.items():
            random_mask[plot_mask] = exact_random_mask(
                np.count_nonzero(plot_mask), keep_ratio,
                seed + 1009 * int(plot_id))
        per_plot[str(int(plot_id))] = selector_metrics(
            local_dual, reliable[plot_mask], critical[plot_mask],
            utility[plot_mask], tree_ids[plot_mask])

    tree_stride = max(int(tree_ids.max(initial=0)) + 1, 1)
    global_tree_ids = tree_ids.copy()
    tree_mask = tree_ids > 0
    global_tree_ids[tree_mask] = (
        (plot_ids[tree_mask] + 1) * tree_stride + tree_ids[tree_mask])
    random_metrics = [
        selector_metrics(
            mask, reliable, critical, utility, global_tree_ids)
        for mask in random_masks.values()]
    random_mean = {
        name: float(np.mean([row[name] for row in random_metrics]))
        for name in random_metrics[0]
    }
    dual = selector_metrics(
        dual_mask, reliable, critical, utility, global_tree_ids)
    reliability_only = selector_metrics(
        reliability_mask, reliable, critical, utility, global_tree_ids)
    oracle = selector_metrics(
        oracle_mask, reliable, critical, utility, global_tree_ids)
    return {
        'reliability': reliability,
        'coverage': coverage,
        'selection_info': selection_info,
        'dual': dual,
        'reliability_only': reliability_only,
        'random_mean': random_mean,
        'oracle': oracle,
        'per_plot': per_plot,
        'min_plot_critical_tree_coverage': float(min(
            row['critical_tree_coverage'] for row in per_plot.values())),
    }

def set_deterministic_seed(seed):
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)

def predict_in_chunks(model, features, mean, scale, batch_size, device, amp):
    import torch

    reliability = np.empty(len(features), dtype=np.float32)
    coverage = np.empty(len(features), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            end = min(start + int(batch_size), len(features))
            values = standardize_features(features[start:end], mean, scale)
            tensor = torch.from_numpy(values).to(device)
            with torch.cuda.amp.autocast(
                    enabled=bool(amp and device.type == 'cuda')):
                reliability_logits, coverage_logits = model(tensor)
            reliability[start:end] = torch.sigmoid(
                reliability_logits).float().cpu().numpy()
            coverage[start:end] = torch.sigmoid(
                coverage_logits).float().cpu().numpy()
    return reliability, coverage


def train_one_seed(dataset, settings, seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch import nn

    set_deterministic_seed(seed)
    train_mask = dataset['split_id'] == 0
    validation_mask = dataset['split_id'] == 1
    training = settings['training']
    mean, scale, standardizer_indices = fit_sampled_standardizer(
        dataset['features'], dataset['plot_id'], train_mask,
        int(training['max_standardizer_points']), int(seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SEED_QUALITY.build_seed_quality_mlp(
        dataset['input_dim'], settings['model']).to(device)
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
        max(np.count_nonzero(train_reliable), 1),
        max_pos_weight)
    coverage_pos_weight = min(
        np.count_nonzero(~train_critical) /
        max(np.count_nonzero(train_critical), 1),
        max_pos_weight)
    reliability_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(reliability_pos_weight, device=device))
    coverage_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(coverage_pos_weight, device=device))
    utility_loss = nn.SmoothL1Loss()

    validation_features = dataset['features'][validation_mask]
    validation_reliable = dataset['target_reliable'][validation_mask]
    validation_critical = dataset[
        'target_coverage_critical'][validation_mask]
    validation_utility = dataset['target_utility'][validation_mask]
    validation_tree_ids = dataset['target_tree_id'][validation_mask]
    validation_plot_ids = dataset['plot_id'][validation_mask]
    best_state = None
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    total_epochs = int(training['epochs'])
    warmup_epochs = int(training['warmup_epochs'])
    base_learning_rate = float(training['learning_rate'])
    min_learning_rate = float(training['min_learning_rate'])
    for epoch in range(1, total_epochs + 1):
        if epoch <= warmup_epochs:
            progress = epoch / max(warmup_epochs, 1)
            learning_rate = (
                min_learning_rate +
                progress * (base_learning_rate - min_learning_rate))
        else:
            progress = (
                (epoch - warmup_epochs) /
                max(total_epochs - warmup_epochs, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            learning_rate = (
                min_learning_rate +
                cosine * (base_learning_rate - min_learning_rate))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        epoch_indices = balanced_plot_sample_indices(
            dataset['plot_id'], train_mask,
            int(training['examples_per_plot']),
            int(seed) * 100000 + epoch)
        model.train()
        losses = []
        for start in range(0, len(epoch_indices), int(training['batch_size'])):
            indices = epoch_indices[
                start:start + int(training['batch_size'])]
            values = standardize_features(
                dataset['features'][indices], mean, scale)
            tensor = torch.from_numpy(values).to(device)
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
                reliability_logits, coverage_logits = model(tensor)
                loss_reliability = reliability_bce(
                    reliability_logits, reliable)
                loss_coverage = coverage_bce(coverage_logits, critical)
                loss_utility = utility_loss(
                    torch.sigmoid(reliability_logits), utility)
                loss = (
                    loss_reliability +
                    float(training['coverage_loss_weight']) * loss_coverage +
                    float(training['utility_loss_weight']) * loss_utility)
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
            model, validation_features, mean, scale,
            int(training['validation_batch_size']), device, amp_enabled)
        metrics = evaluate_seed_scores(
            reliability_scores, coverage_scores,
            validation_reliable, validation_critical,
            validation_utility, validation_tree_ids,
            validation_plot_ids, settings['selector'])
        metrics['epoch'] = epoch
        metrics['training_loss'] = float(np.mean(losses))
        metrics['learning_rate'] = learning_rate
        history.append(metrics)
        dual = metrics['dual']
        key = (
            metrics['min_plot_critical_tree_coverage'],
            dual['critical_recall'],
            dual['selected_mean_utility'],
            metrics['reliability']['average_precision'],
            metrics['coverage']['average_precision'],
        )
        print(
            f'[seed {seed}] epoch {epoch:03d}: '
            f'loss={metrics["training_loss"]:.5f}, '
            f'rel_AP={metrics["reliability"]["average_precision"]:.5f}, '
            f'cov_AP={metrics["coverage"]["average_precision"]:.5f}, '
            f'critical_recall={dual["critical_recall"]:.5f}, '
            f'critical_tree_coverage='
            f'{dual["critical_tree_coverage"]:.5f}',
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
                f'[seed {seed}] early stop at epoch {epoch}; '
                f'best epoch={best_epoch}.', flush=True)
            break

    if best_state is None:
        raise RuntimeError('Seed-MLP training produced no checkpoint.')
    model.load_state_dict(best_state)
    reliability_scores, coverage_scores = predict_in_chunks(
        model, validation_features, mean, scale,
        int(training['validation_batch_size']), device, amp_enabled)
    final_metrics = evaluate_seed_scores(
        reliability_scores, coverage_scores,
        validation_reliable, validation_critical,
        validation_utility, validation_tree_ids,
        validation_plot_ids, settings['selector'])
    return {
        'seed': int(seed),
        'best_epoch': int(best_epoch),
        'state_dict': best_state,
        'input_mean': mean,
        'input_scale': scale,
        'standardizer_sample_size': int(len(standardizer_indices)),
        'metrics': final_metrics,
        'history': history,
        'validation_scores': {
            'reliability': reliability_scores,
            'coverage': coverage_scores,
        },
        'device': str(device),
        'parameter_count': int(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad)),
    }

def flatten_seed_metrics(result):
    metrics = result['metrics']
    row = {
        'seed': result['seed'],
        'best_epoch': result['best_epoch'],
        'device': result['device'],
        'parameter_count': result['parameter_count'],
        'standardizer_sample_size': result['standardizer_sample_size'],
        'reliability_roc_auc': metrics['reliability']['roc_auc'],
        'reliability_ap': metrics['reliability']['average_precision'],
        'coverage_roc_auc': metrics['coverage']['roc_auc'],
        'coverage_ap': metrics['coverage']['average_precision'],
        'coverage_prevalence': metrics['coverage']['prevalence'],
        'min_plot_critical_tree_coverage': (
            metrics['min_plot_critical_tree_coverage']),
    }
    for prefix in ('dual', 'reliability_only', 'random_mean', 'oracle'):
        for name, value in metrics[prefix].items():
            row[f'{prefix}_{name}'] = value
    return row


def aggregate_results(results, settings, dataset):
    rows = [flatten_seed_metrics(result) for result in results]
    excluded = {
        'seed', 'best_epoch', 'device', 'parameter_count',
        'standardizer_sample_size'}
    metric_names = [name for name in rows[0] if name not in excluded]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        aggregate[name] = {
            'mean': float(values.mean()),
            'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max()),
        }
    critical_gain_random = (
        aggregate['dual_critical_recall']['mean'] -
        aggregate['random_mean_critical_recall']['mean'])
    critical_gain_reliability = (
        aggregate['dual_critical_recall']['mean'] -
        aggregate['reliability_only_critical_recall']['mean'])
    random_utility = aggregate['random_mean_selected_mean_utility']['mean']
    utility_relative_gain = (
        (aggregate['dual_selected_mean_utility']['mean'] - random_utility) /
        max(abs(random_utility), 1e-12))
    coverage_ap_lift = (
        aggregate['coverage_ap']['mean'] /
        max(aggregate['coverage_prevalence']['mean'], 1e-12))
    gate_config = settings['gate']
    gate = {
        'reliability_auc_passed': (
            aggregate['reliability_roc_auc']['mean'] >=
            float(gate_config['min_reliability_roc_auc'])),
        'coverage_ap_lift_passed': (
            coverage_ap_lift >= float(
                gate_config['min_coverage_ap_lift'])),
        'critical_gain_over_random_passed': (
            critical_gain_random >= float(
                gate_config['min_critical_recall_gain_over_random'])),
        'coverage_head_gain_passed': (
            critical_gain_reliability >= float(
                gate_config['min_critical_recall_gain_over_reliability'])),
        'utility_gain_passed': (
            utility_relative_gain >= float(
                gate_config['min_selected_utility_relative_gain'])),
        'critical_tree_coverage_passed': (
            aggregate['dual_critical_tree_coverage']['min'] >=
            float(gate_config['min_critical_tree_coverage'])),
        'per_plot_coverage_passed': (
            aggregate['min_plot_critical_tree_coverage']['min'] >=
            float(gate_config['min_plot_critical_tree_coverage'])),
        'seed_stability_passed': (
            aggregate['dual_critical_recall']['sample_std'] <=
            float(gate_config['max_critical_recall_sample_std'])),
        'locked_seed_present': any(
            result['seed'] == int(settings['locked_seed'])
            for result in results),
    }
    gate['passed'] = bool(all(gate.values()))
    train_mask = dataset['split_id'] == 0
    validation_mask = dataset['split_id'] == 1
    return {
        'num_candidates': int(len(dataset['features'])),
        'num_train_candidates': int(train_mask.sum()),
        'num_validation_candidates': int(validation_mask.sum()),
        'input_dim': int(dataset['input_dim']),
        'feature_names': dataset['feature_names'],
        'train_plots': sorted(
            dataset['plot_names'][index]
            for index, split in enumerate(dataset['plot_splits'])
            if split == 'train'),
        'validation_plots': sorted(
            dataset['plot_names'][index]
            for index, split in enumerate(dataset['plot_splits'])
            if split == 'validation'),
        'seeds': [int(value) for value in settings['seeds']],
        'locked_seed': int(settings['locked_seed']),
        'selector': settings['selector'],
        'aggregate': aggregate,
        'effects': {
            'critical_recall_gain_over_random': critical_gain_random,
            'critical_recall_gain_over_reliability': (
                critical_gain_reliability),
            'selected_utility_relative_gain': utility_relative_gain,
            'coverage_ap_lift': coverage_ap_lift,
        },
        'per_seed': rows,
        'gate': gate,
    }


def format_markdown(summary):
    aggregate = summary['aggregate']
    effects = summary['effects']
    lines = [
        '# E1b 冻结主干双头 Seed-MLP 控制实验', '',
        f'- 候选总数：{summary["num_candidates"]:,}',
        f'- 训练候选：{summary["num_train_candidates"]:,}',
        f'- 验证候选：{summary["num_validation_candidates"]:,}',
        f'- 输入维度：{summary["input_dim"]}',
        f'- 随机种子：{summary["seeds"]}',
        f'- 锁定部署 seed：{summary["locked_seed"]}',
        f'- 固定 keep ratio：{summary["selector"]["keep_ratio"]:.3f}',
        f'- Coverage 保护比例：'
        f'{summary["selector"]["coverage_protect_ratio"]:.3f}', '',
        '| Metric | Mean | Std | Min | Max |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in (
            'reliability_roc_auc', 'reliability_ap',
            'coverage_roc_auc', 'coverage_ap',
            'dual_critical_recall', 'dual_critical_tree_coverage',
            'dual_selected_mean_utility', 'dual_selected_non_tree_rate'):
        metric = aggregate[name]
        lines.append(
            f'| {name} | {metric["mean"]:.6f} | '
            f'{metric["sample_std"]:.6f} | {metric["min"]:.6f} | '
            f'{metric["max"]:.6f} |')
    lines.extend([
        '', '## 相对控制组', '',
        f'- Critical recall 相对随机：'
        f'{effects["critical_recall_gain_over_random"]:+.6f}',
        f'- Critical recall 相对 Reliability-only：'
        f'{effects["critical_recall_gain_over_reliability"]:+.6f}',
        f'- Selected utility 相对随机提升：'
        f'{100 * effects["selected_utility_relative_gain"]:+.3f}%',
        f'- Coverage AP / prevalence：'
        f'{effects["coverage_ap_lift"]:.3f}x', '',
        '## Gate', '',
    ])
    lines.extend(
        f'- {name}: **{value}**'
        for name, value in summary['gate'].items())
    lines.extend([
        '',
        ('PASS：实现同输入的 Complexity Seed Attention。'
         if summary['gate']['passed'] else
         'STOP：先重新设计种子特征或 Coverage target，不进入注意力。'),
    ])
    return '\n'.join(lines) + '\n'


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty metrics table.')
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    import torch
    import yaml

    with open(config_path, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    dataset = load_seed_quality_dataset(
        settings['data_root'], settings['manifest_path'],
        int(settings['expected_backbone_dim']),
        int(settings['expected_scalar_dim']))
    if dataset['input_dim'] != int(settings['expected_input_dim']):
        raise ValueError(
            f'Expected input dim {settings["expected_input_dim"]}, '
            f'got {dataset["input_dim"]}.')
    output_dir = Path(settings['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in settings['seeds']:
        print(f'===== START Seed-MLP seed {seed} =====', flush=True)
        result = train_one_seed(dataset, settings, int(seed))
        checkpoint = {
            'state_dict': result['state_dict'],
            'input_mean': result['input_mean'],
            'input_scale': result['input_scale'],
            'feature_names': dataset['feature_names'],
            'model_config': settings['model'],
            'selector_config': settings['selector'],
            'seed': result['seed'],
            'best_epoch': result['best_epoch'],
            'metrics': result['metrics'],
            'parameter_count': result['parameter_count'],
        }
        checkpoint_path = (
            checkpoint_dir / f'seed_mlp_seed{int(seed)}.pth')
        torch.save(checkpoint, checkpoint_path)
        if int(seed) == int(settings['locked_seed']):
            validation_mask = dataset['split_id'] == 1
            np.savez_compressed(
                output_dir / 'locked_validation_scores.npz',
                reliability_scores=result['validation_scores']['reliability'],
                coverage_scores=result['validation_scores']['coverage'],
                target_reliable=dataset['target_reliable'][validation_mask],
                target_coverage_critical=dataset[
                    'target_coverage_critical'][validation_mask],
                target_utility=dataset['target_utility'][validation_mask],
                target_tree_id=dataset['target_tree_id'][validation_mask],
                candidate_index=dataset['candidate_index'][validation_mask],
                plot_id=dataset['plot_id'][validation_mask],
            )
        results.append(result)
        print(
            f'DONE seed {seed}: best epoch={result["best_epoch"]}, '
            f'critical recall='
            f'{result["metrics"]["dual"]["critical_recall"]:.6f}',
            flush=True)

    summary = aggregate_results(results, settings, dataset)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    write_csv(output_dir / 'per_seed_metrics.csv', summary['per_seed'])
    print(markdown, flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError(
            'E1b Seed-MLP gate failed; do not implement attention yet.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the E1b dual-head Seed-MLP control.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
