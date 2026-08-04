import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


IDENTITY_COLUMNS = {
    'source_plot',
    'split',
    'source_instance_id',
    'target_instance_id',
    'source_best_gt_id',
    'target_best_gt_id',
    'source_max_iou',
    'target_max_iou',
    'source_is_true_tree',
    'target_is_true_tree',
    'pair_valid',
    'pair_positive',
}


def load_yaml(path):
    import yaml

    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {'true', '1', 'yes'}:
        return True
    if normalized in {'false', '0', 'no'}:
        return False
    raise ValueError(f'Invalid boolean value: {value}')


def load_pair_dataset(path, configured_feature_names=None):
    with Path(path).open(newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError('E8b pair CSV is empty.')
    missing_identity = IDENTITY_COLUMNS - set(fieldnames)
    if missing_identity:
        raise ValueError(
            f'E8b pair CSV misses fields: {sorted(missing_identity)}')
    inferred = [
        name for name in fieldnames if name not in IDENTITY_COLUMNS]
    feature_names = (
        list(configured_feature_names)
        if configured_feature_names is not None else inferred)
    if not feature_names:
        raise ValueError('No pair features were selected.')
    missing_features = set(feature_names) - set(fieldnames)
    if missing_features:
        raise ValueError(
            f'Configured pair features are missing: {sorted(missing_features)}')
    forbidden = set(feature_names) & IDENTITY_COLUMNS
    if forbidden:
        raise ValueError(
            f'Labels or identifiers cannot be model features: '
            f'{sorted(forbidden)}')

    dataset = {
        'source_plot': np.asarray(
            [row['source_plot'] for row in rows], dtype=str),
        'split': np.asarray([row['split'] for row in rows], dtype=str),
        'source_instance_id': np.asarray(
            [int(row['source_instance_id']) for row in rows],
            dtype=np.int64),
        'target_instance_id': np.asarray(
            [int(row['target_instance_id']) for row in rows],
            dtype=np.int64),
        'pair_valid': np.asarray(
            [parse_bool(row['pair_valid']) for row in rows], dtype=bool),
        'pair_positive': np.asarray(
            [parse_bool(row['pair_positive']) for row in rows], dtype=bool),
        'features': np.asarray([
            [float(row[name]) for name in feature_names]
            for row in rows
        ], dtype=np.float32),
        'feature_names': feature_names,
    }
    if not np.isfinite(dataset['features']).all():
        raise ValueError('Pair features contain NaN or infinity.')
    observed_splits = set(dataset['split'].tolist())
    if observed_splits != {'train', 'validation'}:
        raise ValueError(
            f'Both train and validation splits are required, got '
            f'{sorted(observed_splits)}.')
    train_plots = set(
        dataset['source_plot'][dataset['split'] == 'train'].tolist())
    validation_plots = set(
        dataset['source_plot'][dataset['split'] == 'validation'].tolist())
    overlap = train_plots & validation_plots
    if overlap:
        raise ValueError(f'Plots leak across splits: {sorted(overlap)}')
    if np.any(dataset['pair_positive'] & ~dataset['pair_valid']):
        raise ValueError('A positive pair cannot be marked invalid.')
    dataset['source_key'] = np.asarray([
        f'{plot}:{instance_id}'
        for plot, instance_id in zip(
            dataset['source_plot'], dataset['source_instance_id'])
    ], dtype=str)
    dataset['train_plots'] = sorted(train_plots)
    dataset['validation_plots'] = sorted(validation_plots)
    return dataset


def fit_standardizer(values):
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values, axis=0)
    scale = np.std(values - median, axis=0)
    scale[scale < 1e-6] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def standardize(values, median, scale):
    return ((values - median) / scale).astype(np.float32)


def select_best_target_indices(scores, source_keys, target_ids):
    scores = np.asarray(scores, dtype=np.float64)
    source_keys = np.asarray(source_keys, dtype=str)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    if (
            scores.ndim != 1 or source_keys.ndim != 1 or
            target_ids.ndim != 1 or len(scores) == 0 or
            len(scores) != len(source_keys) or len(scores) != len(target_ids)):
        raise ValueError('Pair scores, source keys, and target IDs must align.')
    if not np.isfinite(scores).all():
        raise ValueError('Pair scores must be finite.')
    order = np.lexsort((target_ids, -scores, source_keys))
    ordered_sources = source_keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = ordered_sources[1:] != ordered_sources[:-1]
    return order[first]


def source_merge_metrics(
        labels, scores, source_keys, target_ids, threshold):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    source_keys = np.asarray(source_keys, dtype=str)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    best = select_best_target_indices(scores, source_keys, target_ids)
    selected = best[scores[best] >= float(threshold)]
    positive_sources = set(source_keys[labels].tolist())
    evaluable_sources = set(source_keys.tolist())
    correct = int(np.count_nonzero(labels[selected]))
    unsafe = int(len(selected) - correct)
    proposed = int(len(selected))
    precision = correct / proposed if proposed else 1.0
    recall = correct / max(len(positive_sources), 1)
    unsafe_rate = unsafe / max(len(evaluable_sources), 1)
    return {
        'threshold': float(threshold),
        'num_evaluable_sources': int(len(evaluable_sources)),
        'num_positive_sources': int(len(positive_sources)),
        'num_proposed_merges': proposed,
        'num_correct_merges': correct,
        'num_unsafe_merges': unsafe,
        'pair_precision': float(precision),
        'positive_source_recall': float(recall),
        'unsafe_merge_rate': float(unsafe_rate),
    }


def select_merge_threshold(
        labels, scores, source_keys, target_ids, constraints):
    best_indices = select_best_target_indices(
        scores, source_keys, target_ids)
    best_scores = np.asarray(scores, dtype=np.float64)[best_indices]
    thresholds = [float('inf')]
    thresholds.extend(
        float(value) for value in sorted(set(best_scores.tolist()), reverse=True))
    feasible = []
    all_rows = []
    for threshold in thresholds:
        metrics = source_merge_metrics(
            labels, scores, source_keys, target_ids, threshold)
        metrics['precision_passed'] = (
            metrics['pair_precision'] >=
            float(constraints['min_pair_precision']))
        metrics['unsafe_rate_passed'] = (
            metrics['unsafe_merge_rate'] <=
            float(constraints['max_unsafe_merge_rate']))
        metrics['recall_passed'] = (
            metrics['positive_source_recall'] >=
            float(constraints['min_positive_source_recall']))
        metrics['passed'] = bool(
            metrics['precision_passed'] and
            metrics['unsafe_rate_passed'] and
            metrics['recall_passed'])
        all_rows.append(metrics)
        if metrics['precision_passed'] and metrics['unsafe_rate_passed']:
            feasible.append(metrics)
    if not feasible:
        raise RuntimeError(
            'No merge threshold satisfies precision and unsafe-rate '
            'constraints.')
    selected = max(
        feasible,
        key=lambda row: (
            row['positive_source_recall'],
            row['pair_precision'],
            -row['unsafe_merge_rate'],
            row['threshold'],
        ))
    return dict(selected), all_rows


def pair_classification_metrics(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise ValueError('Pair metrics require positive and negative labels.')
    return {
        'pair_roc_auc': float(roc_auc_score(labels, scores)),
        'pair_average_precision': float(
            average_precision_score(labels, scores)),
    }


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_logistic(train_x, train_y, validation_x, seed, config):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        class_weight='balanced',
        C=float(config.get('c', 1.0)),
        max_iter=int(config.get('max_iter', 2000)),
        random_state=int(seed),
    )
    model.fit(train_x, train_y)
    return {
        'scores': model.predict_proba(validation_x)[:, 1],
        'coef': model.coef_.astype(np.float32),
        'intercept': model.intercept_.astype(np.float32),
    }


def build_pair_mlp(input_dim, model_config):
    import torch
    from torch import nn

    layers = []
    current_dim = int(input_dim)
    for hidden_dim in model_config['hidden_dims']:
        layers.extend([
            nn.Linear(current_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(model_config['dropout'])),
        ])
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


def train_pair_mlp(
        train_x, train_y, validation_x, validation_y,
        model_config, training_config, seed):
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(int(seed))
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model = build_pair_mlp(train_x.shape[1], model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
    )
    positive = int(np.count_nonzero(train_y))
    negative = int(len(train_y) - positive)
    if positive == 0 or negative == 0:
        raise ValueError('Pair MLP training requires both classes.')
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negative / positive, dtype=torch.float32, device=device))
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(training_config['batch_size']),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state = None
    best_ap = -np.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(training_config['epochs']) + 1):
        model.train()
        for values, targets in loader:
            values = values.to(device)
            targets = targets.to(device)
            logits = model(values).squeeze(1)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(
                model(validation_tensor).squeeze(1)).cpu().numpy()
        validation_ap = float(
            average_precision_score(validation_y, scores))
        if validation_ap > best_ap + 1e-12:
            best_ap = validation_ap
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= int(training_config['patience']):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            model(validation_tensor).squeeze(1)).cpu().numpy()
    return {
        'scores': scores,
        'state_dict': best_state,
        'best_epoch': int(best_epoch),
        'best_validation_ap': float(best_ap),
        'device': str(device),
        'parameter_count': int(sum(
            parameter.numel() for parameter in model.parameters())),
    }


def evaluate_model(
        name, seed, labels, scores, source_keys, target_ids, constraints):
    metrics = pair_classification_metrics(labels, scores)
    selected, _ = select_merge_threshold(
        labels, scores, source_keys, target_ids, constraints)
    metrics.update(selected)
    metrics.update({
        'model': name,
        'seed': int(seed),
        'gate_passed': bool(selected['passed']),
    })
    return metrics


def aggregate_rows(rows):
    metric_names = [
        'pair_roc_auc',
        'pair_average_precision',
        'pair_precision',
        'positive_source_recall',
        'unsafe_merge_rate',
        'num_proposed_merges',
        'num_correct_merges',
        'num_unsafe_merges',
    ]
    models = {}
    for model_name in sorted({row['model'] for row in rows}):
        group = [row for row in rows if row['model'] == model_name]
        summary = {
            'num_seeds': len(group),
            'num_gate_passed': int(sum(
                bool(row['gate_passed']) for row in group)),
        }
        for metric in metric_names:
            values = np.asarray(
                [float(row[metric]) for row in group], dtype=np.float64)
            summary[metric] = {
                'mean': float(values.mean()),
                'sample_std': float(
                    values.std(ddof=1) if len(values) > 1 else 0.0),
                'min': float(values.min()),
                'max': float(values.max()),
            }
        models[model_name] = summary
    return models


def choose_deployment_model(rows, settings):
    models = aggregate_rows(rows)
    seeds = [int(seed) for seed in settings['seeds']]
    locked_seed = int(settings['selection']['locked_seed'])
    if locked_seed not in seeds:
        raise ValueError('The locked deployment seed is not configured.')
    minimum_passes = int(settings['selection']['min_gate_pass_seeds'])
    eligible = {}
    for name, summary in models.items():
        locked_row = next(
            row for row in rows
            if row['model'] == name and int(row['seed']) == locked_seed)
        eligible[name] = bool(
            summary['num_gate_passed'] >= minimum_passes and
            locked_row['gate_passed'])

    selected = None
    if eligible.get('distance_rule', False):
        selected = 'distance_rule'
    if eligible.get('logistic_regression', False):
        if (
                selected is None or
                models['logistic_regression'][
                    'positive_source_recall']['mean'] >=
                models[selected]['positive_source_recall']['mean'] +
                float(settings['selection'][
                    'min_logistic_recall_gain_over_distance'])):
            selected = 'logistic_regression'
    if eligible.get('pair_mlp', False):
        if (
                selected is None or
                models['pair_mlp']['positive_source_recall']['mean'] >=
                models[selected]['positive_source_recall']['mean'] +
                float(settings['selection'][
                    'min_mlp_recall_gain_over_simpler'])):
            selected = 'pair_mlp'

    selected_row = None
    if selected is not None:
        selected_row = next(
            row for row in rows
            if row['model'] == selected and
            int(row['seed']) == locked_seed)
    mlp_stable = bool(
        models['pair_mlp']['positive_source_recall']['sample_std'] <=
        float(settings['gate']['max_mlp_recall_sample_std']))
    gate = {
        'at_least_one_deployable_model': selected is not None,
        'locked_seed_passed': bool(
            selected_row is not None and selected_row['gate_passed']),
        'mlp_stability_passed_if_selected': bool(
            selected != 'pair_mlp' or mlp_stable),
    }
    gate['passed'] = bool(all(gate.values()))
    return models, eligible, selected, selected_row, gate


def format_markdown(summary):
    lines = [
        '# E8c 质量引导实例合并模型比较',
        '',
        f'- 配对数：{summary["num_pairs"]}',
        f'- 有效配对数：{summary["num_valid_pairs"]}',
        f'- 特征数：{summary["num_features"]}',
        f'- 锁定部署 seed：{summary["locked_seed"]}',
        f'- 推荐模型：**{summary["selected_model"]}**',
        f'- 推荐阈值：{summary["selected_threshold"]}',
        '',
        '| Model | Pair AUC | Pair AP | Precision | Positive-source recall | Unsafe merge rate | Pass seeds |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('distance_rule', 'logistic_regression', 'pair_mlp'):
        model = summary['models'][name]
        lines.append(
            f'| {name} | '
            f'{model["pair_roc_auc"]["mean"]:.6f} | '
            f'{model["pair_average_precision"]["mean"]:.6f} | '
            f'{model["pair_precision"]["mean"]:.6f} | '
            f'{model["positive_source_recall"]["mean"]:.6f} | '
            f'{model["unsafe_merge_rate"]["mean"]:.6f} | '
            f'{model["num_gate_passed"]}/{model["num_seeds"]} |')
    lines.extend([
        '',
        '## 模型资格',
        '',
    ])
    lines.extend(
        f'- {name}: **{value}**'
        for name, value in summary['eligible_models'].items())
    lines.extend(['', '## Gate', ''])
    lines.extend(
        f'- {name}: **{value}**'
        for name, value in summary['gate'].items())
    lines.extend([
        '',
        ('PASS：进入 E8d L1W 安全合并集成。'
         if summary['gate']['passed']
         else 'STOP：不要进入 pipeline 合并。'),
        '',
    ])
    return '\n'.join(lines)


def write_rows(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    import torch

    settings = load_yaml(config_path)
    dataset = load_pair_dataset(
        settings['pairs_path'], settings.get('feature_names'))
    valid = dataset['pair_valid']
    train_mask = valid & (dataset['split'] == 'train')
    validation_mask = valid & (dataset['split'] == 'validation')
    train_y = dataset['pair_positive'][train_mask]
    validation_y = dataset['pair_positive'][validation_mask]
    if len(np.unique(train_y)) != 2 or len(np.unique(validation_y)) != 2:
        raise ValueError(
            'Both train and validation pairs require two classes.')

    median, scale = fit_standardizer(dataset['features'][train_mask])
    standardized = standardize(dataset['features'], median, scale)
    train_x = standardized[train_mask]
    validation_x = standardized[validation_mask]
    validation_sources = dataset['source_key'][validation_mask]
    validation_targets = dataset['target_instance_id'][validation_mask]
    distance_index = dataset['feature_names'].index('base_vote_distance')
    distance_scores = -dataset['features'][
        validation_mask, distance_index].astype(np.float64)

    output_dir = Path(settings['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in settings['seeds']:
        seed = int(seed)
        distance_metrics = evaluate_model(
            'distance_rule', seed, validation_y, distance_scores,
            validation_sources, validation_targets,
            settings['constraints'])
        rows.append(distance_metrics)

        logistic = train_logistic(
            train_x, train_y, validation_x, seed,
            settings['logistic_regression'])
        logistic_metrics = evaluate_model(
            'logistic_regression', seed, validation_y,
            logistic['scores'], validation_sources, validation_targets,
            settings['constraints'])
        rows.append(logistic_metrics)
        np.savez_compressed(
            checkpoint_dir / f'logistic_regression_seed{seed}.npz',
            coef=logistic['coef'],
            intercept=logistic['intercept'],
            feature_names=np.asarray(dataset['feature_names']),
            median=median,
            scale=scale,
            seed=np.asarray(seed),
        )

        mlp = train_pair_mlp(
            train_x, train_y, validation_x, validation_y,
            settings['pair_mlp']['model'],
            settings['pair_mlp']['training'],
            seed,
        )
        mlp_metrics = evaluate_model(
            'pair_mlp', seed, validation_y, mlp['scores'],
            validation_sources, validation_targets,
            settings['constraints'])
        mlp_metrics.update({
            'best_epoch': mlp['best_epoch'],
            'best_validation_ap': mlp['best_validation_ap'],
            'device': mlp['device'],
            'parameter_count': mlp['parameter_count'],
        })
        rows.append(mlp_metrics)
        torch.save({
            'state_dict': mlp['state_dict'],
            'feature_names': dataset['feature_names'],
            'median': median,
            'scale': scale,
            'seed': seed,
            'best_epoch': mlp['best_epoch'],
            'model_config': settings['pair_mlp']['model'],
            'training_config': settings['pair_mlp']['training'],
        }, checkpoint_dir / f'pair_mlp_seed{seed}.pth')

    models, eligible, selected, selected_row, gate = (
        choose_deployment_model(rows, settings))
    selected_threshold = (
        None if selected_row is None else float(selected_row['threshold']))
    summary = {
        'config': str(config_path),
        'pairs_path': str(settings['pairs_path']),
        'num_pairs': int(len(dataset['pair_valid'])),
        'num_valid_pairs': int(np.count_nonzero(valid)),
        'num_features': int(len(dataset['feature_names'])),
        'feature_names': dataset['feature_names'],
        'train_plots': dataset['train_plots'],
        'validation_plots': dataset['validation_plots'],
        'seeds': [int(seed) for seed in settings['seeds']],
        'locked_seed': int(settings['selection']['locked_seed']),
        'models': models,
        'eligible_models': eligible,
        'selected_model': selected,
        'selected_threshold': selected_threshold,
        'selected_seed_metrics': selected_row,
        'constraints': settings['constraints'],
        'gate': gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / 'per_seed_metrics.csv', rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8')
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'E8c gate failed; do not integrate pair merging into pipeline.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Train E8c quality-guided instance merge baselines.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()