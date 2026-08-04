import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np


BASELINE_PATH = (
    Path(__file__).resolve().parent / 'train_instance_merge_pairs.py')
BASELINE_SPEC = importlib.util.spec_from_file_location(
    'instance_merge_pointwise', BASELINE_PATH)
POINTWISE = importlib.util.module_from_spec(BASELINE_SPEC)
BASELINE_SPEC.loader.exec_module(POINTWISE)


def build_source_groups(
        features, labels, source_keys, target_ids):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    source_keys = np.asarray(source_keys, dtype=str)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    if (
            features.ndim != 2 or labels.ndim != 1 or
            source_keys.ndim != 1 or target_ids.ndim != 1 or
            len(features) == 0 or len(features) != len(labels) or
            len(features) != len(source_keys) or
            len(features) != len(target_ids)):
        raise ValueError('Groupwise pair arrays must be nonempty and align.')
    order = np.lexsort((target_ids, source_keys))
    ordered_sources = source_keys[order]
    boundaries = np.flatnonzero(
        np.r_[True, ordered_sources[1:] != ordered_sources[:-1], True])
    groups = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:end]
        groups.append({
            'source_key': str(source_keys[indices[0]]),
            'features': features[indices],
            'positive_mask': labels[indices],
            'target_ids': target_ids[indices],
        })
    return groups


def collate_groups(groups):
    import torch

    if not groups:
        raise ValueError('Cannot collate an empty group batch.')
    batch_size = len(groups)
    max_candidates = max(len(group['target_ids']) for group in groups)
    feature_dim = groups[0]['features'].shape[1]
    features = np.zeros(
        (batch_size, max_candidates, feature_dim), dtype=np.float32)
    candidate_mask = np.zeros(
        (batch_size, max_candidates), dtype=bool)
    positive_mask = np.zeros(
        (batch_size, max_candidates), dtype=bool)
    target_ids = np.full(
        (batch_size, max_candidates), -1, dtype=np.int64)
    for row, group in enumerate(groups):
        count = len(group['target_ids'])
        if group['features'].shape != (count, feature_dim):
            raise ValueError('Group feature dimensions differ.')
        features[row, :count] = group['features']
        candidate_mask[row, :count] = True
        positive_mask[row, :count] = group['positive_mask']
        target_ids[row, :count] = group['target_ids']
    return {
        'features': torch.from_numpy(features),
        'candidate_mask': torch.from_numpy(candidate_mask),
        'positive_mask': torch.from_numpy(positive_mask),
        'target_ids': torch.from_numpy(target_ids),
        'source_keys': [group['source_key'] for group in groups],
    }


def build_groupwise_merge_model(input_dim, model_config):
    import torch
    from torch import nn

    hidden_dims = [int(value) for value in model_config['hidden_dims']]
    if not hidden_dims:
        raise ValueError('At least one hidden dimension is required.')
    layers = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(model_config['dropout'])),
        ])
        current_dim = hidden_dim

    class GroupwiseMergeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.candidate_encoder = nn.Sequential(*layers)
            self.candidate_head = nn.Linear(current_dim, 1)
            self.no_merge_head = nn.Sequential(
                nn.Linear(current_dim, current_dim),
                nn.ReLU(),
                nn.Linear(current_dim, 1),
            )

        def forward(self, values, candidate_mask):
            encoded = self.candidate_encoder(values)
            candidate_logits = self.candidate_head(
                encoded).squeeze(-1)
            weights = candidate_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(
                dim=1).clamp_min(1.0)
            no_merge_logit = self.no_merge_head(
                pooled).squeeze(-1)
            return candidate_logits, no_merge_logit

    return GroupwiseMergeModel()


def groupwise_list_loss(
        candidate_logits, no_merge_logits,
        candidate_mask, positive_mask):
    import torch

    candidate_mask = candidate_mask.bool()
    positive_mask = positive_mask.bool() & candidate_mask
    if (
            candidate_logits.shape != candidate_mask.shape or
            positive_mask.shape != candidate_mask.shape or
            no_merge_logits.shape != candidate_mask.shape[:1]):
        raise ValueError('Groupwise loss tensor shapes do not align.')
    negative_infinity = torch.finfo(candidate_logits.dtype).min
    valid_candidate_logits = candidate_logits.masked_fill(
        ~candidate_mask, negative_infinity)
    denominator = torch.logsumexp(
        torch.cat(
            [no_merge_logits[:, None], valid_candidate_logits], dim=1),
        dim=1,
    )
    positive_logits = candidate_logits.masked_fill(
        ~positive_mask, negative_infinity)
    positive_numerator = torch.logsumexp(positive_logits, dim=1)
    has_positive = positive_mask.any(dim=1)
    numerator = torch.where(
        has_positive, positive_numerator, no_merge_logits)
    return (denominator - numerator).mean()


def set_seed(seed):
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def predict_groups(model, groups, batch_size, device):
    import torch

    records = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(groups), int(batch_size)):
            group_batch = groups[start:start + int(batch_size)]
            batch = collate_groups(group_batch)
            values = batch['features'].to(device)
            candidate_mask = batch['candidate_mask'].to(device)
            positive_mask = batch['positive_mask'].to(device)
            candidate_logits, no_merge_logits = model(
                values, candidate_mask)
            negative_infinity = torch.finfo(
                candidate_logits.dtype).min
            masked_candidates = candidate_logits.masked_fill(
                ~candidate_mask, negative_infinity)
            all_logits = torch.cat(
                [no_merge_logits[:, None], masked_candidates], dim=1)
            probabilities = torch.softmax(all_logits, dim=1)
            candidate_probabilities = probabilities[:, 1:]
            top_positions = masked_candidates.argmax(dim=1)
            top_probabilities = candidate_probabilities.gather(
                1, top_positions[:, None]).squeeze(1)
            top_correct = positive_mask.gather(
                1, top_positions[:, None]).squeeze(1)
            top_targets = batch['target_ids'].to(device).gather(
                1, top_positions[:, None]).squeeze(1)
            has_positive = positive_mask.any(dim=1)
            no_merge_probability = probabilities[:, 0]
            for index, source_key in enumerate(batch['source_keys']):
                records.append({
                    'source_key': source_key,
                    'target_instance_id': int(
                        top_targets[index].cpu().item()),
                    'confidence': float(
                        top_probabilities[index].cpu().item()),
                    'no_merge_probability': float(
                        no_merge_probability[index].cpu().item()),
                    'chosen_correct': bool(
                        top_correct[index].cpu().item()),
                    'has_positive_target': bool(
                        has_positive[index].cpu().item()),
                })
    return records


def group_source_metrics(records, threshold):
    selected = [
        row for row in records
        if row['confidence'] >= float(threshold)]
    correct = sum(row['chosen_correct'] for row in selected)
    unsafe = len(selected) - correct
    positive_sources = sum(
        row['has_positive_target'] for row in records)
    precision = correct / len(selected) if selected else 1.0
    recall = correct / max(positive_sources, 1)
    unsafe_rate = unsafe / max(len(records), 1)
    return {
        'threshold': float(threshold),
        'num_evaluable_sources': int(len(records)),
        'num_positive_sources': int(positive_sources),
        'num_proposed_merges': int(len(selected)),
        'num_correct_merges': int(correct),
        'num_unsafe_merges': int(unsafe),
        'pair_precision': float(precision),
        'positive_source_recall': float(recall),
        'unsafe_merge_rate': float(unsafe_rate),
    }


def select_group_threshold(records, constraints):
    thresholds = [float('inf')]
    thresholds.extend(float(value) for value in sorted({
        row['confidence'] for row in records
    }, reverse=True))
    feasible = []
    for threshold in thresholds:
        metrics = group_source_metrics(records, threshold)
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
        if metrics['precision_passed'] and metrics['unsafe_rate_passed']:
            feasible.append(metrics)
    if not feasible:
        raise RuntimeError(
            'No groupwise threshold satisfies the safety constraints.')
    return max(
        feasible,
        key=lambda row: (
            row['positive_source_recall'],
            row['pair_precision'],
            -row['unsafe_merge_rate'],
            row['threshold'],
        ))


def binary_average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError('Average-precision inputs must align.')
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind='mergesort')
    ordered_labels = labels[order].astype(np.float64)
    precision = np.cumsum(ordered_labels) / np.arange(
        1, len(labels) + 1, dtype=np.float64)
    return float(np.sum(precision * ordered_labels) / positives)


def source_average_precision(records):
    return binary_average_precision(
        [row['chosen_correct'] for row in records],
        [row['confidence'] for row in records],
    )


def train_groupwise_model(
        train_groups, validation_groups, input_dim,
        model_config, training_config, constraints, seed):
    import torch

    set_seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model = build_groupwise_merge_model(
        input_dim, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
    )
    rng = np.random.default_rng(int(seed))
    best_state = None
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    batch_size = int(training_config['batch_size'])
    for epoch in range(1, int(training_config['epochs']) + 1):
        model.train()
        order = rng.permutation(len(train_groups))
        for start in range(0, len(order), batch_size):
            group_batch = [
                train_groups[int(index)]
                for index in order[start:start + batch_size]]
            batch = collate_groups(group_batch)
            values = batch['features'].to(device)
            candidate_mask = batch['candidate_mask'].to(device)
            positive_mask = batch['positive_mask'].to(device)
            candidate_logits, no_merge_logits = model(
                values, candidate_mask)
            loss = groupwise_list_loss(
                candidate_logits,
                no_merge_logits,
                candidate_mask,
                positive_mask,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        records = predict_groups(
            model, validation_groups, batch_size, device)
        source_ap = source_average_precision(records)
        selected = select_group_threshold(records, constraints)
        key = (
            source_ap,
            selected['positive_source_recall'],
            selected['pair_precision'],
        )
        if best_key is None or key > best_key:
            best_key = key
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
    records = predict_groups(
        model, validation_groups, batch_size, device)
    selected = select_group_threshold(records, constraints)
    selected.update({
        'source_average_precision': source_average_precision(records),
        'top1_correct_rate_on_positive_sources': float(
            sum(row['chosen_correct'] for row in records) /
            max(sum(
                row['has_positive_target'] for row in records), 1)),
        'best_epoch': int(best_epoch),
        'device': str(device),
        'parameter_count': int(sum(
            parameter.numel() for parameter in model.parameters())),
        'seed': int(seed),
        'gate_passed': bool(selected['passed']),
    })
    return {
        'metrics': selected,
        'records': records,
        'state_dict': best_state,
    }


def aggregate_metrics(rows):
    metric_names = [
        'source_average_precision',
        'top1_correct_rate_on_positive_sources',
        'pair_precision',
        'positive_source_recall',
        'unsafe_merge_rate',
        'num_proposed_merges',
        'num_correct_merges',
        'num_unsafe_merges',
    ]
    summary = {
        'num_seeds': len(rows),
        'num_gate_passed': int(sum(
            row['gate_passed'] for row in rows)),
    }
    for name in metric_names:
        values = np.asarray(
            [float(row[name]) for row in rows], dtype=np.float64)
        summary[name] = {
            'mean': float(values.mean()),
            'sample_std': float(
                values.std(ddof=1) if len(values) > 1 else 0.0),
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return summary


def format_markdown(summary):
    groupwise = summary['groupwise']
    lines = [
        '# E8c2 Groupwise 安全合并头',
        '',
        f'- 训练源实例：{summary["num_train_sources"]}',
        f'- 验证源实例：{summary["num_validation_sources"]}',
        f'- 验证正目标源实例：{summary["num_validation_positive_sources"]}',
        f'- 锁定部署 seed：{summary["locked_seed"]}',
        f'- 锁定阈值：{summary["selected_threshold"]}',
        f'- 可训练参数：{summary["parameter_count"]}',
        f'- Pointwise 安全召回：'
        f'{summary["pointwise_reference_recall"]:.6f}',
        f'- Groupwise 安全召回增益：'
        f'{summary["recall_gain_over_pointwise"]:+.6f}',
        '',
        '| Metric | Mean | Std | Min | Max |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in (
            'source_average_precision',
            'top1_correct_rate_on_positive_sources',
            'pair_precision',
            'positive_source_recall',
            'unsafe_merge_rate'):
        value = groupwise[name]
        lines.append(
            f'| {name} | {value["mean"]:.6f} | '
            f'{value["sample_std"]:.6f} | {value["min"]:.6f} | '
            f'{value["max"]:.6f} |')
    lines.extend([
        '',
        f'- Gate passed seeds：'
        f'{groupwise["num_gate_passed"]}/{groupwise["num_seeds"]}',
        '',
        '## Gate',
        '',
    ])
    lines.extend(
        f'- {name}: **{value}**'
        for name, value in summary['gate'].items())
    lines.extend([
        '',
        ('PASS：进入 E8d L1W Groupwise 合并集成。'
         if summary['gate']['passed']
         else 'STOP：不进入 pipeline 合并。'),
        '',
    ])
    return '\n'.join(lines)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    import torch

    settings = POINTWISE.load_yaml(config_path)
    pointwise_config = POINTWISE.load_yaml(
        settings['pointwise_config_path'])
    dataset = POINTWISE.load_pair_dataset(
        pointwise_config['pairs_path'],
        pointwise_config['feature_names'],
    )
    with open(
            settings['pointwise_summary_path'],
            encoding='utf-8') as file:
        pointwise_summary = json.load(file)
    if pointwise_summary['gate']['passed']:
        raise ValueError(
            'E8c2 is only permitted after the pointwise E8c gate fails.')

    valid = dataset['pair_valid']
    train_mask = valid & (dataset['split'] == 'train')
    validation_mask = valid & (dataset['split'] == 'validation')
    median, scale = POINTWISE.fit_standardizer(
        dataset['features'][train_mask])
    standardized = POINTWISE.standardize(
        dataset['features'], median, scale)
    train_groups = build_source_groups(
        standardized[train_mask],
        dataset['pair_positive'][train_mask],
        dataset['source_key'][train_mask],
        dataset['target_instance_id'][train_mask],
    )
    validation_groups = build_source_groups(
        standardized[validation_mask],
        dataset['pair_positive'][validation_mask],
        dataset['source_key'][validation_mask],
        dataset['target_instance_id'][validation_mask],
    )

    output_dir = Path(settings['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_rows = []
    locked_seed = int(settings['selection']['locked_seed'])
    locked_threshold = None
    parameter_count = None
    for seed in settings['seeds']:
        seed = int(seed)
        result = train_groupwise_model(
            train_groups,
            validation_groups,
            len(dataset['feature_names']),
            settings['model'],
            settings['training'],
            pointwise_config['constraints'],
            seed,
        )
        metrics = result['metrics']
        rows.append(metrics)
        parameter_count = metrics['parameter_count']
        if seed == locked_seed:
            locked_threshold = float(metrics['threshold'])
        torch.save({
            'state_dict': result['state_dict'],
            'feature_names': dataset['feature_names'],
            'median': median,
            'scale': scale,
            'seed': seed,
            'best_epoch': metrics['best_epoch'],
            'selected_threshold': metrics['threshold'],
            'model_config': settings['model'],
            'training_config': settings['training'],
            'constraints': pointwise_config['constraints'],
        }, checkpoint_dir / f'groupwise_merge_seed{seed}.pth')
        prediction_rows.extend({
            'seed': seed,
            **record,
        } for record in result['records'])

    aggregate = aggregate_metrics(rows)
    locked_row = next(
        row for row in rows if int(row['seed']) == locked_seed)
    recall_std = aggregate[
        'positive_source_recall']['sample_std']
    pointwise_reference_recall = float(
        pointwise_summary['models']['pair_mlp'][
            'positive_source_recall']['mean'])
    recall_gain = float(
        aggregate['positive_source_recall']['mean'] -
        pointwise_reference_recall)
    gate = {
        'pointwise_e8c_failed_as_expected': bool(
            not pointwise_summary['gate']['passed']),
        'enough_seed_passes': bool(
            aggregate['num_gate_passed'] >= int(
                settings['selection']['min_gate_pass_seeds'])),
        'locked_seed_passed': bool(locked_row['gate_passed']),
        'recall_stability_passed': bool(
            recall_std <= float(
                settings['gate']['max_recall_sample_std'])),
        'recall_gain_over_pointwise_passed': bool(
            recall_gain >= float(
                settings['gate'][
                    'min_mean_recall_gain_over_pointwise'])),
    }
    gate['passed'] = bool(all(gate.values()))
    summary = {
        'config': str(config_path),
        'pointwise_summary_path': settings['pointwise_summary_path'],
        'num_train_sources': int(len(train_groups)),
        'num_validation_sources': int(len(validation_groups)),
        'num_validation_positive_sources': int(sum(
            group['positive_mask'].any()
            for group in validation_groups)),
        'feature_names': dataset['feature_names'],
        'seeds': [int(seed) for seed in settings['seeds']],
        'locked_seed': locked_seed,
        'selected_threshold': locked_threshold,
        'parameter_count': int(parameter_count),
        'pointwise_reference_recall': pointwise_reference_recall,
        'recall_gain_over_pointwise': recall_gain,
        'groupwise': aggregate,
        'locked_seed_metrics': locked_row,
        'constraints': pointwise_config['constraints'],
        'gate': gate,
    }
    write_csv(output_dir / 'per_seed_metrics.csv', rows)
    write_csv(output_dir / 'validation_predictions.csv', prediction_rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8')
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(
        markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'E8c2 groupwise gate failed; do not integrate merging.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Train the E8c2 groupwise instance merge head.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()