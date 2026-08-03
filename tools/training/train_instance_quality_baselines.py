import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def _as_bool(array):
    return np.asarray(array, dtype=bool)


def load_quality_dataset(data_root, manifest_path):
    root = Path(data_root).resolve()
    with Path(manifest_path).open(newline='', encoding='utf-8') as file:
        manifest_rows = list(csv.DictReader(file))
    if not manifest_rows:
        raise ValueError('The E2 manifest is empty.')
    manifest_instances = {}
    for row in manifest_rows:
        path = Path(row['artifact_path']).resolve()
        manifest_instances.setdefault(path, set()).add(int(row['instance_id']))
    artifact_paths = sorted(manifest_instances)
    expected_plots = {
        (row['source_plot'], row['split']) for row in manifest_rows}
    records = []
    reference_features = None
    observed_plots = set()
    for path in artifact_paths:
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid E2 artifact path: {path}')
        with np.load(path, allow_pickle=False) as data:
            names = data['global_feature_names'].astype(str).tolist()
            if reference_features is None:
                reference_features = names
            elif names != reference_features:
                raise ValueError(f'Global feature schema differs in {path}.')
            plot = str(np.asarray(data['source_plot']).item())
            split = str(np.asarray(data['split']).item())
            observed_plots.add((plot, split))
            instance_ids = data['instance_ids'].astype(np.int64)
            count = len(instance_ids)
            if set(instance_ids.tolist()) != manifest_instances[path]:
                raise ValueError(
                    f'Manifest instance IDs differ from {path}.')
            records.append({
                'instance_id': instance_ids,
                'source_plot': np.repeat(plot, count),
                'split': np.repeat(split, count),
                'features': data['global_feature_values'].astype(np.float32),
                'target_iou': data['target_max_iou'].astype(np.float32),
                'target_valid': _as_bool(data['target_valid']),
                'classification_valid': _as_bool(
                    data['target_classification_valid']),
                'target_true': _as_bool(data['target_is_true_tree']),
            })
    if observed_plots != expected_plots:
        raise ValueError(
            'Manifest and NPZ plot/split sets differ: '
            f'manifest={sorted(expected_plots)}, npz={sorted(observed_plots)}')
    output = {}
    for key in records[0]:
        output[key] = np.concatenate([record[key] for record in records])
    output['feature_names'] = reference_features
    if not np.isfinite(output['features']).all():
        raise ValueError('Global features contain NaN or infinity.')
    if set(np.unique(output['split'])) != {'train', 'validation'}:
        raise ValueError('Both train and validation splits are required.')
    train_plots = set(output['source_plot'][output['split'] == 'train'])
    validation_plots = set(
        output['source_plot'][output['split'] == 'validation'])
    if train_plots & validation_plots:
        raise ValueError('A source plot appears in both splits.')
    return output


def fit_standardizer(values):
    median = np.median(values, axis=0)
    centered = values - median
    scale = np.std(centered, axis=0)
    scale[scale < 1e-6] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def standardize(values, median, scale):
    return ((values - median) / scale).astype(np.float32)


def rank_correlation(first, second):
    def ranks(values):
        values = np.asarray(values)
        order = np.argsort(values, kind='mergesort')
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return result
    first_rank = ranks(first)
    second_rank = ranks(second)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def select_filter_threshold(targets, scores, step, max_completeness_drop):
    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(targets.sum())
    if positives == 0 or positives == len(targets):
        raise ValueError('Threshold selection requires both classes.')
    candidates = []
    threshold_values = np.round(
        np.arange(0.0, 1.0 + step / 2.0, step), decimals=10)
    for threshold in threshold_values:
        retained = scores >= threshold
        tp = int(np.count_nonzero(retained & targets))
        fp = int(np.count_nonzero(retained & ~targets))
        fn = positives - tp
        completeness = tp / positives
        commission = fp / max(tp + fp, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        row = {
            'threshold': float(threshold),
            'tp': tp, 'fp': fp, 'fn': fn,
            'completeness': completeness,
            'commission': commission,
            'f1': f1,
        }
        if completeness >= 1.0 - max_completeness_drop:
            candidates.append(row)
    if not candidates:
        raise RuntimeError('No threshold satisfies the completeness constraint.')
    best = max(
        candidates,
        key=lambda row: (
            row['f1'], row['completeness'], -row['threshold']))
    baseline = candidates[0]
    return best, baseline


def classification_metrics(targets, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score

    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        'roc_auc': float(roc_auc_score(targets, scores)),
        'positive_ap': float(average_precision_score(targets, scores)),
        'fp_ap': float(average_precision_score(1 - targets, 1.0 - scores)),
    }


def select_single_feature(train_x, train_y, feature_names):
    from sklearn.metrics import roc_auc_score

    best = None
    for index, name in enumerate(feature_names):
        values = train_x[:, index]
        if np.all(values == values[0]):
            continue
        auc = float(roc_auc_score(train_y, values))
        direction = 1.0 if auc >= 0.5 else -1.0
        candidate = (max(auc, 1.0 - auc), name, index, direction)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError('Every global feature is constant.')
    return {
        'train_auc': best[0],
        'feature_name': best[1],
        'feature_index': best[2],
        'direction': best[3],
    }


def train_sklearn_baselines(train_x, train_y, validation_x, seed, feature_names):
    from sklearn.linear_model import LogisticRegression

    selected = select_single_feature(train_x, train_y, feature_names)
    feature_index = selected['feature_index']
    single = LogisticRegression(
        class_weight='balanced', max_iter=2000, random_state=seed)
    single.fit(
        selected['direction'] * train_x[:, [feature_index]], train_y)
    single_scores = single.predict_proba(
        selected['direction'] * validation_x[:, [feature_index]])[:, 1]

    logistic = LogisticRegression(
        class_weight='balanced', max_iter=2000, random_state=seed)
    logistic.fit(train_x, train_y)
    logistic_scores = logistic.predict_proba(validation_x)[:, 1]
    return selected, single_scores, logistic_scores


def _set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_global_mlp(
        train_x, train_true, train_iou, train_class_mask, train_valid_mask,
        validation_x, validation_true, validation_iou,
        validation_class_mask, validation_valid_mask, config, seed):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dims = list(config['hidden_dims'])
    layers = []
    input_dim = train_x.shape[1]
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(config['dropout'])),
        ])
        input_dim = hidden_dim

    class QualityMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(*layers)
            self.validity = nn.Linear(input_dim, 1)
            self.iou = nn.Linear(input_dim, 1)

        def forward(self, values):
            encoded = self.encoder(values)
            return self.validity(encoded).squeeze(1), torch.sigmoid(
                self.iou(encoded).squeeze(1))

    model = QualityMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']))
    train_tensors = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_true.astype(np.float32)),
        torch.from_numpy(train_iou.astype(np.float32)),
        torch.from_numpy(train_class_mask),
        torch.from_numpy(train_valid_mask),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_tensors, batch_size=int(config['batch_size']), shuffle=True,
        generator=generator, num_workers=0)
    class_targets = train_true[train_class_mask]
    class_weights = np.asarray([
        len(class_targets) / (2 * max(np.count_nonzero(~class_targets), 1)),
        len(class_targets) / (2 * max(np.count_nonzero(class_targets), 1)),
    ], dtype=np.float32)
    class_weights = torch.from_numpy(class_weights).to(device)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    smooth_l1 = nn.SmoothL1Loss(reduction='none')

    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state = None
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(config['epochs']) + 1):
        model.train()
        for values, true, iou, class_mask, valid_mask in loader:
            values = values.to(device)
            true = true.to(device)
            iou = iou.to(device)
            class_mask = class_mask.to(device)
            valid_mask = valid_mask.to(device)
            logits, predicted_iou = model(values)
            if torch.any(class_mask):
                classification_loss = bce(
                    logits[class_mask], true[class_mask])
                weights = class_weights[true[class_mask].long()]
                classification_loss = (
                    classification_loss * weights).mean()
            else:
                classification_loss = logits.sum() * 0.0
            if torch.any(valid_mask):
                regression_loss = smooth_l1(
                    predicted_iou[valid_mask], iou[valid_mask]).mean()
            else:
                regression_loss = predicted_iou.sum() * 0.0
            loss = classification_loss + float(
                config['iou_loss_weight']) * regression_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, predicted_iou = model(validation_tensor)
            probability = torch.sigmoid(logits).cpu().numpy()
            predicted_iou_np = predicted_iou.cpu().numpy()
        fp_ap = classification_metrics(
            validation_true[validation_class_mask],
            probability[validation_class_mask])['fp_ap']
        iou_mae = float(np.mean(np.abs(
            predicted_iou_np[validation_valid_mask] -
            validation_iou[validation_valid_mask])))
        key = (iou_mae, -fp_ap)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= int(config['patience']):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, predicted_iou = model(validation_tensor)
    return {
        'validity_probability': torch.sigmoid(logits).cpu().numpy(),
        'predicted_iou': predicted_iou.cpu().numpy(),
        'best_epoch': best_epoch,
        'state_dict': best_state,
        'device': str(device),
    }


def evaluate_predictions(
        model_name, seed, true, target_iou, class_mask, valid_mask,
        validity_probability, predicted_iou, threshold_config,
        selected_feature=''):
    final_score = np.clip(validity_probability * predicted_iou, 0.0, 1.0)
    metrics = classification_metrics(
        true[class_mask], final_score[class_mask])
    metrics.update({
        'model': model_name,
        'seed': int(seed),
        'selected_feature': selected_feature,
        'iou_mae': float(np.mean(np.abs(
            predicted_iou[valid_mask] - target_iou[valid_mask]))),
        'spearman': rank_correlation(
            final_score[valid_mask], target_iou[valid_mask]),
    })
    best, baseline = select_filter_threshold(
        true[class_mask], final_score[class_mask],
        float(threshold_config['step']),
        float(threshold_config['max_completeness_drop']))
    metrics.update({
        'selected_threshold': best['threshold'],
        'filtered_f1': best['f1'],
        'filtered_completeness': best['completeness'],
        'filtered_commission': best['commission'],
        'unfiltered_f1': baseline['f1'],
    })
    return metrics, final_score


def summarize(rows, settings, dataset):
    metric_names = [
        'roc_auc', 'positive_ap', 'fp_ap', 'iou_mae', 'spearman',
        'filtered_f1', 'filtered_completeness', 'filtered_commission',
        'unfiltered_f1']
    models = {}
    for model_name in sorted({row['model'] for row in rows}):
        group = [row for row in rows if row['model'] == model_name]
        model_summary = {'num_seeds': len(group)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group], dtype=float)
            model_summary[metric] = {
                'mean': float(values.mean()),
                'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                'min': float(values.min()),
                'max': float(values.max()),
            }
        selected = [row['selected_feature'] for row in group if row['selected_feature']]
        if selected:
            model_summary['selected_features'] = dict(Counter(selected))
        models[model_name] = model_summary

    aggregate_names = ['logistic_regression', 'global_mlp']
    best_name = max(
        aggregate_names,
        key=lambda name: (
            models[name]['roc_auc']['mean'], models[name]['fp_ap']['mean']))
    best = models[best_name]
    mlp_auc_range = (
        models['global_mlp']['roc_auc']['max'] -
        models['global_mlp']['roc_auc']['min'])
    gate_config = settings['gate']
    gate = {
        'aggregate_auc_passed': (
            best['roc_auc']['mean'] >= float(
                gate_config['min_validation_roc_auc'])),
        'aggregate_fp_ap_passed': (
            best['fp_ap']['mean'] >= float(
                gate_config['min_validation_fp_ap'])),
        'filtered_f1_not_below_unfiltered': (
            best['filtered_f1']['mean'] + 1e-12 >=
            best['unfiltered_f1']['mean']),
        'mlp_seed_stability_passed': (
            mlp_auc_range <= float(gate_config['max_mlp_auc_range'])),
    }
    gate['passed'] = all(gate.values())
    return {
        'num_instances': int(len(dataset['instance_id'])),
        'num_features': len(dataset['feature_names']),
        'feature_names': dataset['feature_names'],
        'train_plots': sorted(set(dataset['source_plot'][dataset['split'] == 'train'])),
        'validation_plots': sorted(set(
            dataset['source_plot'][dataset['split'] == 'validation'])),
        'seeds': settings['seeds'],
        'models': models,
        'best_aggregate_model': best_name,
        'mlp_auc_range': mlp_auc_range,
        'gate': gate,
    }


def format_markdown(summary):
    lines = [
        '# E3 global instance-quality baselines', '',
        f'- Instances: {summary["num_instances"]}',
        f'- Global features: {summary["num_features"]}',
        f'- Best aggregate model: {summary["best_aggregate_model"]}', '',
        '| Model | ROC-AUC | FP AP | IoU MAE | Spearman | Filtered F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for model_name in (
            'single_feature', 'logistic_regression', 'global_mlp'):
        model = summary['models'][model_name]
        lines.append(
            f'| {model_name} | {model["roc_auc"]["mean"]:.6f} | '
            f'{model["fp_ap"]["mean"]:.6f} | '
            f'{model["iou_mae"]["mean"]:.6f} | '
            f'{model["spearman"]["mean"]:.6f} | '
            f'{model["filtered_f1"]["mean"]:.6f} |')
    lines.extend(['', '## Gate', ''])
    lines.extend(
        f'- {name}: **{passed}**'
        for name, passed in summary['gate'].items())
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train E3 global instance-quality baselines.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    import torch
    import yaml

    args = parse_args()
    with open(args.config, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    dataset = load_quality_dataset(
        settings['data_root'], settings['manifest_path'])
    train_mask = dataset['split'] == 'train'
    validation_mask = dataset['split'] == 'validation'
    median, scale = fit_standardizer(dataset['features'][train_mask])
    values = standardize(dataset['features'], median, scale)
    train_x = values[train_mask]
    validation_x = values[validation_mask]
    train_true = dataset['target_true'][train_mask]
    validation_true = dataset['target_true'][validation_mask]
    train_iou = dataset['target_iou'][train_mask]
    validation_iou = dataset['target_iou'][validation_mask]
    train_class = dataset['classification_valid'][train_mask]
    validation_class = dataset['classification_valid'][validation_mask]
    train_valid = dataset['target_valid'][train_mask]
    validation_valid = dataset['target_valid'][validation_mask]

    output_dir = Path(settings['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_rows = []
    for seed in settings['seeds']:
        selected, single_probability, logistic_probability = (
            train_sklearn_baselines(
                train_x[train_class], train_true[train_class],
                validation_x, int(seed), dataset['feature_names']))
        for name, probability, feature in (
                ('single_feature', single_probability, selected['feature_name']),
                ('logistic_regression', logistic_probability, '')):
            metrics, score = evaluate_predictions(
                name, seed, validation_true, validation_iou,
                validation_class, validation_valid,
                probability, probability, settings['threshold_selection'],
                selected_feature=feature)
            rows.append(metrics)
            prediction_rows.extend({
                'model': name, 'seed': seed,
                'source_plot': plot, 'instance_id': int(instance_id),
                'score': float(value),
                'target_iou': float(iou),
                'target_true': bool(true),
                'classification_valid': bool(class_valid),
            } for plot, instance_id, value, iou, true, class_valid in zip(
                dataset['source_plot'][validation_mask],
                dataset['instance_id'][validation_mask], score,
                validation_iou, validation_true, validation_class))

        mlp = train_global_mlp(
            train_x, train_true, train_iou, train_class, train_valid,
            validation_x, validation_true, validation_iou,
            validation_class, validation_valid, settings['mlp'], int(seed))
        metrics, score = evaluate_predictions(
            'global_mlp', seed, validation_true, validation_iou,
            validation_class, validation_valid,
            mlp['validity_probability'], mlp['predicted_iou'],
            settings['threshold_selection'])
        metrics['best_epoch'] = mlp['best_epoch']
        metrics['device'] = mlp['device']
        rows.append(metrics)
        torch.save({
            'state_dict': mlp['state_dict'],
            'feature_names': dataset['feature_names'],
            'median': median, 'scale': scale,
            'seed': seed, 'best_epoch': mlp['best_epoch'],
            'config': settings['mlp'],
        }, checkpoint_dir / f'global_mlp_seed{seed}.pth')
        prediction_rows.extend({
            'model': 'global_mlp', 'seed': seed,
            'source_plot': plot, 'instance_id': int(instance_id),
            'score': float(value),
            'target_iou': float(iou),
            'target_true': bool(true),
            'classification_valid': bool(class_valid),
        } for plot, instance_id, value, iou, true, class_valid in zip(
            dataset['source_plot'][validation_mask],
            dataset['instance_id'][validation_mask], score,
            validation_iou, validation_true, validation_class))

    summary = summarize(rows, settings, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'per_seed_metrics.csv').open(
            'w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'validation_predictions.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError('E3 global-baseline gate failed.')
    print('PASS: proceed to E4 Vertical-MLP.', flush=True)


if __name__ == '__main__':
    main()
