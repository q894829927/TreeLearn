import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


BASELINE_PATH = Path(__file__).with_name(
    'train_instance_quality_baselines.py')
BASELINE_SPEC = importlib.util.spec_from_file_location(
    'instance_quality_baselines_for_vertical_mlp', BASELINE_PATH)
BASELINES = importlib.util.module_from_spec(BASELINE_SPEC)
BASELINE_SPEC.loader.exec_module(BASELINES)


def load_vertical_quality_dataset(data_root, manifest_path):
    dataset = BASELINES.load_quality_dataset(data_root, manifest_path)
    root = Path(data_root).resolve()
    with Path(manifest_path).open(newline='', encoding='utf-8') as file:
        manifest_rows = list(csv.DictReader(file))
    artifact_paths = sorted({
        Path(row['artifact_path']).resolve() for row in manifest_rows})

    target_indices = {}
    for index, (plot, instance_id) in enumerate(zip(
            dataset['source_plot'], dataset['instance_id'])):
        key = (str(plot), int(instance_id))
        if key in target_indices:
            raise ValueError(f'Duplicate dataset instance key: {key}')
        target_indices[key] = index

    aligned_tokens = [None] * len(dataset['instance_id'])
    aligned_masks = [None] * len(dataset['instance_id'])
    token_feature_names = None
    reference_shape = None
    for path in artifact_paths:
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid E2 artifact path: {path}')
        with np.load(path, allow_pickle=False) as data:
            required = {
                'instance_ids', 'source_plot', 'vertical_tokens',
                'layer_valid_mask', 'token_feature_names'}
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f'{path} misses vertical arrays: {sorted(missing)}')
            plot = str(np.asarray(data['source_plot']).item())
            instance_ids = data['instance_ids'].astype(np.int64)
            tokens = data['vertical_tokens'].astype(np.float32)
            masks = np.asarray(data['layer_valid_mask'], dtype=bool)
            names = data['token_feature_names'].astype(str).tolist()
            if tokens.ndim != 3 or masks.shape != tokens.shape[:2]:
                raise ValueError(
                    f'Invalid vertical token shape in {path}: '
                    f'{tokens.shape}, mask={masks.shape}.')
            if len(instance_ids) != len(tokens):
                raise ValueError(f'Instance/token count mismatch in {path}.')
            if np.any(masks.sum(axis=1) == 0):
                raise ValueError(f'An instance has no valid layer in {path}.')
            if not np.isfinite(tokens).all():
                raise ValueError(f'Non-finite vertical token in {path}.')
            if token_feature_names is None:
                token_feature_names = names
                reference_shape = tokens.shape[1:]
            elif names != token_feature_names or tokens.shape[1:] != reference_shape:
                raise ValueError(f'Vertical token schema differs in {path}.')
            for row, instance_id in enumerate(instance_ids):
                key = (plot, int(instance_id))
                if key not in target_indices:
                    raise ValueError(f'Unexpected vertical instance key: {key}')
                index = target_indices[key]
                if aligned_tokens[index] is not None:
                    raise ValueError(f'Duplicate vertical instance key: {key}')
                aligned_tokens[index] = tokens[row]
                aligned_masks[index] = masks[row]

    missing_count = sum(value is None for value in aligned_tokens)
    if missing_count:
        raise ValueError(f'{missing_count} instances lack vertical tokens.')
    dataset['vertical_tokens'] = np.stack(aligned_tokens).astype(np.float32)
    dataset['layer_valid_mask'] = np.stack(aligned_masks).astype(bool)
    dataset['token_feature_names'] = token_feature_names
    return dataset


def fit_token_standardizer(tokens, layer_mask):
    tokens = np.asarray(tokens, dtype=np.float32)
    layer_mask = np.asarray(layer_mask, dtype=bool)
    if tokens.ndim != 3 or layer_mask.shape != tokens.shape[:2]:
        raise ValueError('Token and mask shapes are inconsistent.')
    valid_tokens = tokens[layer_mask]
    if len(valid_tokens) == 0:
        raise ValueError('No valid token is available for standardization.')
    median = np.median(valid_tokens, axis=0)
    scale = np.std(valid_tokens - median, axis=0)
    scale[scale < 1e-6] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def standardize_tokens(tokens, layer_mask, median, scale):
    output = (
        (np.asarray(tokens, dtype=np.float32) - median) / scale
    ).astype(np.float32)
    output[~np.asarray(layer_mask, dtype=bool)] = 0.0
    if not np.isfinite(output).all():
        raise ValueError('Standardized vertical tokens are non-finite.')
    return output


def build_vertical_mlp(input_dim, model_config):
    import torch
    from torch import nn

    token_hidden_dim = int(model_config['token_hidden_dim'])
    num_token_layers = int(model_config['token_mlp_layers'])
    if token_hidden_dim <= 0 or num_token_layers <= 0:
        raise ValueError('Token MLP dimensions must be positive.')
    dropout = float(model_config['dropout'])

    class VerticalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            token_layers = []
            current_dim = int(input_dim)
            for _ in range(num_token_layers):
                token_layers.extend([
                    nn.Linear(current_dim, token_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                current_dim = token_hidden_dim
            self.token_encoder = nn.Sequential(*token_layers)

            instance_layers = []
            current_dim = 2 * token_hidden_dim
            for hidden_dim in model_config['instance_hidden_dims']:
                hidden_dim = int(hidden_dim)
                instance_layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                current_dim = hidden_dim
            self.instance_encoder = nn.Sequential(*instance_layers)
            self.validity = nn.Linear(current_dim, 1)
            self.iou = nn.Linear(current_dim, 1)

        def forward(self, tokens, layer_mask):
            layer_mask = layer_mask.bool()
            if tokens.ndim != 3 or layer_mask.shape != tokens.shape[:2]:
                raise ValueError('Token and layer-mask shapes are inconsistent.')
            if not torch.all(layer_mask.any(dim=1)):
                raise ValueError('Every instance must have a valid layer.')
            encoded = self.token_encoder(tokens)
            mask_values = layer_mask.unsqueeze(-1)
            mean = (encoded * mask_values).sum(dim=1) / mask_values.sum(
                dim=1).clamp_min(1)
            maximum = encoded.masked_fill(~mask_values, -torch.inf).max(
                dim=1).values
            pooled = torch.cat([mean, maximum], dim=1)
            instance = self.instance_encoder(pooled)
            return (
                self.validity(instance).squeeze(1),
                torch.sigmoid(self.iou(instance).squeeze(1)),
            )

    return VerticalMLP()


def train_vertical_mlp(
        train_tokens, train_layer_mask, train_true, train_iou,
        train_class_mask, train_valid_mask,
        validation_tokens, validation_layer_mask, validation_true,
        validation_iou, validation_class_mask, validation_valid_mask,
        model_config, training_config, seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    BASELINES._set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_vertical_mlp(train_tokens.shape[-1], model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']))
    dataset = TensorDataset(
        torch.from_numpy(train_tokens),
        torch.from_numpy(train_layer_mask),
        torch.from_numpy(train_true.astype(np.float32)),
        torch.from_numpy(train_iou.astype(np.float32)),
        torch.from_numpy(train_class_mask),
        torch.from_numpy(train_valid_mask),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=int(training_config['batch_size']), shuffle=True,
        generator=generator, num_workers=0)

    class_targets = train_true[train_class_mask]
    class_weights = np.asarray([
        len(class_targets) / (2 * max(np.count_nonzero(~class_targets), 1)),
        len(class_targets) / (2 * max(np.count_nonzero(class_targets), 1)),
    ], dtype=np.float32)
    class_weights = torch.from_numpy(class_weights).to(device)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    smooth_l1 = nn.SmoothL1Loss(reduction='none')

    validation_token_tensor = torch.from_numpy(validation_tokens).to(device)
    validation_mask_tensor = torch.from_numpy(
        validation_layer_mask).to(device)
    best_state = None
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(training_config['epochs']) + 1):
        model.train()
        for tokens, layer_mask, true, iou, class_mask, valid_mask in loader:
            tokens = tokens.to(device)
            layer_mask = layer_mask.to(device)
            true = true.to(device)
            iou = iou.to(device)
            class_mask = class_mask.to(device)
            valid_mask = valid_mask.to(device)
            logits, predicted_iou = model(tokens, layer_mask)
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
                training_config['iou_loss_weight']) * regression_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, predicted_iou = model(
                validation_token_tensor, validation_mask_tensor)
            probability = torch.sigmoid(logits).cpu().numpy()
            predicted_iou_np = predicted_iou.cpu().numpy()
        fp_ap = BASELINES.classification_metrics(
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
        if stale_epochs >= int(training_config['patience']):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, predicted_iou = model(
            validation_token_tensor, validation_mask_tensor)
    return {
        'validity_probability': torch.sigmoid(logits).cpu().numpy(),
        'predicted_iou': predicted_iou.cpu().numpy(),
        'best_epoch': best_epoch,
        'state_dict': best_state,
        'device': str(device),
    }


def summarize_vertical(rows, baseline_summary, settings, dataset):
    metric_names = [
        'roc_auc', 'positive_ap', 'fp_ap', 'iou_mae', 'spearman',
        'filtered_f1', 'filtered_completeness', 'filtered_commission',
        'unfiltered_f1']
    model = {'num_seeds': len(rows)}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        model[metric] = {
            'mean': float(values.mean()),
            'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max()),
        }

    if not baseline_summary.get('gate', {}).get('passed', False):
        raise ValueError('The fixed E3 baseline summary did not pass its gate.')
    baseline_name = baseline_summary['best_aggregate_model']
    baseline = baseline_summary['models'][baseline_name]
    deltas = {
        'filtered_f1': (
            model['filtered_f1']['mean'] -
            baseline['filtered_f1']['mean']),
        'iou_mae_reduction': (
            baseline['iou_mae']['mean'] - model['iou_mae']['mean']),
        'fp_ap': model['fp_ap']['mean'] - baseline['fp_ap']['mean'],
        'roc_auc': model['roc_auc']['mean'] - baseline['roc_auc']['mean'],
        'spearman': model['spearman']['mean'] - baseline['spearman']['mean'],
    }
    gate_config = settings['gate']
    gate = {
        'e3_baseline_passed': True,
        'f1_not_below_global_baseline': (
            deltas['filtered_f1'] >= -float(gate_config['f1_tolerance'])),
        'iou_mae_improved': (
            deltas['iou_mae_reduction'] >= float(
                gate_config['min_iou_mae_improvement'])),
        'fp_ap_improved': (
            deltas['fp_ap'] >= float(
                gate_config['min_fp_ap_improvement'])),
        'f1_seed_stability_passed': (
            model['filtered_f1']['sample_std'] <= float(
                gate_config['max_f1_sample_std'])),
    }
    gate['vertical_information_improved'] = (
        gate['iou_mae_improved'] or gate['fp_ap_improved'])
    gate['passed'] = (
        gate['e3_baseline_passed'] and
        gate['f1_not_below_global_baseline'] and
        gate['vertical_information_improved'] and
        gate['f1_seed_stability_passed'])
    return {
        'num_instances': int(len(dataset['instance_id'])),
        'num_layers': int(dataset['vertical_tokens'].shape[1]),
        'token_dim': int(dataset['vertical_tokens'].shape[2]),
        'token_feature_names': dataset['token_feature_names'],
        'train_plots': sorted(set(
            dataset['source_plot'][dataset['split'] == 'train'])),
        'validation_plots': sorted(set(
            dataset['source_plot'][dataset['split'] == 'validation'])),
        'seeds': settings['seeds'],
        'baseline_name': baseline_name,
        'baseline': baseline,
        'vertical_mlp': model,
        'deltas': deltas,
        'gate': gate,
    }


def format_markdown(summary):
    baseline = summary['baseline']
    model = summary['vertical_mlp']
    delta = summary['deltas']
    lines = [
        '# E4 Vertical-MLP', '',
        f'- Instances: {summary["num_instances"]}',
        f'- Vertical tokens: {summary["num_layers"]} x {summary["token_dim"]}',
        f'- Fixed E3 baseline: {summary["baseline_name"]}', '',
        '| Model | ROC-AUC | FP AP | IoU MAE | Spearman | Filtered F1 | F1 std |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f'| {summary["baseline_name"]} | '
        f'{baseline["roc_auc"]["mean"]:.6f} | '
        f'{baseline["fp_ap"]["mean"]:.6f} | '
        f'{baseline["iou_mae"]["mean"]:.6f} | '
        f'{baseline["spearman"]["mean"]:.6f} | '
        f'{baseline["filtered_f1"]["mean"]:.6f} | '
        f'{baseline["filtered_f1"]["sample_std"]:.6f} |',
        f'| vertical_mlp | {model["roc_auc"]["mean"]:.6f} | '
        f'{model["fp_ap"]["mean"]:.6f} | '
        f'{model["iou_mae"]["mean"]:.6f} | '
        f'{model["spearman"]["mean"]:.6f} | '
        f'{model["filtered_f1"]["mean"]:.6f} | '
        f'{model["filtered_f1"]["sample_std"]:.6f} |', '',
        '## Delta: Vertical-MLP minus fixed baseline', '',
        f'- Filtered F1: {delta["filtered_f1"]:+.6f}',
        f'- IoU MAE reduction: {delta["iou_mae_reduction"]:+.6f}',
        f'- FP AP: {delta["fp_ap"]:+.6f}',
        f'- ROC-AUC: {delta["roc_auc"]:+.6f}',
        f'- Spearman: {delta["spearman"]:+.6f}', '',
        '## Gate', '',
    ]
    lines.extend(
        f'- {name}: **{passed}**'
        for name, passed in summary['gate'].items())
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the E4 vertical-token MLP quality scorer.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import yaml
    with open(args.config, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    dataset = load_vertical_quality_dataset(
        settings['data_root'], settings['manifest_path'])
    if dataset['vertical_tokens'].shape[1] != int(
            settings['expected_num_layers']):
        raise ValueError(
            f'Expected {settings["expected_num_layers"]} vertical layers, '
            f'got {dataset["vertical_tokens"].shape[1]}.')
    with open(settings['baseline_summary_path'], encoding='utf-8') as file:
        baseline_summary = json.load(file)

    train_mask = dataset['split'] == 'train'
    validation_mask = dataset['split'] == 'validation'
    median, scale = fit_token_standardizer(
        dataset['vertical_tokens'][train_mask],
        dataset['layer_valid_mask'][train_mask])
    tokens = standardize_tokens(
        dataset['vertical_tokens'], dataset['layer_valid_mask'], median, scale)
    train_tokens = tokens[train_mask]
    validation_tokens = tokens[validation_mask]
    train_layer_mask = dataset['layer_valid_mask'][train_mask]
    validation_layer_mask = dataset['layer_valid_mask'][validation_mask]
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
        result = train_vertical_mlp(
            train_tokens, train_layer_mask, train_true, train_iou,
            train_class, train_valid,
            validation_tokens, validation_layer_mask, validation_true,
            validation_iou, validation_class, validation_valid,
            settings['model'], settings['training'], int(seed))
        metrics, score = BASELINES.evaluate_predictions(
            'vertical_mlp', seed, validation_true, validation_iou,
            validation_class, validation_valid,
            result['validity_probability'], result['predicted_iou'],
            settings['threshold_selection'])
        metrics['best_epoch'] = result['best_epoch']
        metrics['device'] = result['device']
        rows.append(metrics)
        torch.save({
            'state_dict': result['state_dict'],
            'token_feature_names': dataset['token_feature_names'],
            'token_median': median,
            'token_scale': scale,
            'seed': seed,
            'best_epoch': result['best_epoch'],
            'model_config': settings['model'],
            'training_config': settings['training'],
        }, checkpoint_dir / f'vertical_mlp_seed{seed}.pth')

        prediction_rows.extend({
            'model': 'vertical_mlp', 'seed': seed,
            'source_plot': plot, 'instance_id': int(instance_id),
            'score': float(value), 'target_iou': float(iou),
            'target_true': bool(true),
            'classification_valid': bool(class_valid),
        } for plot, instance_id, value, iou, true, class_valid in zip(
            dataset['source_plot'][validation_mask],
            dataset['instance_id'][validation_mask], score,
            validation_iou, validation_true, validation_class))

    summary = summarize_vertical(rows, baseline_summary, settings, dataset)
    with (output_dir / 'per_seed_metrics.csv').open(
            'w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
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
        raise RuntimeError('E4 Vertical-MLP gate failed; do not run E5.')
    print('PASS: proceed to E5 Vertical-Attention.', flush=True)


if __name__ == '__main__':
    main()
