import argparse
import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np


E4_PATH = Path(__file__).with_name('train_vertical_instance_quality.py')
E4_SPEC = importlib.util.spec_from_file_location(
    'vertical_instance_quality_for_attention', E4_PATH)
E4 = importlib.util.module_from_spec(E4_SPEC)
E4_SPEC.loader.exec_module(E4)
BASELINES = E4.BASELINES


def sinusoidal_height_encoding(num_layers, hidden_dim):
    if num_layers <= 0 or hidden_dim <= 0:
        raise ValueError('Position-encoding dimensions must be positive.')
    positions = (
        (np.arange(num_layers, dtype=np.float32) + 0.5) / num_layers
    )[:, None]
    dimensions = np.arange(0, hidden_dim, 2, dtype=np.float32)
    scales = np.exp(-np.log(10000.0) * dimensions / hidden_dim)
    encoding = np.zeros((num_layers, hidden_dim), dtype=np.float32)
    encoding[:, 0::2] = np.sin(positions * scales)
    if hidden_dim > 1:
        encoding[:, 1::2] = np.cos(
            positions * scales[:encoding[:, 1::2].shape[1]])
    return encoding


def build_vertical_attention(input_dim, model_config):
    import torch
    from torch import nn

    hidden_dim = int(model_config['hidden_dim'])
    num_heads = int(model_config['num_heads'])
    num_layers = int(model_config['num_layers'])
    num_token_layers = int(model_config['token_mlp_layers'])
    num_vertical_layers = int(model_config['num_vertical_layers'])
    dropout = float(model_config['dropout'])
    if hidden_dim <= 0 or hidden_dim % num_heads != 0:
        raise ValueError('hidden_dim must be positive and divisible by num_heads.')
    if num_layers != 1:
        raise ValueError('E5 is preregistered to use exactly one Transformer layer.')

    class VerticalAttention(nn.Module):
        def __init__(self):
            super().__init__()
            token_layers = []
            current_dim = int(input_dim)
            for _ in range(num_token_layers):
                token_layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                current_dim = hidden_dim
            self.token_encoder = nn.Sequential(*token_layers)
            self.register_buffer(
                'height_encoding',
                torch.from_numpy(sinusoidal_height_encoding(
                    num_vertical_layers, hidden_dim)),
                persistent=True)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=int(model_config['feedforward_dim']),
                dropout=dropout,
                activation='gelu',
                batch_first=True)
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers)
            self.pool_score = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1))

            instance_layers = []
            current_dim = hidden_dim
            for instance_dim in model_config['instance_hidden_dims']:
                instance_dim = int(instance_dim)
                instance_layers.extend([
                    nn.Linear(current_dim, instance_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                current_dim = instance_dim
            self.instance_encoder = nn.Sequential(*instance_layers)
            self.validity = nn.Linear(current_dim, 1)
            self.iou = nn.Linear(current_dim, 1)

        def forward(self, tokens, layer_mask):
            layer_mask = layer_mask.bool()
            if tokens.ndim != 3 or layer_mask.shape != tokens.shape[:2]:
                raise ValueError('Token and layer-mask shapes are inconsistent.')
            if tokens.shape[1] != self.height_encoding.shape[0]:
                raise ValueError(
                    'Token layer count differs from fixed height encoding.')
            if not torch.all(layer_mask.any(dim=1)):
                raise ValueError('Every instance must have a valid layer.')
            encoded = self.token_encoder(tokens)
            encoded = encoded + self.height_encoding.unsqueeze(0)
            encoded = self.transformer(
                encoded, src_key_padding_mask=~layer_mask)
            scores = self.pool_score(encoded).squeeze(-1)
            scores = scores.masked_fill(~layer_mask, -torch.inf)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
            instance = self.instance_encoder(pooled)
            return (
                self.validity(instance).squeeze(1),
                torch.sigmoid(self.iou(instance).squeeze(1)),
            )

    return VerticalAttention()


def load_e4_metrics(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    output = {}
    for row in rows:
        if row.get('model') != 'vertical_mlp':
            continue
        seed = int(row['seed'])
        output[seed] = {
            key: float(row[key])
            for key in (
                'filtered_f1', 'filtered_completeness',
                'filtered_commission', 'fp_ap')}
    if not output:
        raise ValueError('No Vertical-MLP rows exist in E4 metrics.')
    return output


def aggregate_metrics(rows):
    metric_names = [
        'roc_auc', 'positive_ap', 'fp_ap', 'iou_mae', 'spearman',
        'filtered_f1', 'filtered_completeness', 'filtered_commission',
        'unfiltered_f1', 'inference_ms_per_instance']
    output = {'num_seeds': len(rows)}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        output[metric] = {
            'mean': float(values.mean()),
            'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return output


def benchmark_model(model, tokens, layer_mask, config):
    import torch

    device = next(model.parameters()).device
    token_tensor = torch.from_numpy(tokens).to(device)
    mask_tensor = torch.from_numpy(layer_mask).to(device)
    model.eval()
    warmup = int(config['warmup_iterations'])
    measured = int(config['measured_iterations'])
    with torch.no_grad():
        for _ in range(warmup):
            model(token_tensor, mask_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(measured):
            model(token_tensor, mask_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / (measured * len(tokens))


def summarize_attention(
        rows, e4_summary, e4_metrics, settings, dataset, parameter_count):
    if not e4_summary.get('gate', {}).get('passed', False):
        raise ValueError('The fixed E4 summary did not pass its gate.')
    e4_model = e4_summary['vertical_mlp']
    attention = aggregate_metrics(rows)
    delta = {
        'filtered_f1': (
            attention['filtered_f1']['mean'] -
            e4_model['filtered_f1']['mean']),
        'commission_reduction': (
            e4_model['filtered_commission']['mean'] -
            attention['filtered_commission']['mean']),
        'completeness_drop': (
            e4_model['filtered_completeness']['mean'] -
            attention['filtered_completeness']['mean']),
        'fp_ap': attention['fp_ap']['mean'] - e4_model['fp_ap']['mean'],
        'iou_mae_reduction': (
            e4_model['iou_mae']['mean'] - attention['iou_mae']['mean']),
        'roc_auc': attention['roc_auc']['mean'] - e4_model['roc_auc']['mean'],
        'spearman': attention['spearman']['mean'] - e4_model['spearman']['mean'],
    }
    tolerance = float(settings['gate']['paired_win_tolerance'])
    paired = {}
    for row in rows:
        seed = int(row['seed'])
        if seed not in e4_metrics:
            raise ValueError(f'E4 metrics miss paired seed {seed}.')
        difference = row['filtered_f1'] - e4_metrics[seed]['filtered_f1']
        paired[str(seed)] = {
            'e4_f1': e4_metrics[seed]['filtered_f1'],
            'e5_f1': row['filtered_f1'],
            'difference': difference,
            'won': difference > tolerance,
        }
    paired_wins = sum(item['won'] for item in paired.values())
    gate_config = settings['gate']
    gate = {
        'e4_baseline_passed': True,
        'mean_f1_gain_passed': (
            delta['filtered_f1'] >= float(
                gate_config['min_f1_improvement'])),
        'commission_tradeoff_passed': (
            delta['commission_reduction'] >= float(
                gate_config['min_commission_reduction']) and
            delta['completeness_drop'] <= float(
                gate_config['max_completeness_drop'])),
        'fp_ap_not_decreased': (
            delta['fp_ap'] >= -float(gate_config['fp_ap_tolerance'])),
        'paired_seed_wins_passed': (
            paired_wins >= int(gate_config['min_paired_seed_wins'])),
    }
    gate['effect_size_passed'] = (
        gate['mean_f1_gain_passed'] or gate['commission_tradeoff_passed'])
    gate['runtime_gate_deferred_to_e6'] = True
    gate['passed'] = (
        gate['e4_baseline_passed'] and
        gate['effect_size_passed'] and
        gate['fp_ap_not_decreased'] and
        gate['paired_seed_wins_passed'])
    return {
        'num_instances': int(len(dataset['instance_id'])),
        'num_layers': int(dataset['vertical_tokens'].shape[1]),
        'token_dim': int(dataset['vertical_tokens'].shape[2]),
        'seeds': settings['seeds'],
        'parameter_count': int(parameter_count),
        'e4_vertical_mlp': e4_model,
        'vertical_attention': attention,
        'deltas': delta,
        'paired_seeds': paired,
        'paired_seed_wins': paired_wins,
        'gate': gate,
    }


def format_markdown(summary):
    e4 = summary['e4_vertical_mlp']
    e5 = summary['vertical_attention']
    delta = summary['deltas']
    lines = [
        '# E5 Vertical-Attention', '',
        f'- Instances: {summary["num_instances"]}',
        f'- Vertical tokens: {summary["num_layers"]} x {summary["token_dim"]}',
        f'- Trainable parameters: {summary["parameter_count"]:,}',
        f'- Paired seed wins: {summary["paired_seed_wins"]}/3', '',
        '| Model | ROC-AUC | FP AP | IoU MAE | Completeness | Commission | Filtered F1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f'| vertical_mlp | {e4["roc_auc"]["mean"]:.6f} | '
        f'{e4["fp_ap"]["mean"]:.6f} | {e4["iou_mae"]["mean"]:.6f} | '
        f'{e4["filtered_completeness"]["mean"]:.6f} | '
        f'{e4["filtered_commission"]["mean"]:.6f} | '
        f'{e4["filtered_f1"]["mean"]:.6f} |',
        f'| vertical_attention | {e5["roc_auc"]["mean"]:.6f} | '
        f'{e5["fp_ap"]["mean"]:.6f} | {e5["iou_mae"]["mean"]:.6f} | '
        f'{e5["filtered_completeness"]["mean"]:.6f} | '
        f'{e5["filtered_commission"]["mean"]:.6f} | '
        f'{e5["filtered_f1"]["mean"]:.6f} |', '',
        '## Delta: Attention minus Vertical-MLP', '',
        f'- Filtered F1: {delta["filtered_f1"]:+.6f}',
        f'- Commission reduction: {delta["commission_reduction"]:+.6f}',
        f'- Completeness drop: {delta["completeness_drop"]:+.6f}',
        f'- FP AP: {delta["fp_ap"]:+.6f}',
        f'- IoU MAE reduction: {delta["iou_mae_reduction"]:+.6f}',
        f'- Validation inference: {e5["inference_ms_per_instance"]["mean"]:.6f} ms/instance', '',
        '## Paired seeds', '',
    ]
    for seed, result in summary['paired_seeds'].items():
        lines.append(
            f'- seed {seed}: {result["e4_f1"]:.6f} -> '
            f'{result["e5_f1"]:.6f}, won={result["won"]}')
    lines.extend(['', '## Gate', ''])
    lines.extend(
        f'- {name}: **{passed}**'
        for name, passed in summary['gate'].items())
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the E5 vertical-attention quality scorer.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def validate_vertical_layer_count(dataset, model_config):
    expected = int(model_config['num_vertical_layers'])
    actual = int(dataset['vertical_tokens'].shape[1])
    if actual != expected:
        raise ValueError(
            f'E5 dataset has {actual} vertical layers, expected {expected}.')
    return expected


def main():
    args = parse_args()

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import yaml

    with open(args.config, encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    dataset = E4.load_vertical_quality_dataset(
        settings['data_root'], settings['manifest_path'])
    validate_vertical_layer_count(dataset, settings['model'])
    with open(settings['e4_summary_path'], encoding='utf-8') as file:
        e4_summary = json.load(file)
    e4_metrics = load_e4_metrics(settings['e4_metrics_path'])
    expected_seeds = {int(seed) for seed in settings['seeds']}
    if set(e4_metrics) != expected_seeds:
        raise ValueError('E4 paired seeds differ from the E5 config.')

    train_mask = dataset['split'] == 'train'
    validation_mask = dataset['split'] == 'validation'
    median, scale = E4.fit_token_standardizer(
        dataset['vertical_tokens'][train_mask],
        dataset['layer_valid_mask'][train_mask])
    tokens = E4.standardize_tokens(
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
    parameter_count = None
    for seed in settings['seeds']:
        result = E4.train_vertical_mlp(
            train_tokens, train_layer_mask, train_true, train_iou,
            train_class, train_valid,
            validation_tokens, validation_layer_mask, validation_true,
            validation_iou, validation_class, validation_valid,
            settings['model'], settings['training'], int(seed),
            model_builder=build_vertical_attention)
        model = build_vertical_attention(
            train_tokens.shape[-1], settings['model'])
        model.load_state_dict(result['state_dict'])
        device = torch.device(result['device'])
        model = model.to(device)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad)
        inference_ms = benchmark_model(
            model, validation_tokens, validation_layer_mask,
            settings['benchmark'])
        metrics, score = BASELINES.evaluate_predictions(
            'vertical_attention', seed, validation_true, validation_iou,
            validation_class, validation_valid,
            result['validity_probability'], result['predicted_iou'],
            settings['threshold_selection'])
        metrics['best_epoch'] = result['best_epoch']
        metrics['device'] = result['device']
        metrics['inference_ms_per_instance'] = inference_ms
        rows.append(metrics)
        torch.save({
            'state_dict': result['state_dict'],
            'token_feature_names': dataset['token_feature_names'],
            'token_median': median, 'token_scale': scale,
            'seed': seed, 'best_epoch': result['best_epoch'],
            'model_config': settings['model'],
            'training_config': settings['training'],
            'inference_ms_per_instance': inference_ms,
        }, checkpoint_dir / f'vertical_attention_seed{seed}.pth')
        prediction_rows.extend({
            'model': 'vertical_attention', 'seed': seed,
            'source_plot': plot, 'instance_id': int(instance_id),
            'score': float(value), 'target_iou': float(iou),
            'target_true': bool(true),
            'classification_valid': bool(class_valid),
        } for plot, instance_id, value, iou, true, class_valid in zip(
            dataset['source_plot'][validation_mask],
            dataset['instance_id'][validation_mask], score,
            validation_iou, validation_true, validation_class))

    summary = summarize_attention(
        rows, e4_summary, e4_metrics, settings, dataset, parameter_count)
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
        raise RuntimeError(
            'E5 Attention gate failed; keep Vertical-MLP and do not use '
            'Attention in the paper method.')
    print('PASS: proceed to E6 quality-filter integration.', flush=True)


if __name__ == '__main__':
    main()
