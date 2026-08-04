import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERTICAL = load_module(
    'vertical_quality_for_fusion',
    SCRIPT_DIR / 'train_vertical_instance_quality.py')
BASELINES = VERTICAL.BASELINES
E10 = load_module(
    'quality_evidence_for_fusion',
    SCRIPT_DIR.parent / 'diagnostics' /
    'evaluate_quality_statistical_evidence.py')


def build_quality_model(global_dim, token_dim, model_type, model_config):
    import torch
    from torch import nn

    dropout = float(model_config['dropout'])

    def mlp(input_dim, hidden_dims):
        layers = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        return nn.Sequential(*layers), current_dim

    class GlobalWide(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, output_dim = mlp(
                global_dim, model_config['global_wide_hidden_dims'])
            self.validity = nn.Linear(output_dim, 1)
            self.iou = nn.Linear(output_dim, 1)

        def forward(self, global_values, tokens, layer_mask):
            encoded = self.encoder(global_values)
            return (
                self.validity(encoded).squeeze(1),
                torch.sigmoid(self.iou(encoded).squeeze(1)),
            )

    class GlobalVerticalFusion(nn.Module):
        def __init__(self):
            super().__init__()
            token_hidden_dim = int(model_config['token_hidden_dim'])
            token_layers = []
            current_dim = int(token_dim)
            for _ in range(int(model_config['token_mlp_layers'])):
                token_layers.extend([
                    nn.Linear(current_dim, token_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                current_dim = token_hidden_dim
            self.token_encoder = nn.Sequential(*token_layers)
            self.global_encoder, global_output_dim = mlp(
                global_dim, [model_config['global_projection_dim']])
            fusion_input_dim = 2 * token_hidden_dim + global_output_dim
            self.fusion_encoder, output_dim = mlp(
                fusion_input_dim, model_config['fusion_hidden_dims'])
            self.validity = nn.Linear(output_dim, 1)
            self.iou = nn.Linear(output_dim, 1)

        def forward(self, global_values, tokens, layer_mask):
            layer_mask = layer_mask.bool()
            if tokens.ndim != 3 or layer_mask.shape != tokens.shape[:2]:
                raise ValueError('Fusion token and mask shapes differ.')
            if not torch.all(layer_mask.any(dim=1)):
                raise ValueError('Every fusion instance needs a valid layer.')
            encoded_tokens = self.token_encoder(tokens)
            mask = layer_mask.unsqueeze(-1)
            token_mean = (
                (encoded_tokens * mask).sum(dim=1) /
                mask.sum(dim=1).clamp_min(1))
            token_max = encoded_tokens.masked_fill(
                ~mask, -torch.inf).max(dim=1).values
            global_encoded = self.global_encoder(global_values)
            fused = self.fusion_encoder(torch.cat(
                [global_encoded, token_mean, token_max], dim=1))
            return (
                self.validity(fused).squeeze(1),
                torch.sigmoid(self.iou(fused).squeeze(1)),
            )

    if model_type == 'global_wide':
        return GlobalWide()
    if model_type == 'global_vertical_fusion':
        return GlobalVerticalFusion()
    raise ValueError(f'Unknown quality model type: {model_type}')


def train_quality_model(
        train_global, train_tokens, train_layer_mask,
        train_true, train_iou, train_class_mask, train_valid_mask,
        validation_global, validation_tokens, validation_layer_mask,
        validation_true, validation_iou,
        validation_class_mask, validation_valid_mask,
        model_type, model_config, training_config, seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    BASELINES._set_seed(int(seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_quality_model(
        train_global.shape[1], train_tokens.shape[2],
        model_type, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']))
    dataset = TensorDataset(
        torch.from_numpy(train_global),
        torch.from_numpy(train_tokens),
        torch.from_numpy(train_layer_mask),
        torch.from_numpy(train_true.astype(np.float32)),
        torch.from_numpy(train_iou.astype(np.float32)),
        torch.from_numpy(train_class_mask),
        torch.from_numpy(train_valid_mask),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset, batch_size=int(training_config['batch_size']),
        shuffle=True, generator=generator, num_workers=0)
    class_targets = train_true[train_class_mask]
    class_weights = np.asarray([
        len(class_targets) / (2 * max(np.count_nonzero(~class_targets), 1)),
        len(class_targets) / (2 * max(np.count_nonzero(class_targets), 1)),
    ], dtype=np.float32)
    class_weights = torch.from_numpy(class_weights).to(device)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    smooth_l1 = nn.SmoothL1Loss(reduction='none')

    validation_tensors = (
        torch.from_numpy(validation_global).to(device),
        torch.from_numpy(validation_tokens).to(device),
        torch.from_numpy(validation_layer_mask).to(device),
    )
    best_state = None
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(training_config['epochs']) + 1):
        model.train()
        for batch in loader:
            (global_values, tokens, layer_mask, true, iou,
             class_mask, valid_mask) = [value.to(device) for value in batch]
            logits, predicted_iou = model(
                global_values, tokens, layer_mask)
            if torch.any(class_mask):
                class_loss = bce(logits[class_mask], true[class_mask])
                class_loss = (
                    class_loss * class_weights[
                        true[class_mask].long()]).mean()
            else:
                class_loss = logits.sum() * 0.0
            if torch.any(valid_mask):
                iou_loss = smooth_l1(
                    predicted_iou[valid_mask], iou[valid_mask]).mean()
            else:
                iou_loss = predicted_iou.sum() * 0.0
            loss = class_loss + float(
                training_config['iou_loss_weight']) * iou_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, predicted_iou = model(*validation_tensors)
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
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= int(training_config['patience']):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, predicted_iou = model(*validation_tensors)
    return {
        'validity_probability': torch.sigmoid(logits).cpu().numpy(),
        'predicted_iou': predicted_iou.cpu().numpy(),
        'best_epoch': int(best_epoch),
        'state_dict': best_state,
        'device': str(device),
        'parameter_count': int(sum(
            parameter.numel() for parameter in model.parameters())),
    }


def summarize_metrics(rows, metric_names):
    output = {}
    for model_name in sorted({row['model'] for row in rows}):
        group = [row for row in rows if row['model'] == model_name]
        output[model_name] = {'num_seeds': len(group)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group], dtype=float)
            output[model_name][metric] = {
                'mean': float(values.mean()),
                'sample_std': float(
                    values.std(ddof=1) if len(values) > 1 else 0.0),
            }
    return output


def selective_comparison(prediction_rows, ratios, seeds, plots):
    by_model = {}
    for model_name in ('global_wide', 'global_vertical_fusion'):
        records = {}
        for row in prediction_rows:
            if row['model'] != model_name:
                continue
            key = (
                int(row['seed']), str(row['source_plot']),
                int(row['instance_id']))
            records[key] = row
        by_model[model_name] = E10.build_group_curves(records, ratios)

    seed_rows = []
    for seed in seeds:
        group_keys = [(int(seed), str(plot)) for plot in plots]
        global_area = E10.curve_areas(E10.aggregate_group_curves(
            by_model['global_wide'], group_keys, ratios))
        fusion_area = E10.curve_areas(E10.aggregate_group_curves(
            by_model['global_vertical_fusion'], group_keys, ratios))
        seed_rows.append({
            'seed': int(seed),
            'commission_area_reduction_pp': float(100 * (
                global_area['commission_area'] -
                fusion_area['commission_area'])),
            'f1_area_gain_pp': float(100 * (
                fusion_area['f1_area'] - global_area['f1_area'])),
        })
    return seed_rows


def format_markdown(summary):
    lines = [
        '# E10b Global + Vertical 互补性验证',
        '',
        f"- 实例数：{summary['num_instances']}",
        f"- Global-Wide 参数：{summary['parameter_counts']['global_wide']:,}",
        f"- Fusion 参数：{summary['parameter_counts']['global_vertical_fusion']:,}",
        f"- 参数差比例：{100 * summary['parameter_difference_ratio']:.3f}%",
        '',
        '| Model | FP AP | IoU MAE | ROC-AUC | Spearman |',
        '|---|---:|---:|---:|---:|',
    ]
    for model_name in ('global_wide', 'global_vertical_fusion'):
        model = summary['models'][model_name]
        lines.append(
            f"| {model_name} | {model['fp_ap']['mean']:.6f} | "
            f"{model['iou_mae']['mean']:.6f} | "
            f"{model['roc_auc']['mean']:.6f} | "
            f"{model['spearman']['mean']:.6f} |")
    lines.extend([
        '',
        '## 固定 validation 选择性曲线差值',
        '',
        '| Seed | Commission area reduction | F1 area gain |',
        '|---:|---:|---:|',
    ])
    for row in summary['selective_seed_rows']:
        lines.append(
            f"| {row['seed']} | "
            f"{row['commission_area_reduction_pp']:+.3f} pp | "
            f"{row['f1_area_gain_pp']:+.3f} pp |")
    lines.extend([
        '',
        f"- Mean Commission-area reduction："
        f"{summary['selective_delta']['commission_area_reduction_pp']:+.3f} pp",
        f"- Mean F1-area gain："
        f"{summary['selective_delta']['f1_area_gain_pp']:+.3f} pp",
        '',
        '## Gate',
        '',
    ])
    for name, passed in summary['gate'].items():
        lines.append(f'- {name}: **{passed}**')
    lines.extend([''])
    if summary['gate']['passed']:
        lines.append('PASS：可以实现锁定 Fusion 的 Wytham score-only 外部验证。')
    else:
        lines.append('STOP：垂直 token 未证明对全局质量特征有互补增益。')
    return '\n'.join(lines) + '\n'
def run(config_path):
    import torch

    with open(config_path, encoding='utf-8') as file:
        import yaml
        settings = yaml.safe_load(file)
    dataset = VERTICAL.load_vertical_quality_dataset(
        settings['data_root'], settings['manifest_path'])
    train_mask = dataset['split'] == 'train'
    validation_mask = dataset['split'] == 'validation'

    global_median, global_scale = BASELINES.fit_standardizer(
        dataset['features'][train_mask])
    global_values = BASELINES.standardize(
        dataset['features'], global_median, global_scale)
    token_median, token_scale = VERTICAL.fit_token_standardizer(
        dataset['vertical_tokens'][train_mask],
        dataset['layer_valid_mask'][train_mask])
    tokens = VERTICAL.standardize_tokens(
        dataset['vertical_tokens'], dataset['layer_valid_mask'],
        token_median, token_scale)

    arrays = {
        'train_global': global_values[train_mask],
        'train_tokens': tokens[train_mask],
        'train_layer_mask': dataset['layer_valid_mask'][train_mask],
        'train_true': dataset['target_true'][train_mask],
        'train_iou': dataset['target_iou'][train_mask],
        'train_class_mask': dataset['classification_valid'][train_mask],
        'train_valid_mask': dataset['target_valid'][train_mask],
        'validation_global': global_values[validation_mask],
        'validation_tokens': tokens[validation_mask],
        'validation_layer_mask': dataset['layer_valid_mask'][validation_mask],
        'validation_true': dataset['target_true'][validation_mask],
        'validation_iou': dataset['target_iou'][validation_mask],
        'validation_class_mask': dataset['classification_valid'][
            validation_mask],
        'validation_valid_mask': dataset['target_valid'][validation_mask],
    }
    output_dir = Path(settings['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    prediction_rows = []
    parameter_counts = {}
    for model_type in ('global_wide', 'global_vertical_fusion'):
        for seed in settings['seeds']:
            print(
                f'===== START {model_type} seed {seed} =====',
                flush=True)
            result = train_quality_model(
                **arrays,
                model_type=model_type,
                model_config=settings['model'],
                training_config=settings['training'],
                seed=int(seed))
            metrics, scores = BASELINES.evaluate_predictions(
                model_type, seed,
                arrays['validation_true'], arrays['validation_iou'],
                arrays['validation_class_mask'],
                arrays['validation_valid_mask'],
                result['validity_probability'], result['predicted_iou'],
                settings['threshold_selection'])
            metrics.update({
                'best_epoch': result['best_epoch'],
                'device': result['device'],
                'parameter_count': result['parameter_count'],
            })
            metric_rows.append(metrics)
            print(
                f"DONE {model_type} seed {seed}: "
                f"epoch={result['best_epoch']}, "
                f"FP-AP={metrics['fp_ap']:.6f}, "
                f"IoU-MAE={metrics['iou_mae']:.6f}",
                flush=True)
            parameter_counts[model_type] = result['parameter_count']
            prediction_rows.extend({
                'model': model_type,
                'seed': int(seed),
                'source_plot': str(plot),
                'instance_id': int(instance_id),
                'score': float(score),
                'target_iou': float(iou),
                'target_true': bool(true),
                'classification_valid': bool(class_valid),
            } for plot, instance_id, score, iou, true, class_valid in zip(
                dataset['source_plot'][validation_mask],
                dataset['instance_id'][validation_mask], scores,
                arrays['validation_iou'], arrays['validation_true'],
                arrays['validation_class_mask']))
            torch.save({
                'state_dict': result['state_dict'],
                'model_type': model_type,
                'model_config': settings['model'],
                'global_feature_names': dataset['feature_names'],
                'global_median': global_median,
                'global_scale': global_scale,
                'token_feature_names': dataset['token_feature_names'],
                'token_median': token_median,
                'token_scale': token_scale,
                'seed': int(seed),
                'best_epoch': result['best_epoch'],
            }, checkpoint_dir / f'{model_type}_seed{seed}.pth')

    grid = settings['ratio_grid']
    ratios = np.round(np.arange(
        float(grid['start']),
        float(grid['stop']) + 0.5 * float(grid['step']),
        float(grid['step'])), 10)
    plots = sorted(set(dataset['source_plot'][validation_mask]))
    seed_rows = selective_comparison(
        prediction_rows, ratios,
        [int(seed) for seed in settings['seeds']], plots)
    commission_values = np.asarray([
        row['commission_area_reduction_pp'] for row in seed_rows])
    f1_values = np.asarray([row['f1_area_gain_pp'] for row in seed_rows])
    parameter_difference = abs(
        parameter_counts['global_wide'] -
        parameter_counts['global_vertical_fusion']) / max(
            parameter_counts.values())
    minimum_effect = float(settings['gate']['min_mean_effect_pp'])
    gate = {
        'parameter_count_matched': bool(
            parameter_difference <= float(
                settings['gate']['max_parameter_difference_ratio'])),
        'commission_mean_not_worse': bool(commission_values.mean() >= 0),
        'f1_mean_not_worse': bool(f1_values.mean() >= 0),
        'commission_seed_wins': bool(np.count_nonzero(
            commission_values > 0) >= int(
                settings['gate']['min_seed_wins'])),
        'f1_seed_wins': bool(np.count_nonzero(
            f1_values > 0) >= int(settings['gate']['min_seed_wins'])),
        'meaningful_effect_passed': bool(
            commission_values.mean() >= minimum_effect or
            f1_values.mean() >= minimum_effect),
    }
    gate['passed'] = bool(all(gate.values()))
    metric_names = ['roc_auc', 'fp_ap', 'iou_mae', 'spearman']
    summary = {
        'config': str(config_path),
        'num_instances': int(len(dataset['instance_id'])),
        'seeds': [int(seed) for seed in settings['seeds']],
        'validation_plots': [str(plot) for plot in plots],
        'parameter_counts': parameter_counts,
        'parameter_difference_ratio': float(parameter_difference),
        'models': summarize_metrics(metric_rows, metric_names),
        'selective_seed_rows': seed_rows,
        'selective_delta': {
            'commission_area_reduction_pp': float(
                commission_values.mean()),
            'f1_area_gain_pp': float(f1_values.mean()),
        },
        'gate': gate,
    }
    with (output_dir / 'per_seed_metrics.csv').open(
            'w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(
            key for row in metric_rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / 'validation_predictions.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8')
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'E10b fusion gate failed; stop the vertical-innovation route.')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Train parameter-matched global/fusion quality heads.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
