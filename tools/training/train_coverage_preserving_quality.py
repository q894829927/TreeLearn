"""Q2 parameter-matched coverage-preserving instance-quality controls.

The experiment compares the original quality/IoU supervision with coverage-CVaR
supervision while keeping the global features, encoder, parameter count,
training split, rejection budget, optimizer and random seeds fixed.  Wytham is
explicitly forbidden and no TreeLearn prediction is changed in this stage.
"""

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / 'tools' / 'training' / (
    'train_instance_quality_baselines.py')
COVERAGE_PATH = ROOT / 'tree_learn' / 'util' / 'coverage_quality.py'


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location(
        'q2_instance_quality_baselines', BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_coverage_module():
    spec = importlib.util.spec_from_file_location(
        'q2_coverage_quality', COVERAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COVERAGE = _load_coverage_module()
build_two_head_instance_mlp = COVERAGE.build_two_head_instance_mlp
count_trainable_parameters = COVERAGE.count_trainable_parameters
coverage_rejection_risk = COVERAGE.coverage_rejection_risk


def _parse_bool(value):
    lowered = str(value).strip().lower()
    if lowered not in {'true', 'false'}:
        raise ValueError(f'Expected boolean value, got {value!r}.')
    return lowered == 'true'


def load_coverage_risk_labels(label_path, dataset):
    """Align the Q1 CSV labels to the E2 feature dataset."""
    path = Path(label_path)
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('Q1 coverage-CVaR label manifest is empty.')
    if any('wytham' in str(row).lower() for row in rows):
        raise ValueError('Q2 must not read Wytham labels.')

    required = {
        'source_plot', 'split', 'instance_id', 'safe_reject',
        'coverage_critical', 'counted_negative', 'redundant_positive',
        'protected_unknown'}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f'Q1 labels lack fields: {sorted(missing)}')
    mapping = {}
    for row in rows:
        key = (row['source_plot'], int(row['instance_id']))
        if key in mapping:
            raise ValueError(f'Duplicate Q1 label key: {key}')
        mapping[key] = {
            name: _parse_bool(row[name]) for name in (
                'safe_reject', 'coverage_critical', 'counted_negative',
                'redundant_positive', 'protected_unknown')}
        mapping[key]['split'] = row['split']

    keys = list(zip(
        dataset['source_plot'].astype(str),
        dataset['instance_id'].astype(np.int64)))
    if set(keys) != set(mapping):
        missing_dataset = sorted(set(mapping) - set(keys))[:5]
        missing_labels = sorted(set(keys) - set(mapping))[:5]
        raise ValueError(
            'Q1 labels and feature instances differ: '
            f'label_only={missing_dataset}, feature_only={missing_labels}')

    output = {
        name: np.asarray([mapping[key][name] for key in keys], dtype=bool)
        for name in (
            'safe_reject', 'coverage_critical', 'counted_negative',
            'redundant_positive', 'protected_unknown')}
    label_splits = np.asarray([mapping[key]['split'] for key in keys])
    if not np.array_equal(label_splits, dataset['split'].astype(str)):
        raise ValueError('Q1 label splits do not align with feature splits.')
    class_valid = np.asarray(dataset['classification_valid'], dtype=bool)
    if np.any(output['safe_reject'] & output['coverage_critical']):
        raise ValueError('Q1 safe and critical targets overlap.')
    if not np.array_equal(
            output['safe_reject'] | output['coverage_critical'], class_valid):
        raise ValueError('Q1 eligible labels do not reproduce class validity.')
    if not np.array_equal(output['protected_unknown'], ~class_valid):
        raise ValueError('Q1 protected-unknown mask does not reproduce labels.')
    return output


def detection_proxy(tp, fp, fn=0):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'completeness': float(tp / max(tp + fn, 1)),
        'commission': float(fp / max(tp + fp, 1)),
        'f1': float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def fixed_ratio_metrics(
        rejection_risk, instance_ids, source_plots, safe_reject,
        coverage_critical, protected_unknown, reject_ratio):
    """Evaluate deterministic per-forest top-risk rejection."""
    risk = np.asarray(rejection_risk, dtype=np.float64).reshape(-1)
    ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    plots = np.asarray(source_plots).astype(str).reshape(-1)
    safe = np.asarray(safe_reject, dtype=bool).reshape(-1)
    critical = np.asarray(coverage_critical, dtype=bool).reshape(-1)
    unknown = np.asarray(protected_unknown, dtype=bool).reshape(-1)
    count = len(risk)
    if not all(len(values) == count for values in (
            ids, plots, safe, critical, unknown)):
        raise ValueError('Q2 metric arrays are not aligned.')
    if not np.isfinite(risk).all():
        raise ValueError('Q2 rejection risk contains NaN or infinity.')
    if not 0 <= float(reject_ratio) < 1:
        raise ValueError('reject_ratio must be within [0, 1).')
    overlaps = (safe & critical) | (safe & unknown) | (critical & unknown)
    if np.any(overlaps) or not np.all(safe | critical | unknown):
        raise ValueError('Q2 target masks are invalid.')

    rejected = np.zeros(count, dtype=bool)
    per_plot = []
    for plot in sorted(np.unique(plots)):
        indices = np.flatnonzero(plots == plot)
        budget = int(np.floor(float(reject_ratio) * len(indices)))
        order = np.lexsort((ids[indices], -risk[indices]))
        chosen = indices[order[:budget]]
        rejected[chosen] = True
        base = detection_proxy(
            int(critical[indices].sum()), int(safe[indices].sum()))
        kept_critical = int((critical[indices] & ~rejected[indices]).sum())
        kept_safe = int((safe[indices] & ~rejected[indices]).sum())
        filtered = detection_proxy(
            kept_critical, kept_safe,
            int((critical[indices] & rejected[indices]).sum()))
        per_plot.append({
            'source_plot': plot,
            'num_instances': int(len(indices)),
            'reject_budget': budget,
            'num_rejected': int(rejected[indices].sum()),
            'safe_rejected': int((safe[indices] & rejected[indices]).sum()),
            'critical_rejected': int(
                (critical[indices] & rejected[indices]).sum()),
            'unknown_rejected': int(
                (unknown[indices] & rejected[indices]).sum()),
            'baseline': base,
            'filtered': filtered,
        })

    baseline = detection_proxy(int(critical.sum()), int(safe.sum()))
    kept_critical = int((critical & ~rejected).sum())
    kept_safe = int((safe & ~rejected).sum())
    filtered = detection_proxy(
        kept_critical, kept_safe, int((critical & rejected).sum()))
    labeled_rejected = int(((safe | critical) & rejected).sum())
    return {
        'baseline': baseline,
        'filtered': filtered,
        'num_rejected': int(rejected.sum()),
        'safe_rejected': int((safe & rejected).sum()),
        'critical_rejected': int((critical & rejected).sum()),
        'unknown_rejected': int((unknown & rejected).sum()),
        'reject_precision': float(
            (safe & rejected).sum() / max(labeled_rejected, 1)),
        'f1_gain_pp': float(100 * (filtered['f1'] - baseline['f1'])),
        'commission_reduction_pp': float(100 * (
            baseline['commission'] - filtered['commission'])),
        'completeness_drop_pp': float(100 * (
            baseline['completeness'] - filtered['completeness'])),
        'per_plot': per_plot,
        'rejected_mask': rejected,
    }


def _balanced_weights(targets):
    targets = np.asarray(targets, dtype=bool)
    count = len(targets)
    negative = max(np.count_nonzero(~targets), 1)
    positive = max(np.count_nonzero(targets), 1)
    return np.asarray([
        count / (2.0 * negative), count / (2.0 * positive),
    ], dtype=np.float32)


def _set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _classification_metrics(targets, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score

    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        'roc_auc': float(roc_auc_score(targets, scores)),
        'average_precision': float(average_precision_score(targets, scores)),
    }


def _risk_from_numpy(first, second, mode):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_probability = 1.0 / (1.0 + np.exp(-first))
    second_probability = 1.0 / (1.0 + np.exp(-second))
    if mode == 'quality_iou_control':
        return 1.0 - first_probability * second_probability
    if mode == 'safe_iou_control':
        return first_probability
    if mode == 'coverage_cvar':
        return first_probability
    raise ValueError(f'Unknown Q2 mode: {mode!r}')


def _training_loss(
        mode, first, second, true, target_iou, safe, critical,
        class_mask, valid_mask, weights, settings):
    import torch
    from torch import nn

    bce = nn.BCEWithLogitsLoss(reduction='none')
    if mode not in {
            'quality_iou_control', 'safe_iou_control', 'coverage_cvar'}:
        raise ValueError(f'Unknown Q2 mode: {mode!r}')
    if torch.any(class_mask):
        target = true if mode == 'quality_iou_control' else safe
        weight_name = 'true' if mode == 'quality_iou_control' else 'safe'
        values = bce(first[class_mask], target[class_mask])
        class_weights = weights[weight_name][target[class_mask].long()]
        classification_loss = (values * class_weights).mean()
    else:
        classification_loss = first.sum() * 0.0
    if torch.any(valid_mask):
        predicted_iou = torch.sigmoid(second[valid_mask])
        regression_loss = nn.functional.smooth_l1_loss(
            predicted_iou, target_iou[valid_mask])
    else:
        regression_loss = second.sum() * 0.0
    loss = classification_loss + float(
        settings['control_iou_loss_weight']) * regression_loss
    if mode == 'coverage_cvar':
        critical_mask = class_mask & critical.bool()
        if torch.any(critical_mask):
            critical_risk = torch.sigmoid(first[critical_mask])
            tail_count = max(1, int(np.ceil(
                float(settings['coverage_tail_fraction']) *
                critical_risk.numel())))
            tail_risk = torch.topk(
                critical_risk, k=tail_count, largest=True).values.mean()
            loss = loss + float(
                settings['coverage_cvar_weight']) * tail_risk
    return loss


def _selection_key(metrics, max_completeness_drop_pp):
    drop = float(metrics['completeness_drop_pp'])
    if drop <= float(max_completeness_drop_pp) + 1e-12:
        return (
            0,
            -float(metrics['filtered']['f1']),
            float(metrics['filtered']['commission']),
            int(metrics['critical_rejected']),
        )
    return (
        1,
        drop,
        -float(metrics['filtered']['f1']),
        int(metrics['critical_rejected']),
    )


def train_model(
        mode, train, validation, model_settings, reject_ratio, seed):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_two_head_instance_mlp(
        train['x'].shape[1], model_settings['hidden_dims'],
        model_settings['dropout']).to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_settings['learning_rate']),
        weight_decay=float(model_settings['weight_decay']))

    tensors = TensorDataset(*(
        torch.from_numpy(train[name]) for name in (
            'x', 'true_float', 'iou', 'safe_float', 'critical_float',
            'class_mask', 'valid_mask')))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        tensors, batch_size=int(model_settings['batch_size']), shuffle=True,
        generator=generator, num_workers=0)
    weights = {
        'true': torch.from_numpy(_balanced_weights(
            train['true'][train['class_mask']])).to(device),
        'safe': torch.from_numpy(_balanced_weights(
            train['safe'][train['class_mask']])).to(device),
        'critical': torch.from_numpy(_balanced_weights(
            train['critical'][train['class_mask']])).to(device),
    }
    validation_tensor = torch.from_numpy(validation['x']).to(device)

    best_state = None
    best_metrics = None
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(model_settings['epochs']) + 1):
        model.train()
        for batch in loader:
            values, true, target_iou, safe, critical, class_mask, valid_mask = (
                item.to(device) for item in batch)
            first, second = model(values)
            loss = _training_loss(
                mode, first, second, true, target_iou, safe, critical,
                class_mask, valid_mask, weights, model_settings)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            first, second = model(validation_tensor)
            risk = coverage_rejection_risk(
                first, second, mode).cpu().numpy()
        metrics = fixed_ratio_metrics(
            risk,
            validation['instance_id'],
            validation['source_plot'],
            validation['safe'],
            validation['critical'],
            validation['unknown'],
            reject_ratio)
        key = _selection_key(
            metrics, model_settings['max_completeness_drop_pp'])
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % int(model_settings['log_interval']) == 0:
            print(
                f'[{mode} seed {seed}] epoch {epoch:03d}: '
                f'F1={100 * metrics["filtered"]["f1"]:.3f}%, '
                f'completeness='
                f'{100 * metrics["filtered"]["completeness"]:.3f}%, '
                f'commission={100 * metrics["filtered"]["commission"]:.3f}%',
                flush=True)
        if stale_epochs >= int(model_settings['patience']):
            break

    if best_state is None:
        raise RuntimeError(f'Q2 {mode} produced no checkpoint.')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        first, second = model(validation_tensor)
        first_np = first.cpu().numpy()
        second_np = second.cpu().numpy()
        risk = coverage_rejection_risk(
            first, second, mode).cpu().numpy()
    metrics = fixed_ratio_metrics(
        risk,
        validation['instance_id'],
        validation['source_plot'],
        validation['safe'],
        validation['critical'],
        validation['unknown'],
        reject_ratio)
    if metrics['filtered'] != best_metrics['filtered']:
        raise AssertionError('Restored Q2 checkpoint metrics changed.')

    class_mask = validation['class_mask']
    safe_metrics = _classification_metrics(
        validation['safe'][class_mask], risk[class_mask])
    critical_metrics = _classification_metrics(
        validation['critical'][class_mask], 1.0 - risk[class_mask])
    metrics.update({
        'safe_roc_auc': safe_metrics['roc_auc'],
        'safe_average_precision': safe_metrics['average_precision'],
        'critical_roc_auc': critical_metrics['roc_auc'],
        'critical_average_precision': critical_metrics['average_precision'],
    })
    return {
        'mode': mode,
        'seed': int(seed),
        'best_epoch': int(best_epoch),
        'device': str(device),
        'parameter_count': parameter_count,
        'state_dict': best_state,
        'first_output': first_np,
        'second_output': second_np,
        'rejection_risk': risk,
        'metrics': metrics,
    }


def _metric_row(result):
    metrics = result['metrics']
    return {
        'model': result['mode'],
        'seed': result['seed'],
        'best_epoch': result['best_epoch'],
        'device': result['device'],
        'parameter_count': result['parameter_count'],
        'safe_roc_auc': metrics['safe_roc_auc'],
        'safe_average_precision': metrics['safe_average_precision'],
        'critical_roc_auc': metrics['critical_roc_auc'],
        'critical_average_precision': metrics[
            'critical_average_precision'],
        'baseline_f1': metrics['baseline']['f1'],
        'filtered_f1': metrics['filtered']['f1'],
        'filtered_completeness': metrics['filtered']['completeness'],
        'filtered_commission': metrics['filtered']['commission'],
        'f1_gain_pp': metrics['f1_gain_pp'],
        'commission_reduction_pp': metrics['commission_reduction_pp'],
        'completeness_drop_pp': metrics['completeness_drop_pp'],
        'reject_precision': metrics['reject_precision'],
        'num_rejected': metrics['num_rejected'],
        'safe_rejected': metrics['safe_rejected'],
        'critical_rejected': metrics['critical_rejected'],
        'unknown_rejected': metrics['unknown_rejected'],
    }


def _summary_stats(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        'min': float(values.min()),
        'max': float(values.max()),
    }


def summarize(rows, per_plot_rows, settings, q1_summary, parameter_counts):
    model_names = (
        'quality_iou_control', 'safe_iou_control', 'coverage_cvar')
    metric_names = (
        'safe_roc_auc', 'safe_average_precision', 'critical_roc_auc',
        'critical_average_precision', 'baseline_f1', 'filtered_f1',
        'filtered_completeness', 'filtered_commission', 'f1_gain_pp',
        'commission_reduction_pp', 'completeness_drop_pp',
        'reject_precision')
    models = {}
    for name in model_names:
        model_rows = [row for row in rows if row['model'] == name]
        models[name] = {
            metric: _summary_stats(model_rows, metric)
            for metric in metric_names}
        models[name]['num_seeds'] = len(model_rows)
        models[name]['parameter_count'] = int(
            model_rows[0]['parameter_count'])

    control = models['safe_iou_control']
    coverage = models['coverage_cvar']
    seeds = sorted({int(row['seed']) for row in rows})
    seed_wins = 0
    for seed in seeds:
        control_row = next(
            row for row in rows
            if row['model'] == 'safe_iou_control' and
            int(row['seed']) == seed)
        coverage_row = next(
            row for row in rows
            if row['model'] == 'coverage_cvar' and int(row['seed']) == seed)
        seed_wins += coverage_row['filtered_f1'] > control_row['filtered_f1']

    plot_names = sorted({row['source_plot'] for row in per_plot_rows})
    plot_deltas = {}
    for plot in plot_names:
        values = {}
        for model in model_names:
            selected = [
                row for row in per_plot_rows
                if row['model'] == model and row['source_plot'] == plot]
            values[model] = float(np.mean([
                row['filtered_f1'] for row in selected]))
        plot_deltas[plot] = float(100 * (
            values['coverage_cvar'] - values['safe_iou_control']))
    plot_wins = sum(value > 0 for value in plot_deltas.values())

    delta = {
        'filtered_f1_gain_pp': float(100 * (
            coverage['filtered_f1']['mean'] -
            control['filtered_f1']['mean'])),
        'completeness_change_pp': float(100 * (
            coverage['filtered_completeness']['mean'] -
            control['filtered_completeness']['mean'])),
        'commission_reduction_pp': float(100 * (
            control['filtered_commission']['mean'] -
            coverage['filtered_commission']['mean'])),
        'safe_ap_gain': float(
            coverage['safe_average_precision']['mean'] -
            control['safe_average_precision']['mean']),
        'seed_wins': int(seed_wins),
        'plot_wins': int(plot_wins),
        'per_plot_f1_delta_pp': plot_deltas,
    }
    gate_cfg = settings['gate']
    gate = {
        'q1_gate_passed': bool(q1_summary['gate']['passed']),
        'expected_seeds_present': (
            seeds == sorted(int(seed) for seed in settings['seeds'])),
        'parameter_count_matched': (
            len(set(int(value) for value in parameter_counts.values())) == 1),
        'coverage_completeness_preserved': (
            coverage['completeness_drop_pp']['max'] <=
            float(gate_cfg['max_coverage_completeness_drop_pp'])),
        'coverage_f1_gain_over_unfiltered_passed': (
            coverage['f1_gain_pp']['mean'] >=
            float(gate_cfg['min_coverage_f1_gain_over_unfiltered_pp'])),
        'coverage_commission_reduction_passed': (
            coverage['commission_reduction_pp']['mean'] >=
            float(gate_cfg['min_coverage_commission_reduction_pp'])),
        'coverage_gain_over_control_passed': (
            delta['filtered_f1_gain_pp'] >=
            float(gate_cfg['min_coverage_f1_gain_over_control_pp'])),
        'coverage_completeness_vs_control_passed': (
            delta['completeness_change_pp'] >=
            -float(gate_cfg['max_coverage_completeness_loss_vs_control_pp'])),
        'seed_consistency_passed': (
            seed_wins >= int(gate_cfg['min_seed_wins'])),
        'plot_consistency_passed': (
            plot_wins >= int(gate_cfg['min_validation_plot_wins'])),
        'coverage_seed_stability_passed': (
            100 * coverage['filtered_f1']['sample_std'] <=
            float(gate_cfg['max_coverage_f1_sample_std_pp'])),
    }
    gate['passed'] = bool(all(gate.values()))
    return {
        'data_role': 'fixed_train_validation_only',
        'reject_ratio': float(settings['reject_ratio']),
        'seeds': seeds,
        'models': models,
        'delta_coverage_minus_safe_iou': delta,
        'parameter_counts': {
            key: int(value) for key, value in parameter_counts.items()},
        'q1_counts': {
            split: {
                key: q1_summary['aggregate'][split][key]
                for key in (
                    'num_instances', 'num_safe_reject',
                    'num_coverage_critical', 'num_protected_unknown')}
            for split in ('train', 'validation')},
        'gate': gate,
    }


def format_markdown(summary):
    delta = summary['delta_coverage_minus_safe_iou']
    lines = [
        '# Q2 参数匹配 Coverage-CVaR 实例质量 MLP', '',
        '- 数据：固定 train/validation forests；未读取 Wytham。',
        f'- 固定拒绝比例：{100 * summary["reject_ratio"]:.1f}%',
        f'- 随机种子：{summary["seeds"]}',
        '- 三个模型使用完全相同的全局特征、encoder 与参数量。', '',
        '| Model | Safe AP | Critical AP | F1 | Completeness | Commission | F1 gain vs unfiltered |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in (
            'quality_iou_control', 'safe_iou_control', 'coverage_cvar'):
        model = summary['models'][name]
        lines.append(
            f'| {name} | '
            f'{model["safe_average_precision"]["mean"]:.6f} | '
            f'{model["critical_average_precision"]["mean"]:.6f} | '
            f'{100 * model["filtered_f1"]["mean"]:.3f}% | '
            f'{100 * model["filtered_completeness"]["mean"]:.3f}% | '
            f'{100 * model["filtered_commission"]["mean"]:.3f}% | '
            f'{model["f1_gain_pp"]["mean"]:+.3f} pp |')
    lines.extend([
        '', '## Coverage-CVaR 相对 safe+IoU 参数匹配控制组', '',
        f'- F1：{delta["filtered_f1_gain_pp"]:+.3f} pp',
        f'- Completeness：{delta["completeness_change_pp"]:+.3f} pp',
        f'- Commission reduction：'
        f'{delta["commission_reduction_pp"]:+.3f} pp',
        f'- Seed wins：{delta["seed_wins"]} / {len(summary["seeds"])}',
        f'- Validation plot wins：'
        f'{delta["plot_wins"]} / '
        f'{len(delta["per_plot_f1_delta_pp"])}', '',
        '## 每森林 F1 差值', '',
        '| Plot | Coverage-CVaR minus safe+IoU |',
        '|---|---:|',
    ])
    for plot, value in delta['per_plot_f1_delta_pp'].items():
        lines.append(f'| {plot} | {value:+.3f} pp |')
    lines.extend(['', '## Gate', ''])
    lines.extend(
        f'- {name}: **{bool(passed)}**'
        for name, passed in summary['gate'].items())
    lines.extend(['', (
        'PASS：进入 Q3 参数匹配关系注意力；checkpoint 与拒绝比例保持锁定。'
        if summary['gate']['passed'] else
        'STOP：Coverage-CVaR 未优于参数匹配 safe+IoU 控制组，不实现关系注意力，'
        '不使用 Wytham 调参。')])
    return '\n'.join(lines) + '\n'


def validate_settings(settings):
    required = {
        'data_root', 'manifest_path', 'q1_summary_path', 'q1_label_path',
        'output_dir', 'seeds', 'models', 'reject_ratio', 'model', 'gate'}
    missing = required - set(settings)
    if missing:
        raise ValueError(f'Missing Q2 config fields: {sorted(missing)}')
    serialized = json.dumps(settings, ensure_ascii=False).lower()
    if 'wytham' in serialized:
        raise ValueError('Q2 configuration must not mention Wytham.')
    if not np.isclose(float(settings['reject_ratio']), 0.15):
        raise ValueError('Q2 reject ratio is preregistered as 0.15.')
    if [int(value) for value in settings['seeds']] != [42, 43, 44]:
        raise ValueError('Q2 seeds are locked to [42, 43, 44].')
    if list(settings['models']) != [
            'quality_iou_control', 'safe_iou_control', 'coverage_cvar']:
        raise ValueError(
            'Q2 models must be quality_iou_control, safe_iou_control and '
            'coverage_cvar.')
    if int(settings['model']['log_interval']) <= 0:
        raise ValueError('Q2 log_interval must be positive.')
    tail_fraction = float(settings['model']['coverage_tail_fraction'])
    if not 0 < tail_fraction <= 1:
        raise ValueError('Q2 coverage_tail_fraction must be within (0, 1].')
    if float(settings['model']['coverage_cvar_weight']) <= 0:
        raise ValueError('Q2 coverage_cvar_weight must be positive.')
    if float(settings['model']['max_completeness_drop_pp']) != float(
            settings['gate']['max_coverage_completeness_drop_pp']):
        raise ValueError(
            'Training selection and final completeness constraints differ.')


def _prepare_split(dataset, labels, values, split):
    mask = dataset['split'].astype(str) == split
    true = np.asarray(dataset['target_true'][mask], dtype=bool)
    safe = np.asarray(labels['safe_reject'][mask], dtype=bool)
    critical = np.asarray(labels['coverage_critical'][mask], dtype=bool)
    return {
        'x': np.asarray(values[mask], dtype=np.float32),
        'instance_id': np.asarray(
            dataset['instance_id'][mask], dtype=np.int64),
        'source_plot': dataset['source_plot'][mask].astype(str),
        'true': true,
        'true_float': true.astype(np.float32),
        'iou': np.asarray(dataset['target_iou'][mask], dtype=np.float32),
        'safe': safe,
        'safe_float': safe.astype(np.float32),
        'critical': critical,
        'critical_float': critical.astype(np.float32),
        'unknown': np.asarray(
            labels['protected_unknown'][mask], dtype=bool),
        'class_mask': np.asarray(
            dataset['classification_valid'][mask], dtype=bool),
        'valid_mask': np.asarray(dataset['target_valid'][mask], dtype=bool),
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f'Cannot write empty CSV: {path}')
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    import torch
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    validate_settings(settings)

    q1_path = Path(settings['q1_summary_path'])
    with q1_path.open(encoding='utf-8') as file:
        q1_summary = json.load(file)
    if not q1_summary.get('gate', {}).get('passed', False):
        raise RuntimeError('Q1 did not pass; Q2 training is forbidden.')
    if not np.isclose(
            float(q1_summary['settings']['reject_ratio']),
            float(settings['reject_ratio'])):
        raise ValueError('Q1 and Q2 reject ratios differ.')

    baseline = _load_baseline_module()
    dataset = baseline.load_quality_dataset(
        settings['data_root'], settings['manifest_path'])
    labels = load_coverage_risk_labels(settings['q1_label_path'], dataset)
    train_mask = dataset['split'].astype(str) == 'train'
    median, scale = baseline.fit_standardizer(
        dataset['features'][train_mask])
    values = baseline.standardize(dataset['features'], median, scale)
    train = _prepare_split(dataset, labels, values, 'train')
    validation = _prepare_split(dataset, labels, values, 'validation')

    q1_validation = q1_summary['aggregate']['validation']
    observed = {
        'num_instances': int(len(validation['x'])),
        'num_safe_reject': int(validation['safe'].sum()),
        'num_coverage_critical': int(validation['critical'].sum()),
        'num_protected_unknown': int(validation['unknown'].sum()),
    }
    for key, value in observed.items():
        if int(q1_validation[key]) != value:
            raise ValueError(
                f'Q1 validation count changed for {key}: '
                f'{q1_validation[key]} != {value}')

    output_dir = Path(settings['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_plot_rows = []
    prediction_rows = []
    parameter_counts = {}
    for mode in settings['models']:
        for seed in settings['seeds']:
            print(f'===== START {mode} seed {seed} =====', flush=True)
            result = train_model(
                mode, train, validation, settings['model'],
                float(settings['reject_ratio']), int(seed))
            row = _metric_row(result)
            rows.append(row)
            parameter_counts[mode] = int(result['parameter_count'])
            for plot_metrics in result['metrics']['per_plot']:
                per_plot_rows.append({
                    'model': mode,
                    'seed': int(seed),
                    'source_plot': plot_metrics['source_plot'],
                    'num_instances': plot_metrics['num_instances'],
                    'reject_budget': plot_metrics['reject_budget'],
                    'safe_rejected': plot_metrics['safe_rejected'],
                    'critical_rejected': plot_metrics['critical_rejected'],
                    'unknown_rejected': plot_metrics['unknown_rejected'],
                    'baseline_f1': plot_metrics['baseline']['f1'],
                    'filtered_f1': plot_metrics['filtered']['f1'],
                    'filtered_completeness': plot_metrics[
                        'filtered']['completeness'],
                    'filtered_commission': plot_metrics[
                        'filtered']['commission'],
                })
            rejected = result['metrics']['rejected_mask']
            prediction_rows.extend({
                'model': mode,
                'seed': int(seed),
                'source_plot': str(plot),
                'instance_id': int(instance_id),
                'rejection_risk': float(risk),
                'safe_reject': bool(safe),
                'coverage_critical': bool(critical),
                'protected_unknown': bool(unknown),
                'rejected': bool(is_rejected),
            } for (
                plot, instance_id, risk, safe, critical, unknown,
                is_rejected) in zip(
                    validation['source_plot'],
                    validation['instance_id'],
                    result['rejection_risk'],
                    validation['safe'],
                    validation['critical'],
                    validation['unknown'],
                    rejected))
            torch.save({
                'state_dict': result['state_dict'],
                'mode': mode,
                'feature_names': dataset['feature_names'],
                'median': median,
                'scale': scale,
                'seed': int(seed),
                'best_epoch': result['best_epoch'],
                'reject_ratio': float(settings['reject_ratio']),
                'model_config': settings['model'],
                'parameter_count': result['parameter_count'],
            }, checkpoint_dir / f'{mode}_seed{seed}.pth')
            print(
                f'DONE {mode} seed {seed}: '
                f'epoch={result["best_epoch"]}, '
                f'F1={100 * row["filtered_f1"]:.3f}%, '
                f'completeness={100 * row["filtered_completeness"]:.3f}%',
                flush=True)

    summary = summarize(
        rows, per_plot_rows, settings, q1_summary, parameter_counts)
    _write_csv(output_dir / 'per_seed_metrics.csv', rows)
    _write_csv(output_dir / 'per_plot_metrics.csv', per_plot_rows)
    _write_csv(output_dir / 'validation_predictions.csv', prediction_rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown, flush=True)
    if not summary['gate']['passed']:
        raise RuntimeError(
            'Q2 coverage-CVaR MLP gate failed; do not implement attention.')
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Q2 parameter-matched coverage quality models.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
