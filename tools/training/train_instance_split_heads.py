"""Train the Q4b1 candidate, child-count and split-safety heads."""

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np

from tree_learn.util.instance_split_learning import (
    detection_metrics_from_contingency,
    evaluate_split_decisions,
    validate_instance_split_learning_artifact,
)
from tree_learn.util.omission_diagnostics import detection_metrics


def load_settings(config_path):
    import yaml
    with Path(config_path).open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    if 'wytham' in json.dumps(settings).lower():
        raise ValueError('Q4b1 training must not reference Wytham.')
    return settings


def load_record(data_root, split, plot_name):
    directory = Path(data_root) / split / plot_name
    npz_path = directory / 'instance_split_learning.npz'
    metadata_path = directory / 'metadata.json'
    metadata = validate_instance_split_learning_artifact(
        npz_path, metadata_path)
    with np.load(npz_path, allow_pickle=False) as payload:
        data = {name: payload[name].copy() for name in payload.files}
    baseline = evaluate_split_decisions(data, [])
    for key in ('tp', 'fp', 'fn'):
        if int(baseline[key]) != int(metadata['baseline'][key]):
            raise ValueError(
                f'Baseline {key} differs for {plot_name}: '
                f'{baseline[key]} vs {metadata["baseline"][key]}.')
    return {
        'plot_name': plot_name,
        'split': split,
        'data': data,
        'metadata': metadata,
    }


def load_dataset(settings):
    generation = Path(settings['data_root']) / 'generation_summary.json'
    if not generation.is_file():
        raise FileNotFoundError(
            'Missing Q4b1 generation_summary.json; run data generation first.')
    with generation.open(encoding='utf-8') as file:
        summary = json.load(file)
    if not summary['gate']['passed']:
        raise ValueError('Q4b1 data gate did not pass.')
    requested = {
        (split, plot_name)
        for split in ('train', 'validation')
        for plot_name in settings['splits'][split]}
    generated = {
        (row['split'], row['source_plot']) for row in summary['rows']}
    if requested != generated:
        raise ValueError(
            'Q4b1 training split differs from the passed generation split.')
    if set(settings['splits']['train']) & set(
            settings['splits']['validation']):
        raise ValueError('Q4b1 train/validation plots overlap.')
    records = []
    for split in ('train', 'validation'):
        for plot_name in settings['splits'][split]:
            records.append(load_record(
                settings['data_root'], split, plot_name))
    return records


def pool_instance_features(data):
    global_values = data['global_feature_values'].astype(np.float32)
    tokens = data['vertical_tokens'].astype(np.float32)
    mask = data['layer_valid_mask'].astype(bool)
    expanded = mask[:, :, None]
    count = np.maximum(expanded.sum(axis=1), 1)
    token_mean = (tokens * expanded).sum(axis=1) / count
    masked = np.where(expanded, tokens, -np.inf)
    token_max = masked.max(axis=1)
    token_max[~np.isfinite(token_max)] = 0.0
    return np.concatenate([
        global_values, token_mean, token_max], axis=1).astype(np.float32)


def robust_fit(values):
    values = np.asarray(values, dtype=np.float32)
    center = np.median(values, axis=0).astype(np.float32)
    scale = (
        np.quantile(values, 0.75, axis=0) -
        np.quantile(values, 0.25, axis=0)).astype(np.float32)
    scale = np.maximum(scale, 1e-3)
    return center, scale


def normalize(values, center, scale):
    return np.clip(
        (np.asarray(values, dtype=np.float32) - center) / scale,
        -10.0, 10.0).astype(np.float32)


def prepare_data(records):
    train_records = [row for row in records if row['split'] == 'train']
    validation_records = [
        row for row in records if row['split'] == 'validation']
    instance_train = np.concatenate([
        pool_instance_features(row['data']) for row in train_records])
    proposal_train = np.concatenate([
        row['data']['proposal_feature_values'] for row in train_records])
    instance_center, instance_scale = robust_fit(instance_train)
    proposal_center, proposal_scale = robust_fit(proposal_train)

    for record in records:
        data = record['data']
        record['instance_x'] = normalize(
            pool_instance_features(data), instance_center, instance_scale)
        record['proposal_x'] = normalize(
            data['proposal_feature_values'], proposal_center, proposal_scale)

    train_instances = []
    train_candidates = []
    train_child_targets = []
    train_child_valid = []
    train_proposals = []
    train_proposal_parents = []
    train_safety = []
    instance_offset = 0
    for record in train_records:
        data = record['data']
        train_instances.append(record['instance_x'])
        train_candidates.append(data['candidate_target'])
        train_child_targets.append(data['child_count_target'])
        train_child_valid.append(data['child_count_valid'])
        train_proposals.append(record['proposal_x'])
        train_proposal_parents.append(
            data['proposal_instance_index'] + instance_offset)
        train_safety.append(data['proposal_safety_target'])
        instance_offset += len(data['instance_ids'])
    train = {
        'instance_x': np.concatenate(train_instances),
        'candidate_target': np.concatenate(train_candidates).astype(np.float32),
        'child_count_target': np.concatenate(train_child_targets),
        'child_count_valid': np.concatenate(train_child_valid),
        'proposal_x': np.concatenate(train_proposals),
        'proposal_parent_index': np.concatenate(train_proposal_parents),
        'safety_target': np.concatenate(train_safety).astype(np.float32),
    }
    norms = {
        'instance_center': instance_center,
        'instance_scale': instance_scale,
        'proposal_center': proposal_center,
        'proposal_scale': proposal_scale,
    }
    return train, validation_records, norms


def build_model(instance_dim, proposal_dim, max_children, model_config):
    import torch
    from torch import nn

    hidden = int(model_config['hidden_dim'])
    safety_hidden = int(model_config['safety_hidden_dim'])
    dropout = float(model_config['dropout'])

    class InstanceSplitHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(instance_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            self.candidate_head = nn.Linear(hidden, 1)
            self.child_count_head = nn.Linear(hidden, max_children - 1)
            self.safety_head = nn.Sequential(
                nn.Linear(hidden + proposal_dim, safety_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(safety_hidden, 1),
            )

        def forward_instances(self, values):
            embedding = self.encoder(values)
            return (
                embedding,
                self.candidate_head(embedding).squeeze(-1),
                self.child_count_head(embedding),
            )

        def forward_safety(self, embedding, proposal, parent_index):
            values = torch.cat([
                embedding[parent_index], proposal], dim=1)
            return self.safety_head(values).squeeze(-1)

    return InstanceSplitHeads()


def balanced_pos_weight(target):
    values = np.asarray(target, dtype=np.float32)
    positives = float(values.sum())
    negatives = float(len(values) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError('Balanced BCE needs both classes.')
    return negatives / positives


def classification_metrics(target, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(target)) < 2:
        return {'roc_auc': float('nan'), 'average_precision': float('nan')}
    return {
        'roc_auc': float(roc_auc_score(target, score)),
        'average_precision': float(average_precision_score(target, score)),
    }


def aggregate_counts(metrics):
    return detection_metrics(
        sum(int(row['tp']) for row in metrics),
        sum(int(row['fp']) for row in metrics),
        sum(int(row['fn']) for row in metrics))


def proposal_lookup(data):
    return {
        (int(parent), int(children)): index
        for index, (parent, children) in enumerate(zip(
            data['proposal_instance_index'], data['proposal_k']))}


def decisions_from_scores(
        data, candidate_scores, child_predictions, safety_scores,
        candidate_threshold, safety_threshold):
    lookup = proposal_lookup(data)
    accepted = []
    for instance_index in np.flatnonzero(
            candidate_scores >= float(candidate_threshold)):
        children = int(child_predictions[instance_index])
        proposal_index = lookup.get((int(instance_index), children))
        if proposal_index is not None and safety_scores[proposal_index] >= float(
                safety_threshold):
            accepted.append(proposal_index)
    return np.asarray(accepted, dtype=np.int64)


def infer_records(model, records, device):
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for record in records:
            instance = torch.from_numpy(record['instance_x']).to(device)
            proposal = torch.from_numpy(record['proposal_x']).to(device)
            parent = torch.from_numpy(
                record['data']['proposal_instance_index']).long().to(device)
            embedding, candidate_logit, child_logit = (
                model.forward_instances(instance))
            safety_logit = model.forward_safety(
                embedding, proposal, parent)
            outputs.append({
                'candidate_score': torch.sigmoid(
                    candidate_logit).cpu().numpy(),
                'child_prediction': (
                    child_logit.argmax(dim=1).cpu().numpy() + 2),
                'safety_score': torch.sigmoid(
                    safety_logit).cpu().numpy(),
            })
    return outputs


def evaluate_thresholds(
        records, outputs, candidate_threshold, safety_threshold,
        match_iou_threshold, min_precision_for_counted_fp):
    plot_rows = []
    for record, output in zip(records, outputs):
        accepted = decisions_from_scores(
            record['data'], output['candidate_score'],
            output['child_prediction'], output['safety_score'],
            candidate_threshold, safety_threshold)
        metrics = evaluate_split_decisions(
            record['data'], accepted,
            match_iou_threshold=match_iou_threshold,
            min_precision_for_counted_fp=min_precision_for_counted_fp)
        baseline = evaluate_split_decisions(
            record['data'], [],
            match_iou_threshold=match_iou_threshold,
            min_precision_for_counted_fp=min_precision_for_counted_fp)
        metrics.update({
            'source_plot': record['plot_name'],
            'baseline_f1': baseline['f1'],
            'f1_gain_pp': 100.0 * (metrics['f1'] - baseline['f1']),
        })
        plot_rows.append(metrics)
    aggregate = aggregate_counts(plot_rows)
    baseline = aggregate_counts([
        evaluate_split_decisions(
            record['data'], [],
            match_iou_threshold=match_iou_threshold,
            min_precision_for_counted_fp=min_precision_for_counted_fp)
        for record in records])
    aggregate.update({
        'baseline': baseline,
        'candidate_threshold': float(candidate_threshold),
        'safety_threshold': float(safety_threshold),
        'accepted_splits': int(sum(
            row['accepted_splits'] for row in plot_rows)),
        'f1_gain_pp': 100.0 * (aggregate['f1'] - baseline['f1']),
        'completeness_drop_pp': 100.0 * (
            baseline['completeness'] - aggregate['completeness']),
        'commission_increase_pp': 100.0 * (
            aggregate['commission'] - baseline['commission']),
        'nonnegative_plots': int(sum(
            row['f1_gain_pp'] >= -1e-9 for row in plot_rows)),
        'plot_metrics': plot_rows,
    })
    return aggregate

def select_thresholds(records, outputs, settings):
    selection = settings['selection']
    candidate_grid = list(map(float, selection['candidate_thresholds'])) + [1.01]
    safety_grid = list(map(float, selection['safety_thresholds'])) + [1.01]
    feasible = []
    all_rows = []
    decision_cache = {}
    for candidate_threshold in candidate_grid:
        for safety_threshold in safety_grid:
            decision_key = tuple(
                tuple(decisions_from_scores(
                    record['data'], output['candidate_score'],
                    output['child_prediction'], output['safety_score'],
                    candidate_threshold, safety_threshold).tolist())
                for record, output in zip(records, outputs))
            if decision_key not in decision_cache:
                decision_cache[decision_key] = evaluate_thresholds(
                    records, outputs, candidate_threshold, safety_threshold,
                    match_iou_threshold=float(
                        selection['match_iou_threshold']),
                    min_precision_for_counted_fp=float(
                        selection['min_precision_for_counted_fp']))
            row = copy.deepcopy(decision_cache[decision_key])
            row['candidate_threshold'] = float(candidate_threshold)
            row['safety_threshold'] = float(safety_threshold)
            row['constraints_passed'] = bool(
                row['completeness_drop_pp'] <= float(
                    selection['max_completeness_drop_pp']) and
                row['commission_increase_pp'] <= float(
                    selection['max_commission_increase_pp']))
            all_rows.append(row)
            if row['constraints_passed']:
                feasible.append(row)
    if not feasible:
        raise RuntimeError('No Q4b1 threshold pair satisfies safety constraints.')
    best = max(feasible, key=lambda row: (
        row['f1'],
        row['completeness'],
        -row['commission'],
        -row['accepted_splits'],
        row['candidate_threshold'],
        row['safety_threshold'],
    ))
    return best, all_rows


def validation_head_metrics(records, outputs):
    candidate_target = np.concatenate([
        row['data']['candidate_target'] for row in records])
    candidate_score = np.concatenate([
        output['candidate_score'] for output in outputs])
    safety_target = np.concatenate([
        row['data']['proposal_safety_target'] for row in records])
    safety_score = np.concatenate([
        output['safety_score'] for output in outputs])
    child_true = []
    child_pred = []
    for record, output in zip(records, outputs):
        valid = (
            record['data']['candidate_target'] &
            record['data']['child_count_valid'])
        child_true.extend(record['data']['child_count_target'][valid].tolist())
        child_pred.extend(output['child_prediction'][valid].tolist())
    candidate = classification_metrics(candidate_target, candidate_score)
    safety = classification_metrics(safety_target, safety_score)
    return {
        'candidate_roc_auc': candidate['roc_auc'],
        'candidate_average_precision': candidate['average_precision'],
        'safety_roc_auc': safety['roc_auc'],
        'safety_average_precision': safety['average_precision'],
        'child_count_accuracy': float(np.mean(
            np.asarray(child_true) == np.asarray(child_pred))),
        'num_child_count_targets': int(len(child_true)),
    }


def set_deterministic(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def train_one_seed(
        seed, train, validation_records, norms, settings, output_dir):
    import torch
    from torch import nn

    set_deterministic(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    max_children = int(settings['model']['max_children'])
    model = build_model(
        train['instance_x'].shape[1], train['proposal_x'].shape[1],
        max_children, settings['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings['optimizer']['lr']),
        weight_decay=float(settings['optimizer']['weight_decay']))

    instance_x = torch.from_numpy(train['instance_x']).to(device)
    candidate_target = torch.from_numpy(
        train['candidate_target']).to(device)
    child_target = torch.from_numpy(
        train['child_count_target']).long().to(device)
    child_mask_np = (
        train['candidate_target'].astype(bool) &
        train['child_count_valid'].astype(bool))
    child_mask = torch.from_numpy(child_mask_np).to(device)
    proposal_x = torch.from_numpy(train['proposal_x']).to(device)
    proposal_parent = torch.from_numpy(
        train['proposal_parent_index']).long().to(device)
    safety_target = torch.from_numpy(train['safety_target']).to(device)

    candidate_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        balanced_pos_weight(train['candidate_target']), device=device))
    safety_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        balanced_pos_weight(train['safety_target']), device=device))
    child_classes = train['child_count_target'][child_mask_np] - 2
    class_counts = np.bincount(
        child_classes, minlength=max_children - 1).astype(np.float64)
    child_weights = np.zeros(max_children - 1, dtype=np.float32)
    present = class_counts > 0
    child_weights[present] = (
        class_counts[present].sum() /
        (present.sum() * class_counts[present])).astype(np.float32)
    child_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(
        child_weights).to(device))

    weights = settings['loss']
    best = None
    best_state = None
    stale_evaluations = 0
    history = []
    epochs = int(settings['training']['epochs'])
    eval_frequency = int(settings['training']['eval_frequency'])
    patience = int(settings['training']['patience_evaluations'])
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embedding, candidate_logit, child_logit = (
            model.forward_instances(instance_x))
        safety_logit = model.forward_safety(
            embedding, proposal_x, proposal_parent)
        loss_candidate = candidate_loss(candidate_logit, candidate_target)
        loss_child = child_loss(
            child_logit[child_mask], child_target[child_mask] - 2)
        loss_safety = safety_loss(safety_logit, safety_target)
        loss = (
            float(weights['candidate']) * loss_candidate +
            float(weights['child_count']) * loss_child +
            float(weights['safety']) * loss_safety)
        if not torch.isfinite(loss):
            raise RuntimeError('Q4b1 training loss is not finite.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings['training']['grad_clip']))
        optimizer.step()

        if epoch % eval_frequency != 0 and epoch != epochs:
            continue
        outputs = infer_records(model, validation_records, device)
        selected, _ = select_thresholds(
            validation_records, outputs, settings)
        head = validation_head_metrics(validation_records, outputs)
        row = {
            'epoch': epoch,
            'loss': float(loss.item()),
            **{key: value for key, value in selected.items()
               if key != 'plot_metrics'},
            **head,
        }
        history.append(row)
        print(
            f'[seed {seed}] epoch {epoch:03d}: '
            f'loss={loss.item():.5f}, F1={100 * selected["f1"]:.3f}%, '
            f'gain={selected["f1_gain_pp"]:+.3f} pp, '
            f'accepted={selected["accepted_splits"]}', flush=True)
        score = (
            selected['f1'], selected['completeness'],
            -selected['commission'], -selected['accepted_splits'])
        best_score = None if best is None else (
            best['f1'], best['completeness'],
            -best['commission'], -best['accepted_splits'])
        if best is None or score > best_score:
            best = copy.deepcopy(selected)
            best['epoch'] = epoch
            best['head_metrics'] = head
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()}
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        if stale_evaluations >= patience:
            print(
                f'[seed {seed}] early stop at epoch {epoch}; '
                f'best epoch={best["epoch"]}.', flush=True)
            break

    model.load_state_dict(best_state)
    outputs = infer_records(model, validation_records, device)
    final = evaluate_thresholds(
        validation_records, outputs,
        best['candidate_threshold'], best['safety_threshold'],
        match_iou_threshold=float(
            settings['selection']['match_iou_threshold']),
        min_precision_for_counted_fp=float(
            settings['selection']['min_precision_for_counted_fp']))
    final['epoch'] = int(best['epoch'])
    final['head_metrics'] = validation_head_metrics(
        validation_records, outputs)
    rule = settings['gate']
    final['gate'] = {
        'f1_gain_passed': final['f1_gain_pp'] >= float(
            rule['min_f1_gain_pp']),
        'completeness_preserved': final['completeness_drop_pp'] <= float(
            rule['max_completeness_drop_pp']),
        'commission_preserved': final['commission_increase_pp'] <= float(
            rule['max_commission_increase_pp']),
        'plot_consistency_passed': final['nonnegative_plots'] >= int(
            rule['min_nonnegative_plots']),
    }
    final['gate']['passed'] = all(final['gate'].values())
    checkpoint = {
        'seed': int(seed),
        'best_epoch': int(best['epoch']),
        'model_state_dict': best_state,
        'model_config': settings['model'],
        'norms': norms,
        'candidate_threshold': float(best['candidate_threshold']),
        'safety_threshold': float(best['safety_threshold']),
        'metrics': final,
    }
    checkpoint_path = output_dir / 'checkpoints' / f'split_heads_seed{seed}.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    (output_dir / f'history_seed{seed}.json').write_text(
        json.dumps(history, indent=2), encoding='utf-8')
    return final, checkpoint_path, sum(
        parameter.numel() for parameter in model.parameters())


def summarize(results, settings, output_dir, parameter_count):
    seeds = sorted(results)
    f1_gains = np.asarray([
        results[seed]['f1_gain_pp'] for seed in seeds])
    passed = [results[seed]['gate']['passed'] for seed in seeds]
    locked_seed = int(settings['locked_seed'])
    rule = settings['gate']
    gate = {
        'data_gate_passed': True,
        'locked_seed_present': locked_seed in results,
        'locked_seed_passed': bool(results[locked_seed]['gate']['passed']),
        'enough_seed_passes': sum(passed) >= int(rule['min_seed_passes']),
        'f1_seed_stability_passed': float(f1_gains.std(ddof=1)) <= float(
            rule['max_f1_gain_std_pp']),
    }
    gate['passed'] = all(gate.values())
    summary = {
        'parameter_count': int(parameter_count),
        'locked_seed': locked_seed,
        'per_seed': {str(seed): results[seed] for seed in seeds},
        'f1_gain_mean_pp': float(f1_gains.mean()),
        'f1_gain_std_pp': float(f1_gains.std(ddof=1)),
        'gate': gate,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    fields = [
        'seed', 'epoch', 'candidate_threshold', 'safety_threshold',
        'accepted_splits', 'f1', 'f1_gain_pp', 'completeness',
        'completeness_drop_pp', 'commission', 'commission_increase_pp',
        'nonnegative_plots', 'candidate_average_precision',
        'child_count_accuracy', 'safety_average_precision', 'passed',
    ]
    with (output_dir / 'per_seed_metrics.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            result = results[seed]
            head = result['head_metrics']
            writer.writerow({
                'seed': seed,
                'epoch': result['epoch'],
                'candidate_threshold': result['candidate_threshold'],
                'safety_threshold': result['safety_threshold'],
                'accepted_splits': result['accepted_splits'],
                'f1': result['f1'],
                'f1_gain_pp': result['f1_gain_pp'],
                'completeness': result['completeness'],
                'completeness_drop_pp': result['completeness_drop_pp'],
                'commission': result['commission'],
                'commission_increase_pp': result[
                    'commission_increase_pp'],
                'nonnegative_plots': result['nonnegative_plots'],
                'candidate_average_precision': head[
                    'candidate_average_precision'],
                'child_count_accuracy': head['child_count_accuracy'],
                'safety_average_precision': head[
                    'safety_average_precision'],
                'passed': result['gate']['passed'],
            })
    lines = [
        '# Q4b1 Candidate + K + Safety 三头 MLP', '',
        '- 数据：固定 13 个 train forests 与 5 个 validation forests。',
        '- Wytham 未参与训练、checkpoint 或阈值选择。',
        f'- 参数量：{parameter_count:,}', '',
        '| Seed | Epoch | Candidate AP | K accuracy | Safety AP | Accepted | F1 gain | Completeness drop | Commission increase | Pass |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for seed in seeds:
        result = results[seed]
        head = result['head_metrics']
        lines.append(
            f'| {seed} | {result["epoch"]} | '
            f'{head["candidate_average_precision"]:.6f} | '
            f'{head["child_count_accuracy"]:.6f} | '
            f'{head["safety_average_precision"]:.6f} | '
            f'{result["accepted_splits"]} | '
            f'{result["f1_gain_pp"]:+.3f} pp | '
            f'{result["completeness_drop_pp"]:+.3f} pp | '
            f'{result["commission_increase_pp"]:+.3f} pp | '
            f'{result["gate"]["passed"]} |')
    lines.extend([
        '', '## Aggregate', '',
        f'- F1 gain mean：{summary["f1_gain_mean_pp"]:+.3f} pp',
        f'- F1 gain std：{summary["f1_gain_std_pp"]:.3f} pp',
        '', '## Gate', ''])
    lines.extend(f'- {key}: **{value}**' for key, value in gate.items())
    lines.extend([
        '',
        'PASS：进入 Q4b2 validation pipeline 集成。'
        if gate['passed'] else
        'STOP：三头 MLP 不可部署，关闭实例拆分路线。'])
    (output_dir / 'summary.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return summary


def run(config_path):
    settings = load_settings(config_path)
    records = load_dataset(settings)
    train, validation, norms = prepare_data(records)
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    parameter_count = None
    for seed in settings['seeds']:
        print(f'===== START Q4b1 seed {seed} =====', flush=True)
        result, checkpoint, count = train_one_seed(
            int(seed), train, validation, norms, settings, output_dir)
        results[int(seed)] = result
        parameter_count = count if parameter_count is None else parameter_count
        if count != parameter_count:
            raise AssertionError('Q4b1 parameter count changed across seeds.')
        print(
            f'DONE seed {seed}: checkpoint={checkpoint}, '
            f'gain={result["f1_gain_pp"]:+.3f} pp', flush=True)
    summary = summarize(results, settings, output_dir, parameter_count)
    if not summary['gate']['passed']:
        raise RuntimeError(
            'Q4b1 three-head gate failed; do not integrate splitting.')


def main():
    parser = argparse.ArgumentParser(
        description='Train Q4b1 instance-split heads.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()