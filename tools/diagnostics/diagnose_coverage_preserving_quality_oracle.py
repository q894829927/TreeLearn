"""Q1 coverage-preserving instance rejection label audit and Oracle.

The diagnostic reuses the fixed instance-quality artifacts generated from the
13 training and five validation forests.  It derives two deployable-learning
targets from GT only for supervision:

* safe_reject: a classification-valid false/redundant predicted instance;
* coverage_critical: the deterministic representative that preserves a
  baseline-matched GT tree.

Wytham is forbidden.  This stage does not train a network or modify a forest.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def derive_dual_risk_targets(
        instance_ids, max_iou, best_gt_ids, classification_valid,
        target_is_true_tree):
    """Derive mutually-exclusive safe-reject and coverage-critical labels."""
    ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    iou = np.asarray(max_iou, dtype=np.float64).reshape(-1)
    gt = np.asarray(best_gt_ids, dtype=np.int64).reshape(-1)
    class_valid = np.asarray(classification_valid, dtype=bool).reshape(-1)
    target_true = np.asarray(target_is_true_tree, dtype=bool).reshape(-1)
    count = len(ids)
    if not all(len(values) == count for values in (
            iou, gt, class_valid, target_true)):
        raise ValueError('Dual-risk target arrays are not aligned.')
    if len(np.unique(ids)) != count or np.any(ids <= 0):
        raise ValueError('Instance IDs must be unique and positive.')
    if not np.isfinite(iou).all() or np.any((iou < 0) | (iou > 1)):
        raise ValueError('IoU targets must be finite and within [0, 1].')

    positive = class_valid & target_true
    negative = class_valid & ~target_true
    if np.any(positive & (gt <= 0)):
        raise ValueError('Every valid positive must have a positive GT id.')
    coverage_critical = np.zeros(count, dtype=bool)
    for gt_id in np.unique(gt[positive]):
        candidates = np.flatnonzero(positive & (gt == gt_id))
        # Highest IoU wins; instance ID is the deterministic tie-breaker.
        order = np.lexsort((ids[candidates], -iou[candidates]))
        coverage_critical[candidates[order[0]]] = True
    redundant_positive = positive & ~coverage_critical
    safe_reject = negative | redundant_positive
    eligible = safe_reject | coverage_critical
    protected_unknown = ~eligible
    if np.any(safe_reject & coverage_critical):
        raise AssertionError('Dual-risk targets overlap.')
    if not np.array_equal(eligible, class_valid):
        raise AssertionError(
            'Every classification-valid instance needs exactly one target.')
    return {
        'safe_reject': safe_reject,
        'coverage_critical': coverage_critical,
        'redundant_positive': redundant_positive,
        'counted_negative': negative,
        'protected_unknown': protected_unknown,
    }


def detection_proxy(tp, fp, fn=0):
    tp, fp, fn = int(tp), int(fp), int(fn)
    denominator = 2 * tp + fp + fn
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'completeness_relative_to_baseline_matches': float(
            tp / max(tp + fn, 1)),
        'commission': float(fp / max(tp + fp, 1)),
        'f1': float(2 * tp / max(denominator, 1)),
    }


def rejection_oracle(instance_ids, targets, reject_ratio):
    """Reject up to a fixed all-instance budget, never a critical instance."""
    ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    safe = np.asarray(targets['safe_reject'], dtype=bool).reshape(-1)
    critical = np.asarray(
        targets['coverage_critical'], dtype=bool).reshape(-1)
    negative = np.asarray(targets['counted_negative'], dtype=bool).reshape(-1)
    redundant = np.asarray(
        targets['redundant_positive'], dtype=bool).reshape(-1)
    if not 0 <= float(reject_ratio) < 1:
        raise ValueError('reject_ratio must be within [0, 1).')
    if not all(len(values) == len(ids) for values in (
            safe, critical, negative, redundant)):
        raise ValueError('Oracle target arrays are not aligned.')
    budget = int(np.floor(float(reject_ratio) * len(ids)))
    safe_indices = np.flatnonzero(safe)
    # Counted negatives precede redundant positives; IDs break all ties.
    order = np.lexsort((ids[safe_indices], redundant[safe_indices]))
    rejected = safe_indices[order[:budget]]
    rejected_mask = np.zeros(len(ids), dtype=bool)
    rejected_mask[rejected] = True
    if np.any(rejected_mask & critical):
        raise AssertionError('Oracle rejected a coverage-critical instance.')

    baseline = detection_proxy(int(critical.sum()), int(safe.sum()))
    remaining_safe = int((safe & ~rejected_mask).sum())
    filtered = detection_proxy(int(critical.sum()), remaining_safe)
    return {
        'reject_budget': budget,
        'num_rejected': int(rejected_mask.sum()),
        'budget_utilization': float(
            rejected_mask.sum() / max(budget, 1)) if budget else 1.0,
        'rejected_counted_negative': int((rejected_mask & negative).sum()),
        'rejected_redundant_positive': int(
            (rejected_mask & redundant).sum()),
        'critical_rejected': int((rejected_mask & critical).sum()),
        'baseline': baseline,
        'filtered': filtered,
        'f1_gain_pp': float(100 * (filtered['f1'] - baseline['f1'])),
        'commission_reduction_pp': float(
            100 * (baseline['commission'] - filtered['commission'])),
        'completeness_drop_pp': float(100 * (
            baseline['completeness_relative_to_baseline_matches'] -
            filtered['completeness_relative_to_baseline_matches'])),
        'rejected_mask': rejected_mask,
    }


def _artifact_identity(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return {
        'path': str(path.resolve()),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
        'sha256': digest.hexdigest(),
    }


def load_artifact_groups(data_root, manifest_path, expected_splits):
    root = Path(data_root).resolve()
    with Path(manifest_path).open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('Instance-quality manifest is empty.')
    if any('wytham' in str(row).lower() for row in rows):
        raise ValueError('Q1 must not read Wytham artifacts.')
    grouped = {}
    for row in rows:
        path = Path(row['artifact_path']).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid quality artifact path: {path}')
        grouped.setdefault(path, []).append(row)

    reports = []
    observed = {'train': [], 'validation': []}
    required = {
        'instance_ids', 'target_max_iou', 'target_best_gt_id',
        'target_valid', 'target_classification_valid',
        'target_is_true_tree', 'source_plot', 'split'}
    for path in sorted(grouped, key=str):
        manifest_rows = grouped[path]
        with np.load(path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f'{path} lacks Q1 fields: {sorted(missing)}')
            plot = str(np.asarray(data['source_plot']).item())
            split = str(np.asarray(data['split']).item())
            ids = data['instance_ids'].astype(np.int64)
            manifest_ids = {int(row['instance_id']) for row in manifest_rows}
            if set(ids.tolist()) != manifest_ids:
                raise ValueError(f'Manifest instance IDs differ for {path}.')
            if {row['source_plot'] for row in manifest_rows} != {plot}:
                raise ValueError(f'Manifest plot differs for {path}.')
            if {row['split'] for row in manifest_rows} != {split}:
                raise ValueError(f'Manifest split differs for {path}.')
            targets = derive_dual_risk_targets(
                ids,
                data['target_max_iou'],
                data['target_best_gt_id'],
                data['target_classification_valid'],
                data['target_is_true_tree'])
            reports.append({
                'source_plot': plot,
                'split': split,
                'instance_ids': ids,
                'max_iou': data['target_max_iou'].astype(np.float64),
                'target_valid': data['target_valid'].astype(bool),
                'classification_valid': data[
                    'target_classification_valid'].astype(bool),
                'targets': targets,
                'artifact_identity': _artifact_identity(path),
            })
        if split not in observed:
            raise ValueError(f'Unexpected split {split!r}.')
        observed[split].append(plot)

    for split, expected in expected_splits.items():
        actual = sorted(observed[split])
        if actual != sorted(expected):
            raise ValueError(
                f'{split} forests differ: observed={actual}, '
                f'expected={sorted(expected)}')
    if set(observed) != set(expected_splits):
        raise ValueError('Q1 requires exactly train and validation splits.')
    return reports


def analyze_group(group, reject_ratio):
    targets = group['targets']
    result = rejection_oracle(
        group['instance_ids'], targets, reject_ratio)
    return {
        'source_plot': group['source_plot'],
        'split': group['split'],
        'num_instances': int(len(group['instance_ids'])),
        'num_target_valid': int(group['target_valid'].sum()),
        'num_classification_valid': int(group['classification_valid'].sum()),
        'num_safe_reject': int(targets['safe_reject'].sum()),
        'num_coverage_critical': int(targets['coverage_critical'].sum()),
        'num_counted_negative': int(targets['counted_negative'].sum()),
        'num_redundant_positive': int(
            targets['redundant_positive'].sum()),
        'num_protected_unknown': int(targets['protected_unknown'].sum()),
        'safe_reject_rate': float(targets['safe_reject'].mean()),
        'classification_coverage': float(group['classification_valid'].mean()),
        'artifact_identity': group['artifact_identity'],
        'oracle': {
            key: value for key, value in result.items()
            if key != 'rejected_mask'
        },
    }


def _pooled_proxy(reports, mode):
    tp = sum(row['oracle'][mode]['tp'] for row in reports)
    fp = sum(row['oracle'][mode]['fp'] for row in reports)
    fn = sum(row['oracle'][mode]['fn'] for row in reports)
    return detection_proxy(tp, fp, fn)


def aggregate_reports(reports, settings):
    split_reports = {
        split: [row for row in reports if row['split'] == split]
        for split in ('train', 'validation')}
    aggregate = {}
    for split, rows in split_reports.items():
        baseline = _pooled_proxy(rows, 'baseline')
        filtered = _pooled_proxy(rows, 'filtered')
        aggregate[split] = {
            'num_plots': int(len(rows)),
            'num_instances': int(sum(row['num_instances'] for row in rows)),
            'num_safe_reject': int(sum(
                row['num_safe_reject'] for row in rows)),
            'num_coverage_critical': int(sum(
                row['num_coverage_critical'] for row in rows)),
            'num_counted_negative': int(sum(
                row['num_counted_negative'] for row in rows)),
            'num_redundant_positive': int(sum(
                row['num_redundant_positive'] for row in rows)),
            'num_protected_unknown': int(sum(
                row['num_protected_unknown'] for row in rows)),
            'baseline': baseline,
            'filtered': filtered,
            'f1_gain_pp': float(100 * (filtered['f1'] - baseline['f1'])),
            'commission_reduction_pp': float(100 * (
                baseline['commission'] - filtered['commission'])),
            'completeness_drop_pp': float(100 * (
                baseline['completeness_relative_to_baseline_matches'] -
                filtered['completeness_relative_to_baseline_matches'])),
        }
    validation = aggregate['validation']
    validation_rows = split_reports['validation']
    rule = settings['gate']
    gate = {
        'expected_train_plots': (
            aggregate['train']['num_plots'] ==
            int(rule['expected_train_plots'])),
        'expected_validation_plots': (
            validation['num_plots'] ==
            int(rule['expected_validation_plots'])),
        'enough_train_safe_reject': (
            aggregate['train']['num_safe_reject'] >=
            int(rule['min_train_safe_reject'])),
        'enough_train_coverage_critical': (
            aggregate['train']['num_coverage_critical'] >=
            int(rule['min_train_coverage_critical'])),
        'enough_validation_safe_reject': (
            validation['num_safe_reject'] >=
            int(rule['min_validation_safe_reject'])),
        'enough_validation_coverage_critical': (
            validation['num_coverage_critical'] >=
            int(rule['min_validation_coverage_critical'])),
        'oracle_f1_gain_passed': (
            validation['f1_gain_pp'] >=
            float(rule['min_validation_f1_gain_pp'])),
        'oracle_commission_reduction_passed': (
            validation['commission_reduction_pp'] >=
            float(rule['min_validation_commission_reduction_pp'])),
        'coverage_preserved': (
            validation['completeness_drop_pp'] <=
            float(rule['max_validation_completeness_drop_pp'])),
        'per_plot_consistency': (
            sum(row['oracle']['f1_gain_pp'] > 0 for row in validation_rows) >=
            int(rule['min_validation_plot_wins'])),
        'labels_mutually_exclusive': all(
            row['num_safe_reject'] + row['num_coverage_critical'] ==
            row['num_classification_valid'] for row in reports),
    }
    gate['passed'] = bool(all(gate.values()))
    return aggregate, gate


def format_markdown(report):
    settings = report['settings']
    aggregate = report['aggregate']
    lines = [
        '# Q1 覆盖保护型实例质量 Oracle', '',
        '- 数据：固定 13 个 train forests 与 5 个 validation forests。',
        '- 未读取 Wytham；未训练网络；未修改实例预测。',
        f'- 固定最大拒绝比例：{100 * settings["reject_ratio"]:.1f}%', '',
        '## 标签定义', '',
        '- `coverage_critical`：每棵已匹配 GT 树中 IoU 最高的唯一代表实例。',
        '- `safe_reject`：classification-valid 假阳性或冗余正实例。',
        '- `protected_unknown`：边界、低标注覆盖或 IoU 模糊实例，Oracle 不删除。', '',
        '## Split 汇总', '',
        '| Split | Instances | Safe reject | Coverage critical | Unknown | Baseline F1 | Oracle F1 | F1 gain | Commission reduction | Completeness drop |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for split in ('train', 'validation'):
        row = aggregate[split]
        lines.append(
            f'| {split} | {row["num_instances"]:,} | '
            f'{row["num_safe_reject"]:,} | '
            f'{row["num_coverage_critical"]:,} | '
            f'{row["num_protected_unknown"]:,} | '
            f'{100 * row["baseline"]["f1"]:.3f}% | '
            f'{100 * row["filtered"]["f1"]:.3f}% | '
            f'{row["f1_gain_pp"]:+.3f} pp | '
            f'{row["commission_reduction_pp"]:+.3f} pp | '
            f'{row["completeness_drop_pp"]:+.3f} pp |')
    lines.extend(['', '## Validation 每森林', '',
                  '| Plot | Instances | Safe reject | Critical | F1 gain | Commission reduction | Completeness drop |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for row in report['plots']:
        if row['split'] != 'validation':
            continue
        oracle = row['oracle']
        lines.append(
            f'| {row["source_plot"]} | {row["num_instances"]:,} | '
            f'{row["num_safe_reject"]:,} | '
            f'{row["num_coverage_critical"]:,} | '
            f'{oracle["f1_gain_pp"]:+.3f} pp | '
            f'{oracle["commission_reduction_pp"]:+.3f} pp | '
            f'{oracle["completeness_drop_pp"]:+.3f} pp |')
    lines.extend(['', '## Gate', ''])
    for name, passed in report['gate'].items():
        lines.append(f'- {name}: **{bool(passed)}**')
    lines.extend(['', (
        'PASS：进入 Q2 双头 MLP 控制实验；先验证覆盖保护目标，不实现注意力。'
        if report['gate']['passed'] else
        'STOP：双风险标签或拒绝上限不足，不进入 Q2。')])
    return '\n'.join(lines) + '\n'


def write_label_manifest(path, groups):
    fields = [
        'source_plot', 'split', 'instance_id', 'safe_reject',
        'coverage_critical', 'counted_negative', 'redundant_positive',
        'protected_unknown']
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            targets = group['targets']
            for index, instance_id in enumerate(group['instance_ids']):
                writer.writerow({
                    'source_plot': group['source_plot'],
                    'split': group['split'],
                    'instance_id': int(instance_id),
                    'safe_reject': bool(targets['safe_reject'][index]),
                    'coverage_critical': bool(
                        targets['coverage_critical'][index]),
                    'counted_negative': bool(
                        targets['counted_negative'][index]),
                    'redundant_positive': bool(
                        targets['redundant_positive'][index]),
                    'protected_unknown': bool(
                        targets['protected_unknown'][index]),
                })


def run_settings(raw, config_path='<memory>'):
    required = {
        'data_root', 'manifest_path', 'output_dir', 'expected_splits',
        'reject_ratio', 'gate'}
    missing = required - set(raw)
    if missing:
        raise ValueError(f'Missing Q1 config fields: {sorted(missing)}')
    settings = {
        'reject_ratio': float(raw['reject_ratio']),
        'gate': raw['gate'],
    }
    if not np.isclose(settings['reject_ratio'], 0.15):
        raise ValueError('Q1 reject ratio is preregistered as 0.15.')
    expected_splits = {
        split: list(plots) for split, plots in raw['expected_splits'].items()}
    if set(expected_splits) != {'train', 'validation'}:
        raise ValueError('Q1 expected_splits must be train/validation only.')
    groups = load_artifact_groups(
        raw['data_root'], raw['manifest_path'], expected_splits)
    reports = [
        analyze_group(group, settings['reject_ratio']) for group in groups]
    aggregate, gate = aggregate_reports(reports, settings)
    report = {
        'config': str(config_path),
        'settings': {'reject_ratio': settings['reject_ratio']},
        'plots': reports,
        'aggregate': aggregate,
        'gate': gate,
    }
    output_dir = Path(raw['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(report)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    write_label_manifest(output_dir / 'dual_risk_labels.csv', groups)
    print(markdown, flush=True)
    if not gate['passed']:
        raise RuntimeError(
            'Q1 coverage-preserving quality gate failed; stop before Q2.')
    return report


def run(config_path):
    import yaml

    with Path(config_path).open(encoding='utf-8') as file:
        raw = yaml.safe_load(file)
    return run_settings(raw, config_path=config_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Q1 coverage-preserving quality Oracle.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
