import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


NON_FEATURE_COLUMNS = {'instance_id', 'target_is_true_tree', 'target_status'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure TP/FP separability of predicted-tree features.')
    parser.add_argument('--features', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--auc_threshold', type=float, default=0.75)
    parser.add_argument('--min_false_positives', type=int, default=20)
    return parser.parse_args()


def load_evaluation(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def attach_detection_targets(features, evaluation):
    detection = evaluation['detection_results']
    matched = set(np.asarray(detection['matched_preds'], dtype=np.int64).tolist())
    counted_false = set(np.asarray(
        detection['non_matched_preds_filtered'], dtype=np.int64).tolist())
    all_unmatched = set(np.asarray(
        detection['non_matched_preds'], dtype=np.int64).tolist())
    ignored_unmatched = all_unmatched - counted_false

    frame = features.copy()
    ids = set(frame['instance_id'].astype(np.int64).tolist())
    referenced = matched | counted_false | ignored_unmatched
    missing = referenced - ids
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            'Evaluation references predicted labels missing from the feature '
            f'table: {preview} (total missing: {len(missing)}).')

    def status(instance_id):
        instance_id = int(instance_id)
        if instance_id in matched:
            return 'tp'
        if instance_id in counted_false:
            return 'fp_counted'
        if instance_id in ignored_unmatched:
            return 'unmatched_ignored'
        return 'not_referenced'

    frame['target_status'] = frame['instance_id'].map(status)
    frame['target_is_true_tree'] = np.where(
        frame['target_status'] == 'tp', 1,
        np.where(frame['target_status'] == 'fp_counted', 0, np.nan))
    return frame


def calculate_feature_auc(frame):
    supervised = frame[frame['target_is_true_tree'].notna()].copy()
    supervised['target_is_true_tree'] = supervised[
        'target_is_true_tree'].astype(np.int64)
    rows = []
    for feature in supervised.columns:
        if feature in NON_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(supervised[feature]):
            continue
        finite = np.isfinite(supervised[feature].to_numpy(dtype=float))
        subset = supervised.loc[finite]
        if subset['target_is_true_tree'].nunique() != 2:
            continue
        values = subset[feature].to_numpy(dtype=float)
        targets = subset['target_is_true_tree'].to_numpy(dtype=np.int64)
        raw_auc = float(roc_auc_score(targets, values))
        higher_for_tp = raw_auc >= 0.5
        separability_auc = raw_auc if higher_for_tp else 1.0 - raw_auc
        tp_values = values[targets == 1]
        fp_values = values[targets == 0]
        rows.append({
            'feature': feature,
            'auc': raw_auc,
            'separability_auc': separability_auc,
            'higher_for_tp': bool(higher_for_tp),
            'num_tp': int(len(tp_values)),
            'num_fp': int(len(fp_values)),
            'tp_median': float(np.median(tp_values)),
            'fp_median': float(np.median(fp_values)),
            'tp_mean': float(np.mean(tp_values)),
            'fp_mean': float(np.mean(fp_values)),
        })
    columns = [
        'feature', 'auc', 'separability_auc', 'higher_for_tp',
        'num_tp', 'num_fp', 'tp_median', 'fp_median',
        'tp_mean', 'fp_mean']
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        'separability_auc', ascending=False).reset_index(drop=True)


def _serializable_row(row):
    output = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        output[key] = value
    return output


def build_summary(labeled, auc_table, args):
    counts = labeled['target_status'].value_counts().to_dict()
    confidence_rows = auc_table[
        auc_table['feature'].str.startswith(('confidence_', 'seed_confidence_'))]
    best_confidence = (
        _serializable_row(confidence_rows.iloc[0].to_dict())
        if len(confidence_rows) else None)
    best_overall = (
        _serializable_row(auc_table.iloc[0].to_dict())
        if len(auc_table) else None)
    num_false_positives = int(counts.get('fp_counted', 0))
    gate = {
        'enough_false_positives': (
            num_false_positives >= args.min_false_positives),
        'confidence_auc_passed': (
            best_confidence is not None and
            best_confidence['separability_auc'] >= args.auc_threshold),
    }
    gate['passed'] = all(gate.values())
    return {
        'features_path': str(args.features),
        'evaluation_path': str(args.evaluation),
        'status_counts': {key: int(value) for key, value in counts.items()},
        'best_confidence_feature': best_confidence,
        'best_overall_feature': best_overall,
        'gate': gate,
        'thresholds': {
            'auc_threshold': args.auc_threshold,
            'min_false_positives': args.min_false_positives,
        },
    }


def format_markdown(summary, auc_table):
    rows = [
        '# Predicted-instance TP/FP separability', '',
        '## Counts', '',
        '| Status | Count |', '|---|---:|']
    for status, count in summary['status_counts'].items():
        rows.append(f'| {status} | {count} |')
    rows.extend([
        '', '## Best features', '',
        '| Feature | Separability AUC | Direction | TP median | FP median |',
        '|---|---:|---|---:|---:|'])
    for _, result in auc_table.head(15).iterrows():
        direction = 'higher for TP' if result['higher_for_tp'] else 'lower for TP'
        rows.append(
            f"| {result['feature']} | {result['separability_auc']:.6f} | "
            f"{direction} | {result['tp_median']:.6f} | "
            f"{result['fp_median']:.6f} |")
    rows.extend(['', '## Gate', ''])
    for name, passed in summary['gate'].items():
        rows.append(f'- {name}: **{passed}**')
    rows.append('')
    return '\n'.join(rows)


def main():
    args = parse_args()
    features = pd.read_csv(args.features)
    evaluation = load_evaluation(args.evaluation)
    labeled = attach_detection_targets(features, evaluation)
    auc_table = calculate_feature_auc(labeled)
    summary = build_summary(labeled, auc_table, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_dir / 'labeled_instance_features.csv', index=False)
    auc_table.to_csv(output_dir / 'feature_auc.csv', index=False)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    markdown = format_markdown(summary, auc_table)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    if summary['gate']['passed']:
        print('PASS: confidence features can support instance validation.')
    else:
        print('STOP: do not implement instance attention with current features.')


if __name__ == '__main__':
    main()
