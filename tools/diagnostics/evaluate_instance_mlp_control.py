import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDED_COLUMNS = {
    'instance_id', 'target_is_true_tree', 'target_status',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Repeated nested-fold control for instance reliability.')
    parser.add_argument('--features', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--min_mlp_auc', type=float, default=0.88)
    parser.add_argument('--min_gain_over_single', type=float, default=0.02)
    parser.add_argument('--logistic_tolerance', type=float, default=0.005)
    return parser.parse_args()


def load_supervised_features(path):
    frame = pd.read_csv(path)
    required = {'target_is_true_tree', 'target_status'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f'Labeled feature table is missing columns: {sorted(missing)}.')
    frame = frame[frame['target_status'].isin(['tp', 'fp_counted'])].copy()
    frame['target_is_true_tree'] = frame[
        'target_is_true_tree'].astype(np.int64)
    if frame['target_is_true_tree'].nunique() != 2:
        raise ValueError('Both TP and counted FP samples are required.')
    feature_names = [
        column for column in frame.columns
        if column not in EXCLUDED_COLUMNS and
        pd.api.types.is_numeric_dtype(frame[column])]
    if not feature_names:
        raise ValueError('No numeric instance features are available.')
    values = frame[feature_names].to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    targets = frame['target_is_true_tree'].to_numpy(dtype=np.int64)
    return frame, values, targets, feature_names


def _median_impute(train_values, test_values):
    medians = np.nanmedian(train_values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train = np.where(np.isfinite(train_values), train_values, medians)
    test = np.where(np.isfinite(test_values), test_values, medians)
    return train, test


def evaluate_nested_single_feature(
        train_values, train_targets, test_values, test_targets,
        feature_names):
    train_values, test_values = _median_impute(train_values, test_values)
    best = None
    for feature_index, feature_name in enumerate(feature_names):
        values = train_values[:, feature_index]
        if np.all(values == values[0]):
            continue
        raw_auc = float(roc_auc_score(train_targets, values))
        direction = 1.0 if raw_auc >= 0.5 else -1.0
        separability = max(raw_auc, 1.0 - raw_auc)
        candidate = (separability, feature_name, feature_index, direction)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError('Every candidate feature is constant in a fold.')
    _, feature_name, feature_index, direction = best
    test_scores = direction * test_values[:, feature_index]
    return {
        'model': 'nested_single_feature',
        'selected_feature': feature_name,
        'roc_auc': float(roc_auc_score(test_targets, test_scores)),
        'tree_average_precision': float(average_precision_score(
            test_targets, test_scores)),
        'fp_average_precision': float(average_precision_score(
            1 - test_targets, -test_scores)),
    }


def build_models(random_state):
    def preprocessing():
        return [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]

    return {
        'logistic_regression': Pipeline(preprocessing() + [
            ('classifier', LogisticRegression(
                class_weight='balanced', max_iter=2000,
                random_state=random_state)),
        ]),
        'instance_mlp': Pipeline(preprocessing() + [
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(32, 16), activation='relu',
                alpha=1e-3, batch_size=64, learning_rate_init=1e-3,
                max_iter=500, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=30,
                random_state=random_state)),
        ]),
    }


def evaluate_model(model_name, model, train_values, train_targets,
                   test_values, test_targets):
    model.fit(train_values, train_targets)
    class_index = list(
        model.named_steps['classifier'].classes_).index(1)
    probabilities = model.predict_proba(test_values)[:, class_index]
    return {
        'model': model_name,
        'selected_feature': '',
        'roc_auc': float(roc_auc_score(test_targets, probabilities)),
        'tree_average_precision': float(average_precision_score(
            test_targets, probabilities)),
        'fp_average_precision': float(average_precision_score(
            1 - test_targets, 1.0 - probabilities)),
    }


def run_repeated_folds(values, targets, feature_names, seeds, folds):
    minority_count = int(np.bincount(targets).min())
    if folds < 2 or folds > minority_count:
        raise ValueError(
            f'folds must be in [2, {minority_count}], received {folds}.')
    rows = []
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_indices, test_indices) in enumerate(
                splitter.split(values, targets), start=1):
            train_values = values[train_indices]
            test_values = values[test_indices]
            train_targets = targets[train_indices]
            test_targets = targets[test_indices]
            result = evaluate_nested_single_feature(
                train_values, train_targets, test_values, test_targets,
                feature_names)
            result.update({'seed': seed, 'fold': fold})
            rows.append(result)
            for model_name, model in build_models(seed * 100 + fold).items():
                result = evaluate_model(
                    model_name, model, train_values, train_targets,
                    test_values, test_targets)
                result.update({'seed': seed, 'fold': fold})
                rows.append(result)
    return pd.DataFrame(rows)


def summarize_results(fold_results, args, frame, feature_names):
    metrics = [
        'roc_auc', 'tree_average_precision', 'fp_average_precision']
    models = {}
    for model_name, group in fold_results.groupby('model'):
        model_summary = {'num_folds': int(len(group))}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            model_summary[metric] = {
                'mean': float(values.mean()),
                'sample_std': float(values.std(ddof=1)),
            }
        if model_name == 'nested_single_feature':
            model_summary['selection_counts'] = dict(Counter(
                group['selected_feature'].tolist()))
        models[model_name] = model_summary

    mlp_auc = models['instance_mlp']['roc_auc']['mean']
    logistic_auc = models['logistic_regression']['roc_auc']['mean']
    single_auc = models['nested_single_feature']['roc_auc']['mean']
    gate = {
        'mlp_auc_at_least_threshold': mlp_auc >= args.min_mlp_auc,
        'mlp_gain_over_nested_single_passed': (
            mlp_auc - single_auc >= args.min_gain_over_single),
        'mlp_not_worse_than_logistic': (
            mlp_auc + args.logistic_tolerance >= logistic_auc),
    }
    gate['passed'] = all(gate.values())
    return {
        'features_path': str(args.features),
        'num_supervised_instances': int(len(frame)),
        'num_true_instances': int((
            frame['target_is_true_tree'] == 1).sum()),
        'num_false_instances': int((
            frame['target_is_true_tree'] == 0).sum()),
        'num_features': int(len(feature_names)),
        'feature_names': feature_names,
        'seeds': args.seeds,
        'folds_per_seed': args.folds,
        'models': models,
        'comparisons': {
            'mlp_minus_nested_single_auc': mlp_auc - single_auc,
            'mlp_minus_logistic_auc': mlp_auc - logistic_auc,
        },
        'thresholds': {
            'min_mlp_auc': args.min_mlp_auc,
            'min_gain_over_single': args.min_gain_over_single,
            'logistic_tolerance': args.logistic_tolerance,
        },
        'gate': gate,
    }


def format_markdown(summary):
    rows = [
        '# 实例可靠性 MLP 控制实验', '',
        f"- 有监督实例：{summary['num_supervised_instances']}",
        f"- TP：{summary['num_true_instances']}",
        f"- FP：{summary['num_false_instances']}",
        f"- 特征数：{summary['num_features']}",
        f"- 重复分层折数：{len(summary['seeds'])} × "
        f"{summary['folds_per_seed']}", '',
        '| 模型 | ROC-AUC | TP AP | FP AP |',
        '|---|---:|---:|---:|']
    order = [
        'nested_single_feature', 'logistic_regression', 'instance_mlp']
    for model_name in order:
        result = summary['models'][model_name]
        rows.append(
            f"| {model_name} | "
            f"{result['roc_auc']['mean']:.6f} ± "
            f"{result['roc_auc']['sample_std']:.6f} | "
            f"{result['tree_average_precision']['mean']:.6f} ± "
            f"{result['tree_average_precision']['sample_std']:.6f} | "
            f"{result['fp_average_precision']['mean']:.6f} ± "
            f"{result['fp_average_precision']['sample_std']:.6f} |")
    rows.extend(['', '## Gate', ''])
    for name, passed in summary['gate'].items():
        rows.append(f'- {name}: **{passed}**')
    rows.extend([
        '', '## 单特征在训练折中的选择次数', '',
        '```json',
        json.dumps(
            summary['models']['nested_single_feature']['selection_counts'],
            indent=2, ensure_ascii=False),
        '```', ''])
    return '\n'.join(rows)


def main():
    args = parse_args()
    frame, values, targets, feature_names = load_supervised_features(
        args.features)
    fold_results = run_repeated_folds(
        values, targets, feature_names, args.seeds, args.folds)
    summary = summarize_results(
        fold_results, args, frame, feature_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(output_dir / 'fold_results.csv', index=False)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    markdown = format_markdown(summary)
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    if summary['gate']['passed']:
        print('PASS: proceed to the attention-pooling control experiment.')
    else:
        print('STOP: aggregate-feature MLP does not justify attention yet.')


if __name__ == '__main__':
    main()
