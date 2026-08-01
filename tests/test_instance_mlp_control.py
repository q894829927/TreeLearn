import importlib.util
import pathlib
import types
import unittest

import numpy as np


HAS_DEPENDENCIES = (
    importlib.util.find_spec('pandas') is not None and
    importlib.util.find_spec('sklearn') is not None)
if HAS_DEPENDENCIES:
    import pandas as pd

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location(
        'instance_mlp_control_standalone',
        repo_root / 'tools' / 'diagnostics' /
        'evaluate_instance_mlp_control.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)


@unittest.skipUnless(
    HAS_DEPENDENCIES,
    'pandas and scikit-learn are required for the MLP control')
class InstanceMlpControlTests(unittest.TestCase):

    def test_nested_single_feature_uses_training_fold_only(self):
        train_targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        train_values = np.column_stack([
            np.arange(8, dtype=float),
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),
        ])
        test_targets = np.array([0, 0, 1, 1])
        test_values = np.array([
            [0.5, 1.0], [1.5, 0.0], [5.5, 0.0], [6.5, 1.0],
        ])

        result = module.evaluate_nested_single_feature(
            train_values, train_targets, test_values, test_targets,
            ['strong_feature', 'noise'])

        self.assertEqual(result['selected_feature'], 'strong_feature')
        self.assertAlmostEqual(result['roc_auc'], 1.0)

    def test_summary_gate_requires_mlp_gain(self):
        rows = []
        values = {
            'nested_single_feature': [0.85, 0.87],
            'logistic_regression': [0.90, 0.91],
            'instance_mlp': [0.91, 0.93],
        }
        for model_name, auc_values in values.items():
            for fold, auc in enumerate(auc_values, start=1):
                rows.append({
                    'model': model_name,
                    'selected_feature': (
                        'confidence_mean'
                        if model_name == 'nested_single_feature' else ''),
                    'roc_auc': auc,
                    'tree_average_precision': auc,
                    'fp_average_precision': auc,
                    'seed': 42,
                    'fold': fold,
                })
        fold_results = pd.DataFrame(rows)
        frame = pd.DataFrame({
            'target_is_true_tree': [1, 1, 0, 0],
        })
        args = types.SimpleNamespace(
            features='features.csv', seeds=[42], folds=2,
            min_mlp_auc=0.88, min_gain_over_single=0.02,
            logistic_tolerance=0.005)

        summary = module.summarize_results(
            fold_results, args, frame, ['confidence_mean'])

        self.assertTrue(summary['gate']['passed'])
        self.assertAlmostEqual(
            summary['comparisons']['mlp_minus_nested_single_auc'], 0.06)


if __name__ == '__main__':
    unittest.main()
