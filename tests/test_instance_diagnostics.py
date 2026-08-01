import importlib.util
import pathlib
import unittest

import numpy as np


HAS_DIAGNOSTIC_DEPENDENCIES = (
    importlib.util.find_spec('pandas') is not None and
    importlib.util.find_spec('sklearn') is not None)
if HAS_DIAGNOSTIC_DEPENDENCIES:
    import pandas as pd

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    feature_spec = importlib.util.spec_from_file_location(
        'instance_diagnostics_standalone',
        repo_root / 'tree_learn' / 'util' / 'instance_diagnostics.py')
    feature_module = importlib.util.module_from_spec(feature_spec)
    feature_spec.loader.exec_module(feature_module)
    compute_instance_features = feature_module.compute_instance_features

    diagnostic_spec = importlib.util.spec_from_file_location(
        'diagnose_instance_separability_standalone',
        repo_root / 'tools' / 'diagnostics' /
        'diagnose_instance_separability.py')
    diagnostic_module = importlib.util.module_from_spec(diagnostic_spec)
    diagnostic_spec.loader.exec_module(diagnostic_module)
    attach_detection_targets = diagnostic_module.attach_detection_targets
    calculate_feature_auc = diagnostic_module.calculate_feature_auc


@unittest.skipUnless(
    HAS_DIAGNOSTIC_DEPENDENCIES,
    'pandas and scikit-learn are required for instance diagnostics')
class InstanceFeatureTests(unittest.TestCase):

    def test_aggregates_compact_features_per_instance(self):
        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0], [1.0, 1.0, 3.0],
            [10.0, 0.0, 0.0], [12.0, 0.0, 2.0],
            [10.0, 2.0, 4.0], [12.0, 2.0, 6.0],
        ])
        predictions = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        initial = np.array([1, 1, -1, -1, 2, 2, -1, -1])
        logits = np.array([[3.0, 0.0]] * 4 + [[1.0, 0.0]] * 4)
        offsets = np.zeros((8, 3), dtype=float)
        verticality = np.array([0.9] * 4 + [0.3] * 4)
        confidence = np.array([0.9] * 4 + [0.2] * 4)

        frame = compute_instance_features(
            coords, predictions, initial, logits, offsets,
            verticality, confidence)

        self.assertEqual(frame['instance_id'].tolist(), [1, 2])
        self.assertEqual(frame['num_points'].tolist(), [4, 4])
        self.assertEqual(frame['num_clustered_seeds'].tolist(), [2, 2])
        self.assertAlmostEqual(frame.loc[0, 'height'], 3.0)
        self.assertAlmostEqual(frame.loc[1, 'height'], 6.0)
        self.assertGreater(
            frame.loc[0, 'confidence_mean'],
            frame.loc[1, 'confidence_mean'])
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())

    def test_attaches_tp_fp_and_calculates_auc(self):
        features = pd.DataFrame({
            'instance_id': [1, 2, 3, 4],
            'confidence_mean': [0.9, 0.8, 0.2, 0.1],
            'height': [10.0, 9.0, 2.0, 1.0],
        })
        evaluation = {
            'detection_results': {
                'matched_preds': np.array([1, 2]),
                'non_matched_preds_filtered': np.array([3, 4]),
                'non_matched_preds': np.array([3, 4]),
            },
        }
        labeled = attach_detection_targets(features, evaluation)
        auc_table = calculate_feature_auc(labeled)
        confidence_auc = auc_table.loc[
            auc_table['feature'] == 'confidence_mean',
            'separability_auc'].item()
        self.assertAlmostEqual(confidence_auc, 1.0)
        self.assertEqual(
            labeled['target_status'].tolist(),
            ['tp', 'tp', 'fp_counted', 'fp_counted'])


if __name__ == '__main__':
    unittest.main()
