import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / 'tree_learn' / 'util' /
    'height_context_diagnostics.py')
SPEC = importlib.util.spec_from_file_location(
    'height_context_diagnostics_standalone', MODULE_PATH)
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


class HeightContextDiagnosticsTests(unittest.TestCase):

    def test_seed_and_flip_accounting(self):
        metrics = DIAGNOSTICS.make_point_metrics(
            base_probability=np.array([0.9, 0.6, 0.4, 0.2]),
            adapted_probability=np.array([0.8, 0.4, 0.6, 0.2]),
            offset_residual=np.array([
                [0.3, 0.4, 0.0], [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            gate=np.full((4, 2), 0.5),
            verticality=np.array([0.8, 0.8, 0.8, 0.8]),
            base_offset=np.zeros((4, 3)),
            adapted_offset=np.zeros((4, 3)))
        accumulator = DIAGNOSTICS.DriftAccumulator()
        accumulator.update(metrics)
        summary = accumulator.summary()
        self.assertEqual(summary['tree_to_non_tree'], 1)
        self.assertEqual(summary['non_tree_to_tree'], 1)
        self.assertEqual(summary['seed_removed'], 1)
        self.assertEqual(summary['seed_added'], 1)
        self.assertAlmostEqual(summary['offset_xy_residual_mean'], 0.125)

    def test_stratified_mask_and_quantiles(self):
        metrics = DIAGNOSTICS.make_point_metrics(
            base_probability=np.array([0.9, 0.9, 0.9]),
            adapted_probability=np.array([0.8, 0.7, 0.6]),
            offset_residual=np.array([
                [0.1, 0.0, 0.1], [0.2, 0.0, 0.2], [0.3, 0.0, 0.3]]),
            gate=np.array([0.1, 0.2, 0.3]),
            verticality=np.ones(3),
            base_offset=np.zeros((3, 3)),
            adapted_offset=np.zeros((3, 3)))
        accumulator = DIAGNOSTICS.DriftAccumulator()
        accumulator.update(metrics, np.array([False, True, True]))
        summary = accumulator.summary()
        self.assertEqual(summary['points'], 2)
        self.assertAlmostEqual(summary['probability_delta_mean'], -0.25)
        self.assertGreater(summary['offset_xy_residual_p90'], 0.2)


if __name__ == '__main__':
    unittest.main()
