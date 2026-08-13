import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).parents[1] / 'tree_learn' / 'model' /
    'height_context_attention.py')
SPEC = importlib.util.spec_from_file_location(
    'height_context_attention_gate_standalone', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HeightContextGateOutputTests(unittest.TestCase):

    def test_gate_is_exposed_and_bounded(self):
        adapter = MODULE.HeightStratifiedContextAdapter(
            in_channels=4, hidden_dim=8, num_height_bins=4,
            adapter_type='attention', num_heads=2, dropout=0.0).eval()
        output = adapter(
            torch.randn(12, 4),
            torch.randn(12, 3),
            torch.rand(12, 1),
            torch.tensor([0] * 6 + [1] * 6))
        gate = output['context_gate']
        self.assertEqual(tuple(gate.shape), (12, 8))
        self.assertTrue(torch.isfinite(gate).all())
        self.assertGreaterEqual(float(gate.min()), 0.0)
        self.assertLessEqual(float(gate.max()), 1.0)


if __name__ == '__main__':
    unittest.main()
