import unittest

from tools.diagnostics.summarize_sparse_unet_locked_evaluation import (
    evaluate_gate,
)


class SparseUNetLockedGateTests(unittest.TestCase):

    def test_l1w_safety_gate(self):
        gate = evaluate_gate('l1w', {
            'f1_score': 98.0,
            'completeness': 100.0,
            'coverage': 97.8,
        })
        self.assertTrue(gate['passed'])

    def test_wytham_accepts_f1_or_safe_coverage_effect(self):
        f1_gate = evaluate_gate('wytham', {
            'f1_score': 72.6,
            'coverage': 57.7,
            'precision': 61.0,
        })
        coverage_gate = evaluate_gate('wytham', {
            'f1_score': 72.1,
            'coverage': 58.8,
            'precision': 61.6,
        })
        self.assertTrue(f1_gate['passed'])
        self.assertTrue(coverage_gate['passed'])

    def test_wytham_rejects_no_effect(self):
        gate = evaluate_gate('wytham', {
            'f1_score': 72.2,
            'coverage': 58.0,
            'precision': 62.5,
        })
        self.assertFalse(gate['passed'])


if __name__ == '__main__':
    unittest.main()
