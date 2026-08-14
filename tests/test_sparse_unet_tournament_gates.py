import tempfile
import unittest
from pathlib import Path

from tools.diagnostics.summarize_sparse_unet_tournament import (
    t1,
    t2,
    t3,
)


def row(model, seed, plot, f1, completeness=95.0, commission=10.0,
        training_log='', parameter_count=''):
    return {
        'model': model,
        'seed': seed,
        'plot': plot,
        'f1_score': f1,
        'completeness': completeness,
        'commission_error_rate': commission,
        'precision': 90.0,
        'recall': 90.0,
        'coverage': 80.0,
        'training_log': training_log,
        'parameter_count': parameter_count,
    }


class SparseUNetTournamentGateTests(unittest.TestCase):

    def test_t1_selects_only_attention_with_nonzero_variance(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'train.log'
            log.write_text(
                'U-Net attention stats: level0_attention_std=0.125\n',
                encoding='utf-8')
            rows = []
            for index in range(5):
                plot = f'P{index}'
                rows.append(row('b1_partial', 42, plot, 80.0))
                rows.append(row(
                    'm1_sparse_se', 42, plot, 80.3,
                    training_log=str(log)))
            report = t1(rows, 'b1_partial')
            self.assertTrue(report['passed'])
            self.assertEqual(report['finalists'], ['m1_sparse_se'])

    def test_t1_rejects_constant_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'train.log'
            log.write_text(
                'U-Net attention stats: level0_attention_std=0.0\n',
                encoding='utf-8')
            rows = []
            for index in range(5):
                plot = f'P{index}'
                rows.append(row('b1_partial', 42, plot, 80.0))
                rows.append(row(
                    'm1_sparse_se', 42, plot, 81.0,
                    training_log=str(log)))
            report = t1(rows, 'b1_partial')
            self.assertFalse(report['passed'])

    def test_t2_requires_three_stable_seed_wins(self):
        rows = []
        for seed in (42, 43, 44):
            for index in range(5):
                plot = f'P{index}'
                rows.append(row('b1_partial', seed, plot, 80.0))
                rows.append(row(
                    'm3_hcag', seed, plot, 80.6,
                    completeness=95.0, commission=9.5))
        report = t2(rows, 'b1_partial')
        self.assertTrue(report['passed'])
        self.assertEqual(report['winner'], 'm3_hcag')

    def test_t3_checks_parameter_match(self):
        rows = []
        for seed in (42, 43, 44):
            for index in range(5):
                plot = f'P{index}'
                rows.append(row(
                    'b2_adapter', seed, plot, 80.0,
                    parameter_count='1000'))
                rows.append(row(
                    'm3_hcag', seed, plot, 80.4,
                    parameter_count='1020'))
        report = t3(rows, 'b2_adapter', 'm3_hcag')
        self.assertTrue(report['passed'])


if __name__ == '__main__':
    unittest.main()
