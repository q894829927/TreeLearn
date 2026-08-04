import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'diagnostics' / 'evaluate_quality_statistical_evidence.py')
SPEC = importlib.util.spec_from_file_location(
    'quality_statistical_evidence_standalone', MODULE_PATH)
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


class QualityStatisticalEvidenceTests(unittest.TestCase):

    @staticmethod
    def records(scores):
        values = {}
        for seed in (42, 43):
            for plot in ('A', 'B'):
                for index, (score, target) in enumerate(zip(
                        scores, (True, True, False, False)), start=1):
                    key = (seed, plot, index)
                    values[key] = {
                        'seed': seed,
                        'source_plot': plot,
                        'instance_id': index,
                        'score': score,
                        'target_true': target,
                        'classification_valid': True,
                    }
        return values

    def test_alignment_rejects_different_targets(self):
        first = self.records([0.9, 0.8, 0.2, 0.1])
        second = self.records([0.9, 0.8, 0.2, 0.1])
        second[(42, 'A', 1)]['target_true'] = False
        with self.assertRaisesRegex(ValueError, 'targets differ'):
            EVIDENCE.align_validation_predictions(first, second)

    def test_good_vertical_ranking_beats_bad_global_for_each_seed(self):
        global_records = self.records([0.2, 0.1, 0.9, 0.8])
        vertical_records = self.records([0.9, 0.8, 0.2, 0.1])
        ratios = [0.5, 0.75, 1.0]
        global_curves = EVIDENCE.build_group_curves(
            global_records, ratios)
        vertical_curves = EVIDENCE.build_group_curves(
            vertical_records, ratios)
        result = EVIDENCE.validation_bootstrap(
            global_curves, vertical_curves, ratios,
            seeds=[42, 43], plots=['A', 'B'],
            repeats=50, random_seed=1)
        self.assertGreater(
            result['point_delta']['commission_area_reduction_pp'], 0)
        self.assertGreater(result['point_delta']['f1_area_gain_pp'], 0)
        self.assertTrue(all(
            row['f1_area_gain_pp'] > 0 for row in result['seed_rows']))

    def test_paired_bootstrap_is_reproducible(self):
        records = self.records([0.9, 0.8, 0.2, 0.1])
        ratios = [0.5, 1.0]
        curves = EVIDENCE.build_group_curves(records, ratios)
        first = EVIDENCE.validation_bootstrap(
            curves, curves, ratios, [42, 43], ['A', 'B'], 10, 7)
        second = EVIDENCE.validation_bootstrap(
            curves, curves, ratios, [42, 43], ['A', 'B'], 10, 7)
        self.assertEqual(first['bootstrap_rows'], second['bootstrap_rows'])

    def test_external_randomization_detects_strong_ordering(self):
        status_data = {
            'instance_ids': np.arange(20),
            'scores': np.r_[np.linspace(1.0, 0.6, 10),
                            np.linspace(0.4, 0.0, 10)],
            'status': np.r_[np.ones(10), -np.ones(10)].astype(np.int8),
            'num_gt': 10,
            'baseline_tp': 10,
            'baseline_fp': 10,
        }
        result = EVIDENCE.external_randomization(
            status_data, [0.5, 0.75, 1.0], 500, 2)
        self.assertTrue(result['baseline_counts_reproduced'])
        self.assertLess(result['commission_one_sided_p'], 0.01)
        self.assertLess(result['f1_one_sided_p'], 0.01)


if __name__ == '__main__':
    unittest.main()
