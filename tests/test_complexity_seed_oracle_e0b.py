import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_complexity_seed_oracle_e0b.py')
SPEC = importlib.util.spec_from_file_location(
    'complexity_seed_oracle_e0b_standalone', MODULE_PATH)
E0B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E0B)


class ComplexitySeedOracleE0BTests(unittest.TestCase):

    def setUp(self):
        self.coords = np.asarray([
            [0.05, 0.05, 0.0],
            [0.10, 0.10, 0.0],
            [0.85, 0.05, 0.0],
            [2.05, 0.05, 0.0],
            [2.10, 0.10, 0.0],
            [4.00, 4.00, 0.0],
        ])
        self.offsets = np.zeros_like(self.coords)
        self.labels = np.asarray([1, 1, 1, 2, 2, -1])
        self.candidates = np.ones(6, dtype=bool)
        self.utility = np.asarray([0.1, 0.9, 0.2, 0.8, 0.7, 0.0])

    def test_label_composition_distinguishes_unknown_and_non_tree(self):
        labels = np.asarray([1, 0, -1, 2, -1])
        result = E0B.label_composition(np.ones(5, dtype=bool), labels)
        self.assertEqual(result, {
            'total': 5, 'tree': 2, 'non_tree': 1, 'unknown': 2,
        })

    def test_spatial_mandatory_preserves_cells_and_tree_quota(self):
        mandatory = E0B.spatial_mandatory_mask(
            self.coords, self.offsets, self.labels, self.candidates,
            self.utility, voxel_size=0.6, min_seeds_per_tree=2)
        np.testing.assert_array_equal(
            np.flatnonzero(mandatory), np.asarray([1, 2, 3, 4]))
        coverage = E0B.spatial_coverage(
            mandatory, self.coords, self.offsets, self.labels,
            self.candidates, voxel_size=0.6)
        self.assertEqual(coverage['coverage'], 1.0)

    def test_rank_scores_prioritize_unknown_then_mandatory(self):
        protected = np.asarray([False, False, False, False, False, True])
        mandatory = np.asarray([False, True, True, True, True, False])
        scores = E0B.compose_rank_scores(
            self.utility, self.candidates, protected, mandatory)
        self.assertEqual(float(scores[5]), 3.0)
        np.testing.assert_array_equal(scores[mandatory], 2.0)
        self.assertTrue(np.all(scores[~(protected | mandatory)] <= 1.0))

    def test_random_scores_are_deterministic_and_seed_dependent(self):
        protected = self.labels < 0
        first = E0B.compose_rank_scores(
            self.utility, self.candidates, protected, random_seed=42)
        repeated = E0B.compose_rank_scores(
            self.utility, self.candidates, protected, random_seed=42)
        different = E0B.compose_rank_scores(
            self.utility, self.candidates, protected, random_seed=43)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))

    def test_selection_plan_keeps_exact_budget_and_constraints(self):
        arrays = {
            'coords': self.coords,
            'offset_predictions': self.offsets,
            'instance_labels': self.labels,
        }
        plan = E0B.build_selection_plan(
            arrays, self.candidates, self.utility, {
                'known_label_min': 0,
                'spatial_vote_cell_size': 0.6,
                'min_seeds_per_tree': 2,
                'keep_ratio': 5 / 6,
                'random_seeds': [42, 43, 44],
            })
        for name, mask in plan['masks'].items():
            self.assertEqual(int(mask.sum()), 5, name)
            self.assertTrue(mask[5], name)
            if name != 'oracle_masked':
                self.assertTrue(np.all(mask[plan['mandatory_mask']]), name)

    def test_selection_plan_rejects_infeasible_budget(self):
        arrays = {
            'coords': self.coords,
            'offset_predictions': self.offsets,
            'instance_labels': self.labels,
        }
        with self.assertRaisesRegex(RuntimeError, 'infeasible'):
            E0B.build_selection_plan(
                arrays, self.candidates, self.utility, {
                    'known_label_min': 0,
                    'spatial_vote_cell_size': 0.6,
                    'min_seeds_per_tree': 2,
                    'keep_ratio': 0.5,
                    'random_seeds': [42],
                })

    def test_gate_requires_spatial_oracle_to_beat_random(self):
        modes = {
            'baseline': {
                'f1': 0.98, 'commission': 0.04, 'completeness': 1.0},
            'oracle_spatial': {
                'f1': 0.99, 'commission': 0.02, 'completeness': 0.998},
        }
        random = [{'f1': 0.985}, {'f1': 0.984}, {'f1': 0.986}]
        gate, effects = E0B.assess_gate(modes, random, {
            'min_f1_gain_pp': 0.5,
            'min_commission_reduction_pp': 1.0,
            'max_completeness_drop_pp': 0.5,
            'min_gain_over_random_pp': 0.25,
        })
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(effects['f1_gain_pp'], 1.0)

    def test_effective_ratio_reproduces_requested_integer_count(self):
        ratio = E0B.effective_ratio_for_count(5, 6)
        self.assertEqual(int(np.ceil(6 * ratio)), 5)

    def test_metrics_csv_accepts_random_seed_column(self):
        metrics = {
            'baseline': {'f1': 0.98, 'tp': 10},
            'random_spatial_s42': {'f1': 0.97, 'tp': 9, 'seed': 42},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            E0B.write_metrics_csv(path, metrics)
            with path.open(encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]['seed'], '')
        self.assertEqual(rows[1]['seed'], '42')


if __name__ == '__main__':
    unittest.main()
