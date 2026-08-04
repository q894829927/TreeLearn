import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'diagnostics' / 'diagnose_groupwise_merge_tradeoff.py')
SPEC = importlib.util.spec_from_file_location(
    'groupwise_merge_tradeoff_standalone', MODULE_PATH)
TRADEOFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRADEOFF)


def record(key, confidence, correct, positive=True):
    return {
        'source_key': key,
        'target_instance_id': 10,
        'confidence': confidence,
        'no_merge_probability': 0.1,
        'chosen_correct': correct,
        'has_positive_target': positive,
    }


class GroupwiseMergeTradeoffTests(unittest.TestCase):

    def test_frontier_uses_all_sources_for_unsafe_rate(self):
        rows = [
            record('P:1', 0.9, True),
            record('P:2', 0.8, False),
            record('P:3', 0.7, False, positive=False),
            record('P:4', 0.6, True),
        ]
        metrics = TRADEOFF.source_metrics(rows, 0.8)
        self.assertEqual(metrics['num_proposed_merges'], 2)
        self.assertEqual(metrics['num_unsafe_merges'], 1)
        self.assertAlmostEqual(metrics['pair_precision'], 0.5)
        self.assertAlmostEqual(metrics['unsafe_merge_rate'], 0.25)
        self.assertAlmostEqual(metrics['positive_source_recall'], 1 / 3)

    def test_selection_maximizes_recall_under_both_safety_constraints(self):
        rows = [
            record('P:1', 0.9, True),
            record('P:2', 0.8, True),
            record('P:3', 0.7, False),
            record('P:4', 0.6, True),
        ]
        selected = TRADEOFF.select_frontier_row(
            TRADEOFF.build_threshold_frontier(rows),
            min_precision=0.95,
            max_unsafe_rate=0.25,
        )
        self.assertAlmostEqual(selected['threshold'], 0.8)
        self.assertEqual(selected['num_correct_merges'], 2)
        self.assertAlmostEqual(selected['positive_source_recall'], 0.5)

    def test_primary_gate_requires_locked_seed_and_two_seed_passes(self):
        records = {
            42: [record('P:1', 0.9, True), record('P:2', 0.8, True)],
            43: [record('P:1', 0.9, True), record('P:2', 0.8, True)],
            44: [record('P:1', 0.9, False), record('P:2', 0.8, False)],
        }
        result = TRADEOFF.evaluate_sensitivity(
            records,
            precision_levels=[0.95],
            unsafe_rate_levels=[0.01],
            primary={
                'min_pair_precision': 0.95,
                'max_unsafe_merge_rate': 0.01,
                'min_positive_source_recall': 0.30,
                'max_recall_sample_std': 0.60,
            },
            locked_seed=42,
            minimum_pass_seeds=2,
        )
        self.assertTrue(result['gate']['locked_seed_passed'])
        self.assertTrue(result['gate']['enough_seed_passes'])
        self.assertTrue(result['gate']['passed'])

    def test_loader_rejects_cross_seed_source_mismatch(self):
        fieldnames = [
            'seed', 'source_key', 'target_instance_id', 'confidence',
            'no_merge_probability', 'chosen_correct',
            'has_positive_target',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            with path.open('w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([
                    {
                        'seed': 42, 'source_key': 'P:1',
                        'target_instance_id': 1, 'confidence': 0.9,
                        'no_merge_probability': 0.1,
                        'chosen_correct': True,
                        'has_positive_target': True,
                    },
                    {
                        'seed': 43, 'source_key': 'P:2',
                        'target_instance_id': 1, 'confidence': 0.9,
                        'no_merge_probability': 0.1,
                        'chosen_correct': True,
                        'has_positive_target': True,
                    },
                ])
            with self.assertRaisesRegex(ValueError, 'do not align'):
                TRADEOFF.load_prediction_records(path, [42, 43])


if __name__ == '__main__':
    unittest.main()
