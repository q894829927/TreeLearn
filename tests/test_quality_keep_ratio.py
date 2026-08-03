import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'diagnostics' / 'select_quality_keep_ratio.py')
SPEC = importlib.util.spec_from_file_location(
    'quality_keep_ratio_standalone', MODULE_PATH)
RATIO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RATIO)


class QualityKeepRatioTests(unittest.TestCase):

    def test_selects_smallest_best_ratio_across_seeds(self):
        rows = []
        for seed in (42, 43):
            for plot in ('A', 'B'):
                rows.extend([
                    {
                        'seed': seed, 'source_plot': plot,
                        'instance_id': 1, 'score': 0.9,
                        'target_true': True,
                        'classification_valid': True,
                    },
                    {
                        'seed': seed, 'source_plot': plot,
                        'instance_id': 2, 'score': 0.1,
                        'target_true': False,
                        'classification_valid': True,
                    },
                ])
        selected, summaries, _ = RATIO.evaluate_keep_ratios(
            rows, [42, 43], [0.5, 1.0], 0.01)
        self.assertEqual(selected['keep_ratio'], 0.5)
        self.assertEqual(selected['mean_f1'], 1.0)
        self.assertEqual(len(summaries), 2)

    def test_ratio_ties_are_deterministic_by_instance_id(self):
        retained = RATIO.top_ratio_mask(
            [0.5, 0.5, 0.5], [30, 10, 20], ['A', 'A', 'A'], 1 / 3)
        self.assertEqual(retained.tolist(), [False, True, False])


if __name__ == '__main__':
    unittest.main()
