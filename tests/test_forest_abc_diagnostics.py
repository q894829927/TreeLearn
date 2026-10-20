"""CPU-only regressions, intended to run on the experiment server."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.forest_abc.diagnose import analyze_table, audit_forest, compare_models, scan_arrays
from tree_learn.mask_refinement.evaluation import evaluate_labels


class ForestABCDiagnosticsTests(unittest.TestCase):
    def sample(self):
        labels = np.array([2] * 6 + [7] * 6 + [900] * 6 + [0] * 4 + [-1] * 4)
        baseline = np.array([10] * 6 + [11] * 6 + [12] * 6 + [0] * 4 + [99] * 4)
        refined = np.array([20] * 12 + [0] * 6 + [20] + [30] * 3 + [99] * 4)
        coords = np.zeros((len(labels), 3), dtype=np.float32)
        coords[:, 2] = np.arange(len(labels))
        return coords, labels, dict(baseline=baseline, refined=refined)

    def analyze(self, coords, labels, predictions, chunk_points):
        with contextlib.redirect_stdout(io.StringIO()):
            geometry, tables = scan_arrays(coords, labels, predictions, chunk_points)
        return geometry, {name: analyze_table(geometry, table) for name, table in tables.items()}

    def test_streamed_metrics_match_existing_evaluator_and_ignore_accounting(self):
        coords, labels, predictions = self.sample()
        previous = None
        for chunk in (1, 7, 100):
            geometry, models = self.analyze(coords, labels, predictions, chunk)
            for name, values in predictions.items():
                expected = evaluate_labels(labels, values)
                for key, actual in models[name]['metrics'].items():
                    self.assertAlmostEqual(expected[key], actual, places=12)
            if previous is not None:
                self.assertEqual(previous, models)
            previous = models
        refined = models['refined']
        self.assertEqual(refined['final_instance_accounting'], dict(
            true_positive=0, false_positive=1, ignored_only=1, excluded_low_tree_fraction=1))
        self.assertEqual(refined['predictions_covering_multiple_gt_at_half_recall'], 1)
        self.assertEqual(refined['tree_categories'], dict(high_recall_low_purity=2, no_assigned_tree_points=1))
        rows, comparison = compare_models(geometry, models)
        self.assertEqual(comparison['transitions'], dict(lost=3))
        self.assertEqual([row['gt_id'] for row in rows], [2, 7, 900])

    def test_fragmentation_background_contamination_and_empty_predictions(self):
        labels = np.array([1] * 10 + [0] * 20)
        coords = np.zeros((len(labels), 3), dtype=np.float32)
        coords[:10, 2] = np.arange(10)
        predictions = dict(fragmented=np.array([10] * 4 + [20] * 3 + [30] * 3 + [0] * 20),
                           background=np.full(len(labels), 5), empty=np.zeros(len(labels), dtype=int))
        geometry, models = self.analyze(coords, labels, predictions, 3)
        self.assertEqual(models['fragmented']['trees'][0]['category'], 'spread_across_predictions')
        self.assertEqual(models['background']['trees'][0]['category'], 'high_recall_low_purity')
        self.assertAlmostEqual(models['background']['trees'][0]['best_background_fraction'], 2 / 3)
        self.assertEqual(models['background']['trees'][0]['best_other_tree_fraction'], 0)
        self.assertEqual(models['background']['predictions_covering_multiple_gt_at_half_recall'], 0)
        self.assertEqual(models['empty']['final_instance_count'], 0)
        for name, values in predictions.items():
            expected = evaluate_labels(labels, values)
            for key, actual in models[name]['metrics'].items():
                self.assertAlmostEqual(expected[key], actual, places=12)
        # 9 m is vertical extent, and no FP attribution is invented for a height group.
        _, comparison = compare_models(geometry, dict(baseline=models['empty'], refined=models['fragmented']))
        group = comparison['z_span_groups']['5_to_10m']
        self.assertEqual(group['num_gt'], 1)
        self.assertNotIn('f1', group['refined'])
        self.assertIsNone(comparison['z_span_groups']['lt_5m']['refined']['coverage'])

    def test_strict_half_iou_is_not_a_match_and_pair_budget_is_enforced(self):
        labels = np.array([1] * 4 + [2] * 4)
        coords = np.zeros((8, 3), dtype=np.float32)
        prediction = np.ones(8, dtype=int)
        _, models = self.analyze(coords, labels, dict(refined=prediction), 2)
        self.assertEqual(models['refined']['metrics']['tp'], 0)
        self.assertEqual(models['refined']['metrics']['fp'], 1)
        self.assertEqual(models['refined']['predictions_covering_multiple_gt_at_half_recall'], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'max_pairs'):
                scan_arrays(coords, labels, dict(refined=prediction), 2, max_pairs=1)

    def test_file_audit_preserves_inputs_and_refuses_stale_metrics(self):
        coords, labels, predictions = self.sample()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, prediction, output = [root / name for name in ('cache', 'prediction', 'audit')]
            cache.mkdir()
            prediction.mkdir()
            for name, values in dict(coords=coords, labels=labels, baseline=predictions['baseline']).items():
                np.save(cache / f'{name}.npy', values)
            np.save(prediction / 'refined_labels.npy', predictions['refined'])
            (prediction / 'prediction.json').write_text(json.dumps(dict(partial=False, final_instances=3)))
            expected = dict(plot='synthetic', **{name: evaluate_labels(labels, values)
                                               for name, values in predictions.items()})
            marker = prediction / 'metrics.json'
            marker.write_text(json.dumps(expected))
            before = {path: path.read_bytes() for path in list(cache.iterdir()) + list(prediction.iterdir())}
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit_forest(cache, prediction, output, 20000, chunk_points=3)
            self.assertEqual(result['checkpoint_step'], 20000)
            self.assertTrue((output / 'trees.csv').exists())
            self.assertTrue((output / 'refined_predictions.csv').exists())
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            expected['refined']['tp'] += 1
            marker.write_text(json.dumps(expected))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    audit_forest(cache, prediction, output, 20000, chunk_points=3)
            self.assertFalse((output / 'brief.json').exists())
            self.assertFalse((output / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
