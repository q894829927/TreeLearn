"""CPU regression tests; execute on the experiment server, not locally."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.treelearn_residual_instance.diagnose import audit_saved, point_assignment_changes
from tree_learn.mask_refinement.evaluation import evaluate_labels


class ResidualInstanceDiagnosticsTests(unittest.TestCase):
    def fixture(self, root):
        cache = root / 'cache'
        cache.mkdir()
        labels = np.array([1] * 4 + [2] * 4 + [3] * 4 + [0] * 2 + [-1] * 2)
        baseline = np.array([10] * 4 + [20] * 4 + [30] * 4 + [0] * 2 + [99] * 2)
        control, combined = baseline.copy(), baseline.copy()
        control[8:12] = 0
        combined[:4] = 0
        # Different arbitrary instance IDs must not be interpreted as point loss.
        combined[combined > 0] += 1000
        predictions = dict(baseline=baseline, control=control, combined=combined)
        coords = np.zeros((len(labels), 3), dtype=np.float32)
        coords[:, 2] = np.arange(len(labels))
        for name, values in dict(coords=coords, labels=labels, baseline=baseline).items():
            np.save(cache / f'{name}.npy', values)
        directories, steps = {}, dict(control=3000, combined=3000)
        for arm in ('control', 'combined'):
            directory = root / arm / 'external_evaluation' / 'synthetic'
            directory.mkdir(parents=True)
            directories[arm] = directory
            np.save(directory / 'predictions.npy', predictions[arm])
            (directory / 'metrics.json').write_text(json.dumps(dict(plot='synthetic',
                baseline=evaluate_labels(labels, baseline), refined=evaluate_labels(labels, predictions[arm]))))
            (directory / 'prediction.json').write_text(json.dumps(dict(num_points=len(labels),
                final_instances=len(np.unique(predictions[arm][predictions[arm] > 0])))))
            (directory.parent / 'request.json').write_text(json.dumps(dict(arm=arm,
                step=steps[arm], plots=['synthetic'])))
        return cache, directories, steps, labels, predictions

    def test_equal_tp_counts_can_hide_lost_and_recovered_trees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directories, steps, _, _ = self.fixture(root)
            sources = list(cache.iterdir())
            for directory in directories.values():
                sources += list(directory.iterdir()) + [directory.parent / 'request.json']
            before = {path: path.read_bytes() for path in sources}
            with contextlib.redirect_stdout(io.StringIO()):
                report = audit_saved(cache, directories, root / 'audit', steps, chunk_points=3)
            self.assertEqual(report['models']['control']['metrics']['tp'], 2)
            self.assertEqual(report['models']['combined']['metrics']['tp'], 2)
            transition = report['pairwise']['control_to_combined']['transitions']
            self.assertEqual(transition, dict(lost=1, both_detected=1, recovered=1))
            self.assertEqual(report['pairwise']['baseline_to_control']['transitions'],
                             dict(both_detected=2, lost=1))
            self.assertEqual(report['pairwise']['baseline_to_control']['lost_target_categories'],
                             dict(no_assigned_tree_points=1))
            self.assertAlmostEqual(report['models']['control']['gt_point_assignment_fraction'], 2 / 3)
            self.assertEqual(before, {path: path.read_bytes() for path in sources})
            self.assertTrue((root / 'audit/control_to_combined_trees.csv').is_file())
            self.assertTrue((root / 'audit/summary.json').is_file())

    def test_assignment_counts_ignore_unknown_background_and_instance_numbering(self):
        labels = np.array([1, 1, 1, 1, 0, -1])
        first = np.array([7, 7, 0, 0, 99, 99])
        second = np.array([123, 0, 12, 0, 0, 0])
        expected = dict(both_assigned=1, became_unassigned=1, became_assigned=1, both_unassigned=1)
        for chunk in (1, 3, 99):
            actual = point_assignment_changes(labels, dict(first=first, second=second), chunk)
            self.assertEqual(actual, {'first_to_second': expected})

    def test_saved_metric_mismatch_rejects_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directories, steps, _, _ = self.fixture(root)
            marker = directories['combined'] / 'metrics.json'
            metadata = json.loads(marker.read_text())
            metadata['refined']['tp'] += 1
            marker.write_text(json.dumps(metadata))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    audit_saved(cache, directories, root / 'audit', steps, 5)
            self.assertFalse((root / 'audit/brief.json').exists())

    def test_request_step_and_full_point_count_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directories, steps, _, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'selected arm/step/plot'):
                audit_saved(cache, directories, root / 'audit', dict(control=500, combined=3000))
            marker = directories['combined'] / 'prediction.json'
            metadata = json.loads(marker.read_text())
            metadata['num_points'] -= 1
            marker.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'point count'):
                audit_saved(cache, directories, root / 'audit', steps)


if __name__ == '__main__':
    unittest.main()
