"""CPU regression cases; run on the experiment server, not the development PC."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.diagnostics.diagnose_structure_voting import audit_saved, PROTOCOL


class StructureVotingDiagnosticsTests(unittest.TestCase):
    def fixture(self, root):
        cache, directory = root / 'cache', root / 'evaluation' / 'fixture'
        cache.mkdir()
        directory.mkdir(parents=True)
        labels = np.array([1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [0] * 2 + [-1] * 2)
        original = np.array([10] * 4 + [20] * 4 + [30] * 8 + [0] * 2 + [99] * 2)
        refined = np.array([100] * 8 + [200] * 4 + [300] * 4 + [0] * 2 + [999] * 2)
        # Same totals, but two old detections are lost and two other trees recovered.
        expected = dict(tp=2, fp=1, fn=2, num_gt=4, num_predictions=3, ignored_points=2,
            f1=4 / 7, completeness=.5, commission=1 / 3, coverage=.75,
            point_precision=.75, point_recall=1.)
        coords = np.zeros((len(labels), 3), dtype=np.float32)
        coords[:, 2] = np.arange(len(labels))
        for name, values in dict(coords=coords, labels=labels, baseline=original).items():
            np.save(cache / f'{name}.npy', values)
        np.save(directory / 'predictions.npy', refined)
        metadata = dict(plot='fixture', baseline=expected, refined=expected)
        (directory / 'metrics.json').write_text(json.dumps(metadata))
        (directory / 'prediction.json').write_text(json.dumps(dict(num_points=len(labels), final_instances=4)))
        (directory / 'foreground_identity.json').write_text(json.dumps(
            dict(changed_foreground_points=0, official_foreground_points=18)))
        (directory.parent / 'request.json').write_text(json.dumps(dict(step=3000, plots=['fixture'])))
        (directory.parent / 'brief.json').write_text(json.dumps(dict(checkpoint_step=3000, protocol=PROTOCOL)))
        return cache, directory, original

    def run_audit(self, cache, directory, output, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit_saved(cache, directory, output, chunk_points=3, **kwargs)

    def test_equal_totals_reveal_lost_recovered_and_merged_original_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directory, _ = self.fixture(root)
            files = list(cache.iterdir()) + list(directory.iterdir()) + list(directory.parent.glob('*.json'))
            before = {p: p.read_bytes() for p in files}
            report = self.run_audit(cache, directory, root / 'audit', watch={'selected': (1, 3)})
            self.assertEqual(report['transitions'], dict(both_detected=0, both_missed=0, lost=2, recovered=2))
            self.assertEqual(report['original_tp_full_mask_accounting'],
                             dict(total=2, exact_masks_preserved=0, masks_changed=2))
            self.assertEqual([r['transition'] for r in report['watchlist']], ['lost', 'recovered'])
            self.assertEqual(before, {p: p.read_bytes() for p in files})
            self.assertTrue((root / 'audit/original_tp_full_masks.csv').is_file())

    def test_renumbering_preserves_complete_original_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directory, original = self.fixture(root)
            np.save(directory / 'predictions.npy', np.where(original > 0, original + 700, 0))
            report = self.run_audit(cache, directory, root / 'audit')
            self.assertEqual(report['transitions']['lost'], 0)
            self.assertEqual(report['transitions']['recovered'], 0)
            self.assertEqual(report['original_tp_full_mask_accounting'],
                             dict(total=2, exact_masks_preserved=2, masks_changed=0))

    def test_foreground_change_is_detected_even_when_metrics_do_not_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directory, _ = self.fixture(root)
            pred = np.load(directory / 'predictions.npy')
            pred[-1] = 0  # Unknown GT is ignored in metrics, but foreground must still be identical.
            np.save(directory / 'predictions.npy', pred)
            with self.assertRaisesRegex(ValueError, 'Foreground changed'):
                self.run_audit(cache, directory, root / 'audit')
            self.assertFalse((root / 'audit/brief.json').exists())

    def test_wrong_step_and_saved_metric_mismatch_reject_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directory, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'request/step/plot/protocol'):
                self.run_audit(cache, directory, root / 'audit', expect_step=1500)
            path = directory / 'metrics.json'
            value = json.loads(path.read_text())
            value['refined']['tp'] = 3
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                self.run_audit(cache, directory, root / 'audit')

    def test_full_point_count_and_existing_output_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, directory, _ = self.fixture(root)
            output = root / 'existing'
            output.mkdir()
            with self.assertRaises(FileExistsError):
                self.run_audit(cache, directory, output)
            path = directory / 'prediction.json'
            value = json.loads(path.read_text())
            value['num_points'] -= 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'full aligned cache'):
                self.run_audit(cache, directory, root / 'audit')


if __name__ == '__main__':
    unittest.main()
