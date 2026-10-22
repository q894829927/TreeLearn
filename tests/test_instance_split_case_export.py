"""Server-side CPU regressions for complete-core split diagnostics; no models."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.forest_abc.diagnose import scan_arrays
from tools.instance_relation_v3.export_split_cases import (
    audit_core, check_output, collect_context, collect_cores, select_cases, write_case)


class InstanceSplitCaseExportTests(unittest.TestCase):
    def fixture(self):
        labels = np.array([1] * 6 + [2] * 6 + [-1, -2, 0] + [3] * 12 + [4] * 16 + [0] * 4)
        baseline = np.array([10] * 15 + [20] * 12 + [30] * 16 + [0] * 4)
        xyz = np.zeros((len(labels), 3), dtype=np.float32)
        xyz[:15, 2] = np.linspace(0, 10, 15)
        xyz[15:27, 0] = .5
        xyz[15:27, 2] = np.linspace(0, 10, 12)
        xyz[27:43, 0] = 4
        xyz[27:43, 2] = np.linspace(0, 15, 16)
        xyz[43:, 2] = [1, 3, 6, 9]
        with contextlib.redirect_stdout(io.StringIO()):
            geometry, tables = scan_arrays(xyz, labels, {'baseline': baseline}, 7)
        targets = dict(**tables['baseline'], gt_ids=geometry['ids'], gt_counts=geometry['counts'],
            z_span=geometry['z_span'], ignored_points=geometry['ignored_points'], num_points=geometry['num_points'])
        targets['original_tp'] = np.array([False, True, True])
        features = dict(ids=targets['ids'],
            low=np.stack([xyz[baseline == i].min(0) for i in targets['ids']]),
            high=np.stack([xyz[baseline == i].max(0) for i in targets['ids']]))
        inputs = dict(coords=xyz, baseline=baseline, votes_xy=xyz[:, :2] + .1,
            tree_probability=np.full(len(labels), .8, dtype=np.float32),
            features=np.arange(len(labels) * 32, dtype=np.float32).reshape(-1, 32))
        return inputs, labels, targets, features

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_complete_core_and_context_are_aligned_and_chunk_independent(self):
        inputs, _, targets, features = self.fixture()
        cases, _ = select_cases(features, targets, [10], 100)
        self.assertEqual([r['original_id'] for r in cases], [10, 20])
        self.assertEqual(cases[1]['role'], 'detected_control')
        outputs = []
        for chunk in (3, 11):
            core = self.quiet(collect_cores, inputs['baseline'], cases, chunk, 100)
            context = self.quiet(collect_context, inputs['coords'], inputs['baseline'], cases,
                core, chunk, 8, 4, 2., 2., 42)
            for case in cases:
                identifier = case['original_id']
                self.assertTrue(np.array_equal(core[identifier], np.flatnonzero(inputs['baseline'] == identifier)))
                sampled = np.concatenate(context[identifier]['indices'])
                self.assertLessEqual(len(sampled), 8)
                self.assertTrue(np.all(inputs['baseline'][sampled] != identifier))
            outputs.append(context)
        for identifier in (10, 20):
            self.assertTrue(np.array_equal(outputs[0][identifier]['seen'], outputs[1][identifier]['seen']))
            for a, b in zip(outputs[0][identifier]['indices'], outputs[1][identifier]['indices']):
                self.assertTrue(np.array_equal(np.sort(a), np.sort(b)))

    def test_inputs_are_saved_before_reading_gt_and_unknown_labels_stay_separate(self):
        inputs, labels, targets, features = self.fixture()
        cases, _ = select_cases(features, targets, [10], 100)
        cores = self.quiet(collect_cores, inputs['baseline'], cases, 7, 100)
        context = self.quiet(collect_context, inputs['coords'], inputs['baseline'], cases, cores, 7, 8, 4, 2., 2., 42)
        case = cases[0]
        indices = np.sort(np.concatenate((cores[10], *context[10]['indices'])))
        core = inputs['baseline'][indices] == 10
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'original_10'
            class GuardedGT:
                def __getitem__(self, at):
                    if not (directory / 'inputs.npz').is_file():
                        raise AssertionError('GT read before prediction-only observations were fixed.')
                    return labels[at]
            report = write_case(temporary, case, inputs, GuardedGT(), indices, core, context[10], targets)
            self.assertEqual((report['core_points'], report['known_points'], report['ignored_points']), (15, 13, 2))
            with np.load(directory / 'inputs.npz', allow_pickle=False) as saved:
                self.assertNotIn('labels', saved.files)
                self.assertTrue(np.array_equal(saved['features'], inputs['features'][saved['source_indices']]))
                self.assertTrue(np.array_equal(saved['source_indices'][saved['is_core']], np.flatnonzero(inputs['baseline'] == 10)))
            with np.load(directory / 'audit.npz', allow_pickle=False) as audit:
                self.assertTrue(np.array_equal(audit['source_indices'], indices))
                self.assertTrue(np.array_equal(audit['labels'], labels[indices]))
                self.assertIn(-2, audit['labels'])

    def test_changed_counts_and_oversized_cores_fail_instead_of_truncating(self):
        inputs, labels, targets, features = self.fixture()
        cases, _ = select_cases(features, targets, [10], 100)
        bad = [dict(cases[0], total_points=14)]
        with self.assertRaisesRegex(ValueError, 'point count changed'):
            self.quiet(collect_cores, inputs['baseline'], bad, 7, 100)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.quiet(collect_cores, inputs['baseline'], cases, 7, 14)
        changed = labels[inputs['baseline'] == 10].copy()
        changed[0] = 2
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            audit_core(changed, np.ones(len(changed), dtype=bool), targets, 0)

    def test_correct_instance_cannot_be_a_mixed_case_and_inputs_are_protected(self):
        _, _, targets, features = self.fixture()
        with self.assertRaisesRegex(ValueError, 'original TP'):
            select_cases(features, targets, [20], 100)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / 'cache'
            with self.assertRaisesRegex(ValueError, 'protected input'):
                check_output(cache / 'export', [cache])
            with self.assertRaisesRegex(ValueError, 'protected input'):
                check_output(Path(temporary), [cache])
            check_output(Path(temporary) / 'separate', [cache])


if __name__ == '__main__':
    unittest.main()
