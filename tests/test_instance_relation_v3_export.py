"""CPU saved-count export regressions for the experiment server."""

import contextlib
import io
import unittest

import numpy as np

from tools.forest_abc.diagnose import scan_arrays
from tools.instance_relation_v3.export_cases import choose_cases, decision
from tree_learn.instance_relation.supervision import make_targets
from tree_learn.instance_relation_v3 import policy
from tree_learn.mask_refinement.evaluation import evaluate_labels


class InstanceRelationV3ExportTests(unittest.TestCase):
    def fixture(self):
        gt = np.asarray([1] * 12 + [2] * 8 + [3] * 8 + [0] * 4)
        original = np.asarray([10] * 6 + [20] * 6 + [30] * 8 + [40] * 8 + [50] * 4)
        pairs = np.asarray([[0, 1], [1, 2], [2, 3]])
        with contextlib.redirect_stdout(io.StringIO()):
            geometry, tables = scan_arrays(np.zeros((len(gt), 3)), gt, {'baseline': original}, 7)
        table = tables['baseline']
        targets, _ = make_targets(geometry, table, pairs,
            dict(min_known_fraction=.2, same_tree_purity=.6, min_union_precision=.65, min_iou_gain=.02))
        targets.update(gt_ids=geometry['ids'], gt_counts=geometry['counts'], z_span=geometry['z_span'],
                       ignored_points=geometry['ignored_points'], num_points=geometry['num_points'], **table)
        values = dict(ids=table['ids'], pairs=pairs)
        scores = dict(action=np.asarray([[.01, .98, .01], [.98, .01, .01], [.01, .98, .01]]))
        settings = dict(destruction_cost=3.)
        scores['accepted'] = policy.select_pairs(pairs, scores, settings, .1)
        return values, targets, scores, settings, gt, original

    def quiet(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return choose_cases(*args, **kwargs)

    def test_exact_recovery_and_destruction_export_without_relabeling_predictions(self):
        values, targets, scores, settings, gt, original = self.fixture()
        records, watched, _, refined = self.quiet(values, targets, scores, settings, .1, {1: (10, 20), 99: (10, 50)})
        changed = np.where(original == 20, 10, np.where(original == 40, 30, original))
        expected = evaluate_labels(gt, changed)
        self.assertEqual((refined['tp'], refined['fp'], refined['fn']), (expected['tp'], expected['fp'], expected['fn']))
        self.assertEqual([r['original_ids'] for r in records], [[10, 20], [30, 40]])
        self.assertEqual(records[0]['isolated_outcome'], 'recover')
        self.assertEqual(records[1]['isolated_outcome'], 'destroy')
        self.assertEqual(records[1]['original_matched_gt'], [2, 3])
        self.assertEqual(records[1]['isolated_lost'], 2)
        self.assertIn('watched_gt_1', records[0]['groups'])
        self.assertIsNone(watched[1]['edge_index'])

    def test_wrong_class_threshold_and_competition_are_distinguished(self):
        self.assertEqual(decision(np.asarray([.7, .2, .1]), -.1, .1, False), 'predicted_other')
        self.assertEqual(decision(np.asarray([.1, .2, .7]), -1.9, .1, False), 'predicted_destroy')
        self.assertEqual(decision(np.asarray([.1, .8, .1]), .5, .6, False), 'below_threshold')
        self.assertEqual(decision(np.asarray([.1, .8, .1]), .5, .1, False), 'mutual_selection')
        self.assertEqual(decision(np.asarray([.1, .8, .1]), .5, .1, True), 'accepted')

    def test_saved_action_mismatch_is_rejected_and_empty_case_sets_are_valid(self):
        values, targets, scores, settings, _, _ = self.fixture()
        scores['accepted'] = values['pairs'][[1]]
        with self.assertRaisesRegex(ValueError, 'Saved V3 actions'):
            self.quiet(values, targets, scores, settings, .1)
        scores['accepted'] = np.empty((0, 2), dtype=np.int64)
        records, watched, baseline, refined = self.quiet(values, targets, scores, settings, 2.)
        self.assertEqual(records, [])
        self.assertEqual(watched, [])
        self.assertEqual(baseline, refined)


if __name__ == '__main__':
    unittest.main()
