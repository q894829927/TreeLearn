"""Exact saved-count audit regressions to execute on the experiment server."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.forest_abc.diagnose import scan_arrays
from tools.instance_relation_v2.diagnose import audit, calibration_history
from tree_learn.instance_relation.supervision import make_targets
from tree_learn.instance_relation_v2.policy import select_pairs
from tree_learn.mask_refinement.evaluation import evaluate_labels


class InstanceRelationV2DiagnosticTests(unittest.TestCase):
    def fixture(self):
        gt = np.asarray([1] * 12 + [2] * 8 + [3] * 8 + [0] * 4)
        original = np.asarray([10] * 6 + [20] * 6 + [30] * 8 + [40] * 8 + [50] * 4)
        pairs = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        with contextlib.redirect_stdout(io.StringIO()):
            geometry, tables = scan_arrays(np.zeros((len(gt), 3)), gt, {'baseline': original}, 7)
        table = tables['baseline']
        targets, _ = make_targets(geometry, table, pairs,
            dict(min_known_fraction=.2, same_tree_purity=.6, min_union_precision=.65, min_iou_gain=.02))
        targets.update(gt_ids=geometry['ids'], gt_counts=geometry['counts'], z_span=geometry['z_span'],
                       ignored_points=geometry['ignored_points'], num_points=geometry['num_points'], **table)
        values = dict(ids=table['ids'], pairs=pairs)
        scores = dict(action=np.asarray([[.01, .98, .01], [.98, .01, .01], [.01, .98, .01]]),
                      risk=np.zeros(3), utility=np.ones(3), quality=np.ones((3, 3)) * .8)
        settings = dict(completion_credit=.25, utility_weight=.5, risk_weight=2.)
        scores['accepted'] = select_pairs(pairs, scores, settings, 1.)
        return values, targets, scores, settings, gt, original

    def call(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit(*args, **kwargs)

    def test_replays_recovery_and_loss_of_two_correct_trees(self):
        values, targets, scores, settings, gt, original = self.fixture()
        brief, actions, watched = self.call(values, targets, scores, settings, 1., {1: (10, 20)})
        refined = np.where(original == 20, 10, np.where(original == 40, 30, original))
        expected = evaluate_labels(gt, refined)
        for key in ('tp', 'fp', 'fn', 'f1', 'coverage', 'point_precision', 'point_recall'):
            self.assertAlmostEqual(brief['refined'][key], expected[key], places=12)
        self.assertEqual(brief['lost_gt_ids'], [2, 3])
        self.assertEqual(brief['recovered_gt_ids'], [1])
        self.assertEqual(brief['accepted_two_original_tp'], 1)
        self.assertEqual(brief['accepted_harmful_pairs_in_isolation'], 1)
        self.assertTrue(brief['isolated_pair_sum_matches_combined_counts'])
        self.assertEqual(actions[1]['isolated_lost_gt_ids'], [2, 3])
        self.assertEqual(watched[0]['pair']['actual_label'], 'recover')
        self.assertTrue(watched[0]['refined']['detected'])
        for action in actions:
            self.assertAlmostEqual(sum(action['score_terms'].values()), action['score'], places=12)

    def test_watch_distinguishes_threshold_mutual_blocking_and_absent_candidate(self):
        values, targets, scores, settings, _, _ = self.fixture()
        scores['action'][1] = [.01, .98, .01]
        scores['utility'][1] = .9
        scores['action'][2] = [.98, .01, .01]
        scores['utility'][2] = 0
        scores['accepted'] = select_pairs(values['pairs'], scores, settings, 1.)
        _, _, watched = self.call(values, targets, scores, settings, 1.,
            {1: (20, 30), 2: (30, 40), 3: (10, 40)})
        self.assertEqual(watched[0]['pair']['decision'], 'mutual_selection')
        self.assertEqual(watched[0]['pair']['blocking_pairs'][0]['original_ids'], [10, 20])
        self.assertEqual(watched[1]['pair']['decision'], 'below_threshold')
        self.assertFalse(watched[2]['case_pair_in_candidates'])

    def test_tampered_actions_and_probabilities_are_rejected_and_empty_policy_works(self):
        values, targets, scores, settings, _, _ = self.fixture()
        scores['accepted'] = values['pairs'][[1]]
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            self.call(values, targets, scores, settings, 1.)
        scores['accepted'] = np.empty((0, 2), dtype=np.int64)
        brief, actions, _ = self.call(values, targets, scores, settings, 2.5)
        self.assertEqual(brief['baseline'], brief['refined'])
        self.assertEqual(actions, [])
        scores['action'][0] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            self.call(values, targets, scores, settings, 2.5)

    def test_individual_union_iou_is_not_a_hungarian_recovery(self):
        # The union has IoU > .5 for GT1, but maximizing total IoU matches
        # two .32 edges instead; do not report an IoU-feasible pair as recovered.
        common = np.asarray([[34, 16], [34, 16], [32, 0]])
        pairs = np.asarray([[0, 1]], dtype=np.int64)
        geometry = dict(ids=np.asarray([1, 2]), counts=np.asarray([100, 32]),
                        z_span=np.ones(2), ignored_points=0, num_points=132)
        table = dict(ids=np.asarray([10, 20, 30]), known=np.asarray([50, 50, 32]),
                     total=np.asarray([50, 50, 32]), intersections=common)
        target, _ = make_targets(geometry, table, pairs,
            dict(min_known_fraction=.2, same_tree_purity=.6, min_union_precision=.65, min_iou_gain=.02))
        target.update(gt_ids=geometry['ids'], gt_counts=geometry['counts'], z_span=geometry['z_span'],
                      ignored_points=0, num_points=132, **table)
        scores = dict(action=np.asarray([[0., 1., 0.]]), risk=np.zeros(1), utility=np.ones(1),
                      quality=np.ones((1, 3)), accepted=pairs.copy())
        settings = dict(completion_credit=.25, utility_weight=.5, risk_weight=2.)
        brief, actions, _ = self.call(dict(ids=table['ids'], pairs=pairs), target, scores, settings, 1.)
        self.assertGreater(actions[0]['actual_union_quality'][2], .5)
        self.assertEqual(actions[0]['isolated_recovered_gt_ids'], [])
        self.assertEqual(brief['recovered_gt_ids'], [])

    def test_history_reports_existing_source_selections_without_external_tuning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'calibration'
            root.mkdir()
            for step, count in ((1000, 4), (500, 2)):
                report = dict(step=step, selected=dict(threshold=1., macro=dict(f1=.8), lost=0,
                    recovered=count, accepted_pairs=10, selected_labeled_counts=dict(keep=8, recover=2, complete=0)),
                    natural_recovery_classifier=dict(tp=2, fp=8, fn=1))
                (root / f'step_{step:06d}.json').write_text(json.dumps(report))
            history = calibration_history(temporary)
        self.assertEqual([r['step'] for r in history], [500, 1000])
        self.assertEqual([r['recovered'] for r in history], [2, 4])


if __name__ == '__main__':
    unittest.main()
