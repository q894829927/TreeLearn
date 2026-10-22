"""Synthetic checks for diagnostic conclusions, with no model or forest data."""
import unittest
import numpy as np

from tools.instance_relation_v3.analyze_split_cases import (
    distance_auc, layer_graph, lower_support_summary, radius_summary, vote_summary)


class SplitCaseAnalysisTests(unittest.TestCase):
    def test_vote_modes_reproduce_in_hash_halves_and_merge_at_coarse_scale(self):
        rng = np.random.default_rng(42)
        votes = np.r_[rng.normal([0, 0], .08, (1000, 2)), rng.normal([.9, 0], .08, (1000, 2))]
        result, _, _ = vote_summary(votes, np.arange(2000), np.tile(np.arange(1000), 2))
        self.assertEqual([len(r['peaks']) for r in result], [2, 2, 1])
        self.assertTrue(all(result[1]['peaks_reproduced_within_0_3m']))

    def test_correct_fork_has_multiple_slab_components_but_one_3d_component(self):
        stem = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0, 2, 60)])
        branches = [np.column_stack([sign * np.linspace(0, 1, 60), np.zeros(60), np.linspace(2, 4, 60)]) for sign in [-1, 1]]
        xyz = np.concatenate([stem, *branches])
        gt = np.ones(len(xyz), dtype=int)
        stats, _, _ = radius_summary(xyz, gt, .2)
        self.assertEqual(stats['substantial_components'], 1)
        graph = layer_graph(xyz, gt)
        self.assertGreater(sum(n['layer'] == 3 for n in graph['nodes']), 1)
        self.assertEqual(stats['different_gt_positive_edge_fraction'], 0.)

    def test_graph_construction_is_label_independent_and_unknown_is_not_negative(self):
        rng = np.random.default_rng(1)
        xyz = rng.normal(0, .1, (100, 3))
        gt = np.r_[np.ones(40, dtype=int), np.full(60, -2)]
        first = layer_graph(xyz, gt)
        second = layer_graph(xyz, np.arange(100) + 1)
        observable_nodes = lambda graph: [{k: n[k] for k in ('id', 'layer', 'points', 'center')} for n in graph['nodes']]
        self.assertEqual(observable_nodes(first), observable_nodes(second))
        self.assertEqual(first['edges'], second['edges'])
        support_a = lower_support_summary(xyz, gt)
        support_b = lower_support_summary(xyz, np.arange(100) + 1)
        self.assertEqual([r['component_counts_first_4m'] for r in support_a],
                         [r['component_counts_first_4m'] for r in support_b])
        stats, _, _ = radius_summary(xyz, gt, .6)
        self.assertEqual(stats['different_gt_positive_edge_fraction'], 0.)
        self.assertGreater(stats['positive_edge_count'], 0)

    def test_distance_rank_statistic_handles_ties_and_missing_negative_edges(self):
        self.assertEqual(distance_auc(np.array([0, 1]), np.array([2, 3])), 1.)
        self.assertEqual(distance_auc(np.array([1]), np.array([1])), .5)
        self.assertIsNone(distance_auc(np.array([1]), np.array([])))


if __name__ == '__main__':
    unittest.main()
