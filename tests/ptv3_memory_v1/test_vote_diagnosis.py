import unittest
import numpy as np
from tree_learn.experiments.ptv3_vote_v1 import vector_metrics, votes, histograms, histogram_report, oracle_arrays


class VoteTests(unittest.TestCase):
    def test_zero_and_perfect_votes(self):
        coords = np.array([[0.,0,0], [1,0,0], [5,0,0], [6,0,0]])
        labels = np.array([1,1,2,2]); anchors = np.array([[.5,0,0],[.5,0,0],[5.5,0,0],[5.5,0,0]])
        gt = anchors-coords; mask = np.ones(4, bool)
        self.assertEqual(vector_metrics(np.zeros_like(gt),gt,mask)['xy_improvement_over_zero'],0)
        self.assertEqual(vector_metrics(gt,gt,mask)['xy_mean_m'],0)
        r = votes(coords,gt,gt,labels,mask,mask)
        self.assertEqual(r['mean_vote_scatter_rms_m'],0)
        self.assertEqual(r['trees_with_seed'],2)

    def test_pooling_retains_minority_tree_counts(self):
        labels = np.array([1,1,2,2,-1,0]); mapping = np.array([0,0,0,1,2,2])
        h = histograms(mapping,labels,3,3); _,r = histogram_report(h,3)
        self.assertEqual(r['multi_tree_groups'],1)
        parent, category, weight = h
        pooled = histograms(np.array([0,0,1])[parent],category,2,3,weight)
        _,r = histogram_report(pooled,2)
        self.assertEqual(r['known_points_in_multi_tree_groups'],4)

    def test_oracle_preserves_unknown_and_unselected_axes(self):
        logits=np.zeros((3,2));offset=np.ones((3,3));target=np.full((3,3),7.)
        valid=np.array([True,False,True]);sem=np.array([0,1,1])
        l,o=oracle_arrays(logits,offset,sem,target,valid,valid,'xy')
        np.testing.assert_array_equal(o[:,2],offset[:,2]);np.testing.assert_array_equal(o[1],offset[1])
        np.testing.assert_array_equal(offset,np.ones((3,3)));np.testing.assert_array_equal(l,logits)


if __name__=='__main__':unittest.main()
