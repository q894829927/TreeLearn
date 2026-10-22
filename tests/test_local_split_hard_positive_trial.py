"""Behavioral checks of the one-change R1 loss and strict other-tree gate."""

import unittest

import torch
from torch.nn import functional as F

from tools.local_split_v1.run_hard_positive_trial import e1_gate, hard_positive_loss


class HardPositiveTrialTests(unittest.TestCase):
    def test_hardest_positive_gets_extra_gradient_without_reweighting_negatives(self):
        logits = torch.tensor([-3., 0., 1., 2., 3., -2.], requires_grad=True)
        targets = torch.tensor([1., 1., 1., 1., 0., 0.])
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        balanced = .5 * (bce[:4].mean() + bce[4:].mean())
        reference = torch.autograd.grad(balanced, logits, retain_graph=True)[0]
        loss, parts = hard_positive_loss(logits, targets)
        actual = torch.autograd.grad(loss, logits)[0]
        self.assertEqual(parts['hard_count'], 1)
        self.assertGreater(abs(float(actual[0])), abs(float(reference[0])))
        torch.testing.assert_close(actual[1:], reference[1:])
        torch.testing.assert_close(loss, balanced + .5 * bce[0])

    def test_fraction_uses_ceiling_and_only_the_positive_pool(self):
        logits = torch.tensor([-3., -2., 0., 1., 2., 8.])
        targets = torch.tensor([1., 1., 1., 1., 1., 0.])
        _, parts = hard_positive_loss(logits, targets)
        self.assertEqual(parts['hard_count'], 2)
        expected = F.binary_cross_entropy_with_logits(logits[:2], targets[:2])
        torch.testing.assert_close(parts['hard_positive'], expected)

    def test_one_positive_is_not_rounded_down_to_zero(self):
        loss, parts = hard_positive_loss(torch.tensor([0., 0.]), torch.tensor([1., 0.]))
        self.assertEqual(parts['hard_count'], 1)
        self.assertTrue(torch.isfinite(loss))

    def test_unknown_targets_and_missing_classes_refuse_training(self):
        for targets in ([-1., 1.], [0., 0.], [1., 1.]):
            with self.assertRaises(ValueError):
                hard_positive_loss(torch.zeros(2), torch.tensor(targets))

    def test_background_recall_cannot_satisfy_other_tree_gate(self):
        rates = dict(same=dict(count=100, cut=5), other_tree=dict(count=100, cut=79),
                     tree_background=dict(count=10000, cut=10000))
        self.assertFalse(e1_gate(rates, 9, 0, self.settings())['passed'])

    def test_gate_keeps_inclusive_edge_boundaries_and_strict_loss_protection(self):
        rates = dict(same=dict(count=100, cut=5), other_tree=dict(count=100, cut=80))
        self.assertTrue(e1_gate(rates, 1, 0, self.settings())['passed'])
        self.assertFalse(e1_gate(rates, 1, 1, self.settings())['passed'])
        self.assertFalse(e1_gate(rates, 0, 0, self.settings())['passed'])
        rates['same']['cut'] = 6
        self.assertFalse(e1_gate(rates, 9, 0, self.settings())['passed'])

    def test_no_other_tree_population_cannot_pass(self):
        rates = dict(same=dict(count=100, cut=0), other_tree=dict(count=0, cut=0))
        self.assertFalse(e1_gate(rates, 1, 0, self.settings())['passed'])

    @staticmethod
    def settings():
        return dict(max_same_false_cut_rate=.05, min_other_tree_detection_rate=.8,
                    min_recovered=1, max_lost=0)


if __name__ == '__main__':
    unittest.main()
