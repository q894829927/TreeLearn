import unittest

import torch

from tree_learn.util.proposal_relation import (
    EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, VerticalRelationAttention,
    build_parameter_matched_edge_mlp, focal_edge_loss, parameter_count)


class ProposalRelationAttentionTests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(3)
        nodes = torch.randn(5, len(NODE_FEATURE_NAMES))
        edges = torch.randn(4, len(EDGE_FEATURE_NAMES))
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        return nodes, edge_index, edges

    def test_attention_is_finite_nonconstant_and_has_gradients(self):
        nodes, edge_index, edges = self._inputs()
        model = VerticalRelationAttention(dropout=0.)
        output = model(nodes, edge_index, edges, return_attention=True)
        loss = focal_edge_loss(
            output['edge_logits'], torch.tensor([1., 1., 0., 0.]))
        loss = loss + output['node_reliability_logits'].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output['edge_logits']).all())
        values = torch.cat([value.flatten() for value in output['attention']])
        self.assertTrue(torch.isfinite(values).all())
        self.assertGreater(float(values.std()), 0.)
        self.assertIsNotNone(model.layers[0].query.weight.grad)
        self.assertIsNotNone(model.layers[0].edge_score.weight.grad)
        self.assertIsNotNone(model.edge_head[0].weight.grad)
        self.assertIsNotNone(model.reliability_head.weight.grad)

    def test_attention_does_not_cross_disconnected_edges(self):
        nodes, edge_index, edges = self._inputs()
        model = VerticalRelationAttention(dropout=0.)
        first = model(nodes, edge_index, edges)['edge_logits']
        changed = nodes.clone()
        changed[4] += 1000
        second = model(changed, edge_index, edges)['edge_logits']
        torch.testing.assert_close(first, second)

    def test_empty_edges_and_single_node(self):
        model = VerticalRelationAttention(dropout=0.)
        output = model(
            torch.randn(1, len(NODE_FEATURE_NAMES)),
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, len(EDGE_FEATURE_NAMES))),
            return_attention=True)
        self.assertEqual(output['edge_logits'].numel(), 0)
        self.assertTrue(torch.isfinite(output['node_embeddings']).all())

    def test_vertical_ablation_changes_features_only(self):
        nodes, edge_index, edges = self._inputs()
        torch.manual_seed(4)
        vertical = VerticalRelationAttention(dropout=0., use_vertical=True)
        ablated = VerticalRelationAttention(dropout=0., use_vertical=False)
        ablated.load_state_dict(vertical.state_dict())
        self.assertFalse(torch.equal(
            vertical(nodes, edge_index, edges)['edge_logits'],
            ablated(nodes, edge_index, edges)['edge_logits']))

    def test_parameter_matched_control(self):
        attention = VerticalRelationAttention()
        control = build_parameter_matched_edge_mlp(
            parameter_count(attention), tolerance=.05)
        difference = abs(parameter_count(attention)-parameter_count(control))
        self.assertLessEqual(difference/parameter_count(attention), .05)


if __name__ == '__main__':
    unittest.main()
