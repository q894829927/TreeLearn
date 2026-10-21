"""Server-side tensor/graph tests; no spconv or model weights are imported."""

from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tree_learn.structure_voting.graph import LocalSupportGraph
from tree_learn.structure_voting.loss import sample_queries, structure_loss, group_average
from tree_learn.structure_voting.model import StructureVoteBranch, apply_xy_correction


SUPERVISION = dict(max_trees=8, points_per_tree=3, vote_weight=1., connection_weight=.3,
                   consistency_weight=.1, regularization_weight=.02)


def fixture():
    torch.manual_seed(42)
    xyz = torch.tensor([[0., 0., 0.], [.2, 0., .4], [.3, 0., .8],
                        [.6, .1, 0.], [.7, .1, .4], [.8, .1, .8],
                        [.4, .3, .4], [.5, -.3, .4]])
    labels = torch.tensor([1, 1, 1, 2, 2, 2, -1, 0])
    target = torch.zeros_like(xyz)
    target[:3] = torch.tensor([0., 0., 0.]) - xyz[:3]
    target[3:6] = torch.tensor([.6, .1, 0.]) - xyz[3:6]
    batch = dict(coords=xyz, instance_labels=labels, batch_ids=torch.zeros(8, dtype=torch.long),
                 masks_off=labels > 0, offset_labels=target)
    features = torch.randn(8, 4)
    offsets = target + torch.tensor([.3, -.2, 0.])
    logits = torch.tensor([[3., -3.]]).repeat(8, 1)
    return batch, features, offsets, logits


def branch(chunk=3):
    return StructureVoteBranch(channels=4, hidden_dim=8, neighbors=4, radius=1.5,
                               support_voxel_size=.1, query_chunk_size=chunk)


def predict(model, batch, features, offsets, logits, selected=None):
    return model(features, batch['coords'], offsets, logits, batch['batch_ids'], selected)


class StructureVotingTests(unittest.TestCase):
    def test_zero_projection_is_exact_identity_including_z(self):
        batch, features, offsets, logits = fixture()
        result = predict(branch(), batch, features, offsets, logits)
        self.assertTrue(torch.equal(apply_xy_correction(offsets, result), offsets))

    def test_only_xy_is_changed_after_projection_opens(self):
        batch, features, offsets, logits = fixture()
        model = branch()
        with torch.no_grad():
            model.correction[-1].bias.copy_(torch.tensor([.4, -.3]))
        actual = apply_xy_correction(offsets, predict(model, batch, features, offsets, logits))
        self.assertTrue(torch.equal(actual[:, 2], offsets[:, 2]))
        self.assertFalse(torch.equal(actual[:, :2], offsets[:, :2]))
        self.assertLessEqual(float((actual[:, :2] - offsets[:, :2]).abs().max()), model.max_delta_xy)

    def test_sparse_query_update_does_not_change_other_rows(self):
        batch, features, offsets, logits = fixture()
        model = branch()
        with torch.no_grad():
            model.correction[-1].bias.fill_(.2)
        result = predict(model, batch, features, offsets, logits, torch.tensor([1, 4]))
        actual = apply_xy_correction(offsets, result)
        self.assertTrue(torch.equal(actual[[0, 2, 3, 5, 6, 7]], offsets[[0, 2, 3, 5, 6, 7]]))

    def test_graph_is_batch_isolated_and_excludes_self(self):
        xyz = np.array([[0, 0, 0], [.2, 0, 0], [0, 0, 0], [.2, 0, 0]], dtype=np.float32)
        batches = np.array([0, 0, 1, 1])
        graph = LocalSupportGraph(xyz, batches, np.ones(4, bool), .1, 4, 1.)
        neighbor, valid = graph.query(np.arange(4))
        for row in range(4):
            self.assertTrue(np.all(batches[neighbor[row, valid[row]]] == batches[row]))
            self.assertNotIn(row, neighbor[row, valid[row]])
            self.assertEqual(int(valid[row].sum()), 1)

    def test_radius_and_empty_foreground_do_not_invent_edges(self):
        xyz = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float32)
        for foreground in (np.ones(2, bool), np.zeros(2, bool)):
            graph = LocalSupportGraph(xyz, np.zeros(2), foreground, .1, 4, .2)
            _, valid = graph.query(np.arange(2))
            self.assertFalse(valid.any())

    def test_supports_cover_every_occupied_cell_without_a_global_cap(self):
        xyz = np.column_stack((np.arange(100), np.zeros(100), np.zeros(100))).astype(np.float32)
        graph = LocalSupportGraph(xyz, np.zeros(100), np.ones(100, bool), .4, 2, 2.)
        self.assertEqual(len(graph.supports[0][0]), 100)

    def test_prediction_is_chunk_invariant_after_learning(self):
        batch, features, offsets, logits = fixture()
        small, large = branch(2), branch(100)
        with torch.no_grad():
            small.correction[-1].weight.normal_(0, .1)
        large.load_state_dict(small.state_dict())
        first = predict(small, batch, features, offsets, logits)
        second = predict(large, batch, features, offsets, logits)
        torch.testing.assert_close(first['delta_xy'], second['delta_xy'], atol=1e-6, rtol=1e-5)

    def test_inference_does_not_depend_on_instance_labels(self):
        batch, features, offsets, logits = fixture()
        model = branch()
        with torch.no_grad():
            model.correction[-1].weight.normal_(0, .1)
        first = predict(model, batch, features, offsets, logits)
        changed = dict(batch, instance_labels=torch.full_like(batch['instance_labels'], 999))
        second = predict(model, changed, features, offsets, logits)
        self.assertTrue(torch.equal(first['delta_xy'], second['delta_xy']))

    def test_sampler_is_deterministic_tree_balanced_and_excludes_unknown_background(self):
        batch, _, _, _ = fixture()
        first = sample_queries(batch, SUPERVISION, 42)
        torch.manual_seed(999)
        second = sample_queries(batch, SUPERVISION, 42)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertTrue(bool((batch['instance_labels'][first] > 0).all()))

    def test_height_strata_retain_bottom_middle_and_top(self):
        count = 90
        batch = dict(coords=torch.stack((torch.zeros(count), torch.zeros(count), torch.arange(count)), dim=1),
            instance_labels=torch.ones(count, dtype=torch.long), batch_ids=torch.zeros(count, dtype=torch.long),
            masks_off=torch.ones(count, dtype=torch.bool))
        rows = sample_queries(batch, dict(max_trees=1, points_per_tree=3), 3)
        self.assertTrue(0 <= rows[0] < 30 and 30 <= rows[1] < 60 and 60 <= rows[2] < 90)

    def test_group_reduction_gives_small_and_large_tree_equal_weight(self):
        values = torch.tensor([1., 3., 3., 3., 3.])
        inverse = torch.tensor([0, 1, 1, 1, 1])
        self.assertAlmostEqual(float(group_average(values, inverse, 2)), 2.)

    def test_connection_targets_ignore_unknown_neighbors(self):
        batch, features, offsets, logits = fixture()
        model = branch()
        result = predict(model, batch, features, offsets, logits, torch.tensor([0, 3]))
        # Directly supply edges: query 0 sees same tree, different tree, unknown, background.
        result['neighbors'] = torch.tensor([[1, 4, 6, 7], [4, 1, 6, 7]])
        result['edge_valid'] = torch.ones(2, 4, dtype=torch.bool)
        result['edge_logits'] = torch.zeros(2, 4, requires_grad=True)
        loss, stats = structure_loss(result, batch, offsets, SUPERVISION)
        loss.backward()
        gradient = result['edge_logits'].grad
        self.assertTrue(torch.equal(gradient[:, 2], torch.zeros(2)))
        self.assertEqual(stats['positive_edges'], 2)
        self.assertEqual(stats['negative_edges'], 4)
        self.assertTrue(bool((gradient[:, 0] < 0).all()))
        self.assertTrue(bool((gradient[:, 1] > 0).all()))

    def test_isolated_single_tree_and_exact_targets_have_finite_gradients(self):
        batch, features, offsets, logits = fixture()
        batch = {name: value[:1] for name, value in batch.items()}
        offsets = batch['offset_labels'].clone()
        model = branch()
        result = predict(model, batch, features[:1], offsets, logits[:1], torch.tensor([0]))
        loss, stats = structure_loss(result, batch, offsets, SUPERVISION)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(stats['positive_edges'], 0)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))

    def test_frozen_inputs_receive_no_gradients(self):
        batch, features, offsets, logits = fixture()
        features.requires_grad_()
        offsets.requires_grad_()
        logits.requires_grad_()
        model = branch()
        rows = sample_queries(batch, SUPERVISION, 42)
        result = predict(model, batch, features, offsets, logits, rows)
        loss, _ = structure_loss(result, batch, offsets, SUPERVISION)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(offsets.grad)
        self.assertIsNone(logits.grad)
        self.assertGreater(float(model.correction[-1].weight.grad.abs().sum()), 0.)

    def test_connections_participate_in_vote_prediction(self):
        batch, features, offsets, logits = fixture()
        model = branch()
        with torch.no_grad():
            model.correction[-1].weight.normal_(0, .1)
        result = predict(model, batch, features, offsets, logits, torch.tensor([0, 3]))
        result['delta_xy'].square().sum().backward()
        magnitude = sum(float(p.grad.abs().sum()) for p in model.connection.parameters() if p.grad is not None)
        self.assertGreater(magnitude, 0.)

    def test_batch_labels_are_separate_trees_in_the_objective(self):
        batch, features, offsets, logits = fixture()
        batch['batch_ids'][3:] = 1
        batch['instance_labels'][3:6] = 1
        model = branch()
        selected = sample_queries(batch, SUPERVISION, 42)
        result = predict(model, batch, features, offsets, logits, selected)
        loss, stats = structure_loss(result, batch, offsets, SUPERVISION)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(stats['sampled_trees'], 2)

    def test_state_and_optimizer_roundtrip_preserve_the_next_update(self):
        batch, features, offsets, logits = fixture()
        model = branch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        rows = sample_queries(batch, SUPERVISION, 42)

        def update(network, opt):
            opt.zero_grad(set_to_none=True)
            loss, _ = structure_loss(predict(network, batch, features, offsets, logits, rows),
                                     batch, offsets, SUPERVISION)
            loss.backward()
            opt.step()

        update(model, optimizer)
        buffer = io.BytesIO()
        torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=False)
        restored = branch()
        restored.load_state_dict(saved['net'])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
        restored_optimizer.load_state_dict(deepcopy(saved['optimizer']))
        update(model, optimizer)
        update(restored, restored_optimizer)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
