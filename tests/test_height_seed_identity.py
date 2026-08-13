import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).parents[1] / 'tree_learn' / 'model' /
    'height_identity.py')
SPEC = importlib.util.spec_from_file_location(
    'height_identity_standalone', MODULE_PATH)
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


class HeightSeedIdentityTests(unittest.TestCase):

    def make_inputs(self):
        base_logits = torch.tensor([
            [2.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.0, 0.0]])
        base_offsets = torch.tensor([
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            [0.0, 0.0, -5.0], [0.0, 0.0, -1.0]])
        input_features = torch.tensor([[0.8], [0.8], [0.8], [0.4]])
        return base_logits, base_offsets, input_features

    def test_base_seed_mask_matches_all_three_conditions(self):
        base_logits, base_offsets, input_features = self.make_inputs()
        mask = IDENTITY.get_base_seed_mask(
            base_logits, base_offsets, input_features)
        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_loss_is_one_sided_and_has_gradient(self):
        base_logits, base_offsets, input_features = self.make_inputs()
        adapted_logits = base_logits.clone().requires_grad_(True)
        safe_loss, _ = IDENTITY.seed_semantic_identity_loss(
            base_logits, adapted_logits, base_offsets, input_features)
        self.assertEqual(float(safe_loss), 0.0)

        damaged_logits = adapted_logits.detach().clone()
        damaged_logits[0] = torch.tensor([-1.0, 1.0])
        damaged_logits.requires_grad_(True)
        loss, _ = IDENTITY.seed_semantic_identity_loss(
            base_logits, damaged_logits, base_offsets, input_features)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertGreater(float(damaged_logits.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(damaged_logits.grad[2].abs().sum()), 0.0)

    def test_retention_separates_semantic_and_full_seed(self):
        base_logits, base_offsets, input_features = self.make_inputs()
        adapted_logits = base_logits.clone()
        adapted_logits[0] = torch.tensor([-1.0, 1.0])
        adapted_offsets = base_offsets.clone()
        adapted_offsets[1, 2] = -5.0
        result = IDENTITY.seed_retention_metrics(
            base_logits, adapted_logits, base_offsets,
            adapted_offsets, input_features)
        self.assertEqual(result['base_seed_count'], 2)
        self.assertEqual(result['semantic_retained_count'], 1)
        self.assertEqual(result['full_retained_count'], 0)


if __name__ == '__main__':
    unittest.main()
