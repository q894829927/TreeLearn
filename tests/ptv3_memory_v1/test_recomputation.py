"""CUDA regression tests for mutable Point and sparse feature semantics."""
import copy
import gc
import unittest
import weakref
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from tree_learn.experiments.ptv3_treelearn.vendor.model import Point, Block, PointSequential
from tree_learn.experiments.ptv3_treelearn.provenance import seed_all, rng_state, restore_rng, state_equal
from tree_learn.experiments.ptv3_memory_v1 import resolved_config
from tree_learn.experiments.ptv3_memory_v1.adapters import (
    ExpandedInversePoint, checkpoint_region, check_blocks, install_stages)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/spconv/Flash required")
class RecomputationTests(unittest.TestCase):
    def setUp(self):
        seed_all(73)
        self.grid = torch.cartesian_prod(torch.arange(6), torch.arange(6), torch.arange(6)).int().cuda()
        self.x = torch.randn(len(self.grid), 32, device="cuda")
        self.y = torch.randn_like(self.x)

    def point(self, cls=Point, divergent=True):
        x = self.x.detach().clone().requires_grad_()
        y = self.y.detach().clone().requires_grad_() if divergent else x
        point = cls(feat=x, grid_coord=self.grid, coord=self.grid.float(),
                    batch=torch.zeros(len(x), device="cuda", dtype=torch.long))
        point.serialization(order=("z", "z-trans", "hilbert", "hilbert-trans"), shuffle_orders=True)
        point.sparsify()
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(y)
        return point, x, y

    def test_two_sparse_inputs_output_gradient_rng_and_no_parent_retention(self):
        block = Block(32, 2, patch_size=16, attn_drop=.1, proj_drop=.1, drop_path=.2,
                      cpe_indice_key="test", enable_flash=True,
                      upcast_attention=False, upcast_softmax=False).cuda().train()
        initial = copy.deepcopy(block.state_dict())
        def evaluate(recompute):
            block.load_state_dict(initial); block.zero_grad(set_to_none=True)
            seed_all(91)
            point, x, y = self.point()
            parent = Point(feat=torch.ones(1, device="cuda"))
            point.pooling_parent = parent
            parent_ref = weakref.ref(parent)
            with torch.cuda.amp.autocast():
                output = checkpoint_region(point, (block,)).feat if recompute else block(point).feat
                loss = output.float().square().mean()
            del parent, point
            gc.collect()
            self.assertIsNone(parent_ref())
            loss.backward()
            return (output.detach(), [x.grad.clone(), y.grad.clone()]+[
                p.grad.clone() for p in block.parameters()], rng_state())
        a, b = evaluate(False), evaluate(True)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        error = sum((x.double()-y.double()).square().sum() for x,y in zip(a[1],b[1]))
        norm = sum(x.double().square().sum() for x in a[1])
        self.assertLess(float((error/norm).sqrt()), 1e-3)
        self.assertTrue(state_equal(a[2], b[2]))
        # Deliberately unifying the two inputs must alter the CPE result.
        seed_all(91)
        bad, _, _ = self.point(divergent=False)
        with torch.no_grad(), torch.cuda.amp.autocast():
            incorrect = block(bad).feat
        self.assertGreater(float((a[0]-incorrect).abs().max()), .01)

    def test_serialization_exact_for_four_orders_and_multiple_batches(self):
        grid = torch.cat([self.grid, self.grid+2])
        batch = torch.arange(2, device="cuda").repeat_interleave(len(self.grid))
        state = rng_state()
        outputs = []
        for cls in (Point, ExpandedInversePoint):
            restore_rng(state)
            point = cls(grid_coord=grid, batch=batch)
            point.serialization(order=("z", "z-trans", "hilbert", "hilbert-trans"), shuffle_orders=True)
            outputs.append((point, rng_state()))
        for key in ("serialized_code", "serialized_order", "serialized_inverse"):
            self.assertTrue(torch.equal(outputs[0][0][key], outputs[1][0][key]))
        self.assertTrue(state_equal(outputs[0][1], outputs[1][1]))

    def test_batchnorm_and_interleaved_stage_are_rejected(self):
        block = Block(8, 2, patch_size=16, norm_layer=nn.BatchNorm1d, enable_flash=False)
        with self.assertRaisesRegex(ValueError, "BatchNorm"):
            check_blocks((block,))
        stage = PointSequential(Block(8, 2, enable_flash=False), nn.Identity(), Block(8, 2, enable_flash=False))
        backbone = SimpleNamespace(enc=nn.Sequential(stage), dec=nn.Sequential())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            install_stages(backbone)

    def test_differentiable_geometry_is_rejected(self):
        point, _, _ = self.point()
        point.coord.requires_grad_()
        with self.assertRaisesRegex(ValueError, "fixed geometric"):
            checkpoint_region(point, (lambda p:p,))

    def test_no_grad_inputs_keep_parameter_gradients(self):
        point, _, _ = self.point()
        point.feat = point.feat.detach()
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.sparse_conv_feat.features.detach())
        projection = nn.Linear(32,32).cuda()
        def forward(p):
            p.feat = projection(p.feat)
            return p
        checkpoint_region(point, (forward,)).feat.sum().backward()
        self.assertIsNotNone(projection.weight.grad)

    def test_configuration_changes_only_checkpoint_policy(self):
        source = {"model": {"backbone":"ptv3", "activation_checkpointing":True,
                            "ptv3":{"patch_size":1024}}, "dataset":{"maximum_points":None}}
        for variant in ("reference", "serialization", "block", "stage", "block_nested", "stage_nested", "offload"):
            changed = resolved_config(source, variant)
            changed["model"]["activation_checkpointing"] = True
            self.assertEqual(source, changed)
            self.assertIsNot(source, changed)

    def test_profiler_peak_without_inherited_context(self):
        from tree_learn.experiments.ptv3_memory_v1.profile import MemoryAudit, _ACTIVE
        with tempfile.TemporaryDirectory() as directory:
            audit = MemoryAudit(Path(directory), operators=True)
            baseline = torch.cuda.memory_allocated()
            with audit.attach(nn.Identity()):
                # Simulate the CUDA autograd worker's absent ContextVar.
                token = _ACTIVE.set(None)
                workspace = torch.ones(32*1024*1024, device="cuda")
                del workspace
                _ACTIVE.reset(token)
            audit.finish()
            self.assertGreaterEqual(audit.peaks["peak_allocated"], baseline+128*1024**2)


if __name__ == "__main__":
    unittest.main()
