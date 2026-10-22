"""Run: python -m unittest discover -s tests/ptv3_treelearn -v (CUDA required)."""
import copy
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from munch import Munch

from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.model.tree_learn import voxelize
from tree_learn.util.train import point_wise_loss
from tree_learn.util.pipeline import ensemble, get_instances, assign_remaining_points_nearest_neighbor
from tree_learn.util.parser import get_config
from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset, resource_indices, sample_digest
from tree_learn.experiments.ptv3_treelearn.model import build_model, canonicalize_voxels
from tree_learn.experiments.ptv3_treelearn.pipeline import safe_instances, safe_assign, safe_save_data
from tree_learn.experiments.ptv3_treelearn.evaluate import evaluate_arrays
from tree_learn.experiments.ptv3_treelearn.provenance import load_initialization, seed_all, rng_state, restore_rng
from tree_learn.experiments.ptv3_treelearn.train import load_config, rewind_update_log
from tree_learn.experiments.ptv3_treelearn.tiles import StreamingSampleGenerator
from tree_learn.experiments.ptv3_treelearn.npz_io import save_npz_with_hash
from tree_learn.experiments.ptv3_treelearn.aggregate import aggregate


class ContractTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.grouping = Munch(dict(tree_conf_thresh=.5, tau_vert=.6, tau_off=4, tau_group=.15, tau_min=50, use_hdbscan=True))

    def test_e0_is_original_class_and_state(self):
        config = load_config("configs/experiments/ptv3_treelearn/e0_scratch.yaml")
        model = build_model(config)
        self.assertIs(type(model), TreeLearn)
        seed_all(42)
        reference = TreeLearn(**config["model"])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)

    def test_constant_features_and_duplicate_voxel(self):
        xyz = torch.tensor([[1., 1., 1.], [1.01, 1.01, 1.01], [2., 2., 2.]], device="cuda")
        feats, indices, inverse, _ = voxelize(torch.cat([xyz, torch.randn(3, 1, device="cuda")], 1),
            torch.zeros(3, dtype=torch.long, device="cuda"), 1, .1, False, False, 3)
        self.assertEqual(feats.shape[1], 4)
        self.assertTrue(torch.equal(feats, torch.ones_like(feats)))
        self.assertEqual(int(inverse[0]), int(inverse[1]))
        self.assertNotEqual(int(inverse[0]), int(inverse[2]))

    def test_canonical_voxel_ids_preserve_each_point_and_order(self):
        features = torch.arange(20, device="cuda").reshape(5, 4).float()
        indices = torch.tensor([[1, 2, 3, 4], [0, 3, 2, 1], [1, 1, 2, 3], [0, 1, 1, 1], [0, 1, 1, 2]], device="cuda")
        point_map = torch.tensor([0, 4, 3, 1, 2, 4], device="cuda")
        shape = torch.tensor([4, 4, 5], device="cuda")
        first = canonicalize_voxels(features, indices, point_map, shape)
        permutation = torch.tensor([3, 1, 4, 0, 2], device="cuda")
        reverse = torch.argsort(permutation)
        second = canonicalize_voxels(features[permutation], indices[permutation], reverse[point_map], shape)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(first[0][first[2]], features[point_map], rtol=0, atol=0)
        torch.testing.assert_close(first[1][first[2]], indices[point_map], rtol=0, atol=0)

    def test_grid_safety_preserves_valid_main_path_and_covers_augmented_extent(self):
        config = load_config("configs/experiments/ptv3_treelearn/e0_scratch_resource.yaml")
        repaired = build_model(config).cuda(); repaired.eval()
        config["model"]["cover_input_spatial_shape"] = False
        reference = build_model(config).cuda(); reference.eval()
        reference.load_state_dict(repaired.state_dict())
        xyz = torch.from_numpy(np.random.RandomState(3).uniform(0, 20, (2048,3)).astype(np.float32))
        batch = dict(coords=xyz, input_feats=torch.zeros(len(xyz),1),
                     batch_ids=torch.zeros(len(xyz),dtype=torch.long), batch_size=1)
        with torch.no_grad():
            left = reference(batch, return_loss=False)
            right = repaired(batch, return_loss=False)
        self.assertEqual(repaired.last_spatial_support["used_shape"], [500,500,1000])
        torch.testing.assert_close(left["semantic_prediction_logits"], right["semantic_prediction_logits"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(left["offset_predictions"], right["offset_predictions"], rtol=1e-5, atol=1e-5)
        batch["coords"] = xyz * torch.tensor([3.3, 3.0, 1.])
        with torch.no_grad():
            result = repaired(batch, return_loss=False)
        support = repaired.last_spatial_support
        self.assertGreater(support["voxels_outside_main_fixed_shape"], 0)
        self.assertTrue(all(a <= b for a,b in zip(support["actual_grid_shape"], support["used_shape"])))
        self.assertTrue(torch.isfinite(result["offset_predictions"]).all())

    def test_ptv3_point_mapping_interleaved_and_duplicates(self):
        model = build_model(load_config("configs/experiments/ptv3_treelearn/e1_ptv3.yaml")).cuda()
        class Echo(torch.nn.Module):
            def forward(self, point):
                return Munch(feat=torch.cat([point["batch"][:, None], point["grid_coord"]], 1).float(),
                             grid_coord=point["grid_coord"], batch=point["batch"])
        model.backbone = Echo()
        xyz = torch.tensor([[1., 1., 1.], [11., 11., 11.], [1.01, 1.01, 1.01], [12., 12., 12.], [2., 2., 2.]], device="cuda")
        batches = torch.tensor([0, 1, 0, 1, 0], device="cuda")
        out, inverse = model.forward_backbone(xyz, torch.zeros(5, 1, device="cuda"), batches, 2)
        restored = out.features[inverse]
        torch.testing.assert_close(restored[:, 0].long(), batches)
        torch.testing.assert_close(restored[0], restored[2])
        self.assertFalse(torch.equal(restored[0], restored[4]))

    def test_standard_ptv3_output_order_and_batch_isolation(self):
        model = build_model(load_config("configs/experiments/ptv3_treelearn/e1_ptv3.yaml")).cuda()
        model.eval()
        n = 256
        grid = torch.stack([(torch.arange(n, device="cuda")*13) % 350,
                            (torch.arange(n, device="cuda")*17) % 350,
                            (torch.arange(n, device="cuda")*19) % 350], 1).int()
        grid = torch.cat([grid, grid], 0)
        feat = torch.randn(2*n, 4, device="cuda")
        batch = torch.arange(2, device="cuda").repeat_interleave(n)
        def forward(features):
            with torch.no_grad():
                return model.backbone({"feat": features, "grid_coord": grid.clone(), "coord": grid.float()*.1,
                    "batch": batch, "offset": torch.tensor([n, 2*n], device="cuda")})
        state = rng_state()
        output = forward(feat)
        changed = feat.clone(); changed[n:] *= 5
        restore_rng(state)
        output2 = forward(changed)
        torch.testing.assert_close(output.grid_coord, grid)
        torch.testing.assert_close(output.batch, batch)
        torch.testing.assert_close(output.feat[:n], output2.feat[:n], atol=1e-4, rtol=1e-4)
        heads = model.forward_head(type("Features", (), {"features": output.feat.float()})(), torch.arange(2*n, device="cuda"))
        self.assertEqual(heads["semantic_prediction_logits"].shape, (2*n, 2))
        self.assertEqual(heads["offset_predictions"].shape, (2*n, 3))

    def test_gt_partition_exact_and_edge_counts(self):
        dataset = object.__new__(TreeDataset)
        for n in (1, 11, 12, 40):
            xyz = np.arange(n*3).reshape(n, 3).astype(float) / 10
            labels, semantic = np.ones(n), np.zeros(n)
            actual, mask = dataset.getOffset(xyz, labels, semantic)
            minz = np.partition(xyz[:, 2], 10)[3] if n > 11 else xyz[:, 2].min()
            anchor = xyz[xyz[:, 2] <= minz + .5].mean(0).astype(np.float32)
            np.testing.assert_allclose(actual, anchor - xyz)
            self.assertTrue(mask.all())

    def test_tensor_checkpointing_gradient_and_bn_parity(self):
        config = load_config("configs/experiments/ptv3_treelearn/e1_ptv3.yaml")
        # Standard widths/depths, a deterministic sparse fixture spanning 35m.
        first = build_model(config).cuda()
        config["model"]["activation_checkpointing"] = True
        second = build_model(config).cuda()
        second.load_state_dict(first.state_dict())
        n = 256
        grid = torch.stack([(torch.arange(n, device="cuda")*13) % 350,
                            (torch.arange(n, device="cuda")*17) % 350,
                            (torch.arange(n, device="cuda")*19) % 350], 1).int()
        data = {"feat": torch.randn(n, 4, device="cuda"), "grid_coord": grid,
                "coord": grid.float()*.1, "batch": torch.zeros(n, dtype=torch.long, device="cuda")}
        state = rng_state()
        first.train(); out1 = first.backbone(dict(data)).feat
        out1.square().mean().backward()
        restore_rng(state)
        second.train(); out2 = second.backbone(dict(data)).feat
        out2.square().mean().backward()
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)
        for (name, p), (name2, q) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(name, name2)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=2e-4)
        for (name, p), (name2, q) in zip(first.named_buffers(), second.named_buffers()):
            self.assertEqual(name, name2)
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_unknown_and_empty_laz_round_trip(self):
        from tree_learn.util.data_preparation import load_data
        with tempfile.TemporaryDirectory() as temp:
            data = np.array([[1., 2., 3., -1.], [2., 3., 4., 1.]])
            safe_save_data(data, "laz", "unknown", temp)
            result = load_data(str(Path(temp)/"unknown.laz"))
            np.testing.assert_array_equal(result[:, 3], [-1, 1])
            safe_save_data(np.empty((0, 4)), "laz", "empty", temp)
            result = load_data(str(Path(temp)/"empty.laz"))
            self.assertEqual(len(result), 0)

    def test_streamed_tiles_match_main_arrays_and_metadata_exactly(self):
        from tree_learn.util.data_preparation import SampleGenerator
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            xyz = np.random.RandomState(5).uniform([-6, -5, 0], [6, 5, 20], (3000, 3)).astype(np.float32)
            labels = (np.arange(len(xyz)) % 7).astype(np.float32)
            np.savez(root/"forest.npz", points=xyz, labels=labels)
            np.savez(root/"features.npz", features=np.random.RandomState(9).rand(len(xyz), 1).astype(np.float32))
            kwargs = dict(plot_path=str(root/"forest.npz"), features_path=str(root/"features.npz"),
                          n_neigh_sor=None, multiplier_sor=None, rad=None, npoints_rad=None)
            reference = SampleGenerator(save_dir=str(root/"original"), **kwargs)
            streamed = StreamingSampleGenerator(save_dir=str(root/"streamed"), **kwargs)
            logger = logging.getLogger("tiles")
            reference.tile_generate_and_save(8, 13.5, .5, logger=logger)
            streamed.tile_generate_and_save(8, 13.5, .5, logger=logger)
            paths = sorted((root/"original/npz").glob("*.npz"))
            self.assertEqual(len(paths), len(list((root/"streamed/npz").glob("*.npz"))))
            for path in paths:
                with np.load(path) as a, np.load(root/"streamed/npz"/path.name) as b:
                    self.assertEqual(a.files, b.files)
                    for field in a.files:
                        np.testing.assert_array_equal(a[field], b[field])
                meta = path.with_suffix(".json").name
                self.assertEqual((root/"original/json"/meta).read_text(), (root/"streamed/json"/meta).read_text())

    def test_seeded_augmentation_ignore_and_shared_subsampling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"forest_0.npz"
            xyz = np.random.uniform(-17.5, 17.5, (1000, 3)).astype(np.float32)
            labels = np.tile([-1, 0, 1, 2], 250)
            np.savez(path, points=xyz, feat=np.zeros((1000, 1), np.float32), instance_label=labels, center=np.zeros(3))
            options = dict(paths=[path], data_augmentations=dict(jitter=True, flip=True, rot=True, scaled=True, point_jitter=True))
            full = PlannedDataset(**options)
            small = PlannedDataset(**options, maximum_points=200)
            before = np.random.get_state()
            sample = full[(0, 1234)]
            after = np.random.get_state()
            np.testing.assert_array_equal(before[1], after[1]); self.assertEqual(before[2], after[2])
            torch.rand(1000); np.random.rand(1000)
            replay = full[(0, 1234)]
            for a, b in zip(sample, replay):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            indices = resource_indices(sample, 200, 1234)
            for a, b in zip(sample, small[(0, 1234)]):
                torch.testing.assert_close(a[indices], b, rtol=0, atol=0)
            self.assertFalse(sample[-1][sample[2] == -1].any())
            self.assertFalse(sample[-2][sample[2] <= 0].any())

    def test_main_loss_formula_empty_masks_and_gradients(self):
        logits = torch.tensor([[2., -1.], [-.5, 1.]], device="cuda", requires_grad=True)
        offsets = torch.tensor([[3., 4., 0.], [0., 0., 12.]], device="cuda", requires_grad=True)
        labels = torch.tensor([0, 1], device="cuda")
        mask = torch.ones(2, dtype=torch.bool, device="cuda")
        sem, off = point_wise_loss(logits, offsets, mask, mask, labels, torch.zeros_like(offsets))
        torch.testing.assert_close(sem, torch.nn.functional.cross_entropy(logits, labels))
        self.assertEqual(float(off), 8.5)
        (50*sem + off).backward()
        self.assertTrue(torch.isfinite(logits.grad).all() and torch.isfinite(offsets.grad).all())
        sem, off = point_wise_loss(logits, offsets, ~mask, ~mask, labels, torch.zeros_like(offsets))
        self.assertEqual(float(sem+off), 0.)

    def test_hais_required_coverage_and_failure(self):
        model = TreeLearn()
        with tempfile.TemporaryDirectory() as temp:
            report = load_initialization(model, "data/source/model_weights/hais_ckpt_spconv2.pth", Path(temp)/"load.json")
            self.assertEqual(report["required_coverage"], 1.)
            self.assertFalse(report["is_complete_initialization"])
            torch.save({"net": {}}, Path(temp)/"empty.pth")
            with self.assertRaises(RuntimeError):
                load_initialization(model, Path(temp)/"empty.pth", Path(temp)/"failed.json")

    def test_grouping_normal_matches_main_and_five_nn(self):
        # Two real-size dense seed clusters plus a few upper-tree query points.
        rng = np.random.RandomState(13)
        xyz = np.vstack([rng.normal(0, .01, (80, 3)), rng.normal(5, .01, (80, 3)), [[0, 0, 7], [5, 5, 7]]]).astype(np.float32)
        logits = np.tile([4., -4.], (len(xyz), 1)).astype(np.float32)
        offsets = np.zeros_like(xyz); offsets[-2:, 2] = -7
        verticality = np.ones(len(xyz))
        actual = safe_instances(xyz, offsets, logits, self.grouping, verticality)
        expected = get_instances(xyz, offsets, logits, self.grouping, verticality, 0, 0, -1, 1)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(np.unique(actual[actual > 0])), 2)
        shifted = xyz + offsets
        np.testing.assert_array_equal(safe_assign(shifted, actual), assign_remaining_points_nearest_neighbor(shifted, actual, -1))

    def test_grouping_empty_no_seed_no_cluster_and_small_knn(self):
        for n in (0, 1, 20, 60):
            xyz = np.zeros((n, 3), np.float32)
            logits = np.tile([3., -3.], (n, 1)).astype(np.float32)
            result = safe_instances(xyz, xyz, logits, self.grouping, np.zeros(n))
            self.assertTrue(np.all(result == -1))
            np.testing.assert_array_equal(safe_assign(xyz, result), result)
            result = safe_instances(xyz, xyz, -logits, self.grouping, np.ones(n))
            self.assertTrue(np.all(result == 0))
        # All-noise HDBSCAN and an insufficient reference set.
        cfg = copy.deepcopy(self.grouping); cfg.tau_min = 50
        result = safe_instances(np.zeros((60, 3), np.float32), np.zeros((60, 3), np.float32),
            np.tile([3., -3.], (60, 1)).astype(np.float32), cfg, np.ones(60))
        self.assertTrue(np.all(result == -1))
        np.testing.assert_array_equal(safe_assign(np.zeros((3, 3)), np.array([1, -1, -1])), [1, 1, 1])

    def test_ensemble_main_averages_overlaps(self):
        xyz = np.array([[1., 2., 3.], [1., 2., 3.], [4., 5., 6.]], np.float32)
        result = ensemble(xyz, np.array([[2., 0.], [4., 0.], [0., 1.]]), np.zeros(3),
            np.arange(9).reshape(3, 3), np.zeros((3, 3)), np.ones(3), np.ones((3, 32)), np.ones((3, 1)))
        self.assertEqual(len(result[0]), 2)
        np.testing.assert_allclose(result[1][0], [3., 0.])

    def test_evaluation_perfect_and_empty_predictions(self):
        rng = np.random.RandomState(1)
        xyz = np.vstack([rng.normal(0, .01, (30, 3)), rng.normal(5, .01, (30, 3))])
        labels = np.repeat([1, 2], 30)
        config = get_config("configs/evaluation/evaluate.yaml")
        metrics, _, _ = evaluate_arrays(xyz, labels, xyz, labels, config)
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (2, 0, 0))
        self.assertEqual(metrics["detection_f1"], 1.)
        metrics, _, _ = evaluate_arrays(xyz, labels, xyz, np.zeros(60), config)
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (0, 0, 2))

    def test_rng_round_trip(self):
        state = rng_state()
        a = (np.random.rand(3), torch.rand(3), torch.rand(3, device="cuda"))
        restore_rng(state)
        b = (np.random.rand(3), torch.rand(3), torch.rand(3, device="cuda"))
        np.testing.assert_array_equal(a[0], b[0])
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)

    def test_resume_log_discards_only_unsaved_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"updates.jsonl"
            path.write_text('{"update":1}\n{"update":2}\n{"update":3}\n{"update":')
            rewind_update_log(temp, 2)
            self.assertEqual([json.loads(line)["update"] for line in path.read_text().splitlines()], [1, 2])
            archives = list(Path(temp).glob("updates_after_checkpoint_*.jsonl"))
            self.assertEqual(len(archives), 1)
            self.assertIn('"update":3', archives[0].read_text())

    def test_streaming_npz_hash_and_arrays_match_numpy(self):
        from tree_learn.experiments.ptv3_treelearn.provenance import sha256
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arrays = {"points": np.arange(300).reshape(100,3).astype(np.float32),
                      "feat": np.ones((100,1), np.float32), "instance_label": np.arange(100, dtype=np.int32),
                      "center": np.array([1.,2.,0.])}
            np.savez(root/"original.npz", **arrays)
            digest = save_npz_with_hash(root/"streamed.npz", arrays)
            self.assertEqual(digest, sha256(root/"streamed.npz"))
            with np.load(root/"original.npz") as left, np.load(root/"streamed.npz") as right:
                self.assertEqual(left.files, right.files)
                for field in left.files:
                    self.assertEqual(left[field].dtype, right[field].dtype)
                    np.testing.assert_array_equal(left[field], right[field])
            digest = save_npz_with_hash(root/"compressed.npz", arrays, compression_level=1)
            self.assertEqual(digest, sha256(root/"compressed.npz"))
            with np.load(root/"compressed.npz") as result:
                for name, value in arrays.items():
                    self.assertEqual(value.dtype, result[name].dtype)
                    np.testing.assert_array_equal(value, result[name])

    def test_aggregate_rejects_smoke_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"train").mkdir()
            (root/"metrics.json").write_text('{}')
            (root/"evaluation_manifest.json").write_text('{"gt_sha256":"gt"}')
            (root/"pipeline.json").write_text('{"checkpoint_sha256":"checkpoint","seed":42}')
            (root/"train/experiment_manifest.json").write_text('{"phase":"smoke"}')
            (root/"train/checkpoint_provenance.json").write_text('{"is_formal_final":false,"global_update":100}')
            rows=[dict(group="E1-ptv3",domain="L1W",seed=42,metrics=str(root/"metrics.json"),
                       pipeline_manifest=str(root/"pipeline.json"),training_run=str(root/"train"))]
            (root/"index.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,"Only complete"):
                aggregate(root/"index.json",root/"report.json")


if __name__ == "__main__":
    unittest.main()
