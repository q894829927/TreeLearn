"""Guard formal training against stress plans, point caps and checkpoint growth."""
import unittest
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from tree_learn.experiments.ptv3_memory_v1.formal import (
    formal_config, validate_config, checkpoint_target, CONFIG, TOTAL_UPDATES)
from tree_learn.experiments.ptv3_treelearn.train import load_config
from tree_learn.experiments.ptv3_memory_v1.run import config
from tree_learn.experiments.ptv3_memory_v1 import formal as integration


class FormalTests(unittest.TestCase):
    def test_only_adapter_allocator_and_checkpoint_metadata_change(self):
        base, formal = config(), formal_config()
        self.assertEqual(load_config(CONFIG), formal)
        self.assertEqual(formal["model"], base["model"])
        self.assertEqual(formal["dataset_train"], base["dataset_train"])
        self.assertEqual(formal["experiment"]["sample_plan"], base["experiment"]["sample_plan"])
        self.assertEqual(formal["epochs"] * formal["examples_per_epoch"] // 2, TOTAL_UPDATES)
        validate_config(formal)

    def test_no_caps_no_stress_schedule_no_model_changes(self):
        for key, value in (("maximum_points", 400000), ("sample_plan", "/stress/s{seed}.npz")):
            changed = formal_config()
            changed["experiment"][key] = value
            with self.assertRaises(ValueError): validate_config(changed)
        changed = formal_config()
        changed["model"]["ptv3"]["enc_channels"][0] = 16
        with self.assertRaises(ValueError): validate_config(changed)

    def test_recovery_is_rolling_and_epoch_final_names_are_preserved(self):
        for update in (50, 100, 249950):
            self.assertEqual(checkpoint_target(f"/run/update_{update}.pth"), Path("/run/recovery.pth"))
        for name in ("last.pth", "epoch_20.pth", "final.pth"):
            self.assertEqual(checkpoint_target("/run/" + name), Path("/run/" + name))

    def test_scoped_integration_preserves_empty_run_guard_and_rolling_saves(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = formal_config()
            manifest = root / "crops.json"
            manifest.write_text(json.dumps(dict(crops=[dict(path="/crop.npz")])))
            cfg["experiment"]["crop_manifest"] = str(manifest)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            output = root / "run"
            model = torch.nn.Module()
            model.backbone = torch.nn.Identity()
            model.bn = torch.nn.BatchNorm1d(2)
            model.bn.num_batches_tracked.fill_(2)
            saved = []
            def save(path, *args):
                saved.append(Path(path).name)
                Path(path).write_bytes(b"checkpoint-format-tested-by-reference-suite")
            def train(args):
                self.assertFalse(output.exists())
                output.mkdir()
                integration.reference.build_model(cfg)
                integration.reference.save_checkpoint(output / "update_50.pth", *([None] * 4), 50, {}, {})
                integration.reference.save_checkpoint(output / "last.pth", *([None] * 4), 250, {}, {})
                integration.reference.save_checkpoint(output / "update_250.pth", *([None] * 4), 250, {}, {})
            with patch.dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=integration.ALLOCATOR), \
                 patch.object(integration, "formal_config", return_value=cfg), \
                 patch.object(integration, "environment", return_value={}), \
                 patch.object(integration, "git_state", return_value={}), \
                 patch.object(torch.cuda, "set_per_process_memory_fraction"), \
                 patch.object(torch.cuda.memory, "get_allocator_backend", return_value="native"), \
                 patch.object(integration, "build_optimized_model", return_value=model), \
                 patch.object(integration.reference, "save_checkpoint", side_effect=save) as saver, \
                 patch.object(integration.reference, "run", side_effect=train):
                builder = integration.reference.build_model
                integration.run(SimpleNamespace(config=str(config_path), seed=42, work_dir=str(output), resume=None))
                self.assertIs(integration.reference.build_model, builder)
                self.assertIs(integration.reference.save_checkpoint, saver)
            self.assertEqual(saved, ["recovery.pth", "last.pth"])
            latest = json.loads((output / "latest_checkpoint.json").read_text())
            self.assertEqual(latest["global_update"], 250)
            self.assertFalse(latest["is_formal_final"])
            self.assertFalse(model.backbone._forward_pre_hooks)


if __name__ == "__main__": unittest.main()
