"""Verify bounded stress scheduling without accessing production crop files."""
import unittest
import tempfile
from pathlib import Path

from tree_learn.experiments.ptv3_memory_v1.stability import selected_crops, schedule, write_config, UPDATES
from tree_learn.experiments.ptv3_memory_v1.run import config
from tree_learn.experiments.ptv3_treelearn.train import load_config


class StabilityPlanTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(path=f"/input/{i}.npz",points=i+1,source_forest=f"f{i%18}",sha256=str(i))
                   for i in range(25000)]
        self.manifest=dict(status="verified",crops=self.rows)

    def test_selected_indices_preserve_real_manifest_records(self):
        selected=selected_crops(self.manifest)
        self.assertEqual(len(selected),6)
        self.assertEqual(selected[3]["points"],25000)
        self.assertEqual(len({c["source_forest"] for c in selected[3:]}),3)
        for item in selected:
            self.assertEqual(item["path"],self.rows[item["index"]]["path"])
            self.assertEqual(item["sha256"],self.rows[item["index"]]["sha256"])

    def test_schedule_is_bounded_reproducible_and_changes_augmentation(self):
        selected=selected_crops(self.manifest)
        a=schedule(selected,42)
        self.assertEqual(a,schedule(selected,42))
        self.assertEqual(UPDATES,12)
        self.assertEqual(len(a),24)
        self.assertEqual({row["update"] for row in a},set(range(1,13)))
        self.assertEqual(sum(row["role"]==3 for row in a),8)
        self.assertEqual(len({row["augmentation_seed"] for row in a}),24)
        self.assertNotEqual([r["augmentation_seed"] for r in a],
                            [r["augmentation_seed"] for r in schedule(selected,43)])
        self.assertEqual([r["crop_index"] for r in a],
                         [r["crop_index"] for r in schedule(selected,43)])

    def test_unverified_data_and_unsupported_seeds_are_rejected(self):
        with self.assertRaises(ValueError):selected_crops(dict(status="pending",crops=self.rows))
        with self.assertRaises(ValueError):selected_crops(dict(status="verified",crops=self.rows[:100]))
        with self.assertRaises(ValueError):schedule(selected_crops(self.manifest),45)

    def test_reference_yaml_reader_preserves_all_config_types(self):
        options=config()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"config.yaml"
            write_config(path,options)
            loaded=load_config(path)
            self.assertEqual(loaded,options)
            self.assertIsInstance(loaded["scheduler"]["warmup_lr_init"],float)


if __name__=="__main__":unittest.main()
