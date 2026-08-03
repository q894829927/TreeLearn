import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools' / 'data_gen' / 'gen_instance_quality_data.py'
)
SPEC = importlib.util.spec_from_file_location(
    'gen_instance_quality_data_standalone', MODULE_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class InstanceQualityGeneratorTests(unittest.TestCase):

    def test_plot_group_keeps_paired_scans_together(self):
        self.assertEqual(GENERATOR.plot_group('A1N'), 'A1')
        self.assertEqual(GENERATOR.plot_group('G4W'), 'G4')
        self.assertEqual(GENERATOR.plot_group('LG1'), 'LG1')

    def test_split_validation_rejects_paired_plot_leakage(self):
        with self.assertRaises(ValueError):
            GENERATOR.validate_splits({
                'train': ['A1N'],
                'validation': ['A1W'],
            })

    def test_split_validation_accepts_independent_groups(self):
        result = GENERATOR.validate_splits({
            'train': ['A1N', 'A1W', 'LG1'],
            'validation': ['G4N', 'G4W'],
        })
        self.assertEqual(result['train_groups'], ['A1', 'LG1'])
        self.assertEqual(result['validation_groups'], ['G4'])

    def test_runtime_cleanup_is_scoped_below_runtime_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'runtime'
            plot = root / 'A1N'
            plot.mkdir(parents=True)
            (plot / 'temporary.bin').write_bytes(b'test')
            GENERATOR.safe_cleanup_runtime(plot, root)
            self.assertFalse(plot.exists())
            with self.assertRaises(ValueError):
                GENERATOR.safe_cleanup_runtime(root, root)


    def test_audit_sample_is_stratified_and_deterministic(self):
        rows = []
        for index in range(80):
            if index < 5:
                iou = 0.0
                is_edge = True
                valid = False
            elif index < 15:
                iou = 0.35
                is_edge = False
                valid = True
            elif index < 40:
                iou = 0.1
                is_edge = False
                valid = True
            else:
                iou = 0.8
                is_edge = False
                valid = True
            rows.append({
                'instance_id': str(index + 1),
                'artifact_path': f'plot_{index // 20}.npz',
                'target_is_edge': str(is_edge),
                'target_valid': str(valid),
                'target_max_iou': str(iou),
                'target_classification_valid': str(
                    valid and (iou < 0.25 or iou >= 0.5)),
                'target_is_true_tree': str(iou >= 0.5),
                'split': 'validation' if index >= 60 else 'train',
            })

        with tempfile.TemporaryDirectory() as directory:
            first_path = GENERATOR.write_audit_sample(
                rows, directory, sample_size=50, seed=42)
            first = first_path.read_text(encoding='utf-8')
            second_path = GENERATOR.write_audit_sample(
                rows, directory, sample_size=50, seed=42)
            second = second_path.read_text(encoding='utf-8')
            with second_path.open(newline='', encoding='utf-8') as file:
                sampled = list(csv.DictReader(file))

        self.assertEqual(first, second)
        self.assertEqual(len(sampled), 50)
        self.assertTrue(any(
            GENERATOR.bool_value(row['target_is_edge'])
            for row in sampled))
        self.assertTrue(any(
            0.25 <= float(row['target_max_iou']) < 0.5
            for row in sampled))
        self.assertTrue(any(
            row['split'] == 'validation' for row in sampled))
    def test_full_gate_requires_every_fixed_plot(self):
        settings = {
            'splits': {
                'train': ['A1N', 'LG1'],
                'validation': ['G4N'],
            },
            'gate': {
                'min_positive_instances': 1,
                'min_negative_instances': 0,
                'min_independent_groups': 1,
                'min_random_audit_instances': 1,
            },
        }
        rows = [{
            'source_plot': 'A1N',
            'split': 'train',
            'target_valid': 'True',
            'target_classification_valid': 'True',
            'target_is_true_tree': 'True',
        }]
        summary = GENERATOR.summarize(
            rows,
            settings,
            selected_plots=['A1N'],
            pilot=False,
            manual_audit_confirmed=True,
        )
        self.assertFalse(summary['gate']['all_fixed_plots_present'])
        self.assertFalse(summary['gate']['passed'])

if __name__ == '__main__':
    unittest.main()
