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
