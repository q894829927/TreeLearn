import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_instance_split_deployability.py')
SPEC = importlib.util.spec_from_file_location(
    'diagnose_instance_split_deployability', MODULE_PATH)
DEPLOYABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOYABILITY)


class InstanceSplitDeployabilityTests(unittest.TestCase):

    @staticmethod
    def metrics(f1_gain=0.6, completeness_gain=0.0,
                commission_reduction=0.0, plot_wins=4):
        return {
            'f1_gain_pp': f1_gain,
            'completeness_gain_pp': completeness_gain,
            'commission_reduction_pp': commission_reduction,
            'nonnegative_f1_plots': plot_wins,
        }

    @staticmethod
    def gate():
        return {
            'min_f1_gain_pp': 0.5,
            'max_completeness_drop_pp': 0.2,
            'max_commission_increase_pp': 0.5,
            'min_nonnegative_plots': 4,
        }

    def test_mode_gate_requires_effect_safety_and_consistency(self):
        self.assertTrue(DEPLOYABILITY.evaluate_mode_gate(
            self.metrics(), self.gate())['passed'])
        self.assertFalse(DEPLOYABILITY.evaluate_mode_gate(
            self.metrics(f1_gain=0.49), self.gate())['passed'])
        self.assertFalse(DEPLOYABILITY.evaluate_mode_gate(
            self.metrics(completeness_gain=-0.21), self.gate())['passed'])
        self.assertFalse(DEPLOYABILITY.evaluate_mode_gate(
            self.metrics(commission_reduction=-0.51), self.gate())['passed'])
        self.assertFalse(DEPLOYABILITY.evaluate_mode_gate(
            self.metrics(plot_wins=3), self.gate())['passed'])

    def test_load_settings_rejects_wytham(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(
                'output_root: out\nq3_reference_root: q3\n'
                'q4a_reference_root: q4a\nsource_root: Wytham\n'
                'pipeline_template: p\ncheckpoint: c\n'
                'validation_plots: [V1]\nsplit_deployability: {}\n'
                'gate: {}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must not mention Wytham'):
                DEPLOYABILITY.load_settings(path)

    def test_artifact_must_reproduce_q4a_known_k_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q4a_dir = root / 'q4a' / 'validation' / 'V1'
            q4b_dir = root / 'q4b' / 'validation' / 'V1'
            q4a_dir.mkdir(parents=True)
            q4b_dir.mkdir(parents=True)
            baseline = {'tp': 1, 'fp': 1, 'fn': 1}
            reference = {
                'tp': 2, 'fp': 1, 'fn': 0,
                'recovered_undersegmented_trees': 1,
                'lost_baseline_trees': 0,
                'proposed_splits': 1,
                'accepted_splits': 1,
                'accepted_prediction_ids': [7],
            }
            (q4a_dir / 'summary.json').write_text(json.dumps({
                'baseline': baseline,
                'num_split_target_predictions': 1,
                'expected_undersegmented_gt_ids': [2],
                'modes': {'raw_xy_kmeans_oracle': reference},
            }), encoding='utf-8')
            modes = {}
            for mode in DEPLOYABILITY.SPLIT_DEPLOYABILITY_MODES:
                modes[mode] = dict(reference)
            # Prediction instance IDs are arbitrary run-local labels. A
            # relabelled but metric-identical rerun must remain valid.
            modes['known_k_oracle_accept']['accepted_prediction_ids'] = [99]
            report = {
                'source_plot': 'V1', 'split': 'validation',
                'baseline': baseline,
                'num_split_target_predictions': 1,
                'num_undersegmented_gt_trees': 1,
                'expected_undersegmented_gt_ids': [2],
                'modes': modes,
            }
            path = q4b_dir / 'summary.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            validated = DEPLOYABILITY.validate_artifact(
                path, 'V1', root / 'q4a')
            self.assertEqual(validated['source_plot'], 'V1')

            report['modes']['known_k_oracle_accept']['tp'] = 1
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                DEPLOYABILITY.validate_artifact(
                    path, 'V1', root / 'q4a')

    def test_aggregate_selects_minimal_head_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q4a_root = root / 'q4a'
            q4b_root = root / 'q4b'
            q4a_plot = q4a_root / 'validation' / 'V1'
            q4b_plot = q4b_root / 'validation' / 'V1'
            q4a_plot.mkdir(parents=True)
            q4b_plot.mkdir(parents=True)
            baseline = {
                'tp': 1, 'fp': 1, 'fn': 1,
                'completeness': 0.5, 'commission': 0.5, 'f1': 0.5,
            }
            split = {
                'tp': 2, 'fp': 1, 'fn': 0,
                'completeness': 1.0, 'commission': 1 / 3, 'f1': 0.8,
                'recovered_undersegmented_trees': 1,
                'lost_baseline_trees': 0,
                'proposed_splits': 1, 'accepted_splits': 1,
                'accepted_prediction_ids': [7],
                'f1_gain_pp': 30.0,
                'completeness_gain_pp': 50.0,
                'commission_reduction_pp': 100 * (0.5 - 1 / 3),
            }
            q4a_plot.joinpath('summary.json').write_text(json.dumps({
                'baseline': baseline,
                'num_split_target_predictions': 1,
                'expected_undersegmented_gt_ids': [2],
                'modes': {'raw_xy_kmeans_oracle': split},
            }), encoding='utf-8')
            q4a_root.joinpath('summary.json').write_text(json.dumps({
                'baseline': baseline,
                'modes': {'raw_xy_kmeans_oracle': split},
                'gate': {'passed': True},
                'recommendation': 'simple_geometry_split',
            }), encoding='utf-8')
            q4b_plot.joinpath('summary.json').write_text(json.dumps({
                'source_plot': 'V1', 'split': 'validation',
                'baseline': baseline,
                'num_split_target_predictions': 1,
                'num_undersegmented_gt_trees': 1,
                'expected_undersegmented_gt_ids': [2],
                'modes': {
                    mode: dict(split)
                    for mode in DEPLOYABILITY.SPLIT_DEPLOYABILITY_MODES
                },
            }), encoding='utf-8')
            settings = {
                'output_root': str(q4b_root),
                'q4a_reference_root': str(q4a_root),
                'validation_plots': ['V1'],
                'gate': {
                    **self.gate(),
                    'expected_validation_plots': 1,
                    'min_nonnegative_plots': 1,
                },
            }
            result = DEPLOYABILITY.aggregate(settings)
            self.assertTrue(result['integrity_gate']['passed'])
            self.assertTrue(
                result['deployment_gates']['fixed_k2_accept_all']['passed'])
            self.assertEqual(
                result['recommendation'], 'candidate_classifier_only')


if __name__ == '__main__':
    unittest.main()
