import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'diagnostics' /
    'diagnose_instance_split_oracle.py')
SPEC = importlib.util.spec_from_file_location(
    'diagnose_instance_split_oracle', MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class InstanceSplitOracleTests(unittest.TestCase):

    @staticmethod
    def mode(tp, fp, fn, recovered, gain, lost=0):
        total_gt = tp + fn
        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'completeness': tp / total_gt,
            'commission': fp / (tp + fp),
            'f1': 2 * tp / (2 * tp + fp + fn),
            'recovered_undersegmented_trees': recovered,
            'lost_baseline_trees': lost,
            'f1_gain_pp': gain,
            'completeness_gain_pp': 0.0,
            'commission_reduction_pp': 0.0,
        }

    def test_load_settings_rejects_wytham(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(
                'output_root: out\nq3_reference_root: q3\n'
                'source_root: Wytham\npipeline_template: p\ncheckpoint: c\n'
                'validation_plots: [V1]\nsplit_oracle: {}\ngate: {}\n',
                encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must not mention Wytham'):
                ORACLE.load_settings(path)

    def test_aggregate_recommends_vertical_only_after_both_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'q4'
            q3 = root / 'q3'
            (output / 'validation' / 'V1').mkdir(parents=True)
            (q3 / 'validation' / 'V1').mkdir(parents=True)
            baseline = {
                'tp': 1, 'fp': 1, 'fn': 1,
                'completeness': 0.5, 'commission': 0.5, 'f1': 0.5,
            }
            report = {
                'source_plot': 'V1', 'split': 'validation',
                'baseline': baseline,
                'num_split_target_predictions': 1,
                'num_undersegmented_gt_trees': 1,
                'modes': {
                    'gt_extraction_ceiling': self.mode(2, 1, 0, 1, 30.0),
                    'raw_xy_kmeans_oracle': self.mode(1, 1, 1, 0, 0.0),
                    'base_vote_kmeans_oracle': self.mode(1, 1, 1, 0, 0.0),
                    'vertical_axis_kmeans_oracle': self.mode(
                        2, 1, 0, 1, 30.0),
                },
            }
            (output / 'validation' / 'V1' / 'summary.json').write_text(
                json.dumps(report), encoding='utf-8')
            (q3 / 'validation' / 'V1' / 'summary.json').write_text(
                json.dumps({
                    'baseline': baseline,
                    'category_counts': {'undersegmentation': 1},
                }), encoding='utf-8')
            settings = {
                'output_root': str(output),
                'q3_reference_root': str(q3),
                'validation_plots': ['V1'],
                'gate': {
                    'expected_validation_plots': 1,
                    'min_gt_ceiling_f1_gain_pp': 1.0,
                    'min_gt_ceiling_completeness_gain_pp': 1.0,
                    'min_geometry_recovered_trees': 1,
                    'min_geometry_f1_gain_pp': 1.0,
                    'max_geometry_commission_increase_pp': 1.0,
                    'max_geometry_completeness_drop_pp': 1.0,
                    'min_geometry_plot_wins': 1,
                    'min_vertical_gain_over_base_vote_pp': 0.3,
                    'min_vertical_additional_recovered_trees': 1,
                },
            }
            result = ORACLE.aggregate(settings)
            self.assertTrue(result['gate']['passed'])
            self.assertTrue(result['vertical_attention_gate']['passed'])
            self.assertEqual(
                result['recommendation'],
                'vertical_topology_split_attention')


if __name__ == '__main__':
    unittest.main()
