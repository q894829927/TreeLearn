"""Server-only CPU regression checks for saved-mask diagnosis; no model needed."""

import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from tree_learn.mask_refinement.evaluation import contingency, evaluate_labels, macro_metrics
from tree_learn.query_segmentation.diagnosis import analyze_instances, subset_metrics


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('query_diagnosis_tool', ROOT / 'tools/query_segmentation/diagnose.py')
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def fixture():
    labels = np.array([1] * 6 + [0] * 4)
    predictions = np.array([1] * 4 + [2] * 2 + [0] * 4)
    records = [dict(instance_id=1, count=4, final_count=4, score=.9),
               dict(instance_id=2, count=10, final_count=2, score=.8)]
    return labels, predictions, records


class DiagnosisTests(unittest.TestCase):
    def test_all_subsets_reproduce_existing_metrics_with_unknowns_strict_iou_and_empty_predictions(self):
        labels = np.array([1] * 6 + [2] * 4 + [0] * 3 + [-1] * 2)
        predictions = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 4, 4, 0, 9, 9])
        table = contingency(labels, predictions)
        self.assertNotIn(9, table[2])
        for bits in itertools.product((False, True), repeat=len(table[2])):
            keep = np.array(bits)
            actual, tp, fp = subset_metrics(table, keep)
            selected = np.where(np.isin(predictions, table[2][keep]), predictions, 0)
            expected = evaluate_labels(labels, selected)
            for key in actual:
                if key == 'matched_gt_ids':
                    self.assertEqual(actual[key], expected[key])
                else:
                    self.assertAlmostEqual(actual[key], expected[key], places=12)
            self.assertEqual(int(tp.sum()), expected['tp'])
            self.assertEqual(int(fp.sum()), expected['fp'])
        full, _, _ = subset_metrics(table, np.ones(len(table[2]), dtype=bool))
        self.assertEqual(full['tp'], 1)  # The two masks with IoU exactly .5 are not matches.

    def test_retention_diagnosis_separates_fragment_from_high_quality_true_mask(self):
        labels, predictions, records = fixture()
        report, details = analyze_instances(contingency(labels, predictions), records)
        self.assertEqual(report['original']['fp'], 1)
        self.assertEqual(report['fp_retained_below_half'], 1)
        self.assertEqual(report['fp_quality_at_least_0_7'], 1)
        self.assertEqual(report['fp_best_gt_already_matched'], 1)
        for name in ('retained_at_least_half', 'quality_times_retention'):
            self.assertEqual(report['policies'][name]['metrics']['tp'], 1)
            self.assertEqual(report['policies'][name]['metrics']['fp'], 0)
            self.assertEqual(report['policies'][name]['removed_tp'], 0)
        self.assertEqual(report['policies']['quality_at_least_0_6']['metrics']['fp'], 1)
        self.assertAlmostEqual(details[1]['retained_fraction'], .2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            analyze_instances(contingency(labels, predictions), records + [records[0]])

    def test_saved_file_diagnosis_preserves_inputs_and_detects_wrong_evaluation(self):
        labels, predictions, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache/validation/G4N'
            evaluation = root / 'work/evaluation'
            cache.mkdir(parents=True)
            (evaluation / 'G4N').mkdir(parents=True)
            np.save(cache / 'labels.npy', labels)
            np.save(evaluation / 'G4N/refined_labels.npy', predictions)
            (evaluation / 'G4N/instances.json').write_text(json.dumps(records))
            (root / 'cache/manifest.json').write_text(json.dumps(dict(forests=[
                dict(plot='G4N', split='validation', cache='validation/G4N')])))
            source_config = root / 'source.yaml'
            source_config.write_text(yaml.safe_dump(dict(output_root=str(root / 'cache'))))
            baseline, refined = evaluate_labels(labels, labels), evaluate_labels(labels, predictions)
            summary = dict(partial=False, checkpoint_step=2000, checkpoint='unused/best.pth',
                           baseline_macro=macro_metrics([baseline]), refined_macro=macro_metrics([refined]),
                           policy=dict(quality_threshold=.4), forests=[dict(plot='G4N', baseline=baseline,
                           refined=refined, prediction=dict(final_instances=2))])
            (evaluation / 'summary.json').write_text(json.dumps(summary))
            paths = [cache / 'labels.npy', evaluation / 'G4N/refined_labels.npy',
                     evaluation / 'G4N/instances.json', evaluation / 'summary.json']
            before = [p.read_bytes() for p in paths]
            settings = dict(work_dir=str(root / 'work'), source_config=str(source_config))
            WORKFLOW.diagnose(settings)
            self.assertEqual(before, [p.read_bytes() for p in paths])
            result = json.loads((root / 'work/failure_diagnosis/summary.json').read_text())
            self.assertTrue(result['analysis_only'])
            self.assertEqual(result['comparisons']['retained_at_least_half']['totals']['fp'], 0)
            self.assertTrue((root / 'work/failure_diagnosis/G4N/instances.csv').exists())
            summary['forests'][0]['refined']['tp'] = 999
            (evaluation / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'do not reproduce source metrics'):
                WORKFLOW.diagnose(settings)


if __name__ == '__main__':
    unittest.main()
