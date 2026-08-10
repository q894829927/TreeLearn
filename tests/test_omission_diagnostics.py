import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tree_learn.util.omission_diagnostics import (
    OMISSION_CATEGORIES,
    compute_gt_omission_diagnostics,
    save_gt_omission_diagnostics,
)


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1] / 'tools' / 'data_gen' /
    'gen_gt_omission_diagnostics.py')
SPEC = importlib.util.spec_from_file_location(
    'gen_gt_omission_diagnostics', GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class OmissionDiagnosticsTests(unittest.TestCase):

    @staticmethod
    def synthetic_case():
        labels = np.repeat(np.arange(1, 8), 5)
        labels = np.concatenate([labels, np.zeros(10, dtype=np.int64)])
        count = len(labels)
        coords = np.zeros((count, 3), dtype=np.float32)
        coords[:, 2] = np.tile(np.arange(5), 9)
        logits = np.column_stack([
            np.full(count, 5.0), np.zeros(count)]).astype(np.float32)
        offsets = np.zeros((count, 3), dtype=np.float32)
        verticality = np.ones(count, dtype=np.float32)
        predictions = np.zeros(count, dtype=np.int64)
        initial = np.full(count, -1, dtype=np.int64)

        predictions[0:5] = 1                    # detected
        logits[5:10] = [0.0, 5.0]             # semantic failure
        verticality[10:15] = 0.0              # seed failure
        # GT 4 has seeds but HDBSCAN rejects all of them.
        initial[20:25] = 5
        predictions[20:25] = 5                # GT 5 undersegmented
        predictions[35:45] = 5                # background in same prediction
        initial[25:30] = 6
        predictions[25:27] = 6                # GT 6 fragmentation
        predictions[27:29] = 7
        initial[30:35] = 8
        predictions[30] = 8                   # GT 7 partial coverage
        initial[0:5] = 1
        return (
            coords, logits, offsets, verticality,
            labels, predictions, initial)

    def test_categories_form_expected_first_failure_partition(self):
        result = compute_gt_omission_diagnostics(
            *self.synthetic_case(),
            tau_min=2,
            min_fragment_overlap_fraction=0.1,
            min_fragment_overlap_points=1)
        expected = {'detected': 1}
        expected.update({category: 1 for category in OMISSION_CATEGORIES})
        self.assertEqual(result['category_counts'], expected)
        self.assertEqual(result['baseline']['tp'], 1)
        self.assertEqual(result['baseline']['fn'], 6)
        self.assertEqual(len(result['rows']), 7)

    def test_match_threshold_is_strict_like_official_evaluation(self):
        coords = np.zeros((4, 3), dtype=np.float32)
        logits = np.tile([5.0, 0.0], (4, 1)).astype(np.float32)
        offsets = np.zeros((4, 3), dtype=np.float32)
        verticality = np.ones(4, dtype=np.float32)
        labels = np.asarray([1, 1, 2, 2])
        predictions = np.ones(4, dtype=np.int64)
        initial = np.ones(4, dtype=np.int64)
        result = compute_gt_omission_diagnostics(
            coords, logits, offsets, verticality,
            labels, predictions, initial,
            tau_min=1, min_fragment_overlap_points=1)
        self.assertEqual(result['baseline']['tp'], 0)
        self.assertEqual(result['category_counts']['undersegmentation'], 2)

    def test_save_roundtrip_is_json_serializable(self):
        result = compute_gt_omission_diagnostics(
            *self.synthetic_case(),
            tau_min=2,
            min_fragment_overlap_fraction=0.1,
            min_fragment_overlap_points=1)
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = save_gt_omission_diagnostics(
                result, directory, 'V1', 'validation')
            self.assertTrue(Path(csv_path).is_file())
            with Path(json_path).open(encoding='utf-8') as file:
                payload = json.load(file)
            self.assertEqual(payload['source_plot'], 'V1')
            self.assertEqual(payload['num_gt_trees'], 7)

    def test_generator_rejects_wytham_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(
                'output_root: out\nsource_root: data\n'
                'pipeline_template: template.yaml\ncheckpoint: x\n'
                'validation_plots: [Wytham]\ngate: {}\n',
                encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must not mention Wytham'):
                GENERATOR.load_settings(path)


if __name__ == '__main__':
    unittest.main()
