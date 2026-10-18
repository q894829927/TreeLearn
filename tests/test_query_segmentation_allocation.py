"""Run on the server: exported labels, diagnosis equivalence and GT-free inference."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from tree_learn.mask_refinement.data import file_digest
from tree_learn.mask_refinement.evaluation import contingency, evaluate_labels, macro_metrics
from tree_learn.query_segmentation.allocation import filter_allocation
from tree_learn.query_segmentation.diagnosis import analyze_instances
from tree_learn.query_segmentation.inference import predict_forest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('materialize_allocation', ROOT / 'tools/query_segmentation/apply_allocation_filter.py')
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def fixture():
    truth = np.array([1] * 6 + [2] * 2 + [-1, 0])
    labels = np.array([1] * 4 + [2] * 2 + [3] * 2 + [0, 0])
    records = [dict(instance_id=1, count=4, final_count=4, score=.9),
               dict(instance_id=2, count=10, final_count=2, score=.8),
               dict(instance_id=3, count=4, final_count=2, score=.8)]
    return truth, labels, records


class TinyForest:
    metadata = dict(num_points=10)
    centers = np.array([[0., 0.]])
    arrays = dict(coords=np.zeros((10, 3), dtype=np.float32))

    def make_block(self, center, config, seed):
        return dict(tokens=np.zeros((2, 38), dtype=np.float32), core=np.ones(2, dtype=bool),
                    indices=np.arange(10), center=np.zeros(3))

    def inputs(self, indices, center, radius):
        values = np.zeros((len(indices), 38), dtype=np.float32)
        values[:, 0] = indices
        return values


class FixedMasks:
    """Two overlapping fixed masks; no learned network or GT."""
    def eval(self):
        return self

    def decode(self, tokens, core, progress):
        return dict(queries=[torch.tensor([[0.], [1.]])],
                    geometry=[dict(center=torch.zeros(2, 2), sigma=torch.ones(2), strength=torch.ones(2))],
                    quality_logits=torch.logit(torch.tensor([.9, .8])))

    def point_predictions(self, inputs, queries, geometry):
        indices = inputs[:, 0]
        first, second = indices < 6, (indices >= 2) & (indices < 9)
        masks = torch.stack([first if int(row) == 0 else second for row in queries[0][:, 0]])
        return dict(masks=[torch.where(masks, 10., -10.)])


class AllocationTests(unittest.TestCase):
    def test_export_matches_diagnosis_and_is_chunk_invariant_with_compacted_ids(self):
        truth, labels, records = fixture()
        diagnosis, _ = analyze_instances(contingency(truth, labels), records)
        original = labels.copy()
        for chunk in (1, 7):
            destination = np.empty_like(labels)
            kept, rejected, statistics = filter_allocation(labels, records, destination, .4, chunk)
            np.testing.assert_array_equal(labels, original)
            np.testing.assert_array_equal(destination, [1, 1, 1, 1, 0, 0, 2, 2, 0, 0])
            self.assertEqual([r['source_instance_id'] for r in kept], [1, 3])
            self.assertAlmostEqual(kept[1]['allocation_score'], .4)  # Inclusive boundary.
            self.assertEqual(rejected[0]['instance_id'], 2)
            self.assertEqual(statistics['removed_points'], 2)
            actual = evaluate_labels(truth, destination)
            expected = diagnosis['policies']['quality_times_retention']['metrics']
            for key in expected:
                if key == 'matched_gt_ids':
                    self.assertEqual(actual[key], expected[key])
                else:
                    self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_empty_all_rejected_and_stale_counts(self):
        _, labels, records = fixture()
        destination = np.empty_like(labels)
        kept, rejected, _ = filter_allocation(labels, records, destination, 1.)
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(rejected), 3)
        self.assertTrue(np.all(destination == 0))
        self.assertEqual(filter_allocation(destination, [], destination)[0], [])
        stale = [dict(r) for r in records]
        stale[0]['final_count'] = 3
        with self.assertRaisesRegex(ValueError, 'counts disagree'):
            filter_allocation(labels, stale, destination)

    def test_fresh_prediction_and_cached_filter_agree_without_reading_gt(self):
        settings = dict(seed=42, blocks=dict(radius=16.), inference=dict(mask_threshold=.5, quality_threshold=.4,
                        nms_iou=.5, nms_containment=.8, chunk_size=3, max_block_memberships=1000,
                        max_forest_memberships=10000, log_every_blocks=1, keep_proposals=False))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_load = np.load

            def guarded(path, *args, **kwargs):
                if Path(path).name in ('labels.npy', 'baseline.npy', 'gt_counts.npz'):
                    raise AssertionError('GT or baseline read during prediction.')
                return original_load(path, *args, **kwargs)

            with mock.patch('numpy.load', side_effect=guarded):
                raw, _ = predict_forest(FixedMasks(), TinyForest(), settings, root / 'raw', torch.device('cpu'))
                records = json.loads((root / 'raw/instances.json').read_text())
                expected = np.empty_like(raw)
                filter_allocation(raw, records, expected, .4)
                settings['inference']['allocation_filter'] = 'quality_times_retention'
                settings['inference']['chunk_size'] = 7
                actual, statistics = predict_forest(FixedMasks(), TinyForest(), settings, root / 'fresh', torch.device('cpu'))
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual, [1] * 6 + [0] * 4)
                self.assertEqual(statistics['allocation_filter']['removed_instances'], 1)

    def test_materialized_files_reproduce_reference_without_touching_source(self):
        truth, labels, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, evaluation = root / 'cache/validation/G4N', root / 'work/evaluation'
            cache.mkdir(parents=True)
            (evaluation / 'G4N').mkdir(parents=True)
            np.save(cache / 'labels.npy', truth)
            np.save(cache / 'baseline.npy', np.where(truth > 0, truth, 0))
            np.save(evaluation / 'G4N/refined_labels.npy', labels)
            (evaluation / 'G4N/instances.json').write_text(json.dumps(records))
            (root / 'cache/manifest.json').write_text(json.dumps(dict(forests=[
                dict(plot='G4N', split='validation', cache='validation/G4N')])))
            source_config = root / 'source.yaml'
            source_config.write_text(yaml.safe_dump(dict(output_root=str(root / 'cache'))))
            baseline, refined = evaluate_labels(truth, np.where(truth > 0, truth, 0)), evaluate_labels(truth, labels)
            source = dict(protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5', partial=False,
                          checkpoint='unused/best.pth', checkpoint_step=2000, policy=dict(quality_threshold=.4),
                          baseline_macro=macro_metrics([baseline]), refined_macro=macro_metrics([refined]),
                          forests=[dict(plot='G4N', baseline=baseline, refined=refined,
                                        prediction=dict(final_instances=3, inference_seconds=.1))])
            (evaluation / 'summary.json').write_text(json.dumps(source))
            diagnosis, _ = analyze_instances(contingency(truth, labels), records)
            diagnosis['plot'] = 'G4N'
            (root / 'work/failure_diagnosis').mkdir()
            (root / 'work/failure_diagnosis/summary.json').write_text(json.dumps(dict(
                source_evaluation_sha256=file_digest(evaluation / 'summary.json'), forests=[diagnosis])))
            before = [p.read_bytes() for p in (evaluation / 'summary.json', evaluation / 'G4N/refined_labels.npy')]
            report = WORKFLOW.materialize(dict(work_dir=str(root / 'work'), source_config=str(source_config)))
            self.assertTrue(report['diagnostic_reference_verified'])
            self.assertEqual(report['forests'][0]['refined']['fp'], 0)
            self.assertEqual(before, [p.read_bytes() for p in (evaluation / 'summary.json', evaluation / 'G4N/refined_labels.npy')])
            self.assertTrue((root / 'work/evaluation_allocation_filtered/G4N/instances.json').exists())


if __name__ == '__main__':
    unittest.main()
