import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED_MODULE_PATH = ROOT / 'tree_learn' / 'util' / 'seed_quality.py'
SEED_SPEC = importlib.util.spec_from_file_location(
    'seed_quality_standalone', SEED_MODULE_PATH)
SEED = importlib.util.module_from_spec(SEED_SPEC)
SEED_SPEC.loader.exec_module(SEED)

GENERATOR_PATH = ROOT / 'tools' / 'data_gen' / 'gen_seed_quality_data.py'
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    'seed_quality_generator_standalone', GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(GENERATOR)


class SeedQualityFeatureTests(unittest.TestCase):

    def test_candidate_mask_matches_strict_treelearn_boundaries(self):
        logits = np.asarray([
            [1.0, 1.0], [1.0, 1.0], [10.0, 0.0], [0.0, 10.0],
        ])
        verticality = np.asarray([0.6001, 0.6, 0.7, 0.7])
        offsets = np.asarray([
            [0.0, 0.0, 3.999], [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0], [0.0, 0.0, 0.0],
        ])
        mask = SEED.candidate_seed_mask(
            logits, verticality, offsets,
            tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0)
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_local_complexity_is_finite_and_cell_isolated(self):
        votes = np.asarray([
            [0.1, 0.1], [0.2, 0.1], [2.1, 2.1],
        ])
        probability = np.asarray([0.8, 0.6, 0.9])
        verticality = np.asarray([0.7, 0.9, 0.5])
        height = np.asarray([0.2, 0.4, 0.8])
        features, names = SEED.compute_local_complexity_features(
            votes, probability, verticality, height, cell_sizes=[1.0])
        self.assertEqual(features.shape, (3, 8))
        self.assertEqual(len(names), 8)
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features[0, 0]), np.log1p(2), places=6)
        self.assertAlmostEqual(float(features[2, 0]), np.log1p(1), places=6)

        changed = probability.copy()
        changed[:2] = 0.1
        modified, _ = SEED.compute_local_complexity_features(
            votes, changed, verticality, height, cell_sizes=[1.0])
        self.assertFalse(np.array_equal(features[:2], modified[:2]))
        np.testing.assert_array_equal(features[2], modified[2])

    def test_coverage_targets_preserve_tree_cells_and_tree_quota(self):
        votes = np.asarray([
            [0.1, 0.1], [0.2, 0.1], [1.1, 0.1],
            [5.1, 0.1], [5.2, 0.1], [9.1, 0.1],
        ])
        labels = np.asarray([1, 1, 1, 2, 2, 0])
        utility = np.asarray([0.3, 0.9, 0.2, 0.4, 0.8, 1.0])
        result = SEED.compute_coverage_targets(
            votes, labels, utility, cell_size=1.0,
            min_seeds_per_tree=1)
        np.testing.assert_array_equal(
            np.flatnonzero(result['cell_representative']), [1, 2, 4])
        np.testing.assert_array_equal(
            np.flatnonzero(result['tree_quota']), [1, 4])
        np.testing.assert_array_equal(
            np.flatnonzero(result['coverage_critical']), [1, 2, 4])
        self.assertFalse(result['coverage_critical'][-1])

    def make_artifact(self, labels=None, offset_labels=None):
        coords = np.asarray([
            [0.0, 0.0, 1.0], [0.1, 0.0, 1.1],
            [2.0, 0.0, 1.0], [2.1, 0.0, 1.1],
            [4.0, 0.0, 1.0], [6.0, 0.0, 1.0],
        ], dtype=np.float32)
        logits = np.tile(
            np.asarray([[4.0, -4.0]], dtype=np.float32), (6, 1))
        offsets = np.zeros((6, 3), dtype=np.float32)
        offsets[:, 2] = -1.0
        targets = (
            np.zeros((6, 3), dtype=np.float32)
            if offset_labels is None else offset_labels)
        target_labels = (
            np.asarray([1, 1, 2, 2, 0, -1])
            if labels is None else np.asarray(labels))
        backbone = np.arange(24, dtype=np.float32).reshape(6, 4)
        verticality = np.full(6, 0.8, dtype=np.float32)
        return SEED.build_seed_quality_artifact(
            coords, logits, offsets, targets, target_labels,
            backbone, verticality,
            complexity_scales=(0.3, 0.6, 1.2),
            min_seeds_per_tree=1)

    def test_full_artifact_has_dual_targets_and_no_gt_input_leakage(self):
        artifact = self.make_artifact()
        arrays = artifact['arrays']
        metadata = artifact['metadata']
        self.assertEqual(arrays['backbone_features'].shape, (6, 4))
        self.assertEqual(arrays['scalar_features'].shape, (6, 29))
        self.assertEqual(metadata['scalar_dim'], 29)
        self.assertEqual(metadata['num_supervised_trees'], 2)
        self.assertEqual(metadata['num_critical_trees'], 2)
        self.assertEqual(metadata['seed_artifact_schema_version'], 2)
        self.assertEqual(arrays['target_base_vote_xy'].shape, (6, 2))
        self.assertEqual(arrays['target_vote_residual_xy'].shape, (6, 2))
        tree = arrays['target_is_tree'].astype(bool)
        np.testing.assert_allclose(
            arrays['base_votes_xy'][tree] +
            arrays['target_vote_residual_xy'][tree],
            arrays['target_base_vote_xy'][tree])
        np.testing.assert_allclose(
            np.linalg.norm(
                arrays['target_vote_residual_xy'][tree], axis=1),
            arrays['target_vote_error_xy'][tree])
        self.assertTrue(np.all(
            arrays['target_coverage_critical'] <= arrays['target_is_tree']))

        changed_targets = np.ones((6, 3), dtype=np.float32)
        changed = self.make_artifact(
            labels=[7, 7, 8, 8, 0, -1],
            offset_labels=changed_targets)
        for name in (
                'candidate_indices', 'coords', 'base_votes_xy',
                'backbone_features', 'scalar_features'):
            np.testing.assert_array_equal(
                arrays[name], changed['arrays'][name])
        self.assertFalse(np.array_equal(
            arrays['target_tree_id'], changed['arrays']['target_tree_id']))
        self.assertFalse(np.array_equal(
            arrays['target_vote_error_xy'],
            changed['arrays']['target_vote_error_xy']))

    def test_round_trip_and_generator_validation(self):
        artifact = self.make_artifact()
        with tempfile.TemporaryDirectory() as directory:
            npz_path, metadata_path = SEED.save_seed_quality_artifact(
                artifact, directory, source_plot='A1N', split='train')
            metadata = GENERATOR.validate_seed_artifact(
                npz_path, metadata_path)
            self.assertEqual(metadata['source_plot'], 'A1N')
            self.assertEqual(metadata['split'], 'train')
            self.assertTrue(metadata['all_supervised_trees_covered'])
            with np.load(npz_path, allow_pickle=False) as data:
                self.assertEqual(data['scalar_features'].shape[1], 29)

    def test_generator_accepts_legacy_schema_but_rejects_partial_targets(self):
        artifact = self.make_artifact()
        with tempfile.TemporaryDirectory() as directory:
            npz_path, metadata_path = SEED.save_seed_quality_artifact(
                artifact, directory, source_plot='A1N', split='train')
            with np.load(npz_path, allow_pickle=False) as data:
                legacy = {
                    name: data[name] for name in data.files
                    if name not in {
                        'target_base_vote_xy',
                        'target_vote_residual_xy',
                    }}
            np.savez_compressed(npz_path, **legacy)
            with Path(metadata_path).open(encoding='utf-8') as file:
                metadata = json.load(file)
            metadata.pop('seed_artifact_schema_version', None)
            Path(metadata_path).write_text(
                json.dumps(metadata), encoding='utf-8')
            recovered = GENERATOR.validate_seed_artifact(
                npz_path, metadata_path)
            self.assertEqual(recovered['source_plot'], 'A1N')

            partial = dict(legacy)
            partial['target_base_vote_xy'] = np.zeros((6, 2))
            np.savez_compressed(npz_path, **partial)
            with self.assertRaisesRegex(ValueError, 'complete pair'):
                GENERATOR.validate_seed_artifact(npz_path, metadata_path)


class SeedQualityGeneratorTests(unittest.TestCase):

    @staticmethod
    def settings():
        return {
            'splits': {
                'train': ['A1N'],
                'validation': ['G4N'],
            },
            'gate': {
                'expected_backbone_dim': 32,
                'expected_scalar_dim': 29,
                'min_train_candidates': 100,
                'min_validation_candidates': 50,
                'min_train_critical_candidates': 10,
                'min_train_non_tree_candidates': 5,
                'min_known_label_rate': 0.99,
                'min_critical_rate': 0.01,
                'max_critical_rate': 0.5,
                'min_independent_groups': 1,
                'min_audit_candidates': 20,
            },
        }

    @staticmethod
    def rows():
        common = {
            'num_valid_candidates': 100,
            'num_tree_candidates': 80,
            'num_non_tree_candidates': 20,
            'num_unknown_candidates': 0,
            'num_reliable_candidates': 50,
            'num_critical_candidates': 20,
            'num_supervised_trees': 5,
            'num_critical_trees': 5,
            'all_supervised_trees_covered': True,
            'backbone_dim': 32,
            'scalar_dim': 29,
            'known_label_rate': 1.0,
            'critical_rate': 0.25,
            'artifact_path': 'unused.npz',
        }
        train = dict(common)
        train.update({
            'source_plot': 'A1N', 'source_group': 'A1',
            'split': 'train', 'num_candidates': 100,
        })
        validation = dict(common)
        validation.update({
            'source_plot': 'G4N', 'source_group': 'G4',
            'split': 'validation', 'num_candidates': 100,
        })
        return [train, validation]

    def test_full_summary_gate_and_dimension_failure(self):
        settings = self.settings()
        rows = self.rows()
        summary = GENERATOR.summarize(
            rows, settings, ['A1N', 'G4N'], False, 20)
        self.assertTrue(summary['gate']['passed'])

        rows[1]['scalar_dim'] = 28
        failed = GENERATOR.summarize(
            rows, settings, ['A1N', 'G4N'], False, 20)
        self.assertFalse(failed['gate']['dimensions_match'])
        self.assertFalse(failed['gate']['passed'])

    def test_summary_markdown_and_json_are_serializable(self):
        summary = GENERATOR.summarize(
            self.rows(), self.settings(), ['A1N', 'G4N'], False, 20)
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = GENERATOR.write_summary(
                summary, directory)
            with json_path.open(encoding='utf-8') as file:
                recovered = json.load(file)
            text = markdown_path.read_text(encoding='utf-8')
            self.assertTrue(recovered['gate']['passed'])
            self.assertIn('## Gate\n\n', text)
            self.assertIn('train candidates: 100', text)


if __name__ == '__main__':
    unittest.main()