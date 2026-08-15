import unittest

import yaml

from tree_learn.util import get_config
from tools.diagnostics.run_sparse_unet_validation_matrix import (
    partition_model_specs,
)


CONFIGS = [
    'configs/experiments/sparse_attention_unet/'
    'train_t1_b1_partial_seed42.yaml',
    'configs/experiments/sparse_attention_unet/'
    'train_t1_m1_sparse_se_seed42.yaml',
    'configs/experiments/sparse_attention_unet/'
    'train_t1_m2_selective_kernel_seed42.yaml',
    'configs/experiments/sparse_attention_unet/'
    'train_t1_m3_hcag_seed42.yaml',
    'configs/experiments/sparse_attention_unet/'
    'train_t1_m4_window_attention_seed42.yaml',
]


class SparseUNetTournamentConfigTests(unittest.TestCase):

    def test_t1_configs_share_controlled_training_settings(self):
        configs = [get_config(path) for path in CONFIGS]
        reference = configs[0]
        for config in configs:
            self.assertEqual(config.seed, 42)
            self.assertEqual(config.dataloader.train.batch_size, 1)
            self.assertEqual(config.gradient_accumulation_steps, 2)
            self.assertEqual(
                list(config.model.unet_finetune.train_decoder_levels),
                [0, 1])
            self.assertTrue(config.model.unet_finetune.freeze_encoder)
            self.assertEqual(
                config.grouping.seed_confidence_keep_ratio, 1.0
            ) if hasattr(config, 'grouping') else None
            self.assertEqual(
                config.optimizer.paramwise.decoder_lr, 0.0001)
            self.assertEqual(
                config.dataset_train.data_root,
                reference.dataset_train.data_root)
            self.assertEqual(
                config.dataset_test.data_root,
                reference.dataset_test.data_root)

    def test_candidate_types_are_unique(self):
        types = [
            get_config(path).model.unet_enhancement.type
            for path in CONFIGS
        ]
        self.assertEqual(types, [
            'identity', 'sparse_se', 'selective_kernel',
            'hcag', 'window_attention'])


    def test_t1_matrix_records_hardware_elimination(self):
        with open(
                'configs/experiments/sparse_attention_unet/'
                't1_validation_matrix.yaml', encoding='utf-8') as file:
            matrix = yaml.safe_load(file)
        active, eliminated = partition_model_specs(matrix['models'])
        self.assertNotIn(
            'm2_selective_kernel', {name for name, _ in active})
        self.assertEqual(len(eliminated), 1)
        self.assertEqual(eliminated[0]['model'], 'm2_selective_kernel')
        self.assertEqual(
            eliminated[0]['stage'], 't1_training_hardware_gate')

    def test_validation_guard_accepts_fixed_forests(self):
        config = get_config(
            'configs/experiments/sparse_attention_unet/'
            '_pipeline_validation_common.yaml')
        self.assertEqual(config.grouping.max_cluster_seed_points, 600000)

if __name__ == '__main__':
    unittest.main()
