import unittest

from tree_learn.util import get_config


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


if __name__ == '__main__':
    unittest.main()
