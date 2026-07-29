import math
import numpy as np
import torch
import os
from torch.utils.data import Dataset

INSTANCE_LABEL_IGNORE_IN_RAW_DATA = -1 # label for unlabeled in raw data
NON_TREE_CLASS_IN_RAW_DATA = 0 # label for non-trees in raw data
NON_TREE_CLASS_IN_PYTORCH_DATASET = 1 # semantic label for non-tree in pytorch dataset
TREE_CLASS_IN_PYTORCH_DATASET = 0 # semantic label for tree in pytorch dataset


class TreeDataset(Dataset):
    def __init__(self,
                 data_root,
                 inner_square_edge_length,
                 training,
                 logger,
                 data_augmentations=None,
                 base_anchor_mode='robust',
                 base_anchor_height=0.5,
                 base_anchor_floor_quantile=0.01,
                 upper_anchor_mode='crown_median',
                 upper_anchor_lower_ratio=0.55,
                 upper_anchor_upper_ratio=0.75,
                 upper_anchor_min_points=3,
                 stem_axis_lower_ratio=0.10,
                 stem_axis_upper_ratio=0.50,
                 stem_axis_num_bins=4,
                 stem_axis_verticality_threshold=0.60,
                 stem_axis_min_points_per_bin=3,
                 stem_axis_min_valid_bins=3,
                 upper_anchor_target_ratio=0.65):

        self.data_paths = sorted(os.path.join(data_root, path) for path in os.listdir(data_root))
        self.inner_square_edge_length = inner_square_edge_length
        self.logger = logger
        self.training = training
        self.data_augmentations = data_augmentations
        self.base_anchor_mode = base_anchor_mode
        self.base_anchor_height = base_anchor_height
        self.base_anchor_floor_quantile = base_anchor_floor_quantile
        self.upper_anchor_mode = upper_anchor_mode
        self.upper_anchor_lower_ratio = upper_anchor_lower_ratio
        self.upper_anchor_upper_ratio = upper_anchor_upper_ratio
        self.upper_anchor_min_points = upper_anchor_min_points
        self.stem_axis_lower_ratio = stem_axis_lower_ratio
        self.stem_axis_upper_ratio = stem_axis_upper_ratio
        self.stem_axis_num_bins = stem_axis_num_bins
        self.stem_axis_verticality_threshold = \
            stem_axis_verticality_threshold
        self.stem_axis_min_points_per_bin = \
            stem_axis_min_points_per_bin
        self.stem_axis_min_valid_bins = stem_axis_min_valid_bins
        self.upper_anchor_target_ratio = upper_anchor_target_ratio
        if self.base_anchor_mode not in ('legacy', 'robust'):
            raise ValueError("base_anchor_mode must be either 'legacy' or 'robust'.")
        if self.upper_anchor_mode not in ('crown_median', 'stem_axis'):
            raise ValueError(
                "upper_anchor_mode must be 'crown_median' or 'stem_axis'.")
        if not 0 <= self.base_anchor_floor_quantile < 0.5:
            raise ValueError('base_anchor_floor_quantile must be in [0, 0.5).')
        if not 0 <= self.upper_anchor_lower_ratio < self.upper_anchor_upper_ratio <= 1:
            raise ValueError('upper anchor ratios must satisfy 0 <= lower < upper <= 1.')
        if self.upper_anchor_min_points < 1:
            raise ValueError('upper_anchor_min_points must be positive.')
        if not 0 <= self.stem_axis_lower_ratio < \
                self.stem_axis_upper_ratio <= 1:
            raise ValueError(
                'stem-axis ratios must satisfy 0 <= lower < upper <= 1.')
        if self.stem_axis_num_bins < 1:
            raise ValueError('stem_axis_num_bins must be positive.')
        if not 0 <= self.stem_axis_verticality_threshold <= 1:
            raise ValueError(
                'stem_axis_verticality_threshold must be in [0, 1].')
        if self.stem_axis_min_points_per_bin < 1:
            raise ValueError(
                'stem_axis_min_points_per_bin must be positive.')
        if not 2 <= self.stem_axis_min_valid_bins <= \
                self.stem_axis_num_bins:
            raise ValueError(
                'stem_axis_min_valid_bins must be between 2 and '
                'stem_axis_num_bins.')
        if not 0 <= self.upper_anchor_target_ratio <= 1:
            raise ValueError(
                'upper_anchor_target_ratio must be in [0, 1].')
        mode = 'train' if training else 'test'
        self.logger.info(f'Load {mode} dataset: {len(self.data_paths)} scans')


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, index):
        # load data
        data_path = self.data_paths[index]
        data = np.load(data_path)
        
        # get entries
        xyz = data['points']
        input_feat = data['feat']

        instance_label = data['instance_label']
        semantic_label = np.empty(len(instance_label))
        semantic_label[instance_label == NON_TREE_CLASS_IN_RAW_DATA] = NON_TREE_CLASS_IN_PYTORCH_DATASET
        semantic_label[instance_label != NON_TREE_CLASS_IN_RAW_DATA] = TREE_CLASS_IN_PYTORCH_DATASET

        # get center of chunk (used for stitching tiles back together)
        if self.training:
            center = np.ones_like(xyz) # dummy value in training
        else:
            center = np.ones_like(xyz) * data['center']

        # transform data
        xyz = self.transform_train(xyz) if self.training else self.transform_test(xyz)

        # Generate base- and upper-anchor offsets online from existing instance labels.
        pt_offset_label, upper_offset_label, mask_valid_offset, mask_valid_upper = \
            self.getOffset(
                xyz, instance_label, semantic_label, input_feat=input_feat)
        
        # get masks for loss calculation
        mask_inner = self.get_mask_inner(xyz)
        mask_not_ignore = self.get_mask_not_ignore(instance_label)
        mask_off = mask_inner & mask_not_ignore & (semantic_label != NON_TREE_CLASS_IN_PYTORCH_DATASET) & mask_valid_offset
        mask_upper = mask_inner & mask_not_ignore & (semantic_label != NON_TREE_CLASS_IN_PYTORCH_DATASET) & mask_valid_upper
        mask_sem = mask_inner & mask_not_ignore

        xyz = torch.from_numpy(xyz)
        instance_label = torch.from_numpy(instance_label)
        semantic_label = torch.from_numpy(semantic_label)
        mask_inner = torch.from_numpy(mask_inner)
        mask_off = torch.from_numpy(mask_off)
        mask_upper = torch.from_numpy(mask_upper)
        mask_sem = torch.from_numpy(mask_sem)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        upper_offset_label = torch.from_numpy(upper_offset_label)
        input_feat = torch.from_numpy(input_feat)
        center = torch.from_numpy(center)

        return (xyz, input_feat, instance_label, semantic_label, pt_offset_label,
                upper_offset_label, center, mask_inner, mask_off, mask_upper, mask_sem)


    def get_mask_not_ignore(self, instance_label):
        mask_ignore = instance_label == INSTANCE_LABEL_IGNORE_IN_RAW_DATA
        mask_not_ignore = np.logical_not(mask_ignore)
        return mask_not_ignore


    def get_mask_inner(self, xyz):
        # mask of inner square
        inf_norm = np.linalg.norm(xyz[:, :-1], ord=np.inf, axis=1)
        mask_inner = inf_norm <= (self.inner_square_edge_length/2)
        return mask_inner


    def point_jitter(self, points, sigma=0.1, clip=0.2):
        jitter = np.clip(sigma * np.random.randn(points.shape[0], 3), -1 * clip, clip)
        points += jitter
        return points


    def transform_train(self, xyz, aug_prob=0.5, aug_prob_point_jitter=0.25):
        if self.data_augmentations["point_jitter"] == True:
            if np.random.random() <= aug_prob_point_jitter:
                xyz = self.point_jitter(xyz)
        xyz = self.dataAugment(xyz, data_augmentations=self.data_augmentations, prob=aug_prob)
        return xyz


    def transform_test(self, xyz):
        return xyz


    @staticmethod
    def _theil_sen_line(values_z, values_xy):
        """Fit x(z), y(z) from a few robust bin centres."""
        slopes = []
        for first in range(len(values_z)):
            for second in range(first + 1, len(values_z)):
                delta_z = values_z[second] - values_z[first]
                if abs(delta_z) > 1e-6:
                    slopes.append(
                        (values_xy[second] - values_xy[first]) / delta_z)
        if not slopes:
            return None
        slope = np.median(np.asarray(slopes), axis=0)
        intercept = np.median(
            values_xy - values_z[:, None] * slope[None, :], axis=0)
        return slope, intercept


    def _get_stem_axis_upper_anchor(
            self, tree_points, tree_verticality, min_z, tree_height):
        """Estimate a stable upper anchor by extrapolating a lower-stem axis."""
        if tree_height <= 0 or tree_verticality is None:
            return None, 0

        relative_height = (tree_points[:, 2] - min_z) / tree_height
        verticality = np.asarray(tree_verticality).reshape(-1)
        candidate_mask = (
            np.isfinite(verticality) &
            (verticality >= self.stem_axis_verticality_threshold) &
            (relative_height >= self.stem_axis_lower_ratio) &
            (relative_height <= self.stem_axis_upper_ratio))

        bin_edges = np.linspace(
            self.stem_axis_lower_ratio,
            self.stem_axis_upper_ratio,
            self.stem_axis_num_bins + 1)
        centre_heights = []
        centre_xy = []
        for bin_index in range(self.stem_axis_num_bins):
            if bin_index == self.stem_axis_num_bins - 1:
                height_mask = (
                    (relative_height >= bin_edges[bin_index]) &
                    (relative_height <= bin_edges[bin_index + 1]))
            else:
                height_mask = (
                    (relative_height >= bin_edges[bin_index]) &
                    (relative_height < bin_edges[bin_index + 1]))
            points_in_bin = tree_points[candidate_mask & height_mask]
            if len(points_in_bin) < self.stem_axis_min_points_per_bin:
                continue
            centre_heights.append(
                np.median((points_in_bin[:, 2] - min_z) / tree_height))
            centre_xy.append(np.median(points_in_bin[:, :2], axis=0))

        num_valid_bins = len(centre_heights)
        if num_valid_bins < self.stem_axis_min_valid_bins:
            return None, num_valid_bins

        centre_heights = np.asarray(centre_heights, dtype=np.float64)
        centre_xy = np.asarray(centre_xy, dtype=np.float64)
        fitted_line = self._theil_sen_line(centre_heights, centre_xy)
        if fitted_line is None:
            return None, num_valid_bins
        slope, intercept = fitted_line

        upper_anchor = np.empty(3, dtype=np.float32)
        upper_anchor[:2] = (
            intercept + slope * self.upper_anchor_target_ratio)
        upper_anchor[2] = (
            min_z + self.upper_anchor_target_ratio * tree_height)
        if not np.isfinite(upper_anchor).all():
            return None, num_valid_bins
        return upper_anchor, num_valid_bins


    # B is a robust base anchor. U is either a crown median or an extrapolated
    # lower-stem axis anchor. B -> U is the tree-axis morphology target.
    def getOffset(
            self, xyz, instance_label, semantic_label, input_feat=None):
        base_position = np.zeros_like(xyz, dtype=np.float32)
        upper_position = np.zeros_like(xyz, dtype=np.float32)
        instances = np.unique(instance_label)
        mask_valid_offset = np.zeros_like(instance_label, dtype=bool)
        mask_valid_upper = np.zeros_like(instance_label, dtype=bool)

        for instance in instances:
            if instance in (INSTANCE_LABEL_IGNORE_IN_RAW_DATA, NON_TREE_CLASS_IN_RAW_DATA):
                continue

            inst_idx = np.where(instance_label == instance)
            first_idx = inst_idx[0][0]

            if semantic_label[first_idx] != NON_TREE_CLASS_IN_PYTORCH_DATASET:
                tree_points = xyz[inst_idx]
                if self.base_anchor_mode == 'legacy':
                    # Preserve the original TreeLearn implementation exactly for
                    # a separately reported reference baseline.
                    if len(tree_points) > 11:
                        min_z = np.partition(tree_points[:, 2], 10)[3]
                    else:
                        min_z = tree_points[:, 2].min()
                    mask_base_band = (
                        tree_points[:, 2] <= min_z + self.base_anchor_height)
                else:
                    min_z = np.quantile(
                        tree_points[:, 2], self.base_anchor_floor_quantile)
                    base_band_top = min_z + self.base_anchor_height
                    mask_base_band = (
                        (tree_points[:, 2] >= min_z) &
                        (tree_points[:, 2] <= base_band_top)
                    )
                base_points = tree_points[mask_base_band]
                if len(base_points) > 0:
                    if self.base_anchor_mode == 'legacy':
                        base_position_instance = np.mean(base_points, axis=0)
                    else:
                        base_position_instance = np.median(base_points, axis=0)
                    mask_valid_offset[inst_idx] = True
                else:
                    base_position_instance = np.zeros(3, dtype=np.float32)

                max_z = tree_points[:, 2].max()
                tree_height = max_z - min_z
                if self.upper_anchor_mode == 'stem_axis':
                    if input_feat is None:
                        tree_verticality = None
                    else:
                        tree_features = input_feat[inst_idx]
                        tree_verticality = (
                            tree_features if tree_features.ndim == 1
                            else tree_features[:, -1])
                    upper_position_instance, _ = \
                        self._get_stem_axis_upper_anchor(
                            tree_points,
                            tree_verticality,
                            min_z,
                            tree_height)
                    if upper_position_instance is not None:
                        mask_valid_upper[inst_idx] = True
                else:
                    if tree_height > 0:
                        relative_height = (
                            tree_points[:, 2] - min_z) / tree_height
                        mask_upper_band = (
                            (relative_height >=
                             self.upper_anchor_lower_ratio) &
                            (relative_height <=
                             self.upper_anchor_upper_ratio)
                        )
                        upper_points = tree_points[mask_upper_band]
                    else:
                        upper_points = np.empty(
                            (0, 3), dtype=tree_points.dtype)

                    if len(upper_points) >= self.upper_anchor_min_points:
                        upper_position_instance = np.empty(
                            3, dtype=np.float32)
                        upper_position_instance[:2] = np.median(
                            upper_points[:, :2], axis=0)
                        upper_anchor_ratio = (
                            self.upper_anchor_lower_ratio +
                            self.upper_anchor_upper_ratio) / 2
                        upper_position_instance[2] = (
                            min_z + upper_anchor_ratio * tree_height)
                        mask_valid_upper[inst_idx] = True
                    else:
                        upper_position_instance = None

                if upper_position_instance is None:
                    upper_position_instance = np.zeros(3, dtype=np.float32)

                base_position[inst_idx] = base_position_instance
                upper_position[inst_idx] = upper_position_instance

        pt_offset_label = base_position - xyz
        upper_offset_label = upper_position - xyz
        return pt_offset_label, upper_offset_label, mask_valid_offset, mask_valid_upper


    def dataAugment(self, xyz, data_augmentations, prob=0.6):
        jitter = data_augmentations["jitter"]
        flip = data_augmentations["flip"] 
        rot = data_augmentations["rot"]
        scale = data_augmentations["scaled"]
        m = np.eye(3)

        if scale and np.random.rand() < prob:
            scale_xy = np.random.uniform(0.8, 1.2, 2)
            scale_z = np.random.uniform(0.95, 1.05, 1)
            scale = np.concatenate([scale_xy, scale_z])
            m = m * scale
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        return np.matmul(xyz, m)


    def collate_fn(self, batch):
        xyzs = []
        input_feats = []
        batch_ids = []
        instance_labels = []
        semantic_labels = []
        pt_offset_labels = []
        upper_offset_labels = []
        centers = []
        masks_inner = []
        masks_off = []
        masks_upper = []
        masks_sem = []

        total_points_num = 0
        batch_id = 0


        for data in batch:
            (xyz, input_feat, instance_label, semantic_label, pt_offset_label,
             upper_offset_label, center, mask_inner, mask_off, mask_upper, mask_sem) = data
            total_points_num += len(xyz)

            xyzs.append(xyz)
            input_feats.append(input_feat)
            batch_ids.append(torch.ones(len(xyz))*batch_id)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            masks_inner.append(mask_inner)
            masks_off.append(mask_off)
            masks_upper.append(mask_upper)
            masks_sem.append(mask_sem)
            pt_offset_labels.append(pt_offset_label)
            upper_offset_labels.append(upper_offset_label)
            centers.append(center)           
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        xyzs = torch.cat(xyzs, 0).to(torch.float32)
        input_feats = torch.cat(input_feats, 0).to(torch.float32)
        batch_ids = torch.cat(batch_ids, 0).long()
        semantic_labels = torch.cat(semantic_labels, 0).long()
        instance_labels = torch.cat(instance_labels, 0).long()
        masks_inner = torch.cat(masks_inner, 0).bool()
        masks_off = torch.cat(masks_off, 0).bool()
        masks_upper = torch.cat(masks_upper, 0).bool()
        masks_sem = torch.cat(masks_sem, 0).bool()
        pt_offset_labels = torch.cat(pt_offset_labels, 0).float()
        upper_offset_labels = torch.cat(upper_offset_labels, 0).float()
        centers = torch.cat(centers, 0).float()
        
        return {
            'coords': xyzs,
            'input_feats': input_feats,
            'batch_ids': batch_ids,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'masks_inner': masks_inner,
            'masks_off': masks_off,
            'masks_upper': masks_upper,
            'masks_sem': masks_sem,
            'offset_labels': pt_offset_labels,
            'upper_offset_labels': upper_offset_labels,
            'batch_size': batch_id,
            'centers': centers
        }
