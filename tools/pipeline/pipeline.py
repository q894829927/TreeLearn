import os
import numpy as np
import argparse
import hashlib
import pickle
import pprint
import shutil
import time
from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import (munch_to_dict, build_dataloader, get_root_logger, load_checkpoint, ensemble, 
                             get_coords_within_shape, get_hull_buffer, get_hull, get_cluster_means,
                             propagate_preds, save_treewise, load_data, save_data, make_labels_consecutive, 
                             get_config, generate_tiles, assign_remaining_points_nearest_neighbor,
                             get_pointwise_preds, get_instances, get_dual_anchor_features,
                             filter_base_seeds_by_confidence,
                             get_axis_fused_features,
                             compute_instance_features, save_instance_features,
                             compute_vertical_instance_tokens,
                             compute_instance_quality_targets,
                             save_instance_quality_data,
                             predict_vertical_instance_quality,
                             apply_instance_quality_filter,
                             remap_instance_predictions,
                             save_instance_quality_scores,
                             propagate_preds_hash_full, propagate_preds_hash_vox)

TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1



def run_treelearn_pipeline(config, config_path=None):
    pipeline_start_time = time.time()
    # make dirs
    source_forest_path = config.forest_path
    plot_name = os.path.splitext(os.path.basename(source_forest_path))[0]
    base_dir = str(getattr(
        config, 'pipeline_base_dir',
        os.path.dirname(os.path.dirname(source_forest_path))))
    documentation_dir = os.path.join(base_dir, 'documentation')
    unvoxelized_data_dir = os.path.join(base_dir, 'forest')
    voxelized_data_dir = os.path.join(base_dir, f'forest_voxelized{config.sample_generation.voxel_size}')
    tiles_dir = os.path.join(base_dir, 'tiles')
    results_dir_name = getattr(config.save_cfg, 'results_dir', 'results')
    results_dir = os.path.join(base_dir, results_dir_name)

    os.makedirs(documentation_dir, exist_ok=True)
    os.makedirs(unvoxelized_data_dir, exist_ok=True)
    os.makedirs(voxelized_data_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # quick and dirty fix for the fact that method throws errors/does not work with high-magnitude coords
    # --> center coords and de-center at the end
    data = load_data(source_forest_path)
    xyz = data[:, :3].astype(np.float64)
    xyz_mean = np.mean(xyz, 0).astype(np.float64)
    xyz_centered = xyz - xyz_mean
    # Never overwrite an NPZ source. np.savez_compressed appends ".npz" when
    # the destination has another suffix, so the old ".npy" fallback created
    # "forest.npy.npz" and then attempted to read the nonexistent "forest.npy".
    centered_forest_path = os.path.join(
        unvoxelized_data_dir, f'{plot_name}.npz')
    if os.path.abspath(centered_forest_path) == os.path.abspath(source_forest_path):
        centered_forest_path = os.path.join(
            unvoxelized_data_dir, f'{plot_name}_centered.npz')
    np.savez_compressed(
        centered_forest_path, points=xyz_centered, labels=data[:, 3])
    config.forest_path = centered_forest_path
    
    # documentation
    logger = get_root_logger(os.path.join(documentation_dir, 'log_pipeline.txt'))
    logger.info(pprint.pformat(munch_to_dict(config), indent=2))
    if config_path is not None:
        shutil.copy(
            config_path,
            os.path.join(documentation_dir, os.path.basename(config_path)))

    # generate tiles used for inference and specify path to it in dataset config
    config.dataset_test.data_root = os.path.join(tiles_dir, 'npz')
    if config.tile_generation:
        logger.info('#################### generating tiles ####################')
        generate_tiles(config.sample_generation, config.forest_path, logger, config.save_cfg.return_type)

    # Make pointwise predictions with pretrained model
    quality_filter_cfg = getattr(config, 'quality_filter', None)
    quality_filter_enabled = bool(
        quality_filter_cfg is not None and
        getattr(quality_filter_cfg, 'enabled', False))
    quality_scoring_enabled = bool(
        quality_filter_cfg is not None and (
            quality_filter_enabled or
            getattr(quality_filter_cfg, 'score_instances', False)))
    logger.info(f'{plot_name}: #################### getting pointwise predictions ####################')
    model = TreeLearn(**config.model).cuda()
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    dataloader = build_dataloader(dataset, training=False, **config.dataloader)
    load_checkpoint(config.pretrain, logger, model)
    pointwise_results = get_pointwise_preds(
        model, dataloader, config.model, logger,
        return_backbone_feats=bool(
            config.save_cfg.save_pointwise or getattr(
                config.save_cfg, 'save_quality_training_data', False) or
            quality_scoring_enabled))
    (semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels,
     upper_offset_predictions, upper_offset_labels, coords, instance_labels,
     backbone_feats, input_feats, axis_xy_predictions,
     axis_log_variances) = pointwise_results
    del model

    # ensemble predictions from overlapping tiles
    logger.info(f'{plot_name}: #################### ensembling predictions ####################')
    data = ensemble(
        coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels,
        upper_offset_predictions, upper_offset_labels, instance_labels, backbone_feats,
        input_feats, axis_xy_predictions, axis_log_variances, logger=logger)
    (coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels,
     upper_offset_predictions, upper_offset_labels, instance_labels, backbone_feats,
     input_feats, axis_xy_predictions, axis_log_variances) = data
    axis_confidence = (
        1.0 / (1.0 + np.exp(axis_log_variances))
        if axis_log_variances is not None else None)

    # get mask of inner coords if outer points should be removed
    if config.shape_cfg.outer_remove:
        logger.info(f'{plot_name}: #################### prepare remove outer points ####################')
        hull_buffer_large = get_hull_buffer(coords[:, :2], config.shape_cfg.alpha, buffersize=config.shape_cfg.outer_remove)
        mask_coords_within_hull_buffer_large = get_coords_within_shape(coords, hull_buffer_large)
        masks_inner_coords = np.logical_not(mask_coords_within_hull_buffer_large)

    # get tree detections
    logger.info(f'{plot_name}: #################### getting predicted instances ####################')
    grouping_cfg = config.grouping
    if not config.model.use_upper_anchor:
        grouping_cfg.upper_anchor_weight = 0.0
        grouping_cfg.axis_height_weight = 0.0
    instance_preds = get_instances(
        coords, offset_predictions, upper_offset_predictions, semantic_prediction_logits,
        grouping_cfg, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET,
        NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS,
        logger=logger, axis_xy=axis_xy_predictions,
        axis_confidence=axis_confidence)
    instance_preds_after_initial_clustering = np.copy(instance_preds)

    # assign remaining points
    logger.info(
        f'{plot_name}: #################### assigning remaining points ####################')
    tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
    if (
        np.any(tree_mask) and
        not np.any(
            instance_preds[tree_mask] != NOT_ASSIGNED_LABEL_IN_GROUPING)
    ):
        raise RuntimeError(
            'Initial clustering produced no valid tree instances. No output was '
            'saved; inspect the clustering seed count and grouping thresholds.')
    if getattr(grouping_cfg, 'use_axis_fusion', False):
        tree_features = get_axis_fused_features(
            coords[tree_mask],
            offset_predictions[tree_mask],
            axis_xy_predictions[tree_mask]
            if axis_xy_predictions is not None else None,
            axis_confidence[tree_mask]
            if axis_confidence is not None else None,
            getattr(grouping_cfg, 'axis_fusion_weight', 0.0),
            getattr(grouping_cfg, 'use_axis_confidence', True))
    else:
        tree_features = get_dual_anchor_features(
            coords[tree_mask], offset_predictions[tree_mask],
            upper_offset_predictions[tree_mask],
            grouping_cfg.upper_anchor_weight, grouping_cfg.axis_height_weight)
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(
        tree_features, instance_preds[tree_mask],
        NOT_ASSIGNED_LABEL_IN_GROUPING,
        chunk_size=getattr(grouping_cfg, 'knn_chunk_size', 200000),
        logger=logger)
    num_unassigned = np.count_nonzero(
        instance_preds == NOT_ASSIGNED_LABEL_IN_GROUPING)
    if num_unassigned:
        raise RuntimeError(
            f'{num_unassigned:,} predicted tree points remain unassigned because '
            'initial clustering produced no usable reference instances. '
            'No output was saved; inspect the clustering seed count and thresholds.')

    quality_token_data = None
    quality_score_result = None
    pre_quality_label_digest = None
    if quality_scoring_enabled:
        if backbone_feats is None:
            raise RuntimeError(
                'Instance-quality scoring requires frozen backbone features.')
        checkpoint_path = str(getattr(
            quality_filter_cfg, 'checkpoint', '')).strip()
        if not checkpoint_path:
            raise ValueError(
                'quality_filter.checkpoint is required when quality scoring '
                'is enabled.')
        threshold = float(getattr(quality_filter_cfg, 'threshold', 0.0))
        quality_filter_mode = str(getattr(
            quality_filter_cfg, 'mode', 'threshold')).lower()
        quality_keep_ratio = float(getattr(
            quality_filter_cfg, 'keep_ratio', 1.0))
        num_layers = int(getattr(quality_filter_cfg, 'num_layers', 8))
        logger.info(
            f'{plot_name}: #################### scoring instance quality '
            '####################')
        quality_start_time = time.time()
        pre_quality_label_digest = hashlib.sha256(
            np.ascontiguousarray(instance_preds).view(np.uint8)).hexdigest()
        quality_token_data = compute_vertical_instance_tokens(
            coords=coords,
            instance_predictions=instance_preds,
            backbone_features=backbone_feats,
            semantic_prediction_logits=semantic_prediction_logits,
            offset_predictions=offset_predictions,
            verticality=input_feats[:, -1],
            axis_confidence=axis_confidence,
            num_layers=num_layers,
            tree_class_index=TREE_CLASS_IN_PYTORCH_DATASET)
        quality_score_result = predict_vertical_instance_quality(
            quality_token_data,
            checkpoint_path,
            device=getattr(quality_filter_cfg, 'device', None))

        label_mapping = None
        rejected_ids = np.empty(0, dtype=np.int64)
        kept_instance_ids = quality_score_result['instance_ids']
        if quality_filter_enabled:
            filter_result = apply_instance_quality_filter(
                instance_preds,
                quality_score_result['instance_ids'],
                quality_score_result['quality_score'],
                threshold,
                non_tree_label=NON_TREES_LABEL_IN_GROUPING,
                mode=quality_filter_mode,
                keep_ratio=quality_keep_ratio)
            if len(filter_result['kept_instance_ids']) == 0:
                raise RuntimeError(
                    'The instance-quality filter rejected every candidate. '
                    'No output was saved; relax the quality filter.')
            instance_preds = filter_result['predictions']
            label_mapping = filter_result['label_mapping']
            rejected_ids = filter_result['rejected_instance_ids']
            kept_instance_ids = filter_result['kept_instance_ids']
            instance_preds_after_initial_clustering = remap_instance_predictions(
                instance_preds_after_initial_clustering,
                label_mapping,
                rejected_ids,
                non_tree_label=NON_TREES_LABEL_IN_GROUPING)
            logger.info(
                f'Quality-filtered instances from '
                f"{len(quality_score_result['instance_ids']):,} to "
                f"{len(filter_result['kept_instance_ids']):,} "
                f'(mode: {quality_filter_mode}, threshold: {threshold:.4f}, '
                f'keep ratio: {quality_keep_ratio:.4f})')
        post_quality_label_digest = hashlib.sha256(
            np.ascontiguousarray(instance_preds).view(np.uint8)).hexdigest()
        if (
                not quality_filter_enabled and
                post_quality_label_digest != pre_quality_label_digest):
            raise RuntimeError(
                'Score-only instance quality unexpectedly changed predictions.')
        quality_score_result['pre_filter_label_sha256'] = (
            pre_quality_label_digest)
        quality_score_result['post_filter_label_sha256'] = (
            post_quality_label_digest)
        quality_elapsed = time.time() - quality_start_time
        quality_score_result['scoring_seconds'] = quality_elapsed
        quality_dir = os.path.join(results_dir, 'instance_quality_scores')
        score_paths = save_instance_quality_scores(
            quality_score_result,
            quality_dir,
            threshold=threshold,
            filter_enabled=quality_filter_enabled,
            label_mapping=label_mapping,
            kept_instance_ids=kept_instance_ids,
            filter_mode=quality_filter_mode,
            keep_ratio=quality_keep_ratio)
        logger.info(
            f'Instance-quality scoring finished in '
            f'{quality_elapsed:.1f}s; scores: {score_paths[0]}')

    save_diagnostics = bool(getattr(
        config.save_cfg, 'save_instance_diagnostics', False))
    save_quality_data = bool(getattr(
        config.save_cfg, 'save_quality_training_data', False))
    instance_features = None
    if save_diagnostics or save_quality_data:
        logger.info(
            f'{plot_name}: #################### computing instance diagnostics '
            '####################')
        instance_features = compute_instance_features(
            coords=coords,
            instance_predictions=instance_preds,
            initial_instance_predictions=instance_preds_after_initial_clustering,
            semantic_prediction_logits=semantic_prediction_logits,
            offset_predictions=offset_predictions,
            verticality=input_feats[:, -1],
            axis_confidence=axis_confidence,
            tree_class_index=TREE_CLASS_IN_PYTORCH_DATASET)

    diagnostic_metadata = {
        'plot_name': plot_name,
        'checkpoint': str(config.pretrain),
        'seed_filter_mode': str(getattr(
            grouping_cfg, 'seed_confidence_filter_mode', 'threshold')),
        'seed_keep_ratio': float(getattr(
            grouping_cfg, 'seed_confidence_keep_ratio', 1.0)),
    }
    if save_diagnostics:
        diagnostics_dir = os.path.join(results_dir, 'instance_diagnostics')
        csv_path, metadata_path = save_instance_features(
            instance_features, diagnostics_dir,
            metadata=diagnostic_metadata)
        logger.info(
            f'Saved {len(instance_features):,} instance feature rows to '
            f'{csv_path} (metadata: {metadata_path})')

    if save_quality_data:
        if backbone_feats is None:
            raise RuntimeError(
                'Quality training data requires frozen backbone features.')
        logger.info(
            f'{plot_name}: #################### saving quality training data '
            '####################')
        token_data = (
            quality_token_data if not quality_filter_enabled else None)
        if token_data is None:
            token_data = compute_vertical_instance_tokens(
                coords=coords,
                instance_predictions=instance_preds,
                backbone_features=backbone_feats,
                semantic_prediction_logits=semantic_prediction_logits,
                offset_predictions=offset_predictions,
                verticality=input_feats[:, -1],
                axis_confidence=axis_confidence,
                num_layers=int(getattr(
                    config.save_cfg, 'quality_num_layers', 8)),
                tree_class_index=TREE_CLASS_IN_PYTORCH_DATASET)
        target_data = compute_instance_quality_targets(
            coords=coords,
            instance_predictions=instance_preds,
            instance_labels=instance_labels,
            min_labeled_fraction=float(getattr(
                config.save_cfg,
                'quality_min_labeled_fraction', 0.5)),
            match_iou_threshold=float(getattr(
                config.save_cfg, 'quality_match_iou_threshold', 0.5)),
            negative_iou_threshold=float(getattr(
                config.save_cfg, 'quality_negative_iou_threshold', 0.25)),
            edge_margin_m=float(getattr(
                config.save_cfg, 'quality_edge_margin_m', 0.5)))
        quality_dir = os.path.join(results_dir, 'instance_quality')
        quality_paths = save_instance_quality_data(
            instance_features, token_data, target_data, quality_dir,
            source_plot=plot_name,
            split=str(getattr(config, 'quality_split', 'unspecified')),
            metadata=diagnostic_metadata)
        valid = target_data['target_valid']
        classification_valid = target_data['target_classification_valid']
        positives = (
            target_data['target_is_true_tree'] & classification_valid)
        negatives = (
            ~target_data['target_is_true_tree'] & classification_valid)
        logger.info(
            f'Saved quality data for {len(valid):,} candidates '
            f'({valid.sum():,} valid, {positives.sum():,} positive, '
            f'{negatives.sum():,} negative) to {quality_paths[0]}')

    if (
            save_quality_data and
            not bool(config.save_cfg.save_pointwise) and
            not bool(getattr(config.save_cfg, 'save_full_forest', True)) and
            not bool(config.save_cfg.save_treewise)):
        logger.info(
            f'{plot_name}: quality artifact complete; skipping full-forest save')
        return
    
    # save pointwise results
    if config.save_cfg.save_pointwise:
        logger.info(
            f'{plot_name}: #################### saving pointwise results ####################')
        pointwise_dir = os.path.join(results_dir, 'pointwise_results')
        os.makedirs(pointwise_dir, exist_ok=True)
        pointwise_results = {
            'coords': coords,
            'offset_predictions': offset_predictions,
            'offset_labels': offset_labels,
            'upper_offset_predictions': upper_offset_predictions,
            'upper_offset_labels': upper_offset_labels,
            'semantic_prediction_logits': semantic_prediction_logits,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'backbone_feats': backbone_feats,
            'input_feats': input_feats,
            'instance_preds': instance_preds,
            'instance_preds_after_initial_clustering': instance_preds_after_initial_clustering
        }
        if axis_xy_predictions is not None:
            pointwise_results.update({
                'axis_xy_predictions': axis_xy_predictions,
                'axis_log_variances': axis_log_variances,
                'axis_confidence': axis_confidence,
            })
        if config.shape_cfg.outer_remove:
            pointwise_results['masks_inner_coords'] = masks_inner_coords
            hull_buffer_large.to_pickle(os.path.join(pointwise_dir, 'hull_buffer_large.pkl'))
        np.savez_compressed(os.path.join(pointwise_dir, 'pointwise_results.npz'), **pointwise_results)
        
        # offset-shifted coordinates filtered by verticality and offset (initial clustering results); save as laz file for visualization
        verticality = input_feats[:, -1]
        verticality_mask = verticality >= config.grouping.tau_vert
        base_seed_mask = verticality_mask & (
            np.abs(offset_predictions[:, 2]) <= config.grouping.tau_off)
        if (
            getattr(config.grouping, 'use_upper_seeds', False) and
            config.model.use_upper_anchor
        ):
            upper_seed_mask = verticality_mask & (
                np.abs(upper_offset_predictions[:, 2]) <= config.grouping.tau_upper_off)
        else:
            upper_seed_mask = np.zeros_like(base_seed_mask)
        sem_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
        base_seed_mask &= sem_mask
        base_seed_mask = filter_base_seeds_by_confidence(
            base_seed_mask,
            axis_confidence=axis_confidence,
            enabled=bool(getattr(
                config.grouping, 'use_seed_confidence_filter', False)),
            threshold=float(getattr(
                config.grouping, 'seed_confidence_threshold', 0.0)),
            mode=str(getattr(
                config.grouping, 'seed_confidence_filter_mode',
                'threshold')),
            keep_ratio=float(getattr(
                config.grouping, 'seed_confidence_keep_ratio', 1.0)),
            random_seed=int(getattr(
                config.grouping, 'seed_confidence_random_seed', 42)))
        mask = base_seed_mask | (upper_seed_mask & sem_mask)
        cluster_coords = coords[mask] + offset_predictions[mask]
        cluster_coords = np.hstack([cluster_coords, instance_preds[mask].reshape(-1, 1)])
        save_data(cluster_coords, 'laz', 'cluster_coords_initial', pointwise_dir)
        
        # complete offset-shifted coordinates with instance predictions (clustering results after assigning remaining points); save as laz file for visualization
        cluster_coords = coords + offset_predictions
        cluster_coords = cluster_coords[instance_preds != NON_TREES_LABEL_IN_GROUPING]
        cluster_coords = np.hstack([cluster_coords, instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING].reshape(-1, 1)])
        save_data(cluster_coords, 'laz', 'cluster_coords', pointwise_dir)

        if config.model.use_upper_anchor:
            upper_cluster_coords = coords + upper_offset_predictions
            upper_cluster_coords = upper_cluster_coords[
                instance_preds != NON_TREES_LABEL_IN_GROUPING]
            upper_cluster_coords = np.hstack([
                upper_cluster_coords,
                instance_preds[
                    instance_preds != NON_TREES_LABEL_IN_GROUPING].reshape(-1, 1)
            ])
            save_data(upper_cluster_coords, 'laz', 'cluster_coords_upper', pointwise_dir)

    # remove outer points with buffer
    if config.shape_cfg.outer_remove:
        coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, upper_offset_predictions, upper_offset_labels, instance_labels, instance_preds, input_feats = \
            coords[masks_inner_coords], semantic_prediction_logits[masks_inner_coords], \
            semantic_labels[masks_inner_coords], offset_predictions[masks_inner_coords], \
            offset_labels[masks_inner_coords], upper_offset_predictions[masks_inner_coords], \
            upper_offset_labels[masks_inner_coords], instance_labels[masks_inner_coords], \
            instance_preds[masks_inner_coords], input_feats[masks_inner_coords]
        if axis_xy_predictions is not None:
            axis_xy_predictions = axis_xy_predictions[masks_inner_coords]
            axis_log_variances = axis_log_variances[masks_inner_coords]
            axis_confidence = axis_confidence[masks_inner_coords]
        instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING], _ = make_labels_consecutive(instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING], start_num=1)

    # get information whether tree clusters are within or outside hull (used for saving tree in different categories later)
    if config.save_cfg.save_treewise:
        logger.info(
            f'{plot_name}: #################### preparing treewise output ####################')
        cluster_means = get_cluster_means(coords[instance_preds != NON_TREES_LABEL_IN_GROUPING] + offset_predictions[instance_preds != NON_TREES_LABEL_IN_GROUPING], 
                                          instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING])
        hull = get_hull(coords[:, :2], config.shape_cfg.alpha)
        cluster_means_within_hull = get_coords_within_shape(cluster_means, hull)

        # get information whether trees have points very close to hull (used for saving trees in different categories later)
        hull_buffer_small = get_hull_buffer(coords[:, :2], config.shape_cfg.alpha, buffersize=config.shape_cfg.buffer_size_to_determine_edge_trees)
        mask_coords_at_edge = get_coords_within_shape(coords, hull_buffer_small)
        instance_preds_at_edge = np.unique(instance_preds[mask_coords_at_edge])
        instance_preds_at_edge = np.delete(instance_preds_at_edge, np.where(instance_preds_at_edge == NON_TREES_LABEL_IN_GROUPING))
        insts_not_at_edge = np.ones(len(cluster_means_within_hull))
        insts_not_at_edge[instance_preds_at_edge-1] = 0
        insts_not_at_edge = insts_not_at_edge.astype('bool')
        
    # propagate predictions to original forest
    if config.save_cfg.return_type == 'original':
        logger.info(f'{plot_name}: Propagating predictions to original points')
        coords_to_return = load_data(config.forest_path)[:, :3]
        hash_mapping_path = os.path.join(voxelized_data_dir, f'{plot_name}_hash_mapping.pkl')
        with open(hash_mapping_path, 'rb') as pickle_file:
            hash_mapping = pickle.load(pickle_file)
        preds_to_return, not_yet_propagated = propagate_preds_hash_full(coords, instance_preds, coords_to_return, hash_mapping)
    elif config.save_cfg.return_type == 'voxelized':
        logger.info(f'{plot_name}: Propagating predictions to voxelized points')
        voxelized_forest_path = os.path.join(voxelized_data_dir, f'{plot_name}.npz')
        coords_to_return = load_data(voxelized_forest_path)[:, :3]
        preds_to_return, not_yet_propagated = propagate_preds_hash_vox(coords, instance_preds, coords_to_return)
    elif config.save_cfg.return_type == 'voxelized_and_filtered': # 'voxelized_and_filtered' is identical to 'voxelized' if no point filtering is specified in configs/_modular/sample_generation.yaml
        coords_to_return = coords
        preds_to_return = instance_preds
        not_yet_propagated = np.zeros(len(coords_to_return), dtype=bool)
    # optionally remove outer points
    if config.shape_cfg.outer_remove:
        mask_coords_to_return_within_hull_buffer_large = get_coords_within_shape(coords_to_return, hull_buffer_large)
        masks_inner_coords_to_return = np.logical_not(mask_coords_to_return_within_hull_buffer_large)
        coords_to_return = coords_to_return[masks_inner_coords_to_return]
        preds_to_return = preds_to_return[masks_inner_coords_to_return]
        not_yet_propagated = not_yet_propagated[masks_inner_coords_to_return]
    # propagate predictions to points that were not yet propagated
    if not_yet_propagated.any():
        preds_to_return[not_yet_propagated] = propagate_preds(coords, instance_preds, coords_to_return[not_yet_propagated], n_neighbors=5)
        
    # # add xyz_mean again which was potentially subtracted at the beginning
    coords_to_return = coords_to_return.astype(np.float64) + xyz_mean
        
    # save
    logger.info(f'{plot_name}: #################### Saving ####################')
    full_dir = os.path.join(results_dir, 'full_forest')
    os.makedirs(full_dir, exist_ok=True)

    for save_format in config.save_cfg.save_formats:
        save_data(np.hstack([coords_to_return, preds_to_return.reshape(-1, 1)]), save_format, plot_name, full_dir)
    if config.save_cfg.save_treewise:
        trees_dir = os.path.join(results_dir, 'individual_trees')
        os.makedirs(trees_dir, exist_ok=True)
        save_treewise(coords_to_return, preds_to_return, cluster_means_within_hull, insts_not_at_edge, "las", trees_dir, NON_TREES_LABEL_IN_GROUPING)
    logger.info(
        f'{plot_name}: pipeline finished in '
        f'{time.time() - pipeline_start_time:.1f}s')
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser('tree_learn')
    parser.add_argument('--config', type=str, help='path to config file for pipeline')
    args = parser.parse_args()
    config = get_config(args.config)
    run_treelearn_pipeline(config, args.config)
