import os.path as osp
import time
import torch
import tqdm
import numpy as np
import random
from collections import defaultdict
from tree_learn.util import (checkpoint_save, init_train_logger, load_checkpoint,
                            is_multiple, get_args_and_cfg, build_cosine_scheduler, build_optimizer,
                            masked_offset_loss, point_wise_loss, get_eval_components,
                            build_dataloader, checkpoint_save_named)
from tree_learn.model import TreeLearn
from tree_learn.model.point_transformer import get_axis_branch_target_xy
from tree_learn.model.height_identity import (
    height_checkpoint_is_eligible, seed_retention_metrics,
    seed_semantic_identity_loss)

from tree_learn.dataset import TreeDataset

TREE_CLASS_IN_DATASET = 0 # semantic label for tree class in pytorch dataset
NON_TREE_CLASS_IN_DATASET = 1 # semantic label for non-tree class in pytorch dataset
TREE_CONF_THRESHOLD = 0.5 # minimum confidence for tree prediction


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _step_optimizer(config, model, optimizer, scaler):
    if config.grad_norm_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_norm_clip, norm_type=2)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train(config, epoch, model, optimizer, scheduler, scaler, train_loader,
          logger, writer, teacher=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    losses_dict = defaultdict(list)
    accumulation_steps = max(
        1, int(getattr(config, 'gradient_accumulation_steps', 1)))
    optimizer.zero_grad(set_to_none=True)
    processed_batches = 0

    for i, batch in enumerate(train_loader, start=1):
        if config.examples_per_epoch < (
                i * config.dataloader.train.batch_size):
            break

        scheduler.step(epoch)
        with torch.cuda.amp.autocast(enabled=config.fp16):
            teacher_output = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_output = teacher(batch, return_loss=False)
            loss, loss_dict = model(
                batch, return_loss=True, teacher_output=teacher_output)
            for key, value in loss_dict.items():
                losses_dict[key].append(value.detach().cpu().item())

        scaler.scale(loss / accumulation_steps).backward()
        processed_batches += 1
        if processed_batches % accumulation_steps == 0:
            _step_optimizer(config, model, optimizer, scaler)

    if processed_batches % accumulation_steps:
        _step_optimizer(config, model, optimizer, scaler)

    epoch_time = time.time() - start
    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() else 0.0)
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('train/learning_rate', lr, epoch)
    writer.add_scalar('train/peak_memory_gb', peak_memory_gb, epoch)
    average_losses_dict = {
        key: sum(values) / len(values)
        for key, values in losses_dict.items() if values
    }
    for key, value in average_losses_dict.items():
        writer.add_scalar(f'train/{key}', value, epoch)

    log_str = (
        f'[TRAINING] [{epoch}/{config.epochs}], time {epoch_time:.2f}s, '
        f'peak_memory_gb: {peak_memory_gb:.3f}')
    for key, value in average_losses_dict.items():
        log_str += f', {key}: {value:.2f}'
    logger.info(log_str)
    checkpoint_save(
        epoch, model, optimizer, config.work_dir, config.save_frequency)


def validate(config, epoch, model, val_loader, logger, writer, teacher=None):
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions = [], []
        semantic_labels, offset_labels = [], []
        upper_offset_predictions, upper_offset_labels = [], []
        axis_cosine_errors = []
        axis_xy_errors = []
        axis_target_xy_lengths = []
        axis_confidences = []
        height_identity_loss_sum = 0.0
        height_identity_seed_count = 0
        height_semantic_retained_count = 0
        height_full_retained_count = 0
        attention_statistics = defaultdict(list)
        teacher_seed_comparisons = 0
        teacher_seed_mismatches = 0
        if teacher is not None:
            teacher.eval()
        for batch in tqdm.tqdm(val_loader):

            # forward
            output = model(batch, return_loss=False)
            for name, value in output.get(
                    'unet_attention_stats', {}).items():
                if np.isfinite(value):
                    attention_statistics[name].append(float(value))
            if teacher is not None:
                teacher_output = teacher(batch, return_loss=False)
                teacher_probability = teacher_output[
                    'semantic_prediction_logits'].float().softmax(
                        dim=-1)[:, TREE_CLASS_IN_DATASET]
                student_probability = output[
                    'semantic_prediction_logits'].float().softmax(
                        dim=-1)[:, TREE_CLASS_IN_DATASET]
                teacher_offsets = teacher_output[
                    'offset_predictions'].float()
                student_offsets = output['offset_predictions'].float()
                verticality = batch['input_feats'][:, -1].to(
                    teacher_probability.device).float()
                tree_threshold = float(config.model.unet_finetune.get(
                    'teacher_seed_tree_conf_thresh', 0.5))
                vertical_threshold = float(config.model.unet_finetune.get(
                    'teacher_seed_tau_vert', 0.6))
                offset_threshold = float(config.model.unet_finetune.get(
                    'teacher_seed_tau_off', 4.0))
                geometry_candidates = verticality >= vertical_threshold
                teacher_seeds = (
                    geometry_candidates &
                    (teacher_probability >= tree_threshold) &
                    (teacher_offsets[:, 2].abs() <= offset_threshold))
                student_seeds = (
                    geometry_candidates &
                    (student_probability >= tree_threshold) &
                    (student_offsets[:, 2].abs() <= offset_threshold))
                teacher_seed_comparisons += int(
                    geometry_candidates.sum().item())
                teacher_seed_mismatches += int(
                    ((teacher_seeds != student_seeds) &
                     geometry_candidates).sum().item())
            offset_prediction, semantic_prediction_logit = output['offset_predictions'], output['semantic_prediction_logits']
            upper_offset_prediction = output.get('upper_offset_predictions')
            axis_xy_prediction = output.get('axis_xy_predictions')
            identity_weight = float(getattr(
                config.model, 'height_seed_identity_weight', 0.0))
            if identity_weight > 0:
                identity_loss, identity_mask = seed_semantic_identity_loss(
                    output['base_semantic_prediction_logits'],
                    semantic_prediction_logit,
                    output['base_offset_predictions'],
                    batch['input_feats'],
                    tree_conf_thresh=float(getattr(
                        config.model, 'height_seed_tree_conf_thresh', 0.5)),
                    tau_vert=float(getattr(
                        config.model, 'height_seed_tau_vert', 0.6)),
                    tau_off=float(getattr(
                        config.model, 'height_seed_tau_off', 4.0)),
                    probability_margin=float(getattr(
                        config.model, 'height_seed_identity_margin', 0.02)))
                retention = seed_retention_metrics(
                    output['base_semantic_prediction_logits'],
                    semantic_prediction_logit,
                    output['base_offset_predictions'],
                    offset_prediction,
                    batch['input_feats'],
                    tree_conf_thresh=float(getattr(
                        config.model, 'height_seed_tree_conf_thresh', 0.5)),
                    tau_vert=float(getattr(
                        config.model, 'height_seed_tau_vert', 0.6)),
                    tau_off=float(getattr(
                        config.model, 'height_seed_tau_off', 4.0)))
                seed_count = int(identity_mask.sum().item())
                height_identity_loss_sum += float(identity_loss.item()) * seed_count
                height_identity_seed_count += seed_count
                height_semantic_retained_count += retention[
                    'semantic_retained_count']
                height_full_retained_count += retention['full_retained_count']

            semantic_prediction_logits.append(
                semantic_prediction_logit[batch['masks_sem']].detach().cpu())
            semantic_labels.append(
                batch['semantic_labels'][batch['masks_sem']].cpu())
            offset_predictions.append(
                offset_prediction[batch['masks_sem']].detach().cpu())
            offset_labels.append(
                batch['offset_labels'][batch['masks_sem']].cpu())
            if upper_offset_prediction is not None:
                upper_offset_predictions.append(
                    upper_offset_prediction[
                        batch['masks_upper']].detach().cpu())
                upper_offset_labels.append(
                    batch['upper_offset_labels'][batch['masks_upper']].cpu())
                masks_axis = batch['masks_off'] & batch['masks_upper']
                if masks_axis.sum() > 0:
                    predicted_axis = (
                        upper_offset_prediction[masks_axis] - offset_prediction[masks_axis])
                    target_axis = (
                        batch['upper_offset_labels'][masks_axis] -
                        batch['offset_labels'][masks_axis]).to(
                            predicted_axis.device)
                    axis_cosine_errors.append(
                        1 - torch.nn.functional.cosine_similarity(
                            predicted_axis.float(), target_axis.float(),
                            dim=1, eps=1e-6).detach().cpu())
            if axis_xy_prediction is not None:
                candidate_mask = output['axis_candidate_mask']
                masks_axis_xy = (
                    batch['masks_off'].to(candidate_mask.device) &
                    candidate_mask)
                axis_target_mode = getattr(
                    config.model, 'axis_target_mode', 'upper_axis')
                if axis_target_mode == 'upper_axis':
                    masks_axis_xy = (
                        masks_axis_xy &
                        batch['masks_upper'].to(candidate_mask.device))
                if masks_axis_xy.sum() > 0:
                    mask_cpu = masks_axis_xy.cpu()
                    selected_upper_labels = (
                        batch['upper_offset_labels'][mask_cpu].to(
                            axis_xy_prediction.device)
                        if axis_target_mode == 'upper_axis' else None)
                    target_axis_xy = get_axis_branch_target_xy(
                        offset_prediction[masks_axis_xy],
                        batch['offset_labels'][mask_cpu].to(
                            axis_xy_prediction.device),
                        selected_upper_labels,
                        axis_target_mode)
                    axis_xy_errors.append(torch.linalg.vector_norm(
                        axis_xy_prediction[masks_axis_xy].float() -
                        target_axis_xy.float(),
                        dim=1).detach().cpu())
                    axis_target_xy_lengths.append(
                        torch.linalg.vector_norm(
                            target_axis_xy.float(), dim=1).detach().cpu())
                    axis_confidences.append(
                        output['axis_confidence'][
                            masks_axis_xy, 0].detach().float().cpu())

    # concatenate all batches
    semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0), torch.cat(semantic_labels, 0)
    offset_predictions, offset_labels = torch.cat(offset_predictions, 0), torch.cat(offset_labels, 0)
    if upper_offset_predictions:
        upper_offset_predictions = torch.cat(upper_offset_predictions, 0)
        upper_offset_labels = torch.cat(upper_offset_labels, 0)
    else:
        upper_offset_predictions, upper_offset_labels = None, None
    axis_cosine_error = (
        torch.cat(axis_cosine_errors, 0).mean() if axis_cosine_errors else None)
    if axis_xy_errors:
        axis_xy_errors = torch.cat(axis_xy_errors, 0)
        axis_target_xy_lengths = torch.cat(axis_target_xy_lengths, 0)
        axis_confidences = torch.cat(axis_confidences, 0)
    else:
        axis_xy_errors, axis_target_xy_lengths, axis_confidences = \
            None, None, None
    # evaluate semantic and offset predictions
    metrics = pointwise_eval(
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels,
        upper_offset_predictions, upper_offset_labels, axis_cosine_error,
        config, epoch, writer, logger, axis_xy_errors, axis_confidences,
        axis_target_xy_lengths, height_identity_loss_sum,
        height_identity_seed_count, height_semantic_retained_count,
        height_full_retained_count)
    for name, values in attention_statistics.items():
        mean_value = float(np.mean(values))
        metrics[f'attention/{name}'] = mean_value
        writer.add_scalar(f'val/UNet_Attention/{name}', mean_value, epoch)
    if attention_statistics:
        logger.info(
            'U-Net attention stats: ' +
            ', '.join(
                f'{name}={np.mean(values):.6f}'
                for name, values in sorted(attention_statistics.items())))
    if teacher is not None:
        mismatch_rate = (
            teacher_seed_mismatches /
            max(teacher_seed_comparisons, 1))
        metrics['teacher_seed_membership_mismatch_rate'] = mismatch_rate
        metrics['teacher_seed_membership_mismatches'] = (
            teacher_seed_mismatches)
        writer.add_scalar(
            'val/Teacher_Seed_Membership_Mismatch_Rate',
            mismatch_rate, epoch)
        logger.info(
            'Teacher seed membership mismatches: '
            f'{teacher_seed_mismatches}/{teacher_seed_comparisons} '
            f'({100 * mismatch_rate:.4f}%).')
    return metrics


def pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels,
                   upper_offset_predictions, upper_offset_labels, axis_cosine_error,
                   config, epoch, writer, logger, axis_xy_errors=None,
                   axis_confidences=None, axis_target_xy_lengths=None,
                   height_identity_loss_sum=0.0,
                   height_identity_seed_count=0,
                   height_semantic_retained_count=0,
                   height_full_retained_count=0):
    # get offset loss
    masks_sem = torch.ones_like(semantic_labels).bool()
    masks_off = semantic_labels == TREE_CLASS_IN_DATASET
    semantic_loss, offset_loss = point_wise_loss(
        semantic_prediction_logits.float(), offset_predictions.float(),
        masks_sem, masks_off, semantic_labels, offset_labels,
        offset_loss_type=config.model.offset_loss_type,
        smooth_l1_beta=config.model.smooth_l1_beta)
    if upper_offset_predictions is None or len(upper_offset_predictions) == 0:
        upper_offset_loss = None
    else:
        upper_offset_labels = upper_offset_labels.to(
            upper_offset_predictions.device)
        upper_masks = torch.ones(
            len(upper_offset_predictions),
            dtype=torch.bool,
            device=upper_offset_predictions.device)
        upper_offset_loss = masked_offset_loss(
            upper_offset_predictions.float(),
            upper_offset_labels.float(),
            upper_masks,
            loss_type=config.model.offset_loss_type,
            smooth_l1_beta=config.model.smooth_l1_beta)
    
    # get semantic accuracy of classification into tree and non-tree
    semantic_prediction_logits, semantic_labels = semantic_prediction_logits.cpu().numpy(), semantic_labels.cpu().numpy()
    tree_pred_mask = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)[:, TREE_CLASS_IN_DATASET] >= TREE_CONF_THRESHOLD
    tree_pred_mask = tree_pred_mask.numpy()
    tree_mask = semantic_labels == TREE_CLASS_IN_DATASET
    tp, fp, tn, fn = get_eval_components(tree_pred_mask, tree_mask)
    acc = (tp + tn) / (tp + fp + fn + tn)

    # log and write to tensorboard
    identity_weight = float(getattr(
        config.model, 'height_seed_identity_weight', 0.0))
    height_identity_loss = (
        height_identity_loss_sum / max(height_identity_seed_count, 1))
    height_semantic_retention = (
        height_semantic_retained_count / max(height_identity_seed_count, 1))
    height_full_retention = (
        height_full_retained_count / max(height_identity_seed_count, 1))
    selection_loss = (
        semantic_loss * 50.0 + offset_loss).item() + (
            identity_weight * height_identity_loss)
    log_str = (
        f'[VALIDATION] [{epoch}/{config.epochs}] val/semantic_acc {acc*100:.2f}, '
        f'val/semantic_loss {semantic_loss.item():.4f}, '
        f'val/offset_loss {offset_loss.item():.3f}')
    if getattr(config.model, 'use_height_context_adapter', False):
        log_str += f', val/height_selection_loss {selection_loss:.4f}'
        if identity_weight > 0:
            log_str += (
                f', val/height_seed_identity_loss '
                f'{height_identity_loss:.6f}, '
                f'val/height_seed_semantic_retention '
                f'{height_semantic_retention:.4f}, '
                f'val/height_seed_full_retention '
                f'{height_full_retention:.4f}')
    if (
        not getattr(config.model, 'use_height_context_adapter', False) and
        bool(getattr(config.model, 'unet_finetune', {}).get(
            'enabled', False))
    ):
        log_str += f', val/unet_selection_loss {selection_loss:.4f}'
    if upper_offset_loss is not None:
        log_str += f', val/upper_offset_loss {upper_offset_loss.item():.3f}'
    if axis_cosine_error is not None:
        log_str += f', val/axis_cosine_error {axis_cosine_error.item():.3f}'
    axis_metrics = None
    if axis_xy_errors is not None and len(axis_xy_errors) > 0:
        errors_np = axis_xy_errors.numpy()
        confidences_np = axis_confidences.numpy()
        axis_metrics = {
            'mean': float(np.mean(errors_np)),
            'median': float(np.median(errors_np)),
            'p90': float(np.percentile(errors_np, 90)),
        }
        if (
            axis_target_xy_lengths is not None and
            len(axis_target_xy_lengths) == len(axis_xy_errors)
        ):
            target_lengths_np = axis_target_xy_lengths.numpy()
            target_mean_length = float(np.mean(target_lengths_np))
            axis_metrics['target_mean_length'] = target_mean_length
            axis_metrics['normalized_mean_error'] = (
                axis_metrics['mean'] / max(target_mean_length, 1e-6))
        if (
            len(errors_np) > 1 and
            np.std(errors_np) > 0 and
            np.std(confidences_np) > 0
        ):
            axis_metrics['confidence_error_corr'] = float(
                np.corrcoef(confidences_np, errors_np)[0, 1])
        else:
            axis_metrics['confidence_error_corr'] = 0.0
        metric_prefix = (
            'base_residual_xy'
            if getattr(
                config.model, 'axis_target_mode', 'upper_axis') ==
            'base_residual'
            else 'axis_xy')
        log_str += (
            f", val/{metric_prefix}_mean_error "
            f"{axis_metrics['mean']:.3f}, "
            f"val/{metric_prefix}_median_error "
            f"{axis_metrics['median']:.3f}, "
            f"val/{metric_prefix}_p90_error "
            f"{axis_metrics['p90']:.3f}, "
            'val/axis_confidence_error_corr '
            f"{axis_metrics['confidence_error_corr']:.3f}")
        if 'target_mean_length' in axis_metrics:
            log_str += (
                ', val/axis_target_xy_mean_length '
                f"{axis_metrics['target_mean_length']:.3f}, "
                'val/axis_normalized_mean_error '
                f"{axis_metrics['normalized_mean_error']:.3f}")

        confidence_order = np.argsort(confidences_np)
        confidence_bins = np.array_split(confidence_order, 4)
        for bin_idx, indices in enumerate(confidence_bins, start=1):
            if len(indices) == 0:
                continue
            mean_confidence = float(np.mean(confidences_np[indices]))
            mean_error = float(np.mean(errors_np[indices]))
            log_str += (
                f', val/conf_bin{bin_idx}_confidence '
                f'{mean_confidence:.3f}, val/conf_bin{bin_idx}_error '
                f'{mean_error:.3f}')
            writer.add_scalar(
                f'val/Axis_Confidence_Bin_{bin_idx}', mean_confidence, epoch)
            writer.add_scalar(
                f'val/Axis_Error_Bin_{bin_idx}', mean_error, epoch)
    logger.info(log_str)
    writer.add_scalar(f'val/acc', acc if not np.isnan(acc) else 0, epoch)
    writer.add_scalar('val/Semantic_Loss', semantic_loss, epoch)
    writer.add_scalar('val/Offset_Loss', offset_loss, epoch)
    if getattr(config.model, 'use_height_context_adapter', False):
        writer.add_scalar(
            'val/Height_Selection_Loss', selection_loss, epoch)
    elif bool(getattr(config.model, 'unet_finetune', {}).get(
            'enabled', False)):
        writer.add_scalar('val/UNet_Selection_Loss', selection_loss, epoch)
        if identity_weight > 0:
            writer.add_scalar(
                'val/Height_Seed_Identity_Loss',
                height_identity_loss, epoch)
            writer.add_scalar(
                'val/Height_Seed_Semantic_Retention',
                height_semantic_retention, epoch)
            writer.add_scalar(
                'val/Height_Seed_Full_Retention',
                height_full_retention, epoch)
    if upper_offset_loss is not None:
        writer.add_scalar('val/Upper_Offset_Loss', upper_offset_loss, epoch)
    if axis_cosine_error is not None:
        writer.add_scalar('val/Axis_Cosine_Error', axis_cosine_error, epoch)
    if axis_metrics is not None:
        writer.add_scalar(
            'val/Axis_XY_Mean_Error', axis_metrics['mean'], epoch)
        writer.add_scalar(
            'val/Axis_XY_Median_Error', axis_metrics['median'], epoch)
        writer.add_scalar(
            'val/Axis_XY_P90_Error', axis_metrics['p90'], epoch)
        writer.add_scalar(
            'val/Axis_Confidence_Error_Correlation',
            axis_metrics['confidence_error_corr'], epoch)
        if 'target_mean_length' in axis_metrics:
            writer.add_scalar(
                'val/Axis_Target_XY_Mean_Length',
                axis_metrics['target_mean_length'], epoch)
            writer.add_scalar(
                'val/Axis_Normalized_Mean_Error',
                axis_metrics['normalized_mean_error'], epoch)
    if getattr(config.model, 'use_height_context_adapter', False):
        return {
            'selection_loss': float(selection_loss),
            'semantic_loss': float(semantic_loss.item()),
            'offset_loss': float(offset_loss.item()),
            'semantic_acc': float(acc),
            'height_seed_identity_loss': float(height_identity_loss),
            'height_seed_semantic_retention': float(
                height_semantic_retention),
            'height_seed_full_retention': float(height_full_retention),
        }
    metrics = {
        'selection_loss': float(selection_loss),
        'semantic_loss': float(semantic_loss.item()),
        'offset_loss': float(offset_loss.item()),
        'semantic_acc': float(acc),
    }
    if axis_metrics is not None:
        metrics.update(axis_metrics)
    return metrics


def main():
    args, config = get_args_and_cfg()
    logger, writer = init_train_logger(config, args)
    seed = int(getattr(config, 'seed', 42))
    set_random_seed(seed)
    logger.info(f'Random seed: {seed}')

    model = TreeLearn(**config.model).cuda()
    if (
        getattr(config.model, 'use_axis_branch', False) and
        getattr(config.model, 'axis_branch_only', True)
    ):
        unexpected_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith('axis_')
        ]
        if unexpected_trainable:
            raise RuntimeError(
                'Axis branch-only training found unfrozen original '
                f'parameters: {", ".join(unexpected_trainable)}')
    if (
        getattr(config.model, 'use_height_context_adapter', False) and
        getattr(config.model, 'height_adapter_only', True)
    ):
        unexpected_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith(
                'height_context_adapter.')
        ]
        if unexpected_trainable:
            raise RuntimeError(
                'Height-adapter-only training found unfrozen original '
                f'parameters: {", ".join(unexpected_trainable)}')
        height_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad)
        logger.info(
            f'Height-context trainable parameters: {height_parameter_count:,}')

    tournament_mode = bool(
        getattr(config.model, 'unet_finetune', {}).get('enabled', False))
    if tournament_mode:
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters() if parameter.requires_grad)
        total_count = sum(parameter.numel() for parameter in model.parameters())
        if not trainable_names:
            raise RuntimeError('U-Net tournament has no trainable parameters.')
        logger.info(
            'U-Net tournament trainable parameters: '
            f'{trainable_count:,}/{total_count:,} '
            f'({100 * trainable_count / total_count:.3f}%).')
        logger.info(
            'U-Net trainable tensors: ' + ', '.join(trainable_names))

    optimizer = build_optimizer(model, config.optimizer)
    for group in optimizer.param_groups:
        logger.info(
            f"Optimizer group {group.get('group_name', 'default')}: "
            f"lr={group['lr']:.7f}, tensors={len(group['params'])}")
    scheduler = build_cosine_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    train_set = TreeDataset(**config.dataset_train, logger=logger)
    val_set = TreeDataset(**config.dataset_test, logger=logger)
    train_loader = build_dataloader(
        train_set, training=True, seed=seed, **config.dataloader.train)
    val_loader = build_dataloader(
        val_set, training=False, seed=seed + 1, **config.dataloader.test)

    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(
            args.resume, logger, model, optimizer=optimizer)
    elif config.pretrain:
        logger.info(f'Load pretrain from {config.pretrain}')
        load_checkpoint(config.pretrain, logger, model)

    teacher = None
    preservation_weight = float(
        getattr(config.model, 'unet_finetune', {}).get(
            'teacher_seed_preservation_weight', 0.0))
    measure_teacher_mismatch = bool(
        getattr(config.model, 'unet_finetune', {}).get(
            'measure_teacher_seed_mismatch', False))
    if preservation_weight > 0 or measure_teacher_mismatch:
        if not config.pretrain:
            raise ValueError(
                'Teacher seed preservation requires the official pretrain.')
        teacher_settings = dict(config.model)
        teacher_settings['unet_enhancement'] = {
            'enabled': False,
            'type': 'identity',
        }
        teacher_settings['unet_finetune'] = {'enabled': False}
        teacher_settings['use_axis_branch'] = False
        teacher_settings['use_height_context_adapter'] = False
        teacher = TreeLearn(**teacher_settings).cuda()
        load_checkpoint(config.pretrain, logger, teacher)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        logger.info(
            'Loaded frozen official teacher for seed preservation '
            f'(weight={preservation_weight:.4f}, validation mismatch '
            f'audit={measure_teacher_mismatch}).')

    logger.info('Training')
    best_axis_xy_mean = float('inf')
    best_height_selection_loss = float('inf')
    best_unet_selection_loss = float('inf')
    bad_validation_checks = 0
    early_stopping_patience = int(
        getattr(config, 'early_stopping_patience', 0))
    best_checkpoint_name = (
        'best_base_residual_xy.pth'
        if getattr(
            config.model, 'axis_target_mode', 'upper_axis') ==
        'base_residual'
        else 'best_axis_xy.pth')

    for epoch in range(start_epoch, config.epochs + 1):
        train(
            config, epoch, model, optimizer, scheduler, scaler,
            train_loader, logger, writer,
            teacher=teacher if preservation_weight > 0 else None)
        should_stop = False
        if is_multiple(epoch, config.validation_frequency):
            optimizer.zero_grad(set_to_none=True)
            logger.info('Validation')
            torch.cuda.empty_cache()
            validation_metrics = validate(
                config, epoch, model, val_loader, logger, writer,
                teacher=teacher)
            improved = False
            if tournament_mode:
                selection_loss = validation_metrics['selection_loss']
                if selection_loss < best_unet_selection_loss:
                    improved = True
                    best_unet_selection_loss = selection_loss
                    checkpoint_save_named(
                        epoch, model, optimizer, config.work_dir,
                        'best_unet.pth')
                    logger.info(
                        f'Saved best_unet.pth at epoch {epoch} '
                        f'(validation selection loss '
                        f'{best_unet_selection_loss:.4f}).')
            elif getattr(
                    config.model, 'use_height_context_adapter', False):
                selection_loss = validation_metrics['selection_loss']
                minimum_retention = float(getattr(
                    config.model,
                    'height_seed_min_semantic_retention', 0.0))
                semantic_retention = validation_metrics[
                    'height_seed_semantic_retention']
                checkpoint_eligible = height_checkpoint_is_eligible(
                    semantic_retention, minimum_retention)
                if not checkpoint_eligible:
                    logger.info(
                        'Skipped height checkpoint at epoch '
                        f'{epoch}: seed semantic retention '
                        f'{semantic_retention:.4f} is below the fixed '
                        f'{minimum_retention:.4f} gate.')
                elif selection_loss < best_height_selection_loss:
                    improved = True
                    best_height_selection_loss = selection_loss
                    height_checkpoint_name = (
                        'best_hsca.pth'
                        if config.model.height_context_type == 'attention'
                        else 'best_height_mlp.pth')
                    checkpoint_save_named(
                        epoch, model, optimizer, config.work_dir,
                        height_checkpoint_name)
                    logger.info(
                        f'Saved {height_checkpoint_name} at epoch {epoch} '
                        f'(validation selection loss '
                        f'{best_height_selection_loss:.4f}, seed semantic '
                        f'retention {semantic_retention:.4f})')
            elif (
                validation_metrics is not None and
                'mean' in validation_metrics and
                validation_metrics['mean'] < best_axis_xy_mean
            ):
                improved = True
                best_axis_xy_mean = validation_metrics['mean']
                checkpoint_save_named(
                    epoch, model, optimizer, config.work_dir,
                    best_checkpoint_name)
                logger.info(
                    f'Saved {best_checkpoint_name} at epoch '
                    f'{epoch} (mean XY error {best_axis_xy_mean:.3f} m)')

            if tournament_mode and early_stopping_patience > 0:
                bad_validation_checks = (
                    0 if improved else bad_validation_checks + 1)
                if bad_validation_checks >= early_stopping_patience:
                    logger.info(
                        'Early stopping U-Net tournament after '
                        f'{bad_validation_checks} validation checks without '
                        'selection-loss improvement.')
                    should_stop = True
        writer.flush()
        if should_stop:
            break


if __name__ == '__main__':
    main()
