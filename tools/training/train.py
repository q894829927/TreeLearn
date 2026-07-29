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


def train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer):
    model.train()
    start = time.time()
    losses_dict = defaultdict(list)


    for i, batch in enumerate(train_loader, start=1):
        # break after a fixed number of samples have been passed
        if config.examples_per_epoch < (i * config.dataloader.train.batch_size):
            break
        
        scheduler.step(epoch)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.fp16):

            # forward
            loss, loss_dict = model(batch, return_loss=True)
            for key, value in loss_dict.items():
                losses_dict[key].append(value.detach().cpu().item())

        # backward
        scaler.scale(loss).backward()
        if config.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip, norm_type=2)
        scaler.step(optimizer)
        scaler.update()

    # log and write to tensorboard
    epoch_time = time.time() - start
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('train/learning_rate', lr, epoch)
    average_losses_dict = {k: sum(v) / len(v) for k, v in losses_dict.items()}
    for k, v in average_losses_dict.items():
        writer.add_scalar(f'train/{k}', v, epoch)

    log_str = f'[TRAINING] [{epoch}/{config.epochs}], time {epoch_time:.2f}s'
    for k, v in average_losses_dict.items():
        log_str += f', {k}: {v:.2f}'
    logger.info(log_str)
    checkpoint_save(epoch, model, optimizer, config.work_dir, config.save_frequency)
    

def validate(config, epoch, model, val_loader, logger, writer):  
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions = [], []
        semantic_labels, offset_labels = [], []
        upper_offset_predictions, upper_offset_labels = [], []
        axis_cosine_errors = []
        axis_xy_errors = []
        axis_target_xy_lengths = []
        axis_confidences = []
        for batch in tqdm.tqdm(val_loader):

            # forward
            output = model(batch, return_loss=False)
            offset_prediction, semantic_prediction_logit = output['offset_predictions'], output['semantic_prediction_logits']
            upper_offset_prediction = output.get('upper_offset_predictions')
            axis_xy_prediction = output.get('axis_xy_predictions')

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
                    batch['masks_upper'].to(candidate_mask.device) &
                    candidate_mask)
                if masks_axis_xy.sum() > 0:
                    target_axis_xy = (
                        batch['upper_offset_labels'][
                            masks_axis_xy.cpu(), :2] -
                        batch['offset_labels'][
                            masks_axis_xy.cpu(), :2]).to(
                                axis_xy_prediction.device)
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
    return pointwise_eval(
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels,
        upper_offset_predictions, upper_offset_labels, axis_cosine_error,
        config, epoch, writer, logger, axis_xy_errors, axis_confidences,
        axis_target_xy_lengths)


def pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels,
                   upper_offset_predictions, upper_offset_labels, axis_cosine_error,
                   config, epoch, writer, logger, axis_xy_errors=None,
                   axis_confidences=None, axis_target_xy_lengths=None):
    # get offset loss
    masks_sem = torch.ones_like(semantic_labels).bool()
    masks_off = semantic_labels == TREE_CLASS_IN_DATASET
    _, offset_loss = point_wise_loss(semantic_prediction_logits.float(), offset_predictions.float(), 
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
    log_str = (
        f'[VALIDATION] [{epoch}/{config.epochs}] val/semantic_acc {acc*100:.2f}, '
        f'val/offset_loss {offset_loss.item():.3f}')
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
        log_str += (
            f", val/axis_xy_mean_error {axis_metrics['mean']:.3f}, "
            f"val/axis_xy_median_error {axis_metrics['median']:.3f}, "
            f"val/axis_xy_p90_error {axis_metrics['p90']:.3f}, "
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
    writer.add_scalar('val/Offset_Loss', offset_loss, epoch)
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
    return axis_metrics


def main():
    args, config = get_args_and_cfg()
    logger, writer = init_train_logger(config, args)
    seed = int(getattr(config, 'seed', 42))
    set_random_seed(seed)
    logger.info(f'Random seed: {seed}')

    # training objects
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
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_cosine_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    train_set = TreeDataset(**config.dataset_train, logger=logger)
    val_set = TreeDataset(**config.dataset_test, logger=logger)
    train_loader = build_dataloader(
        train_set, training=True, seed=seed, **config.dataloader.train)
    val_loader = build_dataloader(
        val_set, training=False, seed=seed + 1, **config.dataloader.test)
    
    # optionally pretrain or resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif config.pretrain:
        logger.info(f'Load pretrain from {config.pretrain}')
        load_checkpoint(config.pretrain, logger, model)

    # train and val
    logger.info('Training')
    best_axis_xy_mean = float('inf')
    for epoch in range(start_epoch, config.epochs + 1):
        train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer)
        if is_multiple(epoch, config.validation_frequency):
            optimizer.zero_grad()
            logger.info('Validation')
            torch.cuda.empty_cache()
            axis_metrics = validate(
                config, epoch, model, val_loader, logger, writer)
            if (
                axis_metrics is not None and
                axis_metrics['mean'] < best_axis_xy_mean
            ):
                best_axis_xy_mean = axis_metrics['mean']
                checkpoint_save_named(
                    epoch, model, optimizer, config.work_dir,
                    'best_axis_xy.pth')
                logger.info(
                    'Saved best_axis_xy.pth at epoch '
                    f'{epoch} (mean XY error {best_axis_xy_mean:.3f} m)')
        writer.flush()


if __name__ == '__main__':
    main()
