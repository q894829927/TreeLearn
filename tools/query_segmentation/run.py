"""Practical TreeLearn query decoder: prepare existing caches, train, evaluate."""

import argparse
import copy
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from tree_learn.mask_refinement.data import digest, file_digest, write_json
from tree_learn.mask_refinement.evaluation import compare_metrics, evaluate_labels, macro_metrics
from tree_learn.query_segmentation.data import Forest, prepare_index
from tree_learn.query_segmentation.inference import predict_forest
from tree_learn.query_segmentation.loss import instance_loss
from tree_learn.query_segmentation.model import QueryInstanceDecoder


def entries(settings, external=False):
    source = yaml.safe_load(Path(settings['source_config']).read_text(encoding='utf-8'))
    if external:
        root = Path(settings['external_cache_root'])
    else:
        root = Path(source['output_root'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if external:
        training_manifest = json.loads((Path(source['output_root']) / 'manifest.json').read_text(encoding='utf-8'))
        if (not manifest.get('baseline_signature') or
                manifest['baseline_signature'] != training_manifest.get('baseline_signature')):
            raise ValueError('External and training caches use different TreeLearn features/checkpoints.')
    result = []
    for record in manifest['forests']:
        if not external and record['split'] not in ('train', 'validation'):
            continue
        cache = root / record['cache']
        if not (cache / 'cache.json').exists():
            raise FileNotFoundError(f'Completed A1 feature cache required: {cache}')
        result.append(dict(plot=record['plot'], split='external' if external else record['split'],
                           cache=str(cache), index=str(Path(settings['index_root']) /
                           ('external' if external else record['split']) / record['plot'])))
    if not result:
        raise ValueError('No forests in the source manifest.')
    if not external:
        actual = {split: [r['plot'] for r in result if r['split'] == split] for split in ('train', 'validation')}
        if actual != source['splits']:
            raise ValueError('A1 manifest and source forest splits disagree.')
    return result


def prepare(settings, external=False):
    rows = entries(settings, external)
    for row in rows:
        print(f'INDEX {row["split"]}/{row["plot"]}: streaming one forest', flush=True)
        prepare_index(row['cache'], row['index'], settings['blocks']['stride'], with_labels=not external)
    write_json(Path(settings['index_root']) / ('external_manifest.json' if external else 'manifest.json'),
               dict(schema=1, forests=rows))
    print('PREPARE_DONE: reused feature caches; no backbone inference.', flush=True)


def sampler_progress(step, training):
    return min(1., max(0., (step - training['sampler_warmup']) / training['sampler_transition']))


def validate_settings(settings):
    blocks, training, infer = (settings[name] for name in ('blocks', 'training', 'inference'))
    if not 0 < blocks['core_radius'] <= blocks['radius']:
        raise ValueError('Require 0 < core_radius <= radius.')
    if not np.isclose(settings['model']['coordinate_scale'], blocks['radius']):
        raise ValueError('coordinate_scale must equal the block radius used to normalize coordinates.')
    for name in ('stride', 'token_voxel', 'max_tokens', 'train_points'):
        if blocks[name] <= 0:
            raise ValueError(f'blocks.{name} must be positive.')
    for name in ('steps', 'smoke_steps', 'sampler_transition', 'statistics_points_per_forest',
                 'evaluate_every', 'save_every', 'log_every'):
        if training[name] <= 0:
            raise ValueError(f'training.{name} must be positive.')
    for name in ('chunk_size', 'max_block_memberships', 'max_forest_memberships', 'log_every_blocks'):
        if infer[name] <= 0:
            raise ValueError(f'inference.{name} must be positive.')
    for name in ('mask_threshold', 'quality_threshold', 'nms_iou', 'nms_containment'):
        if not 0 <= infer[name] <= 1:
            raise ValueError(f'inference.{name} must be in [0,1].')


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def training_signature(settings, rows):
    # Inference thresholds can be explored without retraining. Every evaluation
    # records its policy, and CLI overrides write to a separate output directory.
    return digest(dict(model=settings['model'], blocks=settings['blocks'], training=settings['training'],
                       seed=settings['seed'], data=[dict(plot=r['plot'], split=r['split'],
                       index=json.loads((Path(r['index']) / 'index.json').read_text())) for r in rows]))


def code_sources():
    files = ['tools/query_segmentation/run.py'] + [f'tree_learn/query_segmentation/{name}.py'
              for name in ('data', 'model', 'loss', 'inference')]
    return {name: file_digest(ROOT / name) for name in files}


def memory_report(device):
    try:
        import resource
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1024 if sys.platform != 'darwin' else 1)
    except ImportError:
        rss = None
    return dict(peak_rss_bytes=rss,
                peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0)


def evaluate(settings, model, rows, output, device, progress=1., partial=False):
    output = Path(output)
    reports = []
    for row in rows:
        print(f'EVALUATE {row["plot"]}: fine-point masks + global NMS', flush=True)
        forest = Forest(row['cache'], row['index'], with_labels=False)
        labels, statistics = predict_forest(model, forest, settings, output / row['plot'], device,
                                           progress, max_blocks=1 if partial else None)
        # Prediction has finished before GT or the original instance IDs open.
        truth = np.load(Path(row['cache']) / 'labels.npy', mmap_mode='r', allow_pickle=False)
        baseline_labels = np.load(Path(row['cache']) / 'baseline.npy', mmap_mode='r', allow_pickle=False)
        baseline, refined = evaluate_labels(truth, baseline_labels), evaluate_labels(truth, labels)
        old_fg = baseline_labels > 0
        statistics.update(original_foreground_points=int(old_fg.sum()),
                          abstained_original_foreground=int(np.count_nonzero(old_fg & (labels == 0))),
                          assigned_original_background_points=int(np.count_nonzero((~old_fg) & (labels > 0))),
                          recovered_gt_tree_points_from_original_background=int(np.count_nonzero(
                              (~old_fg) & (labels > 0) & (truth > 0))),
                          abstained_gt_tree_points=int(np.count_nonzero((labels == 0) & (truth > 0))))
        report = dict(plot=row['plot'], baseline=baseline, refined=refined,
                      delta=compare_metrics(baseline, refined), prediction=statistics,
                      lost_gt_ids=sorted(set(baseline['matched_gt_ids']) - set(refined['matched_gt_ids'])),
                      recovered_gt_ids=sorted(set(refined['matched_gt_ids']) - set(baseline['matched_gt_ids'])))
        write_json(output / row['plot'] / 'metrics.json', report)
        reports.append(report)
        print(f'{row["plot"]}: {json.dumps(report["delta"])}', flush=True)
        if statistics['empty_prediction_reason']:
            print(f'WARNING {row["plot"]}: {statistics["empty_prediction_reason"]}; inspect quality scores.', flush=True)
        del forest, labels, truth, baseline_labels, old_fg
    baseline, refined = (macro_metrics([r[name] for r in reports]) for name in ('baseline', 'refined'))
    result = dict(schema=1, protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5',
                  partial=partial, baseline_macro=baseline, refined_macro=refined,
                  delta_pp=compare_metrics(baseline, refined), forests=reports,
                  policy=settings['inference'])
    write_json(output / 'summary.json', result)
    print(f'SAVED {output}/summary.json', flush=True)
    return result


def complete_validation(settings, model, rows, work, device, checkpoint):
    """A pending checkpoint survives inference errors and can retry this step."""
    step = checkpoint['step']
    result = evaluate(settings, model, rows, work / 'validation' / f'step_{step:06d}',
                      device, checkpoint['sampler_progress'])
    improved = result['refined_macro']['f1'] > checkpoint['best_f1']
    checkpoint['best_f1'] = max(checkpoint['best_f1'], result['refined_macro']['f1'])
    checkpoint['pending_validation'] = False
    if improved:
        save_checkpoint(work / 'best.pth', checkpoint)
        write_json(work / 'best_validation.json', dict(step=step, **result))
    save_checkpoint(work / 'last.pth', checkpoint)
    return checkpoint['best_f1']


def train(settings, resume=False, smoke=False):
    rows = entries(settings)
    signature = training_signature(settings, rows)
    work = Path(settings['work_dir']) / 'smoke' if smoke else Path(settings['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'last.pth').exists() and not resume:
        raise FileExistsError(f'{work}/last.pth already exists; use --resume.')
    cfg = settings['training']
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; activate the server TreeLearn environment.')
    torch.manual_seed(settings['seed'])
    np.random.seed(settings['seed'])
    random.seed(settings['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(settings['seed'])
        torch.cuda.reset_peak_memory_stats(device)
    model = QueryInstanceDecoder(**settings['model']).to(device)
    training_rows = [r for r in rows if r['split'] == 'train']
    validation_rows = [r for r in rows if r['split'] == 'validation']
    samples = []
    statistics_rng = np.random.default_rng(settings['seed'])
    for row in training_rows:
        features = np.load(Path(row['cache']) / 'features.npy', mmap_mode='r', allow_pickle=False)
        chosen = statistics_rng.choice(len(features), min(len(features), cfg['statistics_points_per_forest']), replace=False)
        samples.append(features[chosen])
    sampled = np.concatenate(samples).astype(np.float64)
    model.feature_mean.copy_(torch.tensor(sampled.mean(axis=0), dtype=torch.float32, device=device))
    model.feature_scale.copy_(torch.tensor(np.maximum(sampled.std(axis=0), 1e-3), dtype=torch.float32, device=device))
    del sampled, samples, features
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    start, best_f1, history = 0, -1., []
    if resume:
        last = torch.load(work / 'last.pth', map_location=device, weights_only=False)
        if last['signature'] != signature or last['smoke'] != smoke:
            raise ValueError('Resume data/model/training settings differ from the checkpoint.')
        if last['settings']['inference'] != settings['inference']:
            raise ValueError('Keep the validation policy unchanged when resuming; use evaluate for threshold trials.')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        start, best_f1, history = last['step'], last['best_f1'], last['history']
    maximum = int(cfg['smoke_steps'] if smoke else cfg['steps'])
    forests = OrderedDict()
    started = time.monotonic()
    last_log = started
    print(json.dumps(dict(event='train_start', smoke=smoke, steps=maximum, start=start,
                          parameters=sum(p.numel() for p in model.parameters()), device=str(device))), flush=True)
    if resume and last.get('pending_validation', False):
        print(json.dumps(dict(event='resume_validation', step=start)), flush=True)
        best_f1 = complete_validation(settings, model, validation_rows, work, device, last)
    step = start
    for step in range(start + 1, maximum + 1):
        model.train()
        iteration = time.monotonic()
        rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], step]))
        for attempt in range(30):
            row = training_rows[int(rng.integers(len(training_rows)))]
            key = row['plot']
            if key not in forests:
                forests[key] = Forest(row['cache'], row['index'], with_labels=True)
            forests.move_to_end(key)
            while len(forests) > 2:
                forests.popitem(last=False)
            forest = forests[key]
            center = forest.centers[int(rng.integers(len(forest.centers)))].copy()
            center += rng.uniform(-forest.cell_size / 2, forest.cell_size / 2, 2)
            try:
                block = forest.make_block(center, settings['blocks'], int(rng.integers(2 ** 32)), training=True)
                break
            except ValueError as error:
                if str(error) not in ('Empty spatial block.', 'No query anchors in the central region.',
                                      'Block has no known labels.'):
                    raise
        else:
            raise RuntimeError('Could not find a nonempty labeled training block.')
        load_seconds = time.monotonic() - iteration
        progress = sampler_progress(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        values = {name: torch.from_numpy(block[name]).to(device) for name in
                  ('tokens', 'core', 'points', 'point_labels', 'weights', 'token_labels')}
        output = model(values['tokens'], values['core'], values['points'], progress)
        targets = values['token_labels'][output['query_indices']]
        counts = torch.from_numpy(forest.target_sizes(targets.detach().cpu().numpy())).to(device)
        loss, logs = instance_loss(output, targets, values['point_labels'], values['weights'], counts)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss; inspect mask/quality/embedding components.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip_norm'], error_if_nonfinite=True)
        warmup = min(1., step / max(cfg['lr_warmup'], 1))
        decay = max(.05, 1 - step / max(maximum, 1))
        for group in optimizer.param_groups:
            group['lr'] = cfg['learning_rate'] * warmup * decay
        optimizer.step()
        if step == 1 or step % cfg['log_every'] == 0 or step == maximum or time.monotonic() - last_log >= 30:
            report = dict(step=step, plot=key, tokens=len(block['tokens']), points=len(block['points']),
                          sampler_blend=progress, data_seconds=load_seconds,
                          step_seconds=time.monotonic() - iteration,
                          elapsed_seconds=time.monotonic() - started, **logs, **memory_report(device))
            history.append(report)
            print(json.dumps(report), flush=True)
            write_json(work / 'progress.json', report)
            last_log = time.monotonic()
        validation = (not smoke and (step % cfg['evaluate_every'] == 0 or step == maximum))
        if validation or step % cfg['save_every'] == 0 or step == maximum:
            checkpoint = dict(schema=1, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                              best_f1=best_f1, history=history, signature=signature, smoke=smoke,
                              settings=copy.deepcopy(settings), sampler_progress=progress,
                              source_code=code_sources(), pending_validation=validation)
            # Save the updated weights BEFORE forest inference can fail.
            save_checkpoint(work / 'last.pth', checkpoint)
            write_json(work / 'history.json', history)
            print(json.dumps(dict(event='checkpoint_saved', step=step, pending_validation=validation)), flush=True)
            if validation:
                best_f1 = complete_validation(settings, model, validation_rows, work, device, checkpoint)
            elif smoke:
                save_checkpoint(work / 'best.pth', checkpoint)
    summary = dict(smoke=smoke, step=step, parameter_count=sum(p.numel() for p in model.parameters()),
                   best_validation_f1=best_f1 if best_f1 >= 0 else None,
                   checkpoint=str(work / 'best.pth'), training_forests=[r['plot'] for r in training_rows],
                   validation_forests=[r['plot'] for r in validation_rows],
                   ownership_rank=False, **memory_report(device))
    write_json(work / 'training_summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--config', default='configs/experiments/query_instance_decoder/v1.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--external', action='store_true', help='Reuse the existing Wytham cache; evaluation only.')
    parser.add_argument('--quality-threshold', type=float)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the TreeLearn repository root.')
    if (args.resume and args.command != 'train') or (args.external and args.command == 'train'):
        parser.error('--resume applies to train; external forests cannot train.')
    if args.quality_threshold is not None and args.command != 'evaluate':
        parser.error('--quality-threshold applies to evaluate only.')
    settings = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.quality_threshold is not None:
        if not 0 <= args.quality_threshold <= 1:
            parser.error('quality threshold must be in [0,1].')
        settings['inference']['quality_threshold'] = args.quality_threshold
    validate_settings(settings)
    if args.command == 'prepare':
        prepare(settings, args.external)
    elif args.command == 'train':
        train(settings, args.resume, args.smoke)
    else:
        if args.smoke and args.external:
            parser.error('External evaluation requires a full checkpoint.')
        rows = entries(settings, args.external)
        if args.external:
            prepare(settings, True)
        else:
            rows = [r for r in rows if r['split'] == 'validation']
        if args.smoke:
            rows = rows[:1]
        work = Path(settings['work_dir']) / 'smoke' if args.smoke else Path(settings['work_dir'])
        selected = work / 'best.pth'
        checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
        if checkpoint['smoke'] != args.smoke or checkpoint['settings']['model'] != settings['model']:
            raise ValueError('Model/checkpoint mode differs.')
        if checkpoint['signature'] != training_signature(settings, entries(settings)):
            raise ValueError('Checkpoint model, source data or training settings differ.')
        device = torch.device(settings['training']['device'])
        model = QueryInstanceDecoder(**settings['model']).to(device)
        model.load_state_dict(checkpoint['model'])
        name = 'external_evaluation' if args.external else 'evaluation'
        if args.quality_threshold is not None:
            name += f'_q{args.quality_threshold:g}'
        report = evaluate(settings, model, rows, work / name, device,
                          checkpoint['sampler_progress'], partial=args.smoke)
        report.update(checkpoint=str(selected), checkpoint_step=checkpoint['step'],
                      checkpoint_sha256=file_digest(selected), source_code=code_sources(),
                      exploratory_external=bool(args.external))
        write_json(work / name / 'summary.json', report)


if __name__ == '__main__':
    main()
