"""Foreground V2 workflow. Execute model checks/training only on the server."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from tools.forest_abc import run as common
from tree_learn.forest_abc.data import entries
from tree_learn.forest_abc.runtime import MaskGradientAudit, require_trainable_gradients, restore_training_state
from tree_learn.forest_abc_v2.data import CenterForest, center_request, prepare_centers
from tree_learn.forest_abc_v2.model import CenterForestModel, tensor_block
from tree_learn.forest_abc_v2.loss import instance_loss
from tree_learn.forest_abc_v2.inference import predict_forest
from tree_learn.mask_refinement.data import digest, file_digest, write_json
from tree_learn.mask_refinement.evaluation import evaluate_labels, macro_metrics, compare_metrics


def configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    integer_fields = dict(model=('channels', 'dim', 'layers', 'heads', 'feedforward', 'num_queries', 'isa_candidates'),
        blocks=('max_train_voxels', 'max_inference_voxels', 'minimum_points', 'neighbors'),
        training=('steps', 'warmup_steps', 'sampler_teacher_steps', 'evaluate_every', 'save_every',
                  'log_every', 'max_open_forests', 'check_steps', 'check_voxels'),
        inference=('minimum_instance_points', 'chunk_size', 'max_block_memberships', 'max_forest_memberships',
                   'log_every_blocks', 'max_proposals_per_block'))
    boolean_fields = dict(model=('use_geometry', 'use_mfa', 'checkpoint_decoder'), inference=('keep_proposals',))
    for group in ('model', 'blocks', 'training', 'inference', 'loss'):
        for name, value in config[group].items():
            if name == 'device':
                continue
            if name in boolean_fields.get(group, ()):
                if type(value) is not bool:
                    raise ValueError(f'{group}.{name} must be boolean.')
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{group}.{name} must be a finite positive number.')
        for name in integer_fields.get(group, ()):
            if type(config[group][name]) is not int or config[group][name] <= 0:
                raise ValueError(f'{group}.{name} must be a positive integer.')
    if not config['model']['use_mfa'] or not config['model']['use_geometry']:
        raise ValueError('This first V2 run trains the complete combination.')
    if config['model']['layers'] < 2 or config['model']['dim'] % config['model']['heads']:
        raise ValueError('Require layers >= 2 and dim divisible by heads.')
    b, t, infer = config['blocks'], config['training'], config['inference']
    if not (2 <= b['minimum_points'] <= t['check_voxels'] <= b['max_train_voxels'] and
            b['max_inference_voxels'] >= b['minimum_points']):
        raise ValueError('Invalid check/training/inference voxel limits.')
    if not (0 < b['core_radius'] <= b['radius'] and 0 < b['scale_min'] <= b['scale_max'] and
            0 < b['density_min'] <= 1):
        raise ValueError('Invalid crop/augmentation bounds.')
    for name in ('mask_threshold', 'quality_threshold', 'ownership_threshold', 'mutual_overlap',
                 'minimum_retention', 'allocation_score_threshold'):
        if not 0 < infer[name] <= 1:
            raise ValueError(f'inference.{name} must be in (0, 1].')
    if infer['max_proposals_per_block'] > config['model']['num_queries']:
        raise ValueError('Proposal cap cannot exceed query count.')
    train, validation = config['splits']['train'], config['splits']['validation']
    if len(train) != len(set(train)) or len(validation) != len(set(validation)) or set(train) & set(validation):
        raise ValueError('Repeated or overlapping training/validation forests.')
    if Path(config['work_dir']).name == 'forest_abc_v1_s42':
        raise ValueError('V2 must use its own work_dir.')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer.')
    if type(config['initialization']['step']) is not int or config['initialization']['step'] != 20000:
        raise ValueError('This V2 experiment initializes from the frozen V1 step 20000.')
    if t['warmup_steps'] >= t['steps']:
        raise ValueError('Warmup must be shorter than training.')
    return config


def source_hashes():
    paths = list((ROOT / 'tree_learn/forest_abc_v2').glob('*.py')) + [Path(__file__)]
    return dict(common.source_hashes(), **{p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths})


def signature(config, rows):
    return digest(dict(revision=2, base=common.signature(config, rows),
        initialization=config['initialization'],
        center_inputs={r['plot']: dict(inputs=center_request(r),
            targets=file_digest(Path(r['index']) / 'v2_centers.npz')) for r in rows if r['split'] == 'train'}))


def require_initialization_checkpoint(saved, config):
    if (saved.get('schema') != 1 or saved.get('check_only', True) or
            saved.get('step') != config['initialization']['step'] or
            saved['config']['splits'] != config['splits']):
        raise ValueError('Initialization requires the frozen V1 20000-step source-training checkpoint.')


def initialize(model, config):
    path = Path(config['initialization']['checkpoint'])
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require_initialization_checkpoint(saved, config)
    current = model.state_dict()
    incompatible = [key for key, value in saved['model'].items()
                    if key not in current or value.shape != current[key].shape]
    if incompatible:
        raise ValueError(f'V1 initialization architecture mismatch: {incompatible}')
    missing, unexpected = model.load_state_dict(saved['model'], strict=False)
    allowed = ('offset.', 'decoder.source_position.', 'decoder.query_position.',
               'decoder.center_strength', 'decoder.cross_attn_layers.')
    if unexpected or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f'Unexpected initialization omissions: {missing}, {unexpected}')
    with torch.no_grad():
        for module in model.decoder.region_updates.values():
            module.strength.fill_(-4.)
    report = dict(path=str(path), step=saved['step'], sha256=file_digest(path),
                  loaded_tensors=len(saved['model']), new_tensors=missing,
                  mfa_residual_initial_strength='sigmoid(-4)')
    print('INITIALIZE ' + json.dumps(report), flush=True)
    return report


class BlockSampler(common.BlockSampler):
    def draw(self):
        for _ in range(100):
            row = self.rows[int(self.rng.integers(len(self.rows)))]
            key = row['plot']
            if key not in self.forests:
                if len(self.forests) >= self.maximum_open:
                    self.forests.popitem(last=False)
                print(f'LOAD_COORDINATES {key}', flush=True)
                self.forests[key] = CenterForest(row['cache'], row['index'], with_labels=True)
            forest = self.forests[key]
            self.forests.move_to_end(key)
            center = forest.centers[int(self.rng.integers(len(forest.centers)))].copy()
            center += self.rng.uniform(-.5, .5, 2) * self.blocks['stride']
            block = forest.make_block(center, self.blocks, int(self.rng.integers(2 ** 31)), training=True)
            if block is not None and np.count_nonzero(block['labels'] > 0) >= 16 and np.count_nonzero(
                    (block['labels'] > 0) & block['core']) >= 4:
                return key, block
        raise RuntimeError('100 sampled crops had no usable trees.')


def prepare(config, require_external=False):
    common.prepare(config, require_external=require_external)
    for row in entries(config):
        if row['split'] == 'train':
            prepare_centers(row)
            CenterForest(row['cache'], row['index'], with_labels=True)
    print('V2_PREPARE_DONE: training centers ready; validation/external GT not used for centers.', flush=True)


def check(config):
    output = Path(config['work_dir']) / 'check'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').unlink(missing_ok=True)
    device = common.device_for(config)
    common.seed_everything(config['seed'])
    rows = entries(config)
    small = copy.deepcopy(config['blocks'])
    small['max_train_voxels'] = config['training']['check_voxels']
    sampler = BlockSampler([r for r in rows if r['split'] == 'train'], small, config['seed'])
    model = CenterForestModel(config['model']).to(device).train()
    initialization = initialize(model, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    probes = dict(backbone=next(model.input_conv.parameters()), geometry=model.geometry.mixer.final.weight,
                  center=model.offset[-1].weight, position=model.decoder.query_position[-1].weight)
    for index, module in enumerate(model.decoder.cross_attn_layers):
        probes[f'cross.layer_{index}'] = module.attn.in_proj_weight
    for index, module in model.decoder.region_updates.items():
        probes[f'mfa.layer_{index}'] = module.output[0].weight
    audit = MaskGradientAudit(probes)
    reports = []

    def update(teacher, phase, probe=False):
        plot, block = sampler.draw()
        batch = tensor_block(block, device, supervision=True)
        print(f'CHECK_BEGIN phase={phase} plot={plot} voxels={len(block["xyz"])} teacher={teacher}', flush=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, teacher_fraction=teacher)
        loss, stats, components = instance_loss(prediction, batch, config['loss'], return_components=True)
        if not torch.isfinite(loss) or loss.dtype != torch.float32:
            raise FloatingPointError('V2 check loss must be finite float32.')
        gradients = {}
        if probe:
            values = torch.autograd.grad(components['mask'], list(probes.values()), retain_graph=True, allow_unused=True)
            gradients, _ = audit.observe(values)
        loss.backward()
        require_trainable_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        report = dict(phase=phase, plot=plot, voxels=len(block['xyz']), mask_gradient_l1=gradients,
                      **stats, **common.memory(device))
        reports.append(report)
        print('CHECK ' + json.dumps(report), flush=True)

    for _ in range(config['training']['check_steps']):
        update(1., 'small_gradient', probe=True)
    gradient_report = audit.finish()
    sampler.blocks = config['blocks']
    update(1., 'production_training')
    saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 sampler_rng=sampler.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all())
    common.atomic_checkpoint(output / 'roundtrip.pth', saved)
    del saved
    restored = torch.load(output / 'roundtrip.pth', map_location='cpu', weights_only=False)
    restore_training_state(restored, model, optimizer, sampler.rng)
    del restored
    update(0., 'production_after_resume_without_teacher')
    optimizer.zero_grad(set_to_none=True)
    partial = []
    groups = [[r for r in rows if r['split'] == 'validation'], entries(config, external=True)]
    for group in groups:
        def density(row):
            with np.load(Path(row['index']) / 'cells.npz', allow_pickle=False) as cells:
                return int(cells['counts'].max())
        row = max(group, key=density)
        forest = CenterForest(row['cache'], row['index'])
        forest.centers = forest.centers[[int(forest.counts.argmax())]]
        print(f'CHECK_INFERENCE_BEGIN {row["plot"]}', flush=True)
        labels, stats = predict_forest(model, forest, config, output / 'partial' / row['plot'], device, max_blocks=1)
        labels._mmap.close()
        partial.append(dict(plot=row['plot'], check_only=True, prediction=stats))
        del forest, labels
    write_json(output / 'report.json', dict(check_only=True, initialization=initialization,
        gradient_audit=gradient_report, checks=reports, partial_predictions=partial,
        checkpoint_roundtrip=True, code=source_hashes(), **common.memory(device)))
    print('CHECK_PASSED ' + str(output / 'report.json'), flush=True)


def evaluate_model(config, model, rows, output, device, step):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('brief.json', 'summary.json'):
        (output / name).unlink(missing_ok=True)
    reports = []
    for row in rows:
        print(f'EVALUATE {row["plot"]}: center-guided V2 + point competition', flush=True)
        forest = CenterForest(row['cache'], row['index'])
        labels, stats = predict_forest(model, forest, config, output / row['plot'], device)
        truth = np.load(Path(row['cache']) / 'labels.npy', mmap_mode='r', allow_pickle=False)
        baseline = np.load(Path(row['cache']) / 'baseline.npy', mmap_mode='r', allow_pickle=False)
        try:
            old, new = evaluate_labels(truth, baseline), evaluate_labels(truth, labels)
        finally:
            labels._mmap.close()
            truth._mmap.close()
            baseline._mmap.close()
        report = dict(plot=row['plot'], baseline=old, refined=new, delta=compare_metrics(old, new), prediction=stats)
        write_json(output / row['plot'] / 'metrics.json', report)
        reports.append(report)
        print(row['plot'] + ': ' + json.dumps(report['delta']), flush=True)
        del forest
    old, new = [macro_metrics([r[name] for r in reports]) for name in ('baseline', 'refined')]
    summary = dict(schema=2, architecture='ForestABC V2 center-guided ownership',
        protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5', checkpoint_step=step,
        baseline_macro=old, refined_macro=new, delta_pp=compare_metrics(old, new),
        inference=config['inference'], forests=reports, code=source_hashes(), **common.memory(device))
    brief = {key: summary[key] for key in ('checkpoint_step', 'baseline_macro', 'refined_macro', 'delta_pp')}
    brief['totals'] = {name: sum(r['refined'][name] for r in reports) for name in ('tp', 'fp', 'fn')}
    write_json(output / 'summary.json', summary)
    write_json(output / 'brief.json', brief)
    print('SAVED ' + str(output / 'brief.json') + '\n' + json.dumps(brief, indent=2), flush=True)
    return summary


def compatible(saved, config, rows):
    if (saved.get('check_only', True) or saved.get('schema') != 2 or
            saved.get('signature') != signature(config, rows) or
            saved.get('inference') != config['inference'] or saved.get('code') != source_hashes()):
        raise ValueError('V2 checkpoint configuration/data/code/policy differ; do not resume in this work_dir.')


def train(config, resume=False):
    device = common.device_for(config)
    common.seed_everything(config['seed'])
    rows = entries(config)
    validation = [r for r in rows if r['split'] == 'validation']
    settings = config['training']
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    if not resume and (work / 'last.pth').exists():
        raise FileExistsError('V2 last.pth exists. Use resume.')
    sampler = BlockSampler([r for r in rows if r['split'] == 'train'], config['blocks'],
                           config['seed'], settings['max_open_forests'])
    model = CenterForestModel(config['model']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    step, validated, best, initialization = 0, 0, -1., None
    if resume:
        saved = torch.load(work / 'last.pth', map_location='cpu', weights_only=False)
        compatible(saved, config, rows)
        restore_training_state(saved, model, optimizer, sampler.rng)
        step, validated, best = saved['step'], saved['validated_step'], saved['best_validation_f1']
        initialization = saved['initialization']
        del saved
    else:
        initialization = initialize(model, config)
        (work / 'training.jsonl').write_text('', encoding='utf-8')
    identity, code = signature(config, rows), source_hashes()
    started = time.monotonic()
    print(json.dumps(dict(event='training_start', step=step, parameters=sum(p.numel() for p in model.parameters()),
        initialization=initialization, work_dir=str(work), steps=settings['steps'])), flush=True)

    def checkpoint():
        return dict(schema=2, check_only=False, signature=identity, config=config, code=code,
            inference=config['inference'], initialization=initialization, model=model.state_dict(),
            optimizer=optimizer.state_dict(), step=step, validated_step=validated, best_validation_f1=best,
            sampler_rng=sampler.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all())

    def validate():
        nonlocal validated, best
        result = evaluate_model(config, model, validation, work / 'validation' / f'step_{step:06d}', device, step)
        validated = step
        if result['refined_macro']['f1'] > best:
            best = result['refined_macro']['f1']
            common.atomic_checkpoint(work / 'best.pth', checkpoint())
            print(f'BEST step={step} validation_f1={best:.6f}', flush=True)
        common.atomic_checkpoint(work / f'step_{step:06d}.pth', checkpoint())
        common.atomic_checkpoint(work / 'last.pth', checkpoint())
        model.train()

    if not resume:
        common.atomic_checkpoint(work / 'last.pth', checkpoint())
    if step and validated < step and (step % settings['evaluate_every'] == 0 or step == settings['steps']):
        validate()
    while step < settings['steps']:
        iteration = time.monotonic()
        plot, block = sampler.draw()
        batch = tensor_block(block, device, supervision=True)
        data_seconds = time.monotonic() - iteration
        optimizer.zero_grad(set_to_none=True)
        rate = common.learning_rate(step + 1, config)
        for group in optimizer.param_groups:
            group['lr'] = rate
        teacher = max(0., 1 - step / settings['sampler_teacher_steps'])
        prediction = model(batch, teacher_fraction=teacher)
        loss, stats = instance_loss(prediction, batch, config['loss'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite V2 training loss at step {step + 1}.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        log = dict(step=step, plot=plot, input_voxels=len(block['xyz']), available_voxels=block['available_voxels'],
            raw_points=block['raw_points'], augmentation_scale=block['augmentation_scale'],
            augmentation_density=block['augmentation_density'], learning_rate=rate, teacher_fraction=teacher,
            gradient_norm=float(norm), data_seconds=data_seconds, step_seconds=time.monotonic() - iteration,
            elapsed_seconds=time.monotonic() - started, **stats, **common.memory(device))
        with (work / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(log) + '\n')
        if step == 1 or step % settings['log_every'] == 0:
            print(json.dumps(log), flush=True)
        del loss, prediction, block, batch
        if step % settings['save_every'] == 0 or step % settings['evaluate_every'] == 0 or step == settings['steps']:
            common.atomic_checkpoint(work / 'last.pth', checkpoint())
        if step % settings['evaluate_every'] == 0 or step == settings['steps']:
            validate()
    write_json(work / 'train_summary.json', dict(step=step, best_validation_f1=best,
               checkpoint=str(work / 'best.pth'), initialization=initialization, **common.memory(device)))
    print(f'TRAIN_DONE V2 step={step} best_validation_f1={best:.6f}', flush=True)


def evaluate(config, external=False, checkpoint_path=None):
    device = common.device_for(config)
    path = Path(checkpoint_path or Path(config['work_dir']) / 'best.pth')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source_rows = entries(config)
    compatible(saved, config, source_rows)
    rows = entries(config, external=True) if external else [r for r in source_rows if r['split'] == 'validation']
    if external and set(r['plot'] for r in rows) & set(r['plot'] for r in source_rows):
        raise ValueError('External forest overlaps source forests.')
    model = CenterForestModel(config['model']).to(device)
    model.load_state_dict(saved['model'])
    output = Path(config['work_dir']) / ('external_evaluation' if external else 'evaluation')
    summary = evaluate_model(config, model, rows, output, device, saved['step'])
    summary.update(checkpoint=str(path), checkpoint_sha256=file_digest(path), config=config,
                   training_code=saved['code'], initialization=saved['initialization'])
    write_json(output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['doctor', 'prepare', 'check', 'train', 'evaluate', 'external'])
    parser.add_argument('--config', default='configs/experiments/forest_abc_v2/full.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--require-external', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = configuration(args.config)
    if args.command == 'train':
        train(config, resume=args.resume)
    elif args.command in ('evaluate', 'external'):
        evaluate(config, external=args.command == 'external', checkpoint_path=args.checkpoint)
    elif args.command == 'prepare':
        prepare(config, require_external=args.require_external)
    elif args.command == 'doctor':
        if not Path(config['initialization']['checkpoint']).is_file():
            raise FileNotFoundError('Frozen V1 best_step_020000.pth is required for V2 initialization.')
        saved = torch.load(config['initialization']['checkpoint'], map_location='cpu', weights_only=False)
        require_initialization_checkpoint(saved, config)
        del saved
        common.doctor(config)
    else:
        check(config)


if __name__ == '__main__':
    main()
