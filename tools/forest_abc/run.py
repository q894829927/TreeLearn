"""Foreground complete A+B+C workflow. Model execution belongs on the server."""

import argparse
import copy
import importlib
import json
import math
import os
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
from tree_learn.mask_refinement.evaluation import evaluate_labels, macro_metrics, compare_metrics
from tree_learn.forest_abc.data import RawForest, entries, prepare_rows
from tree_learn.forest_abc.model import ForestABC, tensor_block
from tree_learn.forest_abc.loss import instance_loss
from tree_learn.forest_abc.inference import predict_forest
from tree_learn.forest_abc.runtime import MaskGradientAudit, resolve_cuda_device


def configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for group in ('blocks', 'training'):
        for name, value in config[group].items():
            if not isinstance(value, (int, float)) and name != 'device':
                raise ValueError(f'Expected a number: {group}.{name}')
            if isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f'{group}.{name} must be positive.')
    if not 0 < config['blocks']['core_radius'] <= config['blocks']['radius']:
        raise ValueError('Require 0 < core_radius <= radius.')
    if config['model']['dim'] % config['model']['heads']:
        raise ValueError('Decoder dim must be divisible by heads.')
    if set(config['splits']['train']) & set(config['splits']['validation']):
        raise ValueError('Training and validation forests overlap.')
    return config


def source_hashes():
    paths = sorted((ROOT / 'tree_learn/forest_abc').rglob('*.py'))
    paths += [ROOT / 'tools/forest_abc/run.py', ROOT / 'tree_learn/query_segmentation/data.py',
              ROOT / 'tree_learn/query_segmentation/inference.py',
              ROOT / 'tree_learn/mask_refinement/evaluation.py']
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths}


def signature(config, rows):
    return digest(dict(model=config['model'], blocks=config['blocks'], loss=config['loss'],
                       training=config['training'], seed=config['seed'],
                       forests=[dict(plot=r['plot'], split=r['split'],
                                     index=json.loads((Path(r['index']) / 'index.json').read_text()),
                                     labels=(json.loads((Path(r['index']) / 'gt.json').read_text())
                                             if r['split'] == 'train' else None)) for r in rows]))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(config):
    return resolve_cuda_device(config['training']['device'])


def memory(device):
    import resource
    return dict(peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)))


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class BlockSampler:
    def __init__(self, rows, blocks, seed, maximum_open=2):
        self.rows, self.blocks = rows, blocks
        self.rng = np.random.default_rng(seed)
        self.maximum_open = maximum_open
        self.forests = OrderedDict()

    def draw(self):
        for _ in range(100):
            row = self.rows[int(self.rng.integers(len(self.rows)))]
            key = row['plot']
            if key not in self.forests:
                if len(self.forests) >= self.maximum_open:
                    self.forests.popitem(last=False)
                print(f'LOAD_COORDINATES {key}', flush=True)
                self.forests[key] = RawForest(row['cache'], row['index'], with_labels=True)
            forest = self.forests[key]
            self.forests.move_to_end(key)
            center = forest.centers[int(self.rng.integers(len(forest.centers)))].copy()
            center += self.rng.uniform(-.5, .5, 2) * self.blocks['stride']
            block = forest.make_block(center, self.blocks, int(self.rng.integers(2**31)), training=True)
            if (block is not None and np.count_nonzero(block['labels'] > 0) >= 16
                    and np.count_nonzero((block['labels'] > 0) & block['core']) >= 4):
                return row['plot'], block
        raise RuntimeError('100 sampled blocks had no labeled trees. Inspect coordinate/label inputs.')


def doctor(config):
    device = device_for(config)
    for module in ('spconv.pytorch', 'scipy.spatial', 'yaml'):
        importlib.import_module(module)
    print(json.dumps(dict(python=sys.executable, torch=torch.__version__,
                          torch_cuda=torch.version.cuda, device=str(device), device_index=device.index,
                          gpu=torch.cuda.get_device_name(device.index),
                          free_gpu_bytes=torch.cuda.mem_get_info(device.index)[0],
                          inputs='coords only; no frozen features or teacher predictions'), indent=2), flush=True)


def prepare(config):
    rows = entries(config)
    prepare_rows(config, rows)
    external = []
    if (Path(config['external_cache_root']) / 'manifest.json').is_file():
        external = entries(config, external=True)
        if set(r['plot'] for r in external) & set(r['plot'] for r in rows):
            raise ValueError('External forest repeats a source forest name.')
        prepare_rows(config, external)
    else:
        print('External coordinate cache absent; prepare it before external evaluation.', flush=True)
    write_json(Path(config['index_root']) / 'manifest.json', dict(schema=1, forests=rows, external=external))
    print('PREPARE_DONE: no network features were loaded or computed.', flush=True)


def check(config):
    """Three bounded server steps verify mask gradients reach the full network."""
    device = device_for(config)
    seed_everything(config['seed'])
    settings = copy.deepcopy(config)
    settings['blocks']['max_train_voxels'] = config['training']['check_voxels']
    sampler = BlockSampler([r for r in entries(config) if r['split'] == 'train'],
                           settings['blocks'], config['seed'])
    model = ForestABC(config['model']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(1, config),
                                  weight_decay=config['training']['weight_decay'])
    selected = {'backbone': next(model.input_conv.parameters())}
    if model.geometry is not None:
        selected['geometry'] = model.geometry.mixer.final.weight
    region_activity = {}
    hooks = []

    def capture_regions(name):
        def capture(module, inputs):
            populated = (inputs[2].detach() > 0).any(-1)
            semantic_count = model.decoder.num_semantic_queries
            instances = populated[:-semantic_count] if semantic_count else populated
            region_activity[name] = dict(queries=len(populated), nonempty_queries=int(populated.sum()),
                                         instance_queries=len(instances),
                                         nonempty_instance_queries=int(instances.sum()))
        return capture

    for index, module in model.decoder.region_updates.items():
        name = f'mfa.layer_{index}'
        selected[name] = module.output[0].weight
        hooks.append(module.register_forward_pre_hook(capture_regions(name)))
    audit = MaskGradientAudit(selected)
    checks = []
    for step in range(config['training']['check_steps']):
        plot, block = sampler.draw()
        batch = tensor_block(block, device, supervision=True)
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(step + 1, config)
        for group in optimizer.param_groups:
            group['lr'] = rate
        region_activity.clear()
        prediction = model(batch, teacher_fraction=1.)
        loss, statistics, components = instance_loss(prediction, batch, config['loss'], return_components=True)
        if step == 0:
            dtype_report = dict(inputs={name: str(value.dtype) for name, value in batch.items()},
                                mask=str(prediction['layers'][-1]['masks'].dtype),
                                score=str(prediction['layers'][-1]['scores'].dtype),
                                total_loss=str(loss.dtype),
                                components={name: str(value.dtype) for name, value in components.items()})
            print('CHECK_DTYPES ' + json.dumps(dtype_report), flush=True)
        if loss.dtype != prediction['layers'][-1]['masks'].dtype:
            raise TypeError('Loss dtype differs from mask predictions; inspect CHECK_DTYPES.')
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss in runtime check.')
        gradients = torch.autograd.grad(components['mask'], list(selected.values()),
                                        retain_graph=True, allow_unused=True)
        try:
            gradient_report, probe_report = audit.observe(gradients)
        except (RuntimeError, FloatingPointError):
            print('CHECK_MFA_REGIONS ' + json.dumps(region_activity), flush=True)
            raise
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        checks.append(dict(step=step + 1, plot=plot, learning_rate=rate,
                           mask_gradient_l1=gradient_report, mask_gradient_by_probe_l1=probe_report,
                           zero_gradient_groups=[name for name, value in gradient_report.items() if value == 0],
                           mfa_regions=copy.deepcopy(region_activity), **statistics))
        print('CHECK ' + json.dumps(checks[-1]), flush=True)
        del prediction, loss, components, batch, gradients
    gradient_audit = audit.finish()
    for hook in hooks:
        hook.remove()
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_block(block, device, supervision=False))
        checks[-1]['inference_queries'] = len(prediction['anchors'])
    output = Path(config['work_dir']) / 'check'
    write_json(output / 'report.json', dict(check_only=True, checks=checks, gradient_audit=gradient_audit,
               parameter_count=sum(p.numel() for p in model.parameters()),
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               code=source_hashes(), **memory(device)))
    print(f'CHECK_PASSED: {output / "report.json"}; full training starts independently.', flush=True)


def evaluate_model(config, model, rows, output, device, checkpoint_step):
    reports = []
    for row in rows:
        print(f'EVALUATE {row["plot"]}: full backbone + GeoMixer + MFA', flush=True)
        forest = RawForest(row['cache'], row['index'])
        labels, statistics = predict_forest(model, forest, config, Path(output) / row['plot'], device)
        # Ground truth and original baseline are opened only after prediction.
        truth = np.load(Path(row['cache']) / 'labels.npy', mmap_mode='r', allow_pickle=False)
        baseline_labels = np.load(Path(row['cache']) / 'baseline.npy', mmap_mode='r', allow_pickle=False)
        old, new = evaluate_labels(truth, baseline_labels), evaluate_labels(truth, labels)
        report = dict(plot=row['plot'], baseline=old, refined=new, delta=compare_metrics(old, new),
                      prediction=statistics,
                      lost_gt_ids=sorted(set(old['matched_gt_ids']) - set(new['matched_gt_ids'])),
                      recovered_gt_ids=sorted(set(new['matched_gt_ids']) - set(old['matched_gt_ids'])))
        write_json(Path(output) / row['plot'] / 'metrics.json', report)
        reports.append(report)
        print(row['plot'] + ': ' + json.dumps(report['delta']), flush=True)
        del forest, labels, truth, baseline_labels
    old, new = [macro_metrics([r[name] for r in reports]) for name in ('baseline', 'refined')]
    summary = dict(schema=1, architecture='ForestABC full end-to-end binary forest adaptation',
                   protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5',
                   checkpoint_step=checkpoint_step, baseline_macro=old, refined_macro=new,
                   delta_pp=compare_metrics(old, new), inference=config['inference'],
                   forests=reports, code=source_hashes(), **memory(device))
    write_json(Path(output) / 'summary.json', summary)
    brief = {k: summary[k] for k in ('checkpoint_step', 'baseline_macro', 'refined_macro', 'delta_pp')}
    brief['totals'] = {name: sum(r['refined'][name] for r in reports) for name in ('tp', 'fp', 'fn')}
    write_json(Path(output) / 'brief.json', brief)
    print('SAVED ' + str(Path(output) / 'brief.json') + '\n' + json.dumps(brief, indent=2), flush=True)
    return summary


def learning_rate(step, config):
    settings = config['training']
    if step <= settings['warmup_steps']:
        factor = max(.01, step / settings['warmup_steps'])
    else:
        phase = (step - settings['warmup_steps']) / max(1, settings['steps'] - settings['warmup_steps'])
        factor = .01 + .99 * .5 * (1 + math.cos(math.pi * phase))
    return settings['learning_rate'] * factor


def train(config, resume=False):
    device = device_for(config)
    seed_everything(config['seed'])
    rows = entries(config)
    train_rows = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    identity = signature(config, rows)
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    settings = config['training']
    sampler = BlockSampler(train_rows, config['blocks'], config['seed'], settings['max_open_forests'])
    model = ForestABC(config['model']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    step, validated_step, best = 0, 0, -1.
    if resume:
        saved = torch.load(work / 'last.pth', map_location='cpu', weights_only=False)
        if saved['signature'] != identity or saved.get('check_only', False):
            raise ValueError('Resume configuration/data differ from the checkpoint.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step, validated_step, best = saved['step'], saved['validated_step'], saved['best_validation_f1']
        sampler.rng.bit_generator.state = saved['sampler_rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'RESUME step={step} last_validation={validated_step} best_f1={best:.5f}', flush=True)
    elif (work / 'last.pth').exists():
        raise FileExistsError('last.pth exists. Use resume to continue or choose a new work_dir.')
    code = source_hashes()
    started = time.monotonic()
    print(json.dumps(dict(event='training_start', step=step,
                          parameters=sum(p.numel() for p in model.parameters()),
                          trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          architecture=config['model'], inputs='raw xyz only', work_dir=str(work))), flush=True)

    def checkpoint():
        return dict(schema=1, check_only=False, signature=identity, config=config, code=code,
                    model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    validated_step=validated_step, best_validation_f1=best,
                    sampler_rng=sampler.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all())

    def validate():
        nonlocal validated_step, best
        result = evaluate_model(config, model, validation, work / 'validation' / f'step_{step:06d}', device, step)
        validated_step = step
        if result['refined_macro']['f1'] > best:
            best = result['refined_macro']['f1']
            atomic_checkpoint(work / 'best.pth', checkpoint())
            print(f'BEST step={step} validation_f1={best:.6f}', flush=True)
        model.train()
        atomic_checkpoint(work / 'last.pth', checkpoint())

    if not resume:
        (work / 'training.jsonl').write_text('', encoding='utf-8')
        atomic_checkpoint(work / 'last.pth', checkpoint())
    # Save-before-evaluation permits retrying an interrupted validation without
    # repeating completed optimization steps or skipping checkpoint selection.
    if step and validated_step < step and (step % settings['evaluate_every'] == 0 or step == settings['steps']):
        validate()
    while step < settings['steps']:
        iteration = time.monotonic()
        plot, block = sampler.draw()
        batch = tensor_block(block, device, supervision=True)
        data_seconds = time.monotonic() - iteration
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(step + 1, config)
        for group in optimizer.param_groups:
            group['lr'] = rate
        teacher = max(0., 1 - step / settings['sampler_teacher_steps'])
        prediction = model(batch, teacher_fraction=teacher)
        loss, statistics = instance_loss(prediction, batch, config['loss'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at step {step + 1}, forest {plot}.')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        log = dict(step=step, plot=plot, input_voxels=len(block['xyz']),
                   available_voxels=block['available_voxels'], raw_points=block['raw_points'],
                   learning_rate=rate, teacher_fraction=teacher, gradient_norm=float(grad_norm),
                   mean_quality=float(prediction['layers'][-1]['scores'][:-2].sigmoid().mean()),
                   data_seconds=data_seconds, step_seconds=time.monotonic() - iteration,
                   elapsed_seconds=time.monotonic() - started, **statistics, **memory(device))
        with (work / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(log) + '\n')
        if step == 1 or step % settings['log_every'] == 0:
            print(json.dumps(log), flush=True)
        del loss, prediction, batch, block
        if step % settings['save_every'] == 0 or step % settings['evaluate_every'] == 0 or step == settings['steps']:
            atomic_checkpoint(work / 'last.pth', checkpoint())
        if step % settings['evaluate_every'] == 0 or step == settings['steps']:
            validate()
    write_json(work / 'train_summary.json', dict(step=step, best_validation_f1=best,
               checkpoint=str(work / 'best.pth'), model=config['model'], **memory(device)))
    print(f'TRAIN_DONE best_f1={best:.6f}; checkpoint={work / "best.pth"}', flush=True)


def evaluate(config, external=False, checkpoint_path=None):
    device = device_for(config)
    path = Path(checkpoint_path or Path(config['work_dir']) / 'best.pth')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source_rows = entries(config)
    if saved.get('check_only', True) or saved['signature'] != signature(config, source_rows):
        raise ValueError('Evaluation requires a compatible full-training checkpoint.')
    model = ForestABC(config['model']).to(device)
    model.load_state_dict(saved['model'])
    rows = entries(config, external=True) if external else [r for r in source_rows if r['split'] == 'validation']
    if external and set(r['plot'] for r in rows) & set(r['plot'] for r in source_rows):
        raise ValueError('External forest overlaps source forests.')
    output = Path(config['work_dir']) / ('external_evaluation' if external else 'evaluation')
    summary = evaluate_model(config, model, rows, output, device, saved['step'])
    summary.update(checkpoint=str(path), checkpoint_sha256=file_digest(path),
                   training_code=saved['code'], config=config)
    write_json(output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['doctor', 'prepare', 'check', 'train', 'evaluate', 'external'])
    parser.add_argument('--config', default='configs/experiments/forest_abc/full.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = configuration(args.config)
    if args.command == 'train':
        train(config, resume=args.resume)
    elif args.command in ('evaluate', 'external'):
        evaluate(config, external=args.command == 'external', checkpoint_path=args.checkpoint)
    else:
        {'doctor': doctor, 'prepare': prepare, 'check': check}[args.command](config)


if __name__ == '__main__':
    main()
