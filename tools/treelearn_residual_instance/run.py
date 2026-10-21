"""Foreground paired experiment. Execute model commands on the Linux server."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import yaml

from tree_learn.forest_abc.data import entries, prepare_rows
from tree_learn.mask_refinement.data import digest, file_digest, write_json
from tree_learn.mask_refinement.evaluation import evaluate_labels, macro_metrics, compare_metrics
from tree_learn.model import TreeLearn
from tree_learn.model.residual_instance import instance_auxiliary_loss
from tree_learn.residual_instance.data import PairedCropSampler, tile_batch
from tree_learn.util import get_config, munch_to_dict


ARMS = ('control', 'combined')
PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'


def configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    training, auxiliary = config['training'], config['auxiliary']
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer.')
    adapter = config['adapter']
    levels = adapter['levels']
    if (len(levels) != 2 or len(set(levels)) != 2 or
            any(type(level) is not int or not 1 <= level < 7 for level in levels)):
        raise ValueError('Adapter requires two distinct levels in [1, 6].')
    for field in ('projection_dim', 'hidden_dim', 'embedding_dim'):
        if type(adapter[field]) is not int or adapter[field] < 1:
            raise ValueError(f'Invalid adapter.{field}.')
    integer_fields = ('steps', 'branch_steps', 'pilot_step', 'warmup_steps', 'save_every',
                      'log_every', 'max_open_forests', 'max_points', 'minimum_points')
    for field in integer_fields:
        if type(training[field]) is not int or training[field] <= 0:
            raise ValueError(f'training.{field} must be a positive integer.')
    if not 0 < training['branch_steps'] < training['steps']:
        raise ValueError('Require 0 < branch_steps < steps.')
    if not training['warmup_steps'] < training['branch_steps']:
        raise ValueError('Warmup must finish within the branch phase.')
    validations = training['full_validation_steps']
    if (not validations or validations != sorted(set(validations)) or
            any(type(step) is not int or step <= training['pilot_step'] for step in validations)
            or validations[-1] != training['steps']):
        raise ValueError('Full validation steps must increase after pilot and end at total steps.')
    for field in ('adapter_lr', 'head_lr', 'decoder_lr', 'weight_decay', 'gradient_clip',
                  'outer_half_extent', 'inner_half_extent', 'pilot_max_f1_drop'):
        if not math.isfinite(training[field]) or training[field] <= 0:
            raise ValueError(f'Invalid training.{field}.')
    if training['inner_half_extent'] >= training['outer_half_extent']:
        raise ValueError('Inner supervision square must fit within context square.')
    if training['minimum_points'] > training['max_points']:
        raise ValueError('minimum_points exceeds the crop budget.')
    if training['pilot_max_f1_drop'] > 1:
        raise ValueError('Pilot F1-drop guard is a fraction, not a percentage.')
    for field in ('max_trees', 'points_per_tree', 'boundary_neighbors', 'max_boundary_pairs'):
        if type(auxiliary[field]) is not int or auxiliary[field] < 1:
            raise ValueError(f'Invalid auxiliary.{field}.')
    for field in ('weight', 'pull_margin', 'push_margin', 'boundary_radius', 'boundary_weight'):
        if not math.isfinite(auxiliary[field]) or auxiliary[field] <= 0:
            raise ValueError(f'Invalid auxiliary.{field}.')
    if auxiliary['pull_margin'] >= auxiliary['push_margin']:
        raise ValueError('Instance pull margin must be below the separation margin.')
    if type(training['fp16']) is not bool:
        raise ValueError('fp16 must be Boolean.')
    if set(config['splits']['train']) & set(config['splits']['validation']):
        raise ValueError('Train/validation plot names overlap.')
    if training['pilot_plot'] not in config['splits']['validation']:
        raise ValueError('Pilot must use a validation forest.')
    for path_name in ('checkpoint', 'pipeline_template'):
        if not Path(config[path_name]).is_file():
            raise FileNotFoundError(config[path_name])
    actual = hashlib.md5(Path(config['checkpoint']).read_bytes()).hexdigest()
    if actual != config['checkpoint_md5']:
        raise ValueError(f'Official checkpoint MD5 differs: {actual}. Do not silently substitute weights.')
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def source_hashes():
    paths = [ROOT / 'tools/treelearn_residual_instance/run.py', ROOT / 'tools/pipeline/pipeline.py',
             ROOT / 'tree_learn/model/tree_learn.py', ROOT / 'tree_learn/model/blocks.py',
             ROOT / 'tree_learn/model/residual_instance.py', ROOT / 'tree_learn/util/train.py',
             ROOT / 'tree_learn/util/pipeline.py', ROOT / 'tree_learn/util/parser.py',
             ROOT / 'tree_learn/forest_abc/data.py', ROOT / 'tree_learn/query_segmentation/data.py',
             ROOT / 'tree_learn/mask_refinement/evaluation.py', ROOT / 'tree_learn/mask_refinement/data.py',
             ROOT / 'tree_learn/dataset/dataset.py']
    paths += sorted((ROOT / 'tree_learn/residual_instance').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths}


def cache_template(config, row):
    root = Path(config['external_cache_root'] if row['split'] == 'external' else config['cache_root'])
    path = root / 'resolved_configs' / f'{row["plot"]}.yaml'
    template = munch_to_dict(get_config(str(path)))
    tiles = Path(template['pipeline_base_dir']) / 'tiles'
    if not (tiles / 'npz').is_dir() or not list((tiles / 'npz').glob('*.npz')):
        raise FileNotFoundError(f'Prepared official tiles missing: {tiles / "npz"}')
    return template, tiles.resolve(), path


def request(config, rows):
    records = []
    for row in rows:
        template, tiles, path = cache_template(config, row)
        stamps = {}
        for name in ('coords.npy', 'labels.npy', 'cache.json') + (() if row['split'] == 'train' else ('baseline.npy',)):
            value = Path(row['cache']) / name
            stat = value.stat()
            stamps[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        tile_stamps = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                       for p in sorted((tiles / 'npz').glob('*.npz'))]
        records.append(dict(plot=row['plot'], split=row['split'], cache=str(Path(row['cache']).resolve()),
            arrays=stamps, resolved_config=file_digest(path), tiles=digest(tile_stamps)))
    return dict(config=digest(config), checkpoint_sha256=file_digest(config['checkpoint']),
                sources=source_hashes(), forests=records)


def model_config(config, arm):
    settings = munch_to_dict(get_config(config['pipeline_template']))['model']
    # Reassert the exact official single-anchor path, independent of modular defaults.
    settings.update(use_upper_anchor=False, use_axis_branch=False, use_height_context_adapter=False,
                    offset_loss_type='l2', fixed_modules=[], spatial_shape=[500, 500, 1000],
                    unet_enhancement={'enabled': False}, unet_finetune={'enabled': False},
                    residual_instance=dict(config['adapter'], enabled=arm == 'combined'))
    if (settings['channels'] != 32 or settings['num_blocks'] != 7 or settings['voxel_size'] != .1
            or settings['use_coords'] or settings['use_feats']):
        raise ValueError('This experiment requires the official 32-channel, seven-level, 0.1 m model.')
    return settings


def load_official(model, config):
    saved = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    state = saved['net']
    current = model.state_dict()
    mismatch = [name for name, value in state.items()
                if name not in current or current[name].shape != value.shape]
    if mismatch:
        raise ValueError(f'Official tensor mismatch: {mismatch}')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith('residual_instance.') for key in missing):
        raise ValueError(f'Incomplete official initialization: {missing}, {unexpected}')


def build_model(config, arm):
    model = TreeLearn(**model_config(config, arm)).cuda()
    load_official(model, config)
    return model


def decoder_parameter(name):
    return name.startswith(('unet.deconv.', 'unet.blocks_tail.',
                            'unet.u.deconv.', 'unet.u.blocks_tail.'))


def set_stage(model, arm, completed_steps, settings):
    branch_phase = completed_steps < settings['branch_steps']
    for name, parameter in model.named_parameters():
        new = name.startswith('residual_instance.')
        head = name.startswith(('semantic_linear.', 'offset_linear.'))
        parameter.requires_grad = (new or (head and (arm == 'control' or not branch_phase))
                                   or (decoder_parameter(name) and not branch_phase))
    model.train()
    # Freeze ALL BN running statistics in both arms, including trainable affine layers.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    return 'branch_only' if branch_phase and arm == 'combined' else (
           'head_warmup' if branch_phase else 'joint_last_two_decoder_levels')


def optimizer_for(model, settings):
    groups = {key: [] for key in ('adapter', 'head', 'decoder')}
    for name, parameter in model.named_parameters():
        if name.startswith('residual_instance.'):
            groups['adapter'].append(parameter)
        elif name.startswith(('semantic_linear.', 'offset_linear.')):
            groups['head'].append(parameter)
        elif decoder_parameter(name):
            groups['decoder'].append(parameter)
    # Include later-unfrozen parameters now; AdamW skips parameters with grad=None.
    return torch.optim.AdamW([dict(params=parameters, group_name=name,
        initial_rate=settings[name + '_lr'], lr=settings[name + '_lr'])
        for name, parameters in groups.items() if parameters], weight_decay=settings['weight_decay'])


def objective(model, batch, config, step):
    output = model(batch, return_loss=False)
    original, components = model.get_loss(model_output=output, **batch)
    auxiliary = original.new_zeros(())
    statistics = {}
    if model.residual_instance is not None:
        auxiliary, statistics = instance_auxiliary_loss(model.residual_instance,
            output['backbone_feats'], batch, config['auxiliary'], seed=config['seed'] + step)
    total = original + config['auxiliary']['weight'] * auxiliary
    values = dict(original=original, auxiliary=auxiliary, **components)
    values.update({f'instance_{name}': value for name, value in statistics.items()})
    values = {key: float(value.detach()) if torch.is_tensor(value) else value for key, value in values.items()}
    if not torch.isfinite(total) or not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(f'Non-finite training objective: {values}')
    return total, dict(loss=float(total.detach()), **values)


def save_checkpoint(path, model, optimizer, scaler, step, best, config, arm, experiment, **extra):
    payload = dict(schema=1, experiment='treelearn_residual_instance', arm=arm, step=step,
        net={key: value.detach().cpu() for key, value in model.state_dict().items()},
        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), best=best,
        config=config, request=experiment, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
        numpy_rng=np.random.get_state(), **extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pth')
    torch.save(payload, temporary)
    temporary.replace(path)


def require_checkpoint(saved, arm, experiment):
    if (saved.get('schema') != 1 or saved.get('experiment') != 'treelearn_residual_instance'
            or saved.get('arm') != arm or saved.get('request') != experiment
            or saved.get('check_only', False)):
        raise ValueError('Checkpoint arm, data, code, weights or experiment configuration changed.')
    if saved.get('pilot_failed', False):
        raise ValueError('This run failed the destructive-pilot guard; do not resume it.')


def run_live(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('RUN ' + ' '.join(str(part) for part in command), flush=True)
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=environment)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            if process.wait():
                raise RuntimeError(f'Pipeline failed ({process.returncode}); see {log}.')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def aligned_arrays(directory, cache):
    destination = Path(directory)
    coords = np.load(destination / 'coords.npy', mmap_mode='r', allow_pickle=False)
    original = np.load(Path(cache) / 'coords.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(Path(cache) / 'labels.npy', mmap_mode='r', allow_pickle=False)
    exported_gt = np.load(destination / 'gt_labels.npy', mmap_mode='r', allow_pickle=False)
    predictions = np.load(destination / 'predictions.npy', mmap_mode='r', allow_pickle=False)
    if coords.shape != original.shape or labels.shape != predictions.shape or labels.shape != exported_gt.shape:
        raise ValueError('New predictions do not cover the complete original cache.')
    for start in range(0, len(labels), 1_000_000):
        end = start + 1_000_000
        if not np.array_equal(coords[start:end], original[start:end]):
            raise ValueError('Point coordinates/order changed; never compare unaligned labels.')
        if not np.array_equal(labels[start:end], exported_gt[start:end]):
            raise ValueError('Evaluation GT differs from the reference cache.')
    return labels, predictions


def evaluate(config, arm, checkpoint, rows, output, experiment):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require_checkpoint(saved, arm, experiment)
    stamp = dict(arm=arm, checkpoint_sha256=file_digest(checkpoint), step=saved['step'],
                 experiment=digest(experiment), data_request=digest(request(config, rows)),
                 plots=[row['plot'] for row in rows])
    marker = output / 'request.json'
    if marker.exists() and json.loads(marker.read_text()) != stamp:
        raise ValueError(f'Output belongs to another evaluation: {output}')
    write_json(marker, stamp)
    records = []
    for row in rows:
        directory = output / row['plot']
        complete = directory / 'metrics.json'
        if complete.exists():
            records.append(json.loads(complete.read_text()))
            print(f'REUSE {arm}/{row["plot"]}: completed paired evaluation.', flush=True)
            continue
        template, tiles, _ = cache_template(config, row)
        base = (directory / 'runtime').resolve()
        base.mkdir(parents=True, exist_ok=True)
        link = base / 'tiles'
        if link.exists():
            if not link.is_symlink() or link.resolve() != tiles:
                raise ValueError(f'Refusing an unrelated tile directory: {link}')
        else:
            link.symlink_to(tiles, target_is_directory=True)
        template.update(model=model_config(config, arm), pretrain=str(Path(checkpoint).resolve()),
                        pipeline_base_dir=str(base), tile_generation=False)
        template['save_cfg'].update(save_mask_refinement_cache=False, save_pointwise=False,
            save_treewise=False, save_full_forest=False, save_quality_training_data=False,
            save_seed_training_data=False, prediction_export_dir=str(directory.resolve()),
            prediction_export_only=True)
        runtime_config = directory / 'pipeline.yaml'
        runtime_config.write_text(yaml.safe_dump(template, sort_keys=False), encoding='utf-8')
        if not (directory / 'prediction.json').exists():
            run_live([sys.executable, '-u', 'tools/pipeline/pipeline.py', '--config', str(runtime_config)],
                     directory / 'pipeline.log')
        labels, predictions = aligned_arrays(directory, row['cache'])
        refined = evaluate_labels(labels, predictions)
        baseline = evaluate_labels(labels, np.load(Path(row['cache']) / 'baseline.npy', mmap_mode='r'))
        record = dict(plot=row['plot'], baseline=baseline, refined=refined,
                      delta_pp=compare_metrics(baseline, refined))
        write_json(complete, record)
        records.append(record)
        print(row['plot'] + ': ' + json.dumps(record['delta_pp']), flush=True)
        # Only newly generated, redundant evaluation arrays are removed.
        del labels, predictions
        for filename in ('coords.npy', 'gt_labels.npy'):
            (directory / filename).unlink()
    baseline = macro_metrics([record['baseline'] for record in records])
    refined = macro_metrics([record['refined'] for record in records])
    result = dict(schema=1, protocol=PROTOCOL, arm=arm, checkpoint_step=saved['step'],
        baseline_macro=baseline, refined_macro=refined, delta_pp=compare_metrics(baseline, refined),
        totals={key: sum(record['refined'][key] for record in records) for key in ('tp', 'fp', 'fn')})
    write_json(output / 'summary.json', dict(result, forests=records, request=stamp))
    write_json(output / 'brief.json', result)
    print('SAVED ' + str(output / 'brief.json') + '\n' + json.dumps(result, indent=2), flush=True)
    return result


def check(config, rows, experiment):
    seed_everything(config['seed'])
    baseline, combined = build_model(config, 'control'), build_model(config, 'combined')
    baseline.eval()
    combined.eval()
    combined.residual_instance.collect_alignment = True
    identity = []
    cases = [next(row for row in rows if row['plot'] == config['training']['pilot_plot'])]
    try:
        cases += entries(config, external=True)[:1]
    except FileNotFoundError:
        print('External cache is optional at this stage; checking the source tile.', flush=True)
    for row in cases:
        _, tiles, _ = cache_template(config, row)

        def point_count(path):
            with zipfile.ZipFile(path) as archive, archive.open('points.npy') as stream:
                version = np.lib.format.read_magic(stream)
                return np.lib.format._read_array_header(stream, version)[0][0]

        path = max((tiles / 'npz').glob('*.npz'), key=point_count)
        print(f'CHECK full production tile: {path} points={point_count(path)}', flush=True)
        batch = tile_batch(path)
        for mixed_precision in (False, True):
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=mixed_precision):
                expected = baseline(batch, return_loss=False)
                actual = combined(batch, return_loss=False)
            errors = {}
            for name in ('semantic_prediction_logits', 'offset_predictions'):
                errors[name] = float((actual[name].float() - expected[name].float()).abs().max())
                if not torch.equal(actual[name], expected[name]):
                    raise RuntimeError(f'Initial model changes official {name}: {errors[name]}')
            if any(combined.residual_instance.alignment_missing.values()):
                raise RuntimeError('A production fine voxel lacks its expected coarse parent.')
            identity.append(dict(plot=row['plot'], tile=str(path), points=len(batch['coords']),
                mixed_precision=mixed_precision, max_errors=errors,
                missing_parents=dict(combined.residual_instance.alignment_missing)))
            del expected, actual
        del batch
    combined.residual_instance.collect_alignment = False
    del baseline
    torch.cuda.empty_cache()
    sampler = PairedCropSampler([row for row in rows if row['split'] == 'train'], config['training'], config['seed'])
    optimizer = optimizer_for(combined, config['training'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['training']['fp16'], init_scale=1024.)
    gradient_reports = []
    # First step opens gamma; second trains the fusion body. Then test the joint stage.
    for number in range(3):
        completed = 0 if number < 2 else config['training']['branch_steps']
        phase = set_stage(combined, 'combined', completed, config['training'])
        batch, info = sampler.draw(number + 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=config['training']['fp16']):
            loss, values = objective(combined, batch, config, number + 1)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(not torch.isfinite(parameter.grad).all() for parameter in combined.parameters()
               if parameter.grad is not None):
            raise FloatingPointError('Non-finite production gradients.')
        groups = {name: 0. for name in ('fusion', 'embedding', 'decoder')}
        for name, parameter in combined.named_parameters():
            if parameter.grad is None:
                continue
            group = ('embedding' if name.startswith('residual_instance.embedding.') else
                     'fusion' if name.startswith('residual_instance.') else
                     'decoder' if decoder_parameter(name) else None)
            if group:
                groups[group] += float(parameter.grad.abs().sum())
        if number == 1 and (groups['fusion'] <= 0 or groups['embedding'] <= 0):
            raise RuntimeError('Second branch update does not reach both fusion and embedding.')
        if number == 2 and groups['decoder'] <= 0:
            raise RuntimeError('Joint loss does not reach the last decoder levels.')
        torch.nn.utils.clip_grad_norm_(combined.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        gradient_reports.append(dict(phase=phase, gradients=groups, **info, **values))
        print('CHECK ' + json.dumps(gradient_reports[-1]), flush=True)
        del batch, loss
    # Verify the auxiliary loss itself reaches the shared predictive feature path.
    optimizer.zero_grad(set_to_none=True)
    batch, _ = sampler.draw(4)
    with torch.cuda.amp.autocast(enabled=config['training']['fp16']):
        output = combined(batch, return_loss=False)
        auxiliary, _ = instance_auxiliary_loss(combined.residual_instance, output['backbone_feats'],
                                               batch, config['auxiliary'], config['seed'] + 4)
    scaler.scale(auxiliary).backward()
    scaler.unscale_(optimizer)
    auxiliary_decoder_gradient = sum(float(p.grad.abs().sum()) for name, p in combined.named_parameters()
                                     if decoder_parameter(name) and p.grad is not None)
    if auxiliary_decoder_gradient <= 0:
        raise RuntimeError('Auxiliary head is disconnected from the original prediction feature path.')
    del batch, output, auxiliary
    optimizer.zero_grad(set_to_none=True)
    # A check artifact cannot subsequently be used as a training checkpoint.
    roundtrip_path = Path(config['work_dir']) / 'check_roundtrip.pth'
    save_checkpoint(roundtrip_path, combined, optimizer, scaler, 3, None, config,
                    'combined', experiment, check_only=True)
    saved = torch.load(roundtrip_path, map_location='cpu', weights_only=False)
    restored = build_model(config, 'combined')
    restored.load_state_dict(saved['net'], strict=True)
    restored_optimizer = optimizer_for(restored, config['training'])
    restored_optimizer.load_state_dict(saved['optimizer'])
    restored_scaler = torch.cuda.amp.GradScaler(enabled=config['training']['fp16'], init_scale=1024.)
    restored_scaler.load_state_dict(saved['scaler'])
    for parameter_id, state in optimizer.state_dict()['state'].items():
        other = restored_optimizer.state_dict()['state'][parameter_id]
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.equal(value.cpu(), other[name].cpu()):
                raise RuntimeError(f'Optimizer round trip changed {name}.')
    batch, _ = sampler.draw(5)
    combined.eval()
    restored.eval()
    with torch.no_grad():
        expected, actual = combined(batch, return_loss=False), restored(batch, return_loss=False)
    for name in ('semantic_prediction_logits', 'offset_predictions'):
        if not torch.equal(expected[name], actual[name]):
            raise RuntimeError(f'Checkpoint round trip changed {name}.')
    roundtrip_path.unlink()
    report = dict(request=experiment, identity=identity, gradients=gradient_reports,
                  auxiliary_decoder_gradient=auxiliary_decoder_gradient,
                  checkpoint_and_optimizer_roundtrip=True,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  added_parameters=sum(p.numel() for p in combined.residual_instance.parameters()))
    write_json(Path(config['work_dir']) / 'check.json', report)
    print('CHECK PASSED: production identity, branch updates and auxiliary-to-decoder gradient.', flush=True)


def train(config, rows, experiment, arm, resume):
    work = Path(config['work_dir'])
    check_record = json.loads((work / 'check.json').read_text())
    if check_record['request'] != experiment:
        raise ValueError('Production check is stale; rerun check after code/config/data changes.')
    seed_everything(config['seed'])
    model = build_model(config, arm)
    settings = config['training']
    optimizer = optimizer_for(model, settings)
    scaler = torch.cuda.amp.GradScaler(enabled=settings['fp16'], init_scale=1024.)
    directory = work / arm
    directory.mkdir(parents=True, exist_ok=True)
    step, best = 0, None
    last = directory / 'last.pth'
    if last.exists():
        if not resume:
            raise ValueError(f'Existing run: {last}; use resume to continue it.')
        saved = torch.load(last, map_location='cpu', weights_only=False)
        require_checkpoint(saved, arm, experiment)
        model.load_state_dict(saved['net'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        step, best = saved['step'], saved['best']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
    sampler = PairedCropSampler([row for row in rows if row['split'] == 'train'], settings, config['seed'])
    validation_rows = [row for row in rows if row['split'] == 'validation']
    started = time.monotonic()
    print(f'TRAIN arm={arm} step={step}/{settings["steps"]}; original BN statistics stay frozen.', flush=True)

    def validation(at_step):
        nonlocal best
        pilot = at_step == settings['pilot_step']
        selected = [row for row in validation_rows if row['plot'] == settings['pilot_plot']] if pilot else validation_rows
        checkpoint = directory / f'step_{at_step:06d}.pth'
        if checkpoint.exists():
            frozen = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require_checkpoint(frozen, arm, experiment)
            if frozen['step'] != at_step or any(not torch.equal(value.detach().cpu(), frozen['net'][name])
                    for name, value in model.state_dict().items()):
                raise ValueError('Immutable validation snapshot differs from resumed model.')
            del frozen
        else:
            save_checkpoint(checkpoint, model, optimizer, scaler, at_step, best, config, arm, experiment)
        model.cpu()
        torch.cuda.empty_cache()
        result = evaluate(config, arm, checkpoint, selected,
                          directory / ('pilot' if pilot else 'validation') / f'step_{at_step:06d}', experiment)
        model.cuda()
        if pilot and result['refined_macro']['f1'] < result['baseline_macro']['f1'] - settings['pilot_max_f1_drop']:
            save_checkpoint(last, model, optimizer, scaler, at_step, best, config, arm, experiment, pilot_failed=True)
            write_json(directory / 'pilot_failed.json', result)
            raise RuntimeError(f'{arm}: pilot F1 drop exceeds {100 * settings["pilot_max_f1_drop"]:.1f} pp; stop.')
        if not pilot and (best is None or result['refined_macro']['f1'] > best['f1']):
            best = dict(step=at_step, f1=result['refined_macro']['f1'])
            save_checkpoint(directory / 'best.pth', model, optimizer, scaler, at_step, best, config, arm, experiment)
        save_checkpoint(last, model, optimizer, scaler, at_step, best, config, arm, experiment)

    # A pre-validation checkpoint can be resumed after an interrupted pipeline.
    due = set(settings['full_validation_steps']) | {settings['pilot_step']}
    if step in due:
        validation(step)
    with (directory / 'training.jsonl').open('a' if step else 'w', encoding='utf-8') as log:
        while step < settings['steps']:
            phase = set_stage(model, arm, step, settings)
            data_started = time.monotonic()
            batch, info = sampler.draw(step + 1)
            data_seconds = time.monotonic() - data_started
            optimizer.zero_grad(set_to_none=True)
            multiplier = min(1., (step + 1) / settings['warmup_steps']) * (
                .1 + .9 * .5 * (1 + math.cos(math.pi * step / settings['steps'])))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_rate'] * multiplier
            step_started = time.monotonic()
            with torch.cuda.amp.autocast(enabled=settings['fp16']):
                loss, values = objective(model, batch, config, step + 1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'])
            if not torch.isfinite(gradient):
                raise FloatingPointError('Non-finite gradient; checkpoint not advanced.')
            scaler.step(optimizer)
            scaler.update()
            step += 1
            record = dict(step=step, arm=arm, phase=phase, **info, **values,
                gradient_norm=float(gradient), data_seconds=data_seconds,
                step_seconds=time.monotonic() - step_started, elapsed_seconds=time.monotonic() - started,
                gamma=float(model.residual_instance.gamma.tanh()) if model.residual_instance is not None else None,
                learning_rates={group['group_name']: group['lr'] for group in optimizer.param_groups},
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(record) + '\n')
            log.flush()
            if step == 1 or step % settings['log_every'] == 0:
                print(json.dumps(record), flush=True)
            del batch, loss
            if step % settings['save_every'] == 0 or step in due:
                save_checkpoint(last, model, optimizer, scaler, step, best, config, arm, experiment)
            if step in due:
                validation(step)
    write_json(directory / 'train_summary.json', dict(arm=arm, step=step, best=best,
        checkpoint=str(directory / 'best.pth'), status='training_completed_inspect_paired_metrics'))


def compare(config, external=False):
    work = Path(config['work_dir'])
    records = {}
    for arm in ARMS:
        summary = json.loads((work / arm / 'train_summary.json').read_text())
        step = summary['best']['step']
        location = work / arm / 'external_evaluation' if external else (
                   work / arm / 'validation' / f'step_{step:06d}')
        records[arm] = json.loads((location / 'brief.json').read_text())
        if records[arm]['checkpoint_step'] != step:
            raise ValueError('Comparison is not using the source-selected checkpoint.')
    if records['control']['baseline_macro'] != records['combined']['baseline_macro']:
        raise ValueError('Paired arms do not share the same official reference.')
    matched_steps = {}
    for step in ([] if external else config['training']['full_validation_steps']):
        paired = [json.loads((work / arm / 'validation' / f'step_{step:06d}' / 'brief.json').read_text())
                  for arm in ARMS]
        matched_steps[str(step)] = compare_metrics(paired[0]['refined_macro'], paired[1]['refined_macro'])
    result = dict(protocol=PROTOCOL, arms=records, same_step_combined_minus_control_pp=matched_steps,
        combined_minus_control_pp=compare_metrics(records['control']['refined_macro'], records['combined']['refined_macro']))
    write_json(work / ('external_comparison.json' if external else 'comparison.json'), result)
    print('PAIRED COMPARISON\n' + json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor', 'prepare', 'check', 'train', 'evaluate', 'external', 'compare', 'compare-external'))
    parser.add_argument('--config', default='configs/experiments/treelearn_residual_instance/full.yaml')
    parser.add_argument('--arm', choices=ARMS, default='combined')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = configuration(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError('Run this model workflow in the server TreeLearn CUDA environment.')
    torch.cuda.set_device(torch.cuda.current_device())
    seed_everything(config['seed'])
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    if args.command in ('compare', 'compare-external'):
        compare(config, external=args.command == 'compare-external')
        return
    rows = entries(config)
    experiment = request(config, rows)
    if args.command == 'doctor':
        print(json.dumps(dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(torch.cuda.current_device()),
            free_gpu_bytes=torch.cuda.mem_get_info(torch.cuda.current_device())[0],
            official_checkpoint=config['checkpoint'], checkpoint_md5=config['checkpoint_md5'],
            train_forests=[row['plot'] for row in rows if row['split'] == 'train'],
            validation_forests=[row['plot'] for row in rows if row['split'] == 'validation'],
            inputs='coordinates only; official constant feature inputs; original 0.1 m voxelization',
            paired_steps=config['training']['steps']), indent=2), flush=True)
    elif args.command == 'prepare':
        prepare_rows(config, rows)
        path = work / 'experiment.json'
        if path.exists() and json.loads(path.read_text()) != experiment:
            raise ValueError('Existing experiment changed; choose a fresh work_dir.')
        write_json(path, experiment)
        print('PREPARED: coordinate indices; official tiles reused, no frozen model features loaded.', flush=True)
    elif args.command == 'check':
        check(config, rows, experiment)
    elif args.command == 'train':
        train(config, rows, experiment, args.arm, args.resume)
    else:
        checkpoint = work / args.arm / 'best.pth'
        if args.command == 'external':
            selected = entries(config, external=True)
            output = work / args.arm / 'external_evaluation'
        else:
            selected = [row for row in rows if row['split'] == 'validation']
            output = work / args.arm / 'evaluation'
        evaluate(config, args.arm, checkpoint, selected, output, experiment)


if __name__ == '__main__':
    main()
