"""Foreground workflow for a frozen official model plus structural XY voting."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from tree_learn.forest_abc.data import entries, prepare_rows
from tree_learn.mask_refinement.data import digest, file_digest, write_json
from tree_learn.mask_refinement.evaluation import evaluate_labels, macro_metrics, compare_metrics
from tree_learn.model import TreeLearn
from tree_learn.residual_instance.data import PairedCropSampler, tile_batch
from tree_learn.residual_instance.optimization import optimizer_update
from tree_learn.structure_voting.loss import sample_queries, structure_loss
from tree_learn.util import get_config, munch_to_dict
from tools.treelearn_residual_instance.run import (
    aligned_arrays, cache_template, numerical_logger, request as shared_request,
    run_live, seed_everything)

DEFAULT = 'configs/experiments/structure_voting/full.yaml'
PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'


def configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    training, branch, supervision = (config[key] for key in ('training', 'branch', 'supervision'))
    for group, fields in ((training, ('steps', 'pilot_step', 'warmup_steps', 'save_every', 'log_every',
                            'max_open_forests', 'max_points', 'minimum_points')),
                          (branch, ('hidden_dim', 'neighbors', 'query_chunk_size')),
                          (supervision, ('max_trees', 'points_per_tree'))):
        for name in fields:
            if type(group[name]) is not int or group[name] < 1:
                raise ValueError(f'{name} must be a positive integer.')
    for group, fields in ((training, ('pilot_max_f1_drop', 'branch_lr', 'weight_decay',
                            'gradient_clip', 'outer_half_extent', 'inner_half_extent')),
                          (branch, ('support_voxel_size', 'radius', 'max_delta_xy', 'vote_scale')),
                          (supervision, ('vote_weight', 'connection_weight', 'consistency_weight',
                                         'regularization_weight'))):
        for name in fields:
            if not math.isfinite(group[name]) or group[name] <= 0:
                raise ValueError(f'{name} must be finite and positive.')
    due = training['full_validation_steps']
    if (not due or due != sorted(set(due)) or due[-1] != training['steps']
            or any(type(step) is not int or step <= training['pilot_step'] for step in due)
            or not training['warmup_steps'] < training['pilot_step']):
        raise ValueError('Invalid warmup, pilot or validation schedule.')
    if (not 0 < branch['tree_conf_thresh'] < 1 or training['pilot_max_f1_drop'] > 1
            or training['inner_half_extent'] >= training['outer_half_extent']
            or training['minimum_points'] > training['max_points']
            or type(config['seed']) is not int or config['seed'] < 0):
        raise ValueError('Invalid thresholds, crop geometry or seed.')
    if training['fp16'] is not False or training['fp16_retries'] != 0:
        raise ValueError('This version preserves the production FP32 official path.')
    if set(config['splits']['train']) & set(config['splits']['validation']):
        raise ValueError('Training and validation overlap.')
    if training['pilot_plot'] not in config['splits']['validation']:
        raise ValueError('Pilot plot is not in validation.')
    if hashlib.md5(Path(config['checkpoint']).read_bytes()).hexdigest() != config['checkpoint_md5']:
        raise ValueError('Official checkpoint MD5 changed.')
    return config


def request(config, rows):
    value = shared_request(config, rows)
    paths = sorted((ROOT / 'tree_learn/structure_voting').glob('*.py'))
    paths += [Path(__file__).resolve(), ROOT / 'tools/structure_voting/run.sh']
    value['sources'].update({path.relative_to(ROOT).as_posix(): file_digest(path) for path in paths})
    return value


def model_config(config, enabled=True):
    result = munch_to_dict(get_config(config['pipeline_template']))['model']
    result.update(use_upper_anchor=False, use_axis_branch=False, use_height_context_adapter=False,
        offset_loss_type='l2', fixed_modules=[], spatial_shape=[500, 500, 1000],
        unet_enhancement={'enabled': False}, unet_finetune={'enabled': False},
        residual_instance={'enabled': False}, structure_vote=dict(config['branch'], enabled=enabled))
    if (result['channels'] != 32 or result['num_blocks'] != 7 or result['voxel_size'] != .1
            or result['use_coords'] or result['use_feats']):
        raise ValueError('Require the official 32-channel, seven-level, 0.1 m constant-input model.')
    if config['branch']['tree_conf_thresh'] != .5:
        raise ValueError('Support foreground must use the official 0.5 semantic threshold.')
    return result


def build_model(config, enabled=True):
    model = TreeLearn(**model_config(config, enabled)).cuda()
    official = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)['net']
    missing, unexpected = model.load_state_dict(official, strict=False)
    if unexpected or any(not name.startswith('structure_vote.') for name in missing):
        raise ValueError(f'Official initialization mismatch: {missing}, {unexpected}')
    model._official_state = official
    # No original parameter or BatchNorm buffer is trainable, at any stage.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = enabled and name.startswith('structure_vote.')
    model.eval()
    return model


def assert_frozen(model, state=None):
    state = state if state is not None else {name: value.detach().cpu() for name, value in model.state_dict().items()}
    for name, original in model._official_state.items():
        if not torch.equal(state[name], original):
            raise RuntimeError(f'Official parameter or buffer changed: {name}')
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if not name.startswith('structure_vote.')):
        raise RuntimeError('An official parameter became trainable.')


def optimizer_for(model, config):
    settings = config['training']
    return torch.optim.AdamW(model.structure_vote.parameters(), lr=settings['branch_lr'],
                            weight_decay=settings['weight_decay'])


def objective(model, batch, config, step):
    batch = dict(batch, structure_query_indices=sample_queries(batch, config['supervision'],
                                                               config['seed'] + step))
    output = model(batch, return_loss=False)
    base = output['structure_base_offsets']
    if not torch.equal(output['offset_predictions'][:, 2], base[:, 2]):
        raise RuntimeError('Structure voting changed Z offsets.')
    return structure_loss(output['structure_vote'], batch, base, config['supervision'])


def save_checkpoint(path, model, optimizer, step, best, config, experiment, **extra):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    assert_frozen(model, state)
    value = dict(schema=1, experiment='treelearn_structure_voting_v1', step=step, best=best,
        net=state, optimizer=optimizer.state_dict(), config=config, request=experiment,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), **extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pth')
    torch.save(value, temporary)
    temporary.replace(path)


def require_checkpoint(saved, config, experiment):
    if (saved.get('schema') != 1 or saved.get('experiment') != 'treelearn_structure_voting_v1'
            or saved.get('request') != experiment or saved.get('config') != config
            or saved.get('check_only', False) or saved.get('pilot_failed', False)):
        raise ValueError('Checkpoint inputs changed, or it is a check/failed-pilot checkpoint.')


def foreground_identity(predictions, reference):
    changed, foreground = 0, 0
    for start in range(0, len(predictions), 1_000_000):
        current = predictions[start:start + 1_000_000] > 0
        official = reference[start:start + 1_000_000] > 0
        changed += int(np.count_nonzero(current != official))
        foreground += int(official.sum())
    if changed:
        raise RuntimeError(f'{changed} final foreground points differ from the official cache; inspect pipeline identity.')
    return dict(changed_foreground_points=changed, official_foreground_points=foreground)


def evaluate(config, checkpoint, rows, output, experiment):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require_checkpoint(saved, config, experiment)
    stamp = dict(step=saved['step'], checkpoint_sha256=file_digest(checkpoint),
                 experiment=digest(experiment), data_request=digest(request(config, rows)),
                 plots=[row['plot'] for row in rows])
    marker = output / 'request.json'
    if marker.exists() and json.loads(marker.read_text()) != stamp:
        raise ValueError(f'Evaluation directory belongs to other inputs: {output}')
    write_json(marker, stamp)
    del saved
    records = []
    for row in rows:
        directory = output / row['plot']
        complete = directory / 'metrics.json'
        if complete.exists():
            records.append(json.loads(complete.read_text()))
            print(f'REUSE {row["plot"]}: complete evaluation.', flush=True)
            continue
        template, tiles, _ = cache_template(config, row)
        base = (directory / 'runtime').resolve()
        base.mkdir(parents=True, exist_ok=True)
        link = base / 'tiles'
        if link.exists():
            if not link.is_symlink() or link.resolve() != tiles:
                raise ValueError(f'Unrelated tile directory: {link}')
        else:
            link.symlink_to(tiles, target_is_directory=True)
        template.update(model=model_config(config), pretrain=str(Path(checkpoint).resolve()),
                        pipeline_base_dir=str(base), tile_generation=False)
        template['save_cfg'].update(save_mask_refinement_cache=False, save_pointwise=False,
            save_treewise=False, save_full_forest=False, save_quality_training_data=False,
            save_seed_training_data=False, prediction_export_dir=str(directory.resolve()),
            prediction_export_only=True)
        runtime = directory / 'pipeline.yaml'
        runtime.write_text(yaml.safe_dump(template, sort_keys=False), encoding='utf-8')
        if not (directory / 'prediction.json').exists():
            run_live([sys.executable, '-u', 'tools/pipeline/pipeline.py', '--config', str(runtime)],
                     directory / 'pipeline.log')
        labels, predictions = aligned_arrays(directory, row['cache'])
        reference = np.load(Path(row['cache']) / 'baseline.npy', mmap_mode='r')
        identity = foreground_identity(predictions, reference)
        write_json(directory / 'foreground_identity.json', identity)
        refined, baseline = evaluate_labels(labels, predictions), evaluate_labels(labels, reference)
        record = dict(plot=row['plot'], baseline=baseline, refined=refined,
                      delta_pp=compare_metrics(baseline, refined), foreground_identity=identity)
        write_json(complete, record)
        records.append(record)
        print(row['plot'] + ': ' + json.dumps(record['delta_pp']), flush=True)
        del labels, predictions, reference
        for filename in ('coords.npy', 'gt_labels.npy'):
            (directory / filename).unlink()
    baseline = macro_metrics([record['baseline'] for record in records])
    refined = macro_metrics([record['refined'] for record in records])
    result = dict(schema=1, protocol=PROTOCOL, checkpoint_step=stamp['step'], baseline_macro=baseline,
        refined_macro=refined, delta_pp=compare_metrics(baseline, refined),
        totals={key: sum(record['refined'][key] for record in records) for key in ('tp', 'fp', 'fn')})
    write_json(output / 'summary.json', dict(result, forests=records, request=stamp))
    write_json(output / 'brief.json', result)
    print('SAVED ' + str(output / 'brief.json') + '\n' + json.dumps(result, indent=2), flush=True)
    return result


def largest_tile(config, row):
    _, tiles, _ = cache_template(config, row)

    def count(path):
        with zipfile.ZipFile(path) as archive, archive.open('points.npy') as stream:
            version = np.lib.format.read_magic(stream)
            return np.lib.format._read_array_header(stream, version)[0][0]

    return max((tiles / 'npz').glob('*.npz'), key=count)


def check(config, rows, experiment):
    seed_everything(config['seed'])
    baseline, model = build_model(config, False), build_model(config)
    cases = [next(row for row in rows if row['plot'] == config['training']['pilot_plot'])]
    try:
        cases += entries(config, external=True)[:1]
    except FileNotFoundError:
        print('No external cache: source production tile will be checked.', flush=True)
    identity = []
    for row in cases:
        path = largest_tile(config, row)
        batch = tile_batch(path)
        print(f'CHECK production identity: {row["plot"]}, points={len(batch["coords"]):,}', flush=True)
        with torch.no_grad():
            expected, actual = baseline(batch, return_loss=False), model(batch, return_loss=False)
        for key in ('semantic_prediction_logits', 'offset_predictions'):
            if not torch.equal(expected[key], actual[key]):
                raise RuntimeError(f'Initial branch changed official {key}.')
        identity.append(dict(plot=row['plot'], tile=str(path), points=len(batch['coords']), exact_identity=True))
        del batch, expected, actual
    sampler = PairedCropSampler([row for row in rows if row['split'] == 'train'], config['training'], config['seed'])
    optimizer = optimizer_for(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    updates = []
    for step in range(1, 5):
        model.train()  # Fixed official BN buffers must remain in eval here too.
        batch, info = sampler.draw(step)
        values, update = optimizer_update(model, optimizer, scaler,
            lambda: objective(model, batch, config, step), config['training']['gradient_clip'], fp16_retries=0)
        assert_frozen(model)
        updates.append(dict(step=step, **info, **values, **update))
        print('CHECK_UPDATE ' + json.dumps(updates[-1]), flush=True)
    model.eval()
    with torch.no_grad():
        expected, actual = baseline(batch, return_loss=False), model(batch, return_loss=False)
    if (not torch.equal(expected['semantic_prediction_logits'], actual['semantic_prediction_logits'])
            or not torch.equal(expected['offset_predictions'][:, 2], actual['offset_predictions'][:, 2])):
        raise RuntimeError('Trained branch changed official semantics or Z.')
    if torch.equal(expected['offset_predictions'][:, :2], actual['offset_predictions'][:, :2]):
        raise RuntimeError('XY branch did not update.')
    del expected, actual, baseline
    # A vote-only backward must reach connection parameters after the zero head opens.
    optimizer.zero_grad(set_to_none=True)
    selected = sample_queries(batch, config['supervision'], config['seed'] + 4)
    output = model(dict(batch, structure_query_indices=selected), return_loss=False)
    rows_device = output['structure_vote']['rows']
    target = batch['offset_labels'].to(rows_device.device)[rows_device, :2]
    torch.nn.functional.smooth_l1_loss(output['offset_predictions'][rows_device, :2], target).backward()
    connection_gradient = sum(float(parameter.grad.abs().sum()) for parameter in model.structure_vote.connection.parameters()
                              if parameter.grad is not None)
    if not connection_gradient > 0:
        raise RuntimeError('Vote correction has no gradient through learned connections.')
    optimizer.zero_grad(set_to_none=True)
    del output
    temporary = Path(config['work_dir']) / 'check_roundtrip.pth'
    save_checkpoint(temporary, model, optimizer, 4, None, config, experiment, check_only=True)
    restored = build_model(config)
    saved = torch.load(temporary, map_location='cpu', weights_only=False)
    restored.load_state_dict(saved['net'], strict=True)
    restored_optimizer = optimizer_for(restored, config)
    restored_optimizer.load_state_dict(saved['optimizer'])
    with torch.no_grad():
        expected, actual = model(batch, return_loss=False), restored(batch, return_loss=False)
    if any(not torch.equal(expected[key], actual[key]) for key in ('semantic_prediction_logits', 'offset_predictions')):
        raise RuntimeError('Checkpoint round trip changed predictions.')
    temporary.unlink()
    write_json(Path(config['work_dir']) / 'check.json', dict(request=experiment, identity=identity,
        updates=updates, official_parameters_and_buffers_unchanged=True,
        trained_semantics_and_z_unchanged=True, connection_vote_gradient_l1=connection_gradient,
        checkpoint_roundtrip=True, added_parameters=sum(p.numel() for p in model.structure_vote.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print('CHECK PASSED: dense-tile identity, frozen official state, live connection-to-vote gradient and checkpoint round trip.', flush=True)


def train(config, rows, experiment, resume):
    work, settings = Path(config['work_dir']), config['training']
    if json.loads((work / 'check.json').read_text())['request'] != experiment:
        raise ValueError('Production check is stale; run check again.')
    seed_everything(config['seed'])
    model = build_model(config)
    optimizer = optimizer_for(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    step, best = 0, None
    last = work / 'last.pth'
    if last.exists():
        if not resume:
            raise ValueError(f'Existing run: {last}; use resume.')
        saved = torch.load(last, map_location='cpu', weights_only=False)
        require_checkpoint(saved, config, experiment)
        model.load_state_dict(saved['net'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        step, best = saved['step'], saved['best']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        del saved
    assert_frozen(model)
    sampler = PairedCropSampler([row for row in rows if row['split'] == 'train'], settings, config['seed'])
    validation_rows = [row for row in rows if row['split'] == 'validation']
    due = set(settings['full_validation_steps']) | {settings['pilot_step']}
    started = time.monotonic()

    def validate():
        nonlocal best
        pilot = step == settings['pilot_step']
        selected = [row for row in validation_rows if row['plot'] == settings['pilot_plot']] if pilot else validation_rows
        snapshot = work / f'step_{step:06d}.pth'
        if snapshot.exists():
            previous = torch.load(snapshot, map_location='cpu', weights_only=False)
            require_checkpoint(previous, config, experiment)
            if previous['step'] != step or any(not torch.equal(value.detach().cpu(), previous['net'][name])
                                             for name, value in model.state_dict().items()):
                raise ValueError('Existing validation snapshot differs from the resumed model.')
            del previous
        else:
            save_checkpoint(snapshot, model, optimizer, step, best, config, experiment)
        # The frozen 120 MB model stays on device so optimizer parameter
        # identities and Adam moments cannot be disturbed by a CPU/GPU move.
        torch.cuda.empty_cache()
        result = evaluate(config, snapshot, selected,
            work / ('pilot' if pilot else 'validation') / f'step_{step:06d}', experiment)
        if pilot and result['delta_pp']['f1_delta_pp'] < -100 * settings['pilot_max_f1_drop']:
            save_checkpoint(last, model, optimizer, step, best, config, experiment, pilot_failed=True)
            write_json(work / 'pilot_failed.json', result)
            raise RuntimeError('Pilot F1 drop exceeded the failure floor. Inspect metrics before a new experiment.')
        if not pilot and (best is None or result['refined_macro']['f1'] > best['f1']):
            best = dict(step=step, f1=result['refined_macro']['f1'])
            save_checkpoint(work / 'best.pth', model, optimizer, step, best, config, experiment)
        save_checkpoint(last, model, optimizer, step, best, config, experiment)

    if step in due:
        validate()
    print(f'TRAIN structure branch only: step={step}/{settings["steps"]}; official model fully frozen.', flush=True)
    with (work / 'training.jsonl').open('a' if step else 'w', encoding='utf-8') as log:
        while step < settings['steps']:
            model.train()
            data_start = time.monotonic()
            batch, info = sampler.draw(step + 1)
            data_seconds = time.monotonic() - data_start
            multiplier = min(1., (step + 1) / settings['warmup_steps']) * (
                .1 + .9 * .5 * (1 + math.cos(math.pi * step / settings['steps'])))
            for group in optimizer.param_groups:
                group['lr'] = settings['branch_lr'] * multiplier
            update_start = time.monotonic()
            values, update = optimizer_update(model, optimizer, scaler,
                lambda: objective(model, batch, config, step + 1), settings['gradient_clip'], fp16_retries=0,
                on_event=numerical_logger(work / 'numerical_events.jsonl', step=step + 1, **info))
            step += 1
            record = dict(step=step, total_steps=settings['steps'], **info, **values, **update,
                learning_rate=optimizer.param_groups[0]['lr'], data_seconds=data_seconds,
                step_seconds=time.monotonic() - update_start, elapsed_seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(record) + '\n')
            log.flush()
            if step == 1 or step % settings['log_every'] == 0:
                print(json.dumps(record), flush=True)
            del batch
            if step % settings['save_every'] == 0 or step in due:
                save_checkpoint(last, model, optimizer, step, best, config, experiment)
            if step in due:
                validate()
    write_json(work / 'train_summary.json', dict(step=step, best=best, checkpoint=str(work / 'best.pth'),
                                               status='completed_inspect_metrics_for_effectiveness'))
    print('TRAIN DONE ' + json.dumps(dict(step=step, best=best)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor', 'prepare', 'check', 'train', 'evaluate', 'external'))
    parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = configuration(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError('Run model commands on the Linux TreeLearn CUDA server.')
    torch.cuda.set_device(torch.cuda.current_device())
    seed_everything(config['seed'])
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    rows = entries(config)
    experiment = request(config, rows)
    if args.command == 'doctor':
        print(json.dumps(dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(torch.cuda.current_device()),
            free_gpu_bytes=torch.cuda.mem_get_info(torch.cuda.current_device())[0],
            official_checkpoint=config['checkpoint'], checkpoint_md5=config['checkpoint_md5'],
            steps=config['training']['steps'], precision='FP32',
            trainable='local connections and XY corrections only',
            preserved='official features, semantic logits, Z offsets, foreground and seed eligibility'), indent=2), flush=True)
    elif args.command == 'prepare':
        prepare_rows(config, rows)
        marker = work / 'experiment.json'
        if marker.exists() and json.loads(marker.read_text()) != experiment:
            raise ValueError('Experiment changed; choose a fresh work_dir.')
        write_json(marker, experiment)
        print('PREPARED: coordinate indices; reuse official tiles. No new labels or features required.', flush=True)
    elif args.command == 'check':
        check(config, rows, experiment)
    elif args.command == 'train':
        train(config, rows, experiment, args.resume)
    else:
        selected = entries(config, external=True) if args.command == 'external' else [
            row for row in rows if row['split'] == 'validation']
        evaluate(config, work / 'best.pth', selected,
                 work / ('external_evaluation' if args.command == 'external' else 'evaluation'), experiment)


if __name__ == '__main__':
    main()
