"""Foreground V3: aligned outcomes, real hard negatives and ordered height tracks."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from tools.forest_abc.diagnose import analyze_table, compare_models, write_csv
from tree_learn.instance_relation.supervision import merge_table, load_supervision
from tree_learn.instance_relation_v3 import data, supervision, augmentation, policy, calibration
from tree_learn.mask_refinement.data import digest, file_digest, write_json
from tree_learn.mask_refinement.evaluation import compare_metrics, macro_metrics

PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config_read(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    inputs = [Path(config[key]).resolve() for key in ('cache_root', 'external_cache_root', 'base_prepared_root', 'point_prepared_root')]
    outputs = [Path(config[key]).resolve() for key in ('work_dir', 'prepared_root')]
    if any(a == b or a in b.parents or b in a.parents for a in inputs for b in outputs):
        raise ValueError('Use separate V3 outputs; official/V1/V2 inputs are immutable.')
    if outputs[0] == outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:
        raise ValueError('Separate work and prepared directories are required.')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32 or config['chunk_points'] <= 0:
        raise ValueError('Invalid seed or chunk size.')
    if config['geometry']['height_strata'] != 4 or config['geometry']['points_per_stratum'] < 2:
        raise ValueError('Reuse the original four-stratum V2 observations.')
    settings = config['training']
    for key in ('steps', 'batch_size', 'validation_every', 'save_every', 'log_every', 'hard_mining_every', 'hard_pool_size'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError('Invalid training setting: ' + key)
    real = settings['batch_size'] * settings['real_fraction']
    if not .5 < settings['real_fraction'] < 1 or real != int(real) or min(real, settings['batch_size'] - real) < 3:
        raise ValueError('Require a real majority and at least three real/synthetic samples per batch.')
    for key in ('learning_rate', 'weight_decay', 'grad_clip'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError('Invalid training setting: ' + key)
    for key in ('hidden_dim', 'geometric_neighbors', 'vote_neighbors', 'height_bands', 'minimum_layer_points'):
        if type(config['model'][key]) is not int or config['model'][key] < 1:
            raise ValueError('Invalid model setting: ' + key)
    if config['model']['height_bands'] < 3:
        raise ValueError('At least three ordered height bands are required.')
    aug = config['augmentation']
    if aug['bank_points'] % 4 or aug['bank_points'] < 8 or aug['min_bank_points'] < 4:
        raise ValueError('Invalid synthetic point bank dimensions.')
    if not 0 < aug['min_foreground_coverage'] <= 1 or not 0 < aug['density_min'] <= 1:
        raise ValueError('Invalid augmentation fractions.')
    if config['policy']['destruction_cost'] != config['objective']['destruction_cost'] or config['policy']['destruction_cost'] <= 0:
        raise ValueError('Training ranking and deployment must use the same positive destruction cost.')
    if any(not math.isfinite(v) or v < 0 for v in config['objective'].values()):
        raise ValueError('Invalid objective weights.')
    if not config['policy']['thresholds'] or not all(math.isfinite(x) and 0 <= x <= 1 for x in config['policy']['thresholds']):
        raise ValueError('Recovery-score thresholds must lie in [0, 1].')
    return config


def signature(config):
    sources = list((ROOT / 'tree_learn/instance_relation_v3').glob('*.py')) + [Path(__file__).resolve(),
        ROOT / 'tools/instance_relation_v3/run.sh', ROOT / 'tools/forest_abc/diagnose.py',
        *[ROOT / 'tree_learn/instance_relation_v2' / name for name in ('data.py', 'supervision.py', 'augmentation.py', 'model.py')],
        *[ROOT / 'tree_learn/instance_relation' / name for name in ('data.py', 'supervision.py', 'policy.py')],
        ROOT / 'tree_learn/mask_refinement/data.py', ROOT / 'tree_learn/mask_refinement/evaluation.py']
    return digest(dict(config=config, code={p.relative_to(ROOT).as_posix(): file_digest(p) for p in sources}))


def prepare(config):
    for row in data.rows(config):
        print(f'PREPARE {row["split"]}/{row["plot"]}', flush=True)
        values = data.prepare(row, config)
        targets = supervision.prepare(row, values, config)
        if row['split'] == 'train':
            bank = augmentation.prepare_bank(row, values, targets, config)
            print(f'BANK {row["plot"]}: trees={len(bank["ids"])}', flush=True)
        metadata = read_json(Path(row['prepared']) / 'targets.json')
        print('TARGETS ' + row['plot'] + ': ' + json.dumps(dict(action_counts=metadata['action_counts'],
            newly_recovery_labeled=metadata['newly_recovery_labeled'])), flush=True)
    print('PREPARED: reused frozen samples; exact recovery labels without old purity/gain gates.', flush=True)


def pool(config):
    arrays = {key: [] for key in ('points', 'valid', 'weights', 'point_labels', 'pairs', 'action', 'risk', 'utility', 'quality')}
    groups, hard, banks, node_offset, pair_offset = [], [], [], 0, 0
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        values, targets = data.load(row, config), supervision.load(row, config)
        for key in ('points', 'valid', 'weights'):
            arrays[key].append(values[key])
        arrays['point_labels'].append(targets['point_labels'])
        arrays['pairs'].append(values['pairs'] + node_offset)
        for key in ('action', 'risk', 'utility', 'quality'):
            arrays[key].append(targets[key])
        local = [np.flatnonzero(targets['valid'] & (targets['action'] == category)) for category in range(3)]
        groups.append([group + pair_offset for group in local])
        a, b = values['pairs'].T
        intact = targets['original_tp'][a] & targets['original_tp'][b]
        pools = []
        for category in range(3):
            ordered = sorted(local[category].tolist(), key=lambda i: (not bool(intact[i]), float(values['proximity'][i]), i))
            pools.append(np.asarray(ordered[:config['training']['hard_pool_size']], dtype=np.int64) + pair_offset)
        hard.append(pools)
        banks.append(augmentation.prepare_bank(row, values, targets, config))
        node_offset += len(values['ids'])
        pair_offset += len(values['pairs'])
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result.update(groups=groups, hard=hard, banks=banks)
    for category, name in enumerate(supervision.NAMES):
        if not any(len(group[category]) for group in groups):
            raise ValueError('No real ' + name + ' outcomes; inspect source target counts before training.')
    return result


def sample_real(values, desired, rng):
    available = [i for i, group in enumerate(values['groups']) if len(group[desired])]
    if not available:
        raise ValueError('Missing requested real outcome class.')
    forest = int(rng.choice(available))
    choices = values['groups'][forest][desired]
    if desired != 1 and len(values['hard'][forest][desired]) and rng.random() < .75:
        choices = values['hard'][forest][desired]
    return int(rng.choice(choices))


def training_batch(values, config, rng):
    real_count = int(config['training']['batch_size'] * config['training']['real_fraction'])
    examples, real_counts, synthetic_counts = [], [0, 0, 0], [0, 0, 0]
    for i in range(real_count):
        edge = sample_real(values, (1, 2, 0)[i % 3], rng)
        pair = values['pairs'][edge]
        points, valid, weights = augmentation.augment_real(values['points'][pair], values['valid'][pair],
            values['weights'][pair], rng, config['augmentation']['density_min'])
        example = dict(points=points, valid=valid, weights=weights, point_labels=values['point_labels'][pair], is_real=True)
        example.update({key: values[key][edge] for key in ('action', 'risk', 'utility', 'quality')})
        examples.append(example)
        real_counts[int(example['action'])] += 1
    for i in range(config['training']['batch_size'] - real_count):
        bank = values['banks'][int(rng.integers(len(values['banks'])))]
        example = augmentation.synthetic(bank, rng, values['points'].shape[1], config['augmentation'], (1, 2, 0)[i % 3])
        example['is_real'] = False
        examples.append(example)
        synthetic_counts[int(example['action'])] += 1
    rng.shuffle(examples)
    return {key: np.stack([example[key] for example in examples]) for key in examples[0]}, real_counts, synthetic_counts


def make_model(config):
    import torch
    from tree_learn.instance_relation_v3.model import StructureRelationHead
    device = torch.device(config['training']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; change device only if a CPU run is intended.')
    return StructureRelationHead(**config['model']).to(device=device, dtype=torch.float32), device


def tensors(batch, device):
    import torch
    return {key: torch.as_tensor(value, device=device,
        dtype=torch.bool if key in ('valid', 'is_real') else torch.int64 if key in ('point_labels', 'action') else torch.float32)
        for key, value in batch.items()}


def predict_scores(model, values, device, batch_size):
    import torch
    model.eval()
    result = dict(action=[], risk=[], utility=[], quality=[])
    with torch.no_grad():
        for start in range(0, len(values['pairs']), batch_size):
            pair = values['pairs'][start:start + batch_size]
            batch = tensors({key: values[key][pair] for key in ('points', 'valid', 'weights')}, device)
            output = model(batch['points'], batch['valid'], batch['weights'])
            result['action'].append(output['action_logits'].softmax(-1).cpu().numpy())
            result['risk'].append(output['risk_logit'].sigmoid().cpu().numpy())
            result['utility'].append(output['utility'].cpu().numpy())
            result['quality'].append(output['quality'].cpu().numpy())
            if start == 0 or start + batch_size >= len(values['pairs']) or (start // batch_size) % 25 == 0:
                print(f'SCORE_PAIRS {min(start + batch_size, len(values["pairs"]))}/{len(values["pairs"])}', flush=True)
    result = {key: np.concatenate(value) if value else np.empty((0, 3) if key in ('action', 'quality') else 0, dtype=np.float32)
              for key, value in result.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite V3 predictions.')
    return result


def thresholds(config):
    return [2.] + config['policy']['thresholds']  # score <= 1; 2 is explicit keep-all


def validate(model, config, device, step):
    saved = []
    validation_rows = [r for r in data.rows(config) if r['split'] == 'validation']
    for row in validation_rows:
        values, targets = data.load(row, config), supervision.load(row, config)
        print(f'VALIDATE {row["plot"]}: natural candidate pairs={len(values["pairs"])}', flush=True)
        scores = predict_scores(model, values, device, config['training']['batch_size'])
        geometry, table = merge_table(targets, [])
        saved.append((values, targets, scores, geometry, analyze_table(geometry, table)))
    baseline = macro_metrics([record[4]['metrics'] for record in saved])
    natural_counts = {name: sum(int(np.count_nonzero(targets['valid'] & (targets['action'] == i)))
                               for _, targets, _, _, _ in saved)
                      for i, name in enumerate(('other', 'recover', 'destroy'))}
    classifier = dict(tp=0, fp=0, fn=0)
    for _, targets, scores, _, _ in saved:
        actual = targets['valid'] & (targets['action'] == 1)
        predicted = targets['valid'] & (scores['action'].argmax(1) == 1)
        classifier['tp'] += int(np.count_nonzero(actual & predicted))
        classifier['fp'] += int(np.count_nonzero(~actual & predicted))
        classifier['fn'] += int(np.count_nonzero(actual & ~predicted))
    classifier['precision'] = classifier['tp'] / (classifier['tp'] + classifier['fp']) if classifier['tp'] + classifier['fp'] else None
    classifier['recall'] = classifier['tp'] / (classifier['tp'] + classifier['fn']) if classifier['tp'] + classifier['fn'] else None
    if abs(baseline['f1'] - config['expected_validation_baseline_f1']) > 1e-10:
        raise ValueError('Source baseline differs from the intended official evaluation.')
    trials = []
    for threshold in thresholds(config):
        metrics, lost, recovered, edits, selected_counts, two_tp = [], 0, 0, 0, [0, 0, 0], 0
        forests = []
        for row, (values, targets, scores, geometry, original) in zip(validation_rows, saved):
            accepted = policy.select_pairs(values['pairs'], scores, config['policy'], threshold)
            _, table = merge_table(targets, accepted)
            refined = analyze_table(geometry, table)
            _, comparison = compare_models(geometry, dict(baseline=original, refined=refined))
            metrics.append(refined['metrics'])
            forests.append(dict(plot=row['plot'], baseline=original['metrics'], refined=refined['metrics'],
                lost=comparison['transitions'].get('lost', 0), recovered=comparison['transitions'].get('recovered', 0), accepted_pairs=len(accepted)))
            lost += comparison['transitions'].get('lost', 0)
            recovered += comparison['transitions'].get('recovered', 0)
            edits += len(accepted)
            lookup = {tuple(pair): i for i, pair in enumerate(values['pairs'])}
            for pair in accepted:
                edge = lookup[tuple(pair)]
                if targets['valid'][edge]:
                    selected_counts[int(targets['action'][edge])] += 1
                two_tp += int(targets['original_tp'][pair].all())
        macro = macro_metrics(metrics)
        trials.append(dict(threshold=threshold, macro=macro, lost=lost, recovered=recovered, accepted_pairs=edits,
            selected_labeled_counts=dict(zip(('other', 'recover', 'destroy'), selected_counts)),
            selected_unlabeled_pairs=edits - sum(selected_counts),
            labeled_selection_precision=(selected_counts[1] / sum(selected_counts)) if sum(selected_counts) else None,
            recovery_pair_recall=selected_counts[1] / natural_counts['recover'] if natural_counts['recover'] else None,
            accepted_two_original_tp=two_tp,
            eligible=lost == 0 and macro['f1'] >= baseline['f1'] - 1e-12, forests=forests))
    chosen, stress = calibration.choose(trials)
    report = dict(step=step, baseline_macro=baseline, natural_validation_action_counts=natural_counts,
                  natural_recovery_classifier=classifier, selected=chosen, trials=trials, leave_one_forest_out=stress,
                  delta_pp=compare_metrics(baseline, chosen['macro']))
    write_json(Path(config['work_dir']) / 'calibration' / f'step_{step:06d}.json', report)
    print('VALIDATION ' + json.dumps(dict(step=step, selected={k: v for k, v in chosen.items() if k != 'forests'},
        natural_recovery_classifier=classifier, leave_one_forest_out=stress)), flush=True)
    return report


def normalize(model, values, device):
    import torch
    count, total, square = 0, np.zeros(32), np.zeros(32)
    for start in range(0, len(values['points']), 256):
        block = values['points'][start:start + 256, :, 6:][values['valid'][start:start + 256]].astype(np.float64)
        count += len(block)
        total += block.sum(0)
        square += np.square(block).sum(0)
    mean = total / count
    scale = np.sqrt(np.maximum(square / count - np.square(mean), 0))
    with torch.no_grad():
        model.feature_mean.copy_(torch.as_tensor(mean, device=device, dtype=torch.float32))
        model.feature_scale.copy_(torch.as_tensor(np.maximum(scale, .05), device=device, dtype=torch.float32))


def save(path, model, optimizer, config, step, best, rng, source_signature, hard):
    import torch
    state = dict(schema=3, signature=signature(config), source_signature=source_signature,
        model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best, hard=hard,
        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None)
    temporary = Path(path).with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint(config, path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['signature'] != signature(config):
        raise ValueError('V3 checkpoint belongs to different code/configuration.')
    model, device = make_model(config)
    model.load_state_dict(state['model'], strict=True)
    return model, device, state


def train(config, resume=False):
    import torch
    from tree_learn.instance_relation_v3.model import loss as objective
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    values = pool(config)
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    source_signature = digest({r['plot']: [file_digest(Path(r['prepared']) / name) for name in ('points.json', 'targets.json')]
                               for r in data.rows(config)})
    best, step = None, 0
    if resume:
        model, device, state = checkpoint(config, work / 'last.pth')
        if state['source_signature'] != source_signature:
            raise ValueError('Source samples/targets changed since the checkpoint.')
        step, best, values['hard'] = state['step'], state['best'], state['hard']
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        if (work / 'last.pth').exists():
            raise FileExistsError('Use resume or a new work_dir for existing training checkpoints.')
        model, device = make_model(config)
        normalize(model, values, device)
    settings = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    if resume:
        optimizer.load_state_dict(state['optimizer'])
    started = time.monotonic()
    while step < settings['steps']:
        began = time.monotonic()
        batch, real_counts, synthetic_counts = training_batch(values, config, rng)
        batch = tensors(batch, device)
        data_seconds = time.monotonic() - began
        model.train()
        output = model(batch['points'], batch['valid'], batch['weights'])
        total, components = objective(output, batch, config['objective'])
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite V3 loss; checkpoint not advanced.')
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'])
        if not torch.isfinite(gradient):
            raise FloatingPointError('Nonfinite V3 gradients; checkpoint not advanced.')
        optimizer.step()
        step += 1
        if step == 1 or step % settings['log_every'] == 0:
            record = dict(step=step, loss=float(total.detach()), gradient_norm=float(gradient), **components,
                real_action_counts=real_counts, synthetic_action_counts=synthetic_counts,
                data_seconds=data_seconds, step_seconds=time.monotonic() - began,
                elapsed_seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
            print(json.dumps(record), flush=True)
            with (work / 'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
        if step % settings['hard_mining_every'] == 0 and step < settings['steps']:
            negative = np.unique(np.concatenate([group[c] for group in values['groups'] for c in (0, 2)]))
            probe = dict(values, pairs=values['pairs'][negative])
            print(f'HARD_MINING step={step} negative_pairs={len(negative)}', flush=True)
            scores = policy.action_score(predict_scores(model, probe, device, settings['batch_size']), config['policy'])
            lookup = {int(edge): float(score) for edge, score in zip(negative, scores)}
            for i, group in enumerate(values['groups']):
                for category in (0, 2):
                    ordered = sorted(group[category].tolist(), key=lambda edge: (-lookup[edge], edge))
                    values['hard'][i][category] = np.asarray(ordered[:settings['hard_pool_size']], dtype=np.int64)
        if step % settings['validation_every'] == 0 or step == settings['steps']:
            report = validate(model, config, device, step)
            chosen = report['selected']
            rank = (chosen['recovered'], chosen['macro']['f1'], chosen['macro']['coverage'], -chosen['accepted_pairs'])
            if best is None or rank > tuple(best['rank']):
                best = dict(step=step, rank=list(rank), threshold=chosen['threshold'], validation=chosen,
                            natural_validation_action_counts=report['natural_validation_action_counts'],
                            natural_recovery_classifier=report['natural_recovery_classifier'],
                            delta_pp=report['delta_pp'], leave_one_forest_out=report['leave_one_forest_out'])
                save(work / 'best.pth', model, optimizer, config, step, best, rng, source_signature, values['hard'])
            save(work / 'last.pth', model, optimizer, config, step, best, rng, source_signature, values['hard'])
        elif step % settings['save_every'] == 0:
            save(work / 'last.pth', model, optimizer, config, step, best, rng, source_signature, values['hard'])
    write_json(work / 'train_summary.json', dict(step=step, best=best, checkpoint=str(work / 'best.pth'),
        parameter_count=sum(p.numel() for p in model.parameters()), official_model_trainable_parameters=0,
        real_action_counts={name: sum(len(group[i]) for group in values['groups'])
                            for i, name in enumerate(('other', 'recover', 'destroy'))},
        newly_recovery_labeled=sum(read_json(Path(row['prepared']) / 'targets.json')['newly_recovery_labeled']
                                  for row in data.rows(config) if row['split'] == 'train'),
        note='Outcome-aligned recovery first; source leave-one-forest-out safety stress; keep-all is not an improvement.'))


def predict_one(config, row, values, model, device, state, directory):
    """Prediction-only: no labels, point GT or counterfactual targets are opened."""
    directory = Path(directory)
    wanted = dict(signature=signature(config), plot=row['plot'], step=state['step'],
        threshold=state['best']['threshold'], policy=config['policy'],
        checkpoint_sha256=file_digest(Path(config['work_dir']) / 'best.pth'),
        points_sha256=read_json(Path(row['prepared']) / 'points.json')['sha256'])
    marker = directory / 'prediction.json'
    if marker.exists():
        saved = read_json(marker)
        if (saved['request'] != wanted or saved['scores_sha256'] != file_digest(directory / 'scores.npz')
                or saved['prediction_sha256'] != file_digest(directory / 'predictions.npy')):
            raise ValueError('Saved V3 prediction belongs to different inputs or policy.')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / 'request.json'
    if pending.exists():
        if read_json(pending) != wanted:
            raise ValueError('Incomplete V3 prediction belongs to a different request.')
    elif (directory / 'predictions.npy').exists():
        raise FileExistsError('Unrelated prediction file exists without its input request.')
    write_json(pending, wanted)
    print(f'PREDICT {row["plot"]}: frozen-feature pairs={len(values["pairs"])}', flush=True)
    scores = predict_scores(model, values, device, config['training']['batch_size'])
    accepted = policy.select_pairs(values['pairs'], scores, config['policy'], state['best']['threshold'])
    identity = policy.write_prediction(Path(row['cache']) / 'baseline.npy', directory / 'predictions.npy',
        values['ids'], accepted, config['chunk_points'], replace_existing=True)
    np.savez(directory / 'scores.npz', **scores, accepted=accepted)
    lookup = {tuple(pair): i for i, pair in enumerate(values['pairs'])}
    combined = policy.action_score(scores, config['policy'])
    actions = []
    for pair in accepted:
        edge = lookup[tuple(pair)]
        actions.append(dict(original_ids=values['ids'][pair].tolist(), score=float(combined[edge]),
            action_scores=scores['action'][edge].tolist(), predicted_risk=float(scores['risk'][edge]),
            predicted_utility=float(scores['utility'][edge]), auxiliary_quality=scores['quality'][edge].tolist()))
    write_json(directory / 'accepted_merges.json', actions)
    saved = dict(request=wanted, **identity, gt_opened=False, candidates_exactly_v1=True,
        final_instances=len(values['ids']) - len(accepted), scores_sha256=file_digest(directory / 'scores.npz'),
        prediction_sha256=file_digest(directory / 'predictions.npy'))
    write_json(marker, saved)
    return saved


def predict(config):
    model, device, state = checkpoint(config, Path(config['work_dir']) / 'best.pth')
    for row in data.rows(config, external=True):
        values = data.prepare(row, config)
        predict_one(config, row, values, model, device, state,
                    Path(config['work_dir']) / 'external_evaluation' / row['plot'])


def evaluate(config, external=False):
    model, device, state = checkpoint(config, Path(config['work_dir']) / 'best.pth')
    output = Path(config['work_dir']) / ('external_evaluation' if external else 'evaluation')
    rows = data.rows(config, external)
    if not external:
        rows = [row for row in rows if row['split'] == 'validation']
    records = []
    for row in rows:
        values = data.prepare(row, config)
        directory = output / row['plot']
        marker = predict_one(config, row, values, model, device, state, directory)
        # Open GT supervision only after the deployable prediction is saved.
        targets = load_supervision(row['base_prepared'], config['supervision'])
        geometry, table = merge_table(targets, [])
        original = analyze_table(geometry, table)
        if external and row['plot'] == 'wytham_vox0.1':
            if [original['metrics'][key] for key in ('tp', 'fp', 'fn')] != config['expected_wytham_baseline_counts']:
                raise ValueError('External official baseline differs from the intended protocol.')
        with np.load(directory / 'scores.npz', allow_pickle=False) as stored:
            accepted = stored['accepted'].copy()
        _, table = merge_table(targets, accepted)
        refined = analyze_table(geometry, table)
        all_trees, comparison = compare_models(geometry, dict(baseline=original, refined=refined))
        record = dict(plot=row['plot'], baseline=original['metrics'], refined=refined['metrics'],
            delta_pp=compare_metrics(original['metrics'], refined['metrics']), transitions=comparison['transitions'],
            accepted_pairs=len(accepted), lost_gt_ids=[r['gt_id'] for r in all_trees if r['transition'] == 'lost'],
            recovered_gt_ids=[r['gt_id'] for r in all_trees if r['transition'] == 'recovered'],
            accepted_two_original_tp=int(np.count_nonzero(targets['original_tp'][accepted].all(1))),
            selected_original_tp_masks=int(targets['original_tp'][np.unique(accepted)].sum()),
            changed_foreground_points=marker['changed_foreground_points'],
            changed_unselected_points=marker['changed_unselected_points'])
        isolated = supervision.action_targets(targets['intersections'], targets['known'], targets['gt_counts'], accepted)
        actual = supervision.outcomes(isolated['lost_count'], isolated['recovered_count'])
        record['accepted_observed_outcome_counts'] = {name: int(np.count_nonzero(actual == i))
                                                     for i, name in enumerate(supervision.NAMES)}
        write_json(directory / 'action_diagnostics.json', [dict(original_ids=values['ids'][pair].tolist(),
            isolated_outcome=supervision.NAMES[int(supervision.outcomes(isolated['lost_count'][i], isolated['recovered_count'][i]))],
            isolated_delta_tp=int(isolated['delta_tp'][i]), isolated_delta_fp=int(isolated['delta_fp'][i]),
            isolated_lost=int(isolated['lost_count'][i]), isolated_recovered=int(isolated['recovered_count'][i]),
            actual_union_quality=isolated['quality'][i].tolist()) for i, pair in enumerate(accepted)])
        write_json(directory / 'metrics.json', record)
        write_csv(directory / 'all_gt_trees.csv', all_trees)
        write_json(directory / 'watchlist.json', [r for r in all_trees if row['plot'] == 'wytham_vox0.1'
            and r['gt_id'] in (259, 375, 419, 629, 720, 801)])
        records.append(record)
        print(row['plot'] + ': ' + json.dumps(record), flush=True)
    baseline, refined = (macro_metrics([r[name] for r in records]) for name in ('baseline', 'refined'))
    brief = dict(schema=3, protocol=PROTOCOL, checkpoint_step=state['step'],
        source_leave_one_forest_out=state['best']['leave_one_forest_out'],
        threshold=state['best']['threshold'], baseline_macro=baseline, refined_macro=refined,
        delta_pp=compare_metrics(baseline, refined),
        totals={key: sum(r['refined'][key] for r in records) for key in ('tp', 'fp', 'fn')},
        accepted_pairs=sum(r['accepted_pairs'] for r in records),
        accepted_two_original_tp=sum(r['accepted_two_original_tp'] for r in records),
        accepted_observed_outcome_counts={name: sum(r['accepted_observed_outcome_counts'][name] for r in records)
                                         for name in supervision.NAMES},
        lost_gt=sum(len(r['lost_gt_ids']) for r in records), recovered_gt=sum(len(r['recovered_gt_ids']) for r in records),
        lost_by_plot={r['plot']: r['lost_gt_ids'] for r in records},
        recovered_by_plot={r['plot']: r['recovered_gt_ids'] for r in records},
        changed_foreground_points=sum(r['changed_foreground_points'] for r in records),
        changed_unselected_points=sum(r['changed_unselected_points'] for r in records),
        no_edit_result=not any(r['accepted_pairs'] for r in records))
    write_json(output / 'summary.json', dict(brief, forests=records))
    write_json(output / 'brief.json', brief)
    print('SAVED ' + str(output / 'brief.json') + '\n' + json.dumps(brief, indent=2), flush=True)


def doctor(config):
    import torch
    if hashlib.md5(Path(config['checkpoint']).read_bytes()).hexdigest() != config['checkpoint_md5']:
        raise ValueError('Official checkpoint MD5 differs from the configured reference.')
    rows = data.rows(config)
    missing = []
    for row in rows + data.rows(config, external=True):
        for path in (Path(row['cache']) / 'features.npy', Path(row['base_prepared']) / 'features.json',
                     Path(row['base_prepared']) / 'features.npz', Path(row['base_prepared']) / 'supervision.json',
                     Path(row['point_prepared']) / 'points.json'):
            if not path.is_file():
                missing.append(str(path))
        if row['split'] != 'external':
            for name in ('targets.json', 'targets.npz'):
                path = Path(row['point_prepared']) / name
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError('Completed V1/V2 caches required before V3: ' + ', '.join(missing))
    print(json.dumps(dict(python=sys.executable, torch=torch.__version__, device=config['training']['device'],
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        source_forests=len(rows), official_network_loaded=False, official_model_trainable_parameters=0,
        frozen_input_channels=32, candidates='exactly V1', quality_regression='auxiliary only',
        outcomes=list(supervision.NAMES), reused='V2 point samples and exact counterfactual counts',
        score=f"p(recover) - {config['policy']['destruction_cost']}*p(destroy)",
        real_fraction=config['training']['real_fraction']), indent=2), flush=True)


def check(config):
    import torch
    from tree_learn.instance_relation_v3.model import loss as objective
    torch.manual_seed(config['seed'])
    values = pool(config)
    model, device = make_model(config)
    normalize(model, values, device)
    batch, real_counts, synthetic_counts = training_batch(values, config, np.random.default_rng(config['seed']))
    if any(count == 0 for count in real_counts):
        raise RuntimeError('Check batch lacks real other/recover/destroy classes; inspect exact outcome counts.')
    batch = tensors(batch, device)
    model.eval()
    first = model(batch['points'], batch['valid'], batch['weights'])
    swapped = model(batch['points'].flip(1), batch['valid'].flip(1), batch['weights'].flip(1))
    for key in ('action_logits', 'risk_logit', 'utility', 'quality'):
        if not torch.allclose(first[key], swapped[key], rtol=1e-5, atol=2e-6):
            raise RuntimeError('Point relation output depends on pair ordering: ' + key)
    total, components = objective(first, batch, config['objective'])
    total.backward()
    gradients = {}
    for name in ('point_encoder', 'connection_encoder', 'structure_encoder', 'track_encoder', 'relation'):
        parameters = list(getattr(model, name).parameters())
        gradients[name] = sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)
        if gradients[name] <= 0 or not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Loss does not reach the real point/connection branch: ' + name)
    print('CHECK_PASSED ' + json.dumps(dict(loss=float(total.detach()), gradients=gradients, **components,
        real_action_counts=real_counts, synthetic_action_counts=synthetic_counts,
        parameter_count=sum(p.numel() for p in model.parameters()), pair_symmetric=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor', 'prepare', 'check', 'train', 'predict', 'evaluate', 'external'))
    parser.add_argument('--config', default='configs/experiments/instance_relation_v3/full.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = config_read(args.config)
    if args.command == 'train':
        train(config, args.resume)
    elif args.command == 'predict':
        predict(config)
    elif args.command in ('evaluate', 'external'):
        evaluate(config, args.command == 'external')
    else:
        dict(doctor=doctor, prepare=prepare, check=check)[args.command](config)


if __name__ == '__main__':
    main()
