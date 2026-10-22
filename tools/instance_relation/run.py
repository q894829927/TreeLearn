"""Complete frozen-cache keep/merge workflow. Never imports the official network."""

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
from tree_learn.instance_relation.data import entries, feature_request, load_features, prepare_features
from tree_learn.instance_relation.supervision import load_supervision, merge_table, prepare_supervision
from tree_learn.instance_relation.policy import select_pairs, write_prediction
from tree_learn.mask_refinement.data import digest, file_digest, write_json
from tree_learn.mask_refinement.evaluation import macro_metrics, compare_metrics

PROTOCOL = 'aligned_voxels_ignore_negative_gt_iou_gt_0.5'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config_read(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    inputs = [Path(config[k]).resolve() for k in ('cache_root', 'external_cache_root')]
    outputs = [Path(config[k]).resolve() for k in ('prepared_root', 'work_dir')]
    if any(a == b or a in b.parents or b in a.parents for a in inputs for b in outputs):
        raise ValueError('Relation outputs must be separate from read-only official cache roots.')
    for key in ('height_strata', 'points_per_stratum', 'max_pairs', 'max_instances'):
        if type(config['geometry'][key]) is not int or config['geometry'][key] <= 0:
            raise ValueError(f'Invalid geometry {key}.')
    for key in ('vote_radius_m', 'lower_radius_m', 'vote_scale_m'):
        if not math.isfinite(config['geometry'][key]) or config['geometry'][key] <= 0:
            raise ValueError(f'Invalid geometry {key}.')
    for key in ('steps', 'batch_size', 'node_batch_size', 'validation_every', 'save_every', 'log_every', 'hidden_dim'):
        if type(config['training'][key]) is not int or config['training'][key] <= 0:
            raise ValueError(f'Invalid training {key}.')
    if config['training']['batch_size'] % 2 or config['training']['node_batch_size'] % 2 or config['chunk_points'] < 1:
        raise ValueError('Use an even pair batch size and a positive scan chunk.')
    for group in ('supervision',):
        if any(not 0 < float(value) <= 1 for value in config[group].values()):
            raise ValueError(f'Invalid {group} fractions.')
    if any(not 0 < v < 1 for v in config['policy']['merge_thresholds'] + config['policy']['protect_thresholds']):
        raise ValueError('Policy grid thresholds must lie strictly between zero and one.')
    for key in ('same_threshold', 'min_union_precision', 'min_union_recall', 'min_union_iou', 'min_iou_gain'):
        if not 0 < config['policy'][key] <= 1:
            raise ValueError(f'Invalid policy {key}.')
    if type(config['policy']['max_source_lost_gt']) is not int or config['policy']['max_source_lost_gt'] < 0:
        raise ValueError('Invalid source lost-GT budget.')
    for key in ('learning_rate', 'weight_decay', 'grad_clip'):
        if not math.isfinite(config['training'][key]) or config['training'][key] <= 0:
            raise ValueError(f'Invalid training {key}.')
    return config


def code_signature(config):
    paths = list((ROOT / 'tree_learn/instance_relation').glob('*.py')) + [Path(__file__).resolve(),
        ROOT / 'tools/instance_relation/run.sh', ROOT / 'tools/forest_abc/diagnose.py']
    return digest(dict(config=config, sources={p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths}))


def prepare(config, external=False):
    rows = entries(config, external)
    for row in rows:
        print(f'PREPARE {row["split"]}/{row["plot"]}: official predictions only', flush=True)
        features = prepare_features(row['cache'], row['prepared'], config['geometry'], config['seed'], config['chunk_points'])
        print(f'FEATURES {row["plot"]}: nodes={len(features["ids"])} pairs={len(features["pairs"])}', flush=True)
        if not external:
            targets = prepare_supervision(row['cache'], row['prepared'], features,
                                          config['supervision'], config['chunk_points'])
            print(f'TARGETS {row["plot"]}: positive_merge_pairs=' +
                  str(int(np.count_nonzero(targets['pair_valid'] & (targets['pair_merge'] > 0)))), flush=True)
    return rows


def minimal(row, config):
    meta = read_json(Path(row['prepared']) / 'features.json')
    if meta['request'] != feature_request(row['cache'], dict(config['geometry'], seed=config['seed'])):
        raise ValueError(f'Stale official-cache features: {row["plot"]}')
    all_values = load_features(row['prepared'])
    return {key: all_values[key] for key in ('ids', 'node_features', 'pair_features', 'pairs', 'population')}


def training_pool(rows, config):
    lists = {key: [] for key in ('nodes', 'pairs', 'features', 'merge', 'same', 'quality', 'gain', 'node_quality')}
    pair_groups, node_groups, offset, pair_offset = [], [], 0, 0
    for row in rows:
        values, target = minimal(row, config), load_supervision(row['prepared'], config['supervision'])
        p = len(values['pairs'])
        lists['nodes'].append(values['node_features'])
        lists['pairs'].append(values['pairs'] + offset)
        lists['features'].append(values['pair_features'])
        for key in ('merge', 'same', 'quality', 'gain'):
            lists[key].append(target['pair_' + key])
        lists['node_quality'].append(target['node_quality'])
        pair_groups.append([np.flatnonzero(target['pair_valid'] & (target['pair_merge'] == label)) + pair_offset
                            for label in (0, 1)])
        node_groups.append([np.flatnonzero(target['node_valid'] & ((target['node_quality'] >= .5) == bool(label))) + offset
                            for label in (0, 1)])
        offset += len(values['ids'])
        pair_offset += p
    result = {key: np.concatenate(value) for key, value in lists.items()}
    result.update(pair_groups=pair_groups, node_groups=node_groups)
    if not any(len(group[1]) for group in pair_groups):
        raise ValueError('No real beneficial same-tree merge pairs in training caches. Inspect candidate/target counts before training.')
    if not any(len(group[0]) for group in pair_groups):
        raise ValueError('No valid keep/negative pairs for training.')
    return result


def balanced_draw(groups, count, rng):
    result = []
    for label in (0, 1):
        available = [group[label] for group in groups if len(group[label])]
        if not available:
            available = [group[1-label] for group in groups if len(group[1-label])]
        if not available:
            raise ValueError('No labeled samples for the relation objective.')
        forests = rng.integers(len(available), size=count // 2)
        for forest in np.unique(forests):
            result.extend(rng.choice(available[int(forest)], size=int(np.count_nonzero(forests == forest)), replace=True).tolist())
    rng.shuffle(result)
    return np.asarray(result, dtype=np.int64)


def make_model(config, node_dim, pair_dim):
    import torch
    from tree_learn.instance_relation.model import InstanceRelationHead
    device = torch.device(config['training']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; change training.device to cpu only if intended.')
    return InstanceRelationHead(node_dim, pair_dim, config['training']['hidden_dim']).to(device=device, dtype=torch.float32), device


def predict_scores(model, values, device, batch=2048):
    import torch
    model.eval()
    nodes = values['node_features']
    result = dict(merge=[], same=[], quality=[], gain=[])
    node_quality = []
    with torch.no_grad():
        for start in range(0, len(nodes), batch):
            tensor = torch.as_tensor(nodes[start:start + batch], device=device, dtype=torch.float32)
            node_quality.append(model.node_quality(model.encode(tensor)).sigmoid().squeeze(-1).cpu().numpy())
        for start in range(0, len(values['pairs']), batch):
            pairs = values['pairs'][start:start + batch]
            output = model(torch.as_tensor(nodes[pairs[:, 0]], device=device, dtype=torch.float32),
                           torch.as_tensor(nodes[pairs[:, 1]], device=device, dtype=torch.float32),
                           torch.as_tensor(values['pair_features'][start:start + batch], device=device, dtype=torch.float32))
            result['merge'].append(output['merge_logit'].sigmoid().cpu().numpy())
            result['same'].append(output['same_logit'].sigmoid().cpu().numpy())
            result['quality'].append(output['quality'].cpu().numpy())
            result['gain'].append(output['gain'].cpu().numpy())
    result = {key: np.concatenate(value) if value else np.empty((0, 3) if key == 'quality' else (0,), dtype=np.float32)
              for key, value in result.items()}
    result['node_quality'] = np.concatenate(node_quality)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Relation inference produced nonfinite scores.')
    return result


def policy_grid(config):
    settings = config['policy']
    base = {key: settings[key] for key in ('same_threshold', 'min_union_precision', 'min_union_recall', 'min_union_iou', 'min_iou_gain')}
    # An explicit keep-all option prevents validation from forcing harmful edits.
    yield dict(base, merge_threshold=1.1, protect_threshold=.65)
    for threshold in settings['merge_thresholds']:
        for protect in settings['protect_thresholds']:
            yield dict(base, merge_threshold=threshold, protect_threshold=protect)


def validate(model, rows, config, device, step):
    saved = []
    for row in rows:
        values = minimal(row, config)
        scores = predict_scores(model, values, device)
        target = load_supervision(row['prepared'], config['supervision'])
        geometry, table = merge_table(target, [])
        saved.append((values, scores, target, geometry, analyze_table(geometry, table)))
    baseline_macro = macro_metrics([value[4]['metrics'] for value in saved])
    if abs(baseline_macro['f1'] - config['expected_validation_baseline_f1']) > 1e-10:
        raise ValueError('Validation baseline differs from the intended official cache protocol.')
    trials = []
    for policy in policy_grid(config):
        metrics, lost, recovered, edited = [], 0, 0, 0
        for values, scores, target, geometry, original in saved:
            accepted = select_pairs(values['pairs'], scores, scores['node_quality'], policy)
            _, table = merge_table(target, accepted)
            result = analyze_table(geometry, table)
            _, transitions = compare_models(geometry, dict(baseline=original, refined=result))
            lost += transitions['transitions'].get('lost', 0)
            recovered += transitions['transitions'].get('recovered', 0)
            edited += len(accepted)
            metrics.append(result['metrics'])
        macro = macro_metrics(metrics)
        trials.append(dict(policy=policy, macro=macro, lost=lost, recovered=recovered, accepted_pairs=edited,
                           eligible=lost <= config['policy']['max_source_lost_gt']))
    eligible = [r for r in trials if r['eligible']]
    best = max(eligible, key=lambda r: (r['macro']['f1'], r['macro']['coverage'], -r['accepted_pairs']))
    report = dict(step=step, baseline_macro=baseline_macro, selected=best, trials=trials,
                  delta_pp=compare_metrics(baseline_macro, best['macro']))
    work = Path(config['work_dir'])
    write_json(work / 'calibration' / f'step_{step:06d}.json', report)
    print('VALIDATION ' + json.dumps(dict(step=step, selected=best, delta_pp=report['delta_pp'])), flush=True)
    return report


def save_checkpoint(path, model, optimizer, step, best, rng, config, data_signature):
    import torch
    state = dict(schema=1, signature=code_signature(config), data_signature=data_signature,
        dimensions=model.dimensions, model=model.state_dict(), optimizer=optimizer.state_dict(),
        step=step, best=best, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(config, path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['signature'] != code_signature(config):
        raise ValueError('Checkpoint belongs to different relation code/configuration.')
    model, device = make_model(config, state['dimensions']['node_dim'], state['dimensions']['pair_dim'])
    model.load_state_dict(state['model'], strict=True)
    return model, device, state


def train(config, resume=False):
    import torch
    from torch.nn import functional as F
    from tree_learn.instance_relation.model import relation_loss
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.device(config['training']['device']).type == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    rows = entries(config)
    training_rows, validation_rows = ([r for r in rows if r['split'] == split] for split in ('train', 'validation'))
    pool = training_pool(training_rows, config)
    signature = digest({row['plot']: file_digest(Path(row['prepared']) / 'supervision.json') for row in rows})
    work, settings = Path(config['work_dir']), config['training']
    work.mkdir(parents=True, exist_ok=True)
    rng, step, best = np.random.default_rng(config['seed']), 0, None
    if resume:
        model, device, state = load_checkpoint(config, work / 'last.pth')
        if state['data_signature'] != signature:
            raise ValueError('Training labels/prepared inputs changed since checkpoint.')
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        step, best = state['step'], state['best']
    else:
        if (work / 'last.pth').exists():
            raise FileExistsError('Existing training checkpoint; use resume or a new work_dir.')
        model, device = make_model(config, pool['nodes'].shape[1], pool['features'].shape[1])
        with torch.no_grad():
            for name, values in (('node', pool['nodes']), ('pair', pool['features'])):
                mean, scale = values.mean(0, dtype=np.float64), values.std(0, dtype=np.float64)
                getattr(model, name + '_mean').copy_(torch.as_tensor(mean, device=device, dtype=torch.float32))
                getattr(model, name + '_scale').copy_(torch.as_tensor(np.maximum(scale, .05), device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    if resume:
        optimizer.load_state_dict(state['optimizer'])
    started = time.monotonic()
    while step < settings['steps']:
        model.train()
        indices = balanced_draw(pool['pair_groups'], settings['batch_size'], rng)
        pairs = pool['pairs'][indices]
        def tensor(values):
            return torch.as_tensor(values, device=device, dtype=torch.float32)
        result = model(tensor(pool['nodes'][pairs[:, 0]]), tensor(pool['nodes'][pairs[:, 1]]), tensor(pool['features'][indices]))
        loss, components = relation_loss(result, tensor(pool['merge'][indices]), tensor(pool['same'][indices]),
                                         tensor(pool['quality'][indices]), tensor(pool['gain'][indices]))
        nodes = balanced_draw(pool['node_groups'], settings['node_batch_size'], rng)
        quality = model.node_quality(model.encode(tensor(pool['nodes'][nodes]))).sigmoid().squeeze(-1)
        node_loss = F.smooth_l1_loss(quality, tensor(pool['node_quality'][nodes]))
        loss = loss + node_loss
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite relation loss; checkpoint not advanced.')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'])
        if not torch.isfinite(gradient):
            raise FloatingPointError('Nonfinite relation gradients; checkpoint not advanced.')
        optimizer.step()
        step += 1
        if step == 1 or step % settings['log_every'] == 0:
            record = dict(step=step, loss=float(loss.detach()), node_quality_loss=float(node_loss.detach()),
                gradient_norm=float(gradient), elapsed_seconds=time.monotonic() - started, **components)
            print(json.dumps(record), flush=True)
            with (work / 'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
        if step % settings['validation_every'] == 0 or step == settings['steps']:
            report = validate(model, validation_rows, config, device, step)
            current = report['selected']
            rank = (current['macro']['f1'], current['macro']['coverage'], -current['accepted_pairs'])
            if best is None or rank > tuple(best['rank']):
                best = dict(step=step, rank=list(rank), policy=current['policy'], validation=current,
                            delta_pp=report['delta_pp'])
                save_checkpoint(work / 'best.pth', model, optimizer, step, best, rng, config, signature)
            save_checkpoint(work / 'last.pth', model, optimizer, step, best, rng, config, signature)
        elif step % settings['save_every'] == 0:
            save_checkpoint(work / 'last.pth', model, optimizer, step, best, rng, config, signature)
    write_json(work / 'train_summary.json', dict(step=step, best=best,
        checkpoint=str(work / 'best.pth'), parameter_count=sum(p.numel() for p in model.parameters()),
        official_model_trainable_parameters=0, note='Keep-all policy is explicit; zero accepted pairs is not an improvement.'))


def predict_one(config, row, destination, model, device, state):
    """No labels or supervision files may be read from this function or its callees."""
    destination = Path(destination)
    values = minimal(row, config)
    request = dict(signature=code_signature(config), checkpoint_sha256=file_digest(Path(config['work_dir']) / 'best.pth'),
        features=read_json(Path(row['prepared']) / 'features.json')['npz_sha256'],
        step=state['step'], policy=state['best']['policy'], plot=row['plot'])
    if (destination / 'prediction.json').exists():
        marker = read_json(destination / 'prediction.json')
        if (marker['request'] != request or marker['prediction_sha256'] != file_digest(destination / 'predictions.npy')
                or marker['scores_sha256'] != file_digest(destination / 'scores.npz')):
            raise ValueError('Saved merged prediction belongs to other model/features/policy.')
        return marker
    destination.mkdir(parents=True, exist_ok=True)
    pending = destination / 'request.json'
    if pending.exists():
        if read_json(pending) != request:
            raise ValueError('Incomplete prediction belongs to different inputs; use a new evaluation directory.')
    elif (destination / 'predictions.npy').exists():
        raise FileExistsError('Unrelated prediction file exists without its input request.')
    write_json(pending, request)
    scores = predict_scores(model, values, device)
    accepted = select_pairs(values['pairs'], scores, scores['node_quality'], state['best']['policy'])
    identity = write_prediction(Path(row['cache']) / 'baseline.npy', destination / 'predictions.npy',
                                values['ids'], accepted, config['chunk_points'], replace_existing=True)
    np.savez(destination / 'scores.npz', **scores, accepted=accepted)
    lookup = {tuple(pair): edge for edge, pair in enumerate(values['pairs'])} if len(accepted) else {}
    actions = []
    for pair in accepted:
        edge = lookup[tuple(pair)]
        actions.append(dict(original_ids=values['ids'][pair].tolist(), kept_id=int(values['ids'][min(pair)]),
            merge_probability=float(scores['merge'][edge]), same_tree_probability=float(scores['same'][edge]),
            predicted_union_quality=scores['quality'][edge].tolist(),
            predicted_iou_gain=float(scores['gain'][edge]),
            predicted_original_iou=scores['node_quality'][pair].tolist()))
    write_json(destination / 'accepted_merges.json', actions)
    marker = dict(request=request, **identity, gt_opened=False, scores_sha256=file_digest(destination / 'scores.npz'),
                  final_instances=len(values['ids']) - len(accepted),
                  prediction_sha256=file_digest(destination / 'predictions.npy'))
    write_json(destination / 'prediction.json', marker)
    print('PREDICT ' + row['plot'] + ': ' + json.dumps(identity), flush=True)
    return marker


def predict(config):
    model, device, state = load_checkpoint(config, Path(config['work_dir']) / 'best.pth')
    for row in entries(config, external=True):
        prepare_features(row['cache'], row['prepared'], config['geometry'], config['seed'], config['chunk_points'])
        predict_one(config, row, Path(config['work_dir']) / 'external_evaluation' / row['plot'], model, device, state)


def evaluate(config, external=False):
    model, device, state = load_checkpoint(config, Path(config['work_dir']) / 'best.pth')
    rows = entries(config, external)
    if not external:
        rows = [r for r in rows if r['split'] == 'validation']
    output = Path(config['work_dir']) / ('external_evaluation' if external else 'evaluation')
    reports = []
    for row in rows:
        # External extraction is prediction-only; labels are opened after prediction is saved.
        prepare_features(row['cache'], row['prepared'], config['geometry'], config['seed'], config['chunk_points'])
        directory = output / row['plot']
        marker = predict_one(config, row, directory, model, device, state)
        values = minimal(row, config)
        target = prepare_supervision(row['cache'], row['prepared'], values, config['supervision'], config['chunk_points'])
        with np.load(directory / 'scores.npz', allow_pickle=False) as scores:
            accepted = scores['accepted'].copy()
        geometry, table = merge_table(target, [])
        original = analyze_table(geometry, table)
        _, table = merge_table(target, accepted)
        refined = analyze_table(geometry, table)
        if external and row['plot'] == 'wytham_vox0.1' and [original['metrics'][k] for k in ('tp', 'fp', 'fn')] != config['expected_wytham_baseline_counts']:
            raise ValueError('Wytham baseline counts differ from the intended official cache.')
        comparisons, comparison = compare_models(geometry, dict(baseline=original, refined=refined))
        selected = set(accepted.flatten().tolist())
        selected_tp = int(target['original_tp'][sorted(selected)].sum()) if selected else 0
        record = dict(plot=row['plot'], baseline=original['metrics'], refined=refined['metrics'],
            delta_pp=compare_metrics(original['metrics'], refined['metrics']), transitions=comparison['transitions'],
            selected_original_tp_masks=selected_tp, original_tp_masks_exactly_preserved=original['metrics']['tp'] - selected_tp,
            invariants={key: marker[key] for key in ('changed_foreground_points', 'changed_unselected_points', 'accepted_pairs')},
            candidate_merge_positive_oracle_pairs=int(np.count_nonzero(target['pair_valid'] & (target['pair_merge'] > 0))))
        positive = target['pair_valid'] & (target['pair_merge'] > 0)
        record['candidate_merge_positive_oracle_gt_ids'] = np.unique(target['pair_target_gt'][positive]).tolist()
        detected_ids = {r['gt_id'] for r in original['trees'] if r['detected']}
        record['candidate_recovery_oracle_gt_ids'] = [identifier for identifier in record['candidate_merge_positive_oracle_gt_ids']
                                                     if identifier not in detected_ids]
        record['lost_gt_ids'] = [r['gt_id'] for r in comparisons if r['transition'] == 'lost']
        record['recovered_gt_ids'] = [r['gt_id'] for r in comparisons if r['transition'] == 'recovered']
        write_json(directory / 'metrics.json', record)
        write_csv(directory / 'all_gt_trees.csv', comparisons)
        write_json(directory / 'watchlist.json', [r for r in comparisons if row['plot'] == 'wytham_vox0.1' and r['gt_id'] in (259, 375, 419, 629, 720, 801)])
        reports.append(record)
        print(row['plot'] + ': ' + json.dumps(record), flush=True)
    baseline, refined = (macro_metrics([r[name] for r in reports]) for name in ('baseline', 'refined'))
    brief = dict(schema=1, protocol=PROTOCOL, checkpoint_step=state['step'], policy=state['best']['policy'],
        baseline_macro=baseline, refined_macro=refined, delta_pp=compare_metrics(baseline, refined),
        totals={k: sum(r['refined'][k] for r in reports) for k in ('tp', 'fp', 'fn')},
        accepted_pairs=sum(r['invariants']['accepted_pairs'] for r in reports),
        lost_gt=sum(r['transitions'].get('lost', 0) for r in reports), recovered_gt=sum(r['transitions'].get('recovered', 0) for r in reports),
        changed_foreground_points=sum(r['invariants']['changed_foreground_points'] for r in reports),
        changed_unselected_points=sum(r['invariants']['changed_unselected_points'] for r in reports),
        selected_original_tp_masks=sum(r['selected_original_tp_masks'] for r in reports),
        lost_by_plot={r['plot']: r['lost_gt_ids'] for r in reports},
        recovered_by_plot={r['plot']: r['recovered_gt_ids'] for r in reports},
        no_edit_result=not any(r['invariants']['accepted_pairs'] for r in reports))
    write_json(output / 'summary.json', dict(brief, forests=reports))
    write_json(output / 'brief.json', brief)
    print('SAVED ' + str(output / 'brief.json') + '\n' + json.dumps(brief, indent=2), flush=True)


def doctor(config):
    import torch
    if hashlib.md5(Path(config['checkpoint']).read_bytes()).hexdigest() != config['checkpoint_md5']:
        raise ValueError('Official checkpoint MD5 differs from the configured reference.')
    rows = entries(config)
    print(json.dumps(dict(python=sys.executable, torch=torch.__version__, device=config['training']['device'],
        source_forests=len(rows), official_network_loaded=False, backbone_trainable_parameters=0,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        inputs='official instances, coordinates, probability, XY votes; no GT in descriptors/inference'), indent=2), flush=True)


def check(config):
    import torch
    from tree_learn.instance_relation.model import relation_loss
    rows = [r for r in entries(config) if r['split'] == 'train']
    pool = training_pool(rows, config)
    model, device = make_model(config, pool['nodes'].shape[1], pool['features'].shape[1])
    rng = np.random.default_rng(config['seed'])
    index = balanced_draw(pool['pair_groups'], 32, rng)
    pairs = pool['pairs'][index]
    a, b, pair = [torch.as_tensor(v, device=device, dtype=torch.float32) for v in
                 (pool['nodes'][pairs[:, 0]], pool['nodes'][pairs[:, 1]], pool['features'][index])]
    model.eval()
    first, swapped = model(a, b, pair), model(b, a, pair)
    if any(not torch.equal(first[key], swapped[key]) for key in first):
        raise RuntimeError('Relation predictions depend on pair ordering.')
    loss, _ = relation_loss(first, torch.as_tensor(pool['merge'][index], device=device),
        torch.as_tensor(pool['same'][index], device=device), torch.as_tensor(pool['quality'][index], device=device),
        torch.as_tensor(pool['gain'][index], device=device))
    quality = model.node_quality(model.encode(a)).sigmoid().squeeze(-1)
    loss = loss + torch.nn.functional.smooth_l1_loss(quality,
        torch.as_tensor(pool['node_quality'][pairs[:, 0]], device=device, dtype=torch.float32))
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Relation check has nonfinite real-data gradients.')
    print('CHECK_PASSED ' + json.dumps(dict(real_pairs=len(index), loss=float(loss.detach()),
          train_merge_positives=sum(len(g[1]) for g in pool['pair_groups']), symmetry=True,
          parameter_count=sum(p.numel() for p in model.parameters()))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor', 'prepare', 'check', 'train', 'predict', 'evaluate', 'external'))
    parser.add_argument('--config', default='configs/experiments/instance_relation/full.yaml')
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
