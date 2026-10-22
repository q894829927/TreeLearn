"""Gated source-only edge supervision, training, and complete-core evaluation."""

from collections import defaultdict
import hashlib
import os
from pathlib import Path
import random

import numpy as np

from tree_learn.instance_relation.data import priorities
from tree_learn.mask_refinement.data import file_digest, load_cache, write_json
from . import data
from .audit import advancement, baseline_table, comparison, edge_targets, split_table
from .geometry import BudgetExceeded, graph_edges, split_proposal, two_means


def require_gate(config, stage):
    record = data.read_json(Path(config['work_dir']) / stage / 'summary.json')
    if record['signature'] != data.signature(config):
        raise ValueError(f'Stale {stage} gate.')
    if not record['gate']['passed']:
        raise ValueError(f'{stage} gate failed; later stages are forbidden.')
    # Every source preparation remains tied to immutable observables.
    for row in data.rows(config):
        data.load_prepared(row, config)
    if stage == 'e0':
        for row in data.rows(config):
            expected = record['preparation_sha256'][row['plot']]
            if file_digest(Path(row['prepared']) / 'candidates.json') != expected:
                raise ValueError('Preparation changed after E0 audit.')
            baseline_table(row, config)  # also verifies the GT source stamps
    else:
        require_gate(config, 'e0')
    return record


def normalization(config):
    mean_sum, square_sum, total = np.zeros(32), np.zeros(32), 0
    sources = {}
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        arrays, _ = load_cache(row['cache'], with_labels=False)
        try:
            # Uniform index hashes on assigned points, independent of GT.
            selected, values = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)
            limit = config['training']['normalization_points_per_forest']
            for start in range(0, len(arrays['baseline']), config['chunk_points']):
                indices = start + np.flatnonzero(arrays['baseline'][start:start + config['chunk_points']] > 0)
                selected = np.r_[selected, indices]
                values = np.r_[values, priorities(indices, config['seed'])]
                if len(selected) > limit:
                    at = np.argpartition(values, limit - 1)[:limit]
                    selected, values = selected[at], values[at]
            features = arrays['features'][selected].astype(float)
            if not np.isfinite(features).all():
                raise ValueError('Nonfinite training features.')
            mean_sum += features.sum(0)
            square_sum += np.square(features).sum(0)
            total += len(features)
            sources[row['plot']] = len(features)
            print(f'NORMALIZE {row["plot"]} points={len(features)}', flush=True)
        finally:
            data.close_arrays(arrays)
    mean = mean_sum / total
    scale = np.sqrt(np.maximum(square_sum / total - mean ** 2, 1e-6))
    return dict(mean=mean.tolist(), scale=scale.tolist(), points=total, sources=sources)


def fork_evidence(xyz, cfg):
    """Observable high-branch support for selecting source TP hard controls."""
    low, high = np.quantile(xyz[:, 2], [.5, .95])
    for bottom in np.arange(low, high, cfg['slab_m']):
        xy = xyz[(xyz[:, 2] >= bottom) & (xyz[:, 2] < bottom + cfg['slab_m']), :2]
        fit = two_means(xy) if len(xy) >= 2 * cfg['cluster_points'] else None
        if fit is None:
            continue
        sse, centers, groups = fit
        if np.bincount(groups, minlength=2).min() < cfg['cluster_points']:
            continue
        rms = [np.sqrt(np.square(xy[groups == k] - centers[k]).sum(1).mean()) for k in (0, 1)]
        total = np.square(xy - xy.mean(0)).sum()
        distance = np.linalg.norm(centers[0] - centers[1])
        if distance >= max(cfg['separation_m'], cfg['separation_rms'] * max(rms)) and sse <= cfg['sse_ratio'] * total:
            return True
    return False


def bank(config, e0):
    """Bounded 8,192-edge banks; complete parent graphs still drive evaluation."""
    from .model import raw_edges
    directory = Path(config['work_dir']) / 'banks'
    marker = directory / 'bank.json'
    if marker.exists():
        saved = data.read_json(marker)
        if saved['signature'] != data.signature(config):
            raise ValueError('Stale training bank.')
        for record in saved['records']:
            if file_digest(record['path']) != record['sha256']:
                raise ValueError('Changed training bank.')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    norm = normalization(config)
    records, rng = [], np.random.default_rng(config['seed'])
    reliable = {(f['plot'], c['parent_id']) for f in e0['forests'] for c in f['candidates']
                if c['reliable'] and c['recoverable_gt_ids'] and f['split'] == 'train'}
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        arrays, _ = load_cache(row['cache'], with_labels=True)
        try:
            geometry, table, before = baseline_table(row, config)
            ids, starts, counts, order = data.source_groups(arrays['baseline'])
            core = {int(i): order[s:s + c] for i, s, c in zip(ids, starts, counts)}
            recall = table['intersections'] / geometry['counts'][None, :]
            mixed = ids[(recall > .5).sum(1) >= 2].tolist()
            correct = []
            for r in before['predictions']:
                if r['category'] != 'true_positive':
                    continue
                indices = core[r['pred_id']]
                if len(indices) <= config['candidate']['max_points'] and fork_evidence(arrays['coords'][indices], config['candidate']):
                    correct.append(r['pred_id'])
                if len(correct) == 8:
                    break
            specifications = [('mixed', [int(i)]) for i in mixed] + [('correct', [int(i)]) for i in correct]
            # Real coordinates and predictions from the same forest; no translated features.
            with np.load(Path(config['base_prepared_root']) / row['split'] / row['plot'] / 'features.npz', allow_pickle=False) as z:
                pairs = z['pairs']; node_ids = z['ids']
            synthetic = 0
            for a, b in pairs:
                aa, bb = core[int(node_ids[a])], core[int(node_ids[b])]
                if len(aa) + len(bb) > config['candidate']['max_points']:
                    continue
                from scipy.spatial import cKDTree
                distances = cKDTree(arrays['coords'][aa]).query(arrays['coords'][bb], workers=1)[0]
                if distances.min() > config['graph']['radius_m']:
                    continue
                if table['intersections'][a].argmax() == table['intersections'][b].argmax():
                    continue
                specifications.append(('synthetic', [int(node_ids[a]), int(node_ids[b])]))
                synthetic += 1
                if synthetic == 8:
                    break
            for category, parents in specifications:
                indices = np.concatenate([core[i] for i in parents])
                if len(indices) > config['candidate']['max_points']:
                    continue
                xyz = arrays['coords'][indices].astype(float)
                try:
                    edges = graph_edges(xyz, config['graph'])
                except BudgetExceeded:
                    continue
                valid, targets = edge_targets(arrays['labels'][indices], edges)
                pools = [np.flatnonzero(valid & (targets == k)) for k in (0, 1)]
                if not len(pools[1]) or (category != 'correct' and not len(pools[0])):
                    continue
                active = [p for p in pools if len(p)]
                each = config['training']['bank_edges_per_parent'] // len(active)
                selected = np.concatenate([rng.choice(p, min(each, len(p)), replace=False) for p in active])
                filename = directory / f'{row["plot"]}_{category}_{"_".join(map(str, parents))}.npz'
                np.savez(filename, raw=raw_edges(arrays, indices, edges[selected]), target=targets[selected])
                record = dict(plot=row['plot'], category=category, parents=parents, path=str(filename),
                    sha256=file_digest(filename), edge_count=len(selected),
                    overfit_mixed=(row['plot'], parents[0]) in reliable if category == 'mixed' else False)
                records.append(record)
                print(f'BANK {row["plot"]} {category} {parents} edges={len(selected)}', flush=True)
        finally:
            data.close_arrays(arrays)
    result = dict(signature=data.signature(config), normalization=norm, records=records)
    write_json(marker, result)
    return result


def setup_model(config, norm, checkpoint=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from .model import EdgeHead
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config['training']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA unavailable; no implicit environment change.')
    model = EdgeHead(norm['mean'], norm['scale']).to(device)
    if checkpoint:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['signature'] != data.signature(config):
            raise ValueError('Checkpoint signature differs.')
        model.load_state_dict(saved['state_dict'])
    return model, device


def forest_proposals(row, config, model=None, device=None, allowed=None):
    prepared = data.load_prepared(row, config)
    arrays, _ = load_cache(row['cache'], with_labels=False)
    proposals, reports = [], []
    fingerprint = 'geometry'
    if model is not None:
        content = b''.join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
        fingerprint = hashlib.sha256(content).hexdigest()
    output = Path(config['work_dir']) / 'proposals' / fingerprint / row['split'] / row['plot']
    output.mkdir(parents=True, exist_ok=True)
    try:
        for record in prepared['records']:
            if record['reason'] != 'candidate' or (allowed is not None and record['parent_id'] not in allowed):
                continue
            payload = data.load_parent(row, record, arrays)
            indices = payload['indices']
            xyz = arrays['coords'][indices].astype(float)
            def scorer(edges):
                if model is None:
                    return np.ones(len(edges))
                from .model import predict_edges
                return predict_edges(model, arrays, indices, edges, device)
            try:
                mask, report = split_proposal(xyz, indices, [payload['seed0'], payload['seed1']],
                    config['graph'], config['policy'], scorer, geometry_only=model is None)
            except BudgetExceeded as error:
                mask, report = None, dict(reason=str(error))
            report.update(parent_id=record['parent_id'], entrance=record['entrance'])
            if mask is not None:
                proposals.append(dict(parent_id=record['parent_id'], indices=indices[mask],
                                      mean_cut_score=report['mean_cut_score']))
            path = output / f'{record["parent_id"]}.npz'
            np.savez(path, indices=indices, seed0=payload['seed0'], seed1=payload['seed1'],
                     separated=indices[mask] if mask is not None else np.empty(0, dtype=np.int64))
            report.update(payload=str(path), payload_sha256=file_digest(path))
            reports.append(report)
            print(f'PARTITION {row["plot"]}/{record["parent_id"]}: {report["reason"]}', flush=True)
    finally:
        data.close_arrays(arrays)
    write_json(output / 'decisions.json', dict(signature=data.signature(config), fingerprint=fingerprint, reports=reports))
    return proposals, reports


def evaluate_proposals(row, config, proposals, threshold):
    geometry, base, _ = baseline_table(row, config)
    selected = [] if threshold is None else [r for r in proposals if r['mean_cut_score'] <= threshold]
    edits = [(r['parent_id'], r['indices']) for r in selected]
    labels = np.load(Path(row['cache']) / 'labels.npy', mmap_mode='r', allow_pickle=False)
    try:
        result = comparison(geometry, base, split_table(geometry, base, edits, labels))
        result.update(plot=row['plot'], accepted=len(edits), changed_points=sum(len(i) for _, i in edits))
    finally:
        labels._mmap.close()
    return result


def evaluate(config, model=None, device=None, allowed=None, split='validation'):
    collected = []
    for row in [r for r in data.rows(config) if r['split'] == split]:
        parents = allowed.get(row['plot'], set()) if allowed is not None else None
        proposals, reports = forest_proposals(row, config, model, device, parents)
        collected.append((row, proposals, reports))
    choices = []
    for threshold in [None] + config['policy']['cut_thresholds']:
        reports = [evaluate_proposals(row, config, proposals, threshold) for row, proposals, _ in collected]
        gate = advancement(reports)
        choices.append(dict(threshold=threshold, gate=gate, forests=reports,
            changed_points=sum(r['changed_points'] for r in reports), accepted=sum(r['accepted'] for r in reports)))
    chosen = max(choices, key=lambda r: (r['gate']['passed'], r['gate']['utility'], r['gate']['refined_macro']['f1'],
        -r['changed_points'], -(r['threshold'] if r['threshold'] is not None else -1)))
    chosen = dict(chosen, choices=choices,
                  partition_reports={row['plot']: reports for row, _, reports in collected})
    return chosen


def claim_training(directory, steps):
    """An interrupted attempt cannot silently spend the authorized steps again."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'attempt.json'
    if marker.exists() or list(directory.glob('*.pth')) or (directory / 'summary.json').exists():
        raise FileExistsError('Training attempt already exists; preserve it and audit the remaining budget before any restart.')
    write_json(marker, dict(steps=steps, started=True))


def fit(config, overfit=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch.nn import functional as F
    e0 = require_gate(config, 'e0')
    if not overfit:
        require_gate(config, 'e1')
    saved_bank = bank(config, e0)
    records = saved_bank['records']
    allowed = None
    if overfit:
        mixed = [r for r in records if r['overfit_mixed']][:8]
        correct = [r for r in records if r['category'] == 'correct'][:8]
        if len(mixed) < 8 or len(correct) < 8:
            result = dict(signature=data.signature(config), stage='E1',
                gate=dict(passed=False, reason='insufficient_e1_mixed_or_fork_controls', mixed=len(mixed), correct=len(correct)))
            write_json(Path(config['work_dir']) / 'e1/summary.json', result)
            return result
        records = mixed + correct
        allowed = defaultdict(set)
        for record in records:
            allowed[record['plot']].update(record['parents'])
    loaded = []
    for record in records:
        with np.load(record['path'], allow_pickle=False) as z:
            loaded.append(dict(record, raw=z['raw'], target=z['target']))
    pools = defaultdict(lambda: defaultdict(list))
    for record in loaded:
        pools[record['category']][record['plot']].append(record)
    categories = ['mixed', 'mixed', 'correct', 'correct' if overfit else 'synthetic']
    if any(not pools[k] for k in categories):
        raise ValueError('Missing training category; do not silently change the sample mixture.')
    model, device = setup_model(config, saved_bank['normalization'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    directory = Path(config['work_dir']) / ('e1' if overfit else 'e2')
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'summary.json'
    if result_path.exists():
        raise FileExistsError('Completed training exists; never silently spend the budget again.')
    steps = config['training']['overfit_steps' if overfit else 'steps']
    claim_training(directory, steps)
    history, best, rng = [], None, np.random.default_rng(config['seed'])
    for step in range(1, steps + 1):
        raw, target = [], []
        for category in categories:
            plots = sorted(pools[category])
            plot = plots[int(rng.integers(len(plots)))]
            record = pools[category][plot][int(rng.integers(len(pools[category][plot])))]
            at = rng.integers(len(record['target']), size=config['training']['edges_per_step'] // 4)
            raw.append(record['raw'][at]); target.append(record['target'][at])
        raw = torch.as_tensor(np.concatenate(raw), device=device)
        target = torch.as_tensor(np.concatenate(target), device=device)
        model.train()
        logits = model(raw)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        loss = torch.stack([losses[target == label].mean() for label in (0, 1) if (target == label).any()]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % config['training']['log_every'] == 0:
            print(f'TRAIN {directory.name} step={step}/{steps} loss={float(loss):.6f}', flush=True)
        if step == steps or (not overfit and step % config['training']['validation_every'] == 0):
            checkpoint = directory / f'step_{step}.pth'
            torch.save(dict(state_dict=model.state_dict(), signature=data.signature(config), step=step,
                normalization=saved_bank['normalization'], optimizer=optimizer.state_dict(),
                rng_numpy=rng.bit_generator.state, rng_torch=torch.get_rng_state()), checkpoint)
            validation = evaluate(config, model, device, allowed, 'train' if overfit else 'validation')
            validation.update(step=step, checkpoint=str(checkpoint), checkpoint_sha256=file_digest(checkpoint))
            history.append(validation)
            rank = lambda r: (r['gate']['passed'], r['gate']['utility'], r['gate']['refined_macro']['f1'], -r['changed_points'], -r['step'])
            if best is None or rank(validation) > rank(best):
                best = validation
            print(f'VALIDATE {directory.name} step={step} gate={validation["gate"]}', flush=True)
    if overfit:
        rates = full_edge_rates(config, model, device, records)
        # All full-forest lost IDs are checked, stricter than just the eight controls.
        gate = dict(passed=rates['same_false_cut_rate'] <= .05 and rates['different_detection_rate'] >= .8
            and best['gate']['recovered'] >= 1 and best['gate']['lost'] == 0,
            **rates, recovered=best['gate']['recovered'], lost=best['gate']['lost'])
        result = dict(signature=data.signature(config), stage='E1', gate=gate, selected=best,
                      parent_records=[{k: r[k] for k in ('plot', 'category', 'parents')} for r in records])
    else:
        control = evaluate(config)
        result = dict(signature=data.signature(config), stage='E2', gate=best['gate'], selected=best,
                      geometry_control=control, history=history)
    write_json(result_path, result)
    print('TRAINING_GATE ' + str(result['gate']), flush=True)
    return result


def full_edge_rates(config, model, device, records):
    from .model import predict_edges
    totals = dict(same=0, different=0, same_cut=0, different_cut=0)
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        selected = [r for r in records if r['plot'] == row['plot']]
        if not selected:
            continue
        arrays, _ = load_cache(row['cache'], with_labels=True)
        try:
            ids, starts, counts, order = data.source_groups(arrays['baseline'])
            cores = {int(i): order[s:s + c] for i, s, c in zip(ids, starts, counts)}
            for record in selected:
                indices = cores[record['parents'][0]]
                edges = graph_edges(arrays['coords'][indices].astype(float), config['graph'])
                valid, target = edge_targets(arrays['labels'][indices], edges)
                predicted = predict_edges(model, arrays, indices, edges, device)
                for name, label in [('same', 1), ('different', 0)]:
                    at = valid & (target == label)
                    totals[name] += int(at.sum())
                    totals[name + '_cut'] += int(np.count_nonzero(at & (predicted < .5)))
        finally:
            data.close_arrays(arrays)
    return dict(totals, same_false_cut_rate=totals['same_cut'] / max(totals['same'], 1),
                different_detection_rate=totals['different_cut'] / max(totals['different'], 1))


def validate_saved(config):
    require_gate(config, 'e1')
    saved = data.read_json(Path(config['work_dir']) / 'e2/summary.json')
    chosen = saved['selected']
    if saved['signature'] != data.signature(config) or file_digest(chosen['checkpoint']) != chosen['checkpoint_sha256']:
        raise ValueError('Changed locked checkpoint.')
    norm = data.read_json(Path(config['work_dir']) / 'banks/bank.json')['normalization']
    model, device = setup_model(config, norm, chosen['checkpoint'])
    result = evaluate(config, model, device)
    if result['threshold'] != chosen['threshold'] or result['gate'] != chosen['gate']:
        raise ValueError('Locked source selection did not reproduce.')
    write_json(Path(config['work_dir']) / 'e2/reproduced_validation.json', result)
    return result


def external(config):
    locked = require_gate(config, 'e2')
    output = Path(config['work_dir']) / 'e3'
    if output.exists():
        raise FileExistsError('Wytham exploration already started; inspect its saved outputs, do not rerun/tune.')
    output.mkdir(parents=True)
    chosen = locked['selected']
    if file_digest(chosen['checkpoint']) != chosen['checkpoint_sha256']:
        raise ValueError('Locked checkpoint changed.')
    write_json(output / 'lock.json', dict(signature=data.signature(config), selected=chosen,
        geometry_control_threshold=locked['geometry_control']['threshold'], exploratory=True))
    model, device = setup_model(config, data.read_json(Path(config['work_dir']) / 'banks/bank.json')['normalization'], chosen['checkpoint'])
    reports = []
    for row in data.rows(config, external=True):
        data.prepare_forest(row, config)
        geometry, base, before = baseline_table(row, config)
        if [before['metrics'][k] for k in ('tp', 'fp', 'fn')] != config['expected_wytham_baseline_counts']:
            raise ValueError('Wytham official baseline changed.')
        for name, current, threshold in [('learned', model, chosen['threshold']),
            ('geometry', None, locked['geometry_control']['threshold'])]:
            proposals, partition_reports = forest_proposals(row, config, current, device)
            result = evaluate_proposals(row, config, proposals, threshold)
            edits = [] if threshold is None else [(r['parent_id'], r['indices']) for r in proposals if r['mean_cut_score'] <= threshold]
            arrays, _ = load_cache(row['cache'], with_labels=False)
            try:
                predictions = data.apply_edits(arrays['baseline'], edits)
                np.save(output / f'{name}_predictions.npy', predictions, allow_pickle=False)
                for parent, indices in edits:
                    np.save(output / f'{name}_parent_{parent}_separated.npy', indices, allow_pickle=False)
            finally:
                data.close_arrays(arrays)
            result.update(method=name, partition_reports=partition_reports, threshold=threshold)
            reports.append(result)
            write_json(output / f'{name}.json', result)
    result = dict(signature=data.signature(config), exploratory=True, results=reports)
    write_json(output / 'summary.json', result)
    return result
