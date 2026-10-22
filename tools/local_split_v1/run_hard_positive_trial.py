"""One fresh 500-step E1-R1 trial; preserve V1 signatures and stop after E1.

The sole training change is an extra mean BCE over the hardest quarter of
positive edges. Source banks, normalization, sampling and deployment functions
are reused verbatim. This entry point has no E2 or external stage.
"""

import argparse
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from tree_learn.local_split_v1 import data, learning
from tree_learn.local_split_v1.audit import advancement
from tree_learn.local_split_v1.geometry import BudgetExceeded, graph_edges, split_proposal
from tree_learn.local_split_v1.model import predict_edges
from tree_learn.instance_relation.data import stamp
from tree_learn.mask_refinement.data import digest, file_digest, load_cache, write_json
from tree_learn.mask_refinement.evaluation import evaluate_labels


def hard_positive_loss(logits, targets, fraction=.25, weight=.5):
    """Class-balanced BCE plus top-quarter same-positive-GT BCE, no GT remapping."""
    if logits.shape != targets.shape or logits.ndim != 1 or not len(logits):
        raise ValueError('Require aligned nonempty logits and targets.')
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError('Mask unknown and 0/0 pairs before loss construction.')
    if not 0 < fraction <= 1 or weight < 0:
        raise ValueError('Invalid hard-positive loss settings.')
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    positive, negative = losses[targets == 1], losses[targets == 0]
    if not len(positive) or not len(negative):
        raise ValueError('Locked mixed/correct batches must contain both classes.')
    count = max(1, math.ceil(len(positive) * fraction))
    hard = positive.topk(count).values.mean()
    balanced = .5 * (positive.mean() + negative.mean())
    return balanced + weight * hard, dict(balanced=balanced, hard_positive=hard,
        positive=positive.mean(), negative=negative.mean(), hard_count=count,
        positive_count=len(positive), negative_count=len(negative))


def e1_gate(rates, recovered, lost, trial):
    same, other = rates['same'], rates['other_tree']
    same_rate = same['cut'] / same['count'] if same['count'] else None
    other_rate = other['cut'] / other['count'] if other['count'] else None
    passed = (same_rate is not None and other_rate is not None and
        same_rate <= trial['max_same_false_cut_rate'] and
        other_rate >= trial['min_other_tree_detection_rate'] and
        recovered >= trial['min_recovered'] and lost <= trial['max_lost'])
    return dict(passed=bool(passed), same_false_cut_rate=same_rate,
        strict_other_tree_detection_rate=other_rate, recovered=recovered, lost=lost)


def preflight(trial_path):
    trial = yaml.safe_load(Path(trial_path).read_text())
    locked = dict(schema=1, trial='E1-R1', optimizer_steps=500,
        hard_positive_fraction=.25, hard_positive_weight=.5, edge_threshold=.5,
        max_same_false_cut_rate=.05, min_other_tree_detection_rate=.8,
        min_recovered=1, max_lost=0)
    if any(trial.get(k) != v for k, v in locked.items()):
        raise ValueError('This trial permits exactly the single preregistered setting.')
    config = data.load_config(trial['source_config'])
    if data.signature(config) != trial['source_signature']:
        raise ValueError('Original V1 code/config changed.')
    output = Path(trial['output']).resolve()
    for source in [config[k] for k in ('work_dir', 'prepared_root', 'cache_root',
                                      'external_cache_root', 'base_prepared_root')]:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('R1 output must be separate from immutable V1 inputs.')
    data.checkpoint_md5(config)
    e0 = learning.require_gate(config, 'e0')
    e1 = data.read_json(Path(config['work_dir']) / 'e1/summary.json')
    if e1['signature'] != trial['source_signature']:
        raise ValueError('Stale original E1.')
    checkpoint = e1['selected']['checkpoint']
    if file_digest(checkpoint) != trial['baseline_checkpoint_sha256']:
        raise ValueError('Changed original E1 checkpoint.')
    if not (Path(config['work_dir']) / 'banks/bank.json').exists():
        raise FileNotFoundError('R1 requires the original frozen banks; rebuilding is forbidden.')
    bank = learning.bank(config, e0)  # verify every existing bank, never rebuild
    proposal = data.read_json(trial['proposal'])
    if proposal['source_signature'] != trial['source_signature'] or proposal['parent_records'] != e1['parent_records']:
        raise ValueError('Proposal differs from the locked E1 parents.')
    by_key = {(r['plot'], r['category'], tuple(r['parents'])): r for r in bank['records']}
    records = [by_key[r['plot'], r['category'], tuple(r['parents'])] for r in e1['parent_records']]
    if [r['category'] for r in records] != ['mixed'] * 8 + ['correct'] * 8:
        raise ValueError('Require the identical eight mixed and eight fork controls.')
    label_sources = {r['plot']: stamp(Path(r['cache']) / 'labels.npy') for r in data.rows(config)}
    if label_sources != {f['plot']: f['label_source'] for f in e0['forests']}:
        raise ValueError('GT sources differ from the original E0 audit.')
    sources = dict(source_signature=trial['source_signature'],
        e0_sha256=file_digest(Path(config['work_dir']) / 'e0/summary.json'),
        e1_sha256=file_digest(Path(config['work_dir']) / 'e1/summary.json'),
        bank_manifest_sha256=file_digest(Path(config['work_dir']) / 'banks/bank.json'),
        baseline_checkpoint_sha256=file_digest(checkpoint), label_sources=label_sources)
    signature = digest(dict(trial=trial, source=sources, code_sha256=file_digest(__file__),
                            proposal_sha256=file_digest(trial['proposal'])))
    print(f'PREFLIGHT signature={signature} original={sources["source_signature"]} parents={len(records)}', flush=True)
    return trial, config, e1, bank, records, dict(signature=signature, sources=sources,
        trial=trial, code_sha256=file_digest(__file__), proposal_sha256=file_digest(trial['proposal']))


def train(trial, config, bank, records, manifest):
    directory = Path(trial['output'])
    # Reserve the whole one-attempt budget before optimization. No silent retry.
    learning.claim_training(directory, trial['optimizer_steps'])
    write_json(directory / 'manifest.json', manifest)
    loaded = []
    for record in records:
        with np.load(record['path'], allow_pickle=False) as z:
            loaded.append(dict(record, raw=z['raw'], target=z['target']))
    pools = defaultdict(lambda: defaultdict(list))
    for record in loaded:
        pools[record['category']][record['plot']].append(record)
    model, device = learning.setup_model(config, bank['normalization'])
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 5265:
        raise ValueError('Changed model architecture.')
    initial_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                               for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    rng = np.random.default_rng(config['seed'])
    history = []
    for step in range(1, trial['optimizer_steps'] + 1):
        raw, target = [], []
        # Identical sequence and sampling to V1 E1: mixed/mixed/correct/correct.
        for category in ('mixed', 'mixed', 'correct', 'correct'):
            plots = sorted(pools[category])
            plot = plots[int(rng.integers(len(plots)))]
            record = pools[category][plot][int(rng.integers(len(pools[category][plot])))]
            at = rng.integers(len(record['target']), size=config['training']['edges_per_step'] // 4)
            raw.append(record['raw'][at]); target.append(record['target'][at])
        raw = torch.as_tensor(np.concatenate(raw), device=device)
        target = torch.as_tensor(np.concatenate(target), device=device)
        model.train()
        loss, parts = hard_positive_loss(model(raw), target,
            trial['hard_positive_fraction'], trial['hard_positive_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss; preserve this spent attempt.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % config['training']['log_every'] == 0:
            row = dict(step=step, loss=float(loss.detach()), **{k: float(v.detach()) if torch.is_tensor(v) else v
                                                              for k, v in parts.items()})
            history.append(row)
            write_json(directory / 'history.json', dict(signature=manifest['signature'], history=history))
            print('TRAIN E1-R1 ' + str(row), flush=True)
    checkpoint = directory / 'step_500.pth'
    torch.save(dict(state_dict=model.state_dict(), signature=manifest['signature'],
        source_signature=manifest['sources']['source_signature'], step=trial['optimizer_steps'],
        normalization=bank['normalization'], optimizer=optimizer.state_dict(),
        rng_numpy=rng.bit_generator.state, rng_torch=torch.get_rng_state(),
        initial_state_sha256=initial_sha), checkpoint)
    write_json(directory / 'trained.json', dict(signature=manifest['signature'],
        optimizer_steps=trial['optimizer_steps'], checkpoint=str(checkpoint),
        checkpoint_sha256=file_digest(checkpoint), parameter_count=parameters,
        initial_state_sha256=initial_sha, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None))
    print(f'TRAINED optimizer_steps=500 checkpoint_sha256={file_digest(checkpoint)}', flush=True)


def edge_audit(config, model, device, records):
    totals = {name: dict(count=0, cut=0) for name in ('same', 'other_tree', 'tree_background')}
    per_parent = []
    bins = np.linspace(0, 1, 41)
    hist = {k: np.zeros(40, dtype=np.int64) for k in totals}
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
                scores = predict_edges(model, arrays, indices, edges, device)
                a, b = arrays['labels'][indices[edges[:, 0]]], arrays['labels'][indices[edges[:, 1]]]
                masks = dict(same=(a == b) & (a > 0), other_tree=(a > 0) & (b > 0) & (a != b),
                    tree_background=((a == 0) & (b > 0)) | ((a > 0) & (b == 0)))
                result = dict(plot=row['plot'], parent_id=record['parents'][0], category=record['category'])
                for name, mask in masks.items():
                    counts = dict(count=int(mask.sum()), cut=int(np.count_nonzero(mask & (scores < .5))))
                    result[name] = counts
                    for key in counts:
                        totals[name][key] += counts[key]
                    hist[name] += np.histogram(scores[mask], bins=bins)[0]
                per_parent.append(result)
                print('FULL_EDGE ' + str(result), flush=True)
        finally:
            data.close_arrays(arrays)
    return dict(totals=totals, per_parent=per_parent,
                score_histogram=dict(bins=bins.tolist(), **{k: v.tolist() for k, v in hist.items()}))


def proposals(row, config, allowed, output, model, device):
    """Same V1 partition; explicit output destination avoids changing its signature."""
    prepared = data.load_prepared(row, config)
    arrays, _ = load_cache(row['cache'], with_labels=False)
    accepted, reports = [], []
    directory = output / row['plot']
    directory.mkdir(parents=True)
    try:
        for record in prepared['records']:
            if record['parent_id'] not in allowed:
                continue
            if record['reason'] != 'candidate':
                reports.append(dict(parent_id=record['parent_id'], reason=record['reason'], candidate=False))
                continue
            payload = data.load_parent(row, record, arrays)
            indices = payload['indices']
            def scorer(edges):
                return np.ones(len(edges)) if model is None else predict_edges(model, arrays, indices, edges, device)
            try:
                mask, report = split_proposal(arrays['coords'][indices].astype(float), indices,
                    [payload['seed0'], payload['seed1']], config['graph'], config['policy'], scorer,
                    geometry_only=model is None)
            except BudgetExceeded as error:
                mask, report = None, dict(reason=str(error))
            separated = indices[mask] if mask is not None else np.empty(0, dtype=np.int64)
            path = directory / f'{record["parent_id"]}.npz'
            np.savez(path, indices=indices, seed0=payload['seed0'], seed1=payload['seed1'], separated=separated)
            report.update(parent_id=record['parent_id'], entrance=record['entrance'], candidate=True,
                          payload=str(path), payload_sha256=file_digest(path))
            if mask is not None:
                accepted.append(dict(parent_id=record['parent_id'], indices=separated,
                                     mean_cut_score=report['mean_cut_score']))
            reports.append(report)
            print(f'PARTITION {output.name} {row["plot"]}/{record["parent_id"]}: {report["reason"]}', flush=True)
    finally:
        data.close_arrays(arrays)
    write_json(directory / 'decisions.json', dict(gt_free=True, reports=reports))
    return accepted, reports


def evaluate(config, records, output, model, device):
    allowed = defaultdict(set)
    for record in records:
        allowed[record['plot']].update(record['parents'])
    collected = []
    for row in [r for r in data.rows(config) if r['split'] == 'train']:
        current, reports = proposals(row, config, allowed[row['plot']], output, model, device)
        collected.append((row, current, reports))
    choices = []
    for threshold in [None] + config['policy']['cut_thresholds']:
        forests = [learning.evaluate_proposals(row, config, current, threshold) for row, current, _ in collected]
        gate = advancement(forests)
        choices.append(dict(threshold=threshold, gate=gate, forests=forests,
            accepted=sum(r['accepted'] for r in forests), changed_points=sum(r['changed_points'] for r in forests)))
    # Preserve the V1 tie ordering; this fitted E1 result never authorizes E2.
    selected = max(choices, key=lambda r: (r['gate']['passed'], r['gate']['utility'],
        r['gate']['refined_macro']['f1'], -r['changed_points'],
        -(r['threshold'] if r['threshold'] is not None else -1)))
    return dict(selected=selected, choices=choices,
                partition_reports={row['plot']: reports for row, _, reports in collected}), collected


def verify_point_evaluation(config, evaluation, collected):
    """Independently evaluate actual complete forest labels for every edited forest."""
    threshold, checks = evaluation['selected']['threshold'], []
    expected = {r['plot']: r for r in evaluation['selected']['forests']}
    for row, current, _ in collected:
        edits = [] if threshold is None else [(r['parent_id'], r['indices']) for r in current if r['mean_cut_score'] <= threshold]
        if not edits:
            continue
        arrays, _ = load_cache(row['cache'], with_labels=True)
        try:
            predicted = data.apply_edits(arrays['baseline'], edits)
            changed = np.flatnonzero(predicted != arrays['baseline'])
            wanted = np.sort(np.concatenate([indices for _, indices in edits]))
            np.testing.assert_array_equal(changed, wanted)
            # apply_edits also checks that foreground and all original points survive.
            before = evaluate_labels(arrays['labels'], arrays['baseline'])
            after = evaluate_labels(arrays['labels'], predicted)
            for method, direct in [('baseline', before), ('refined', after)]:
                for key in ('tp', 'fp', 'fn', 'f1', 'coverage', 'point_precision', 'point_recall'):
                    if not np.isclose(direct[key], expected[row['plot']][method][key], atol=1e-10, rtol=0):
                        raise AssertionError(f'Incremental versus point evaluation differs: {row["plot"]}/{method}/{key}')
            recovered = sorted(set(after['matched_gt_ids']) - set(before['matched_gt_ids']))
            lost = sorted(set(before['matched_gt_ids']) - set(after['matched_gt_ids']))
            if recovered != expected[row['plot']]['recovered_gt_ids'] or lost != expected[row['plot']]['lost_gt_ids']:
                raise AssertionError('Point-level full forest recovered/lost IDs differ.')
            checks.append(dict(plot=row['plot'], points=len(predicted), changed_points=len(changed),
                foreground_unchanged=True, all_unedited_labels_equal=True, exact_metrics=True,
                recovered_gt_ids=recovered, lost_gt_ids=lost))
            print(f'POINT_VERIFY {row["plot"]}: {checks[-1]}', flush=True)
        finally:
            data.close_arrays(arrays)
    return checks


def finish(trial, config, e1, bank, records, manifest):
    directory = Path(trial['output'])
    if (directory / 'audit').exists():
        raise FileExistsError('Evaluation already started; inspect preserved outputs instead of overwriting.')
    if data.read_json(directory / 'manifest.json') != manifest:
        raise ValueError('Training provenance changed.')
    trained = data.read_json(directory / 'trained.json')
    if file_digest(trained['checkpoint']) != trained['checkpoint_sha256']:
        raise ValueError('R1 weights changed.')
    model, device = learning.setup_model(config, bank['normalization'])
    saved = torch.load(trained['checkpoint'], map_location=device, weights_only=False)
    if saved['signature'] != manifest['signature'] or saved['step'] != 500 or saved['normalization'] != bank['normalization']:
        raise ValueError('Invalid R1 checkpoint or normalization.')
    model.load_state_dict(saved['state_dict'])
    audit = directory / 'audit'
    audit.mkdir()
    rates = edge_audit(config, model, device, records)
    # The strict edge populations must exactly match the previously audited V1.
    if {k: v['count'] for k, v in rates['totals'].items()} != dict(same=5003262, other_tree=17702, tree_background=7878):
        raise ValueError('Complete E1 edge populations changed.')
    write_json(audit / 'edge_rates.json', rates)
    learned, collected = evaluate(config, records, directory / 'deployment/learned', model, device)
    write_json(audit / 'learned.json', learned)
    learned_checks = verify_point_evaluation(config, learned, collected)
    geometry, geometric_collected = evaluate(config, records, directory / 'deployment/geometry', None, device)
    write_json(audit / 'geometry.json', geometry)
    geometry_checks = verify_point_evaluation(config, geometry, geometric_collected)
    gate = e1_gate(rates['totals'], learned['selected']['gate']['recovered'], learned['selected']['gate']['lost'], trial)
    result = dict(schema=1, stage='E1-R1', signature=manifest['signature'], source_signature=trial['source_signature'],
        optimizer_steps=500, gate=gate, eligible_for_advancement=False,
        scope='GT-selected fitted source training parents; not independent validation',
        checkpoint=trained, edge_rates=rates['totals'], learned=learned, geometry=geometry,
        point_verification=dict(learned=learned_checks, geometry=geometry_checks),
        baseline_e1=dict(checkpoint_sha256=e1['selected']['checkpoint_sha256'], gate=e1['gate'], selected=e1['selected']),
        parent_records=e1['parent_records'], external_executed=False, e2_executed=False)
    write_json(audit / 'summary.json', result)
    print('R1_GATE ' + str(gate), flush=True)
    print('STOP single-setting R1 complete; no E2 or Wytham stage in this entry point.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/e1_r1.yaml')
    parser.add_argument('--stage', choices=['run', 'evaluate'], default='run')
    args = parser.parse_args()
    trial, config, e1, bank, records, manifest = preflight(args.config)
    if args.stage == 'run':
        train(trial, config, bank, records, manifest)
    finish(trial, config, e1, bank, records, manifest)
    _, _, _, _, _, after = preflight(args.config)
    if after != manifest:
        raise ValueError('Immutable inputs changed during the trial.')
    write_json(Path(trial['output']) / 'verified.json', dict(signature=manifest['signature'], immutable_inputs_verified=True))
    print('VERIFIED original V1 caches, labels, banks, code and checkpoint unchanged.', flush=True)


if __name__ == '__main__':
    main()
