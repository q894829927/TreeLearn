"""Server-only A1 workflow: prepare, audit, train, evaluate.

Run from the repository root. Importing this driver does not start an experiment.
"""

import argparse
import copy
import csv
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from tree_learn.mask_refinement.data import (
    ForestLRU, digest, file_digest, load_cache, load_candidates, point_inputs,
    prepare_regions, stratified_sample, write_json,
)
from tree_learn.mask_refinement.evaluation import (
    audit_forest, compare_metrics, evaluate_labels, macro_metrics,
)


def validate_settings(settings):
    splits = settings['splits']
    if set(splits) != {'train', 'validation'}:
        raise ValueError('Exactly train and validation splits are required.')
    groups = {}
    for split, names in splits.items():
        if not names or len(names) != len(set(names)):
            raise ValueError('Empty or duplicate forest split.')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
            raise ValueError('Forest names must be safe file stems.')
        groups[split] = {
            re.sub(r'^(A1|G[1-4]|L[12]|O1)[NW]$', r'\1', name)
            for name in names}
    if groups['train'] & groups['validation']:
        raise ValueError('Forest groups leak across train and validation.')
    all_plots = set(splits['train'] + splits['validation'])
    if not set(settings['pilot_plots']) <= all_plots:
        raise ValueError('Unknown pilot plot.')
    for split in splits:
        if not set(splits[split]) & set(settings['pilot_plots']):
            raise ValueError('Pilot requires both train and validation.')
    region = settings['regions']
    if region['margin_xy_m'] < 0 or region['minimum_scale_m'] <= 0 or region['max_region_points'] < 1:
        raise ValueError('Invalid region geometry settings.')
    training = settings['training']
    for key in ('hidden_dim', 'epochs', 'batch_size', 'statistics_points_per_forest'):
        if int(training[key]) < 1:
            raise ValueError(f'{key} must be positive.')
    if int(training['points_per_candidate']) < 3:
        raise ValueError('At least three sampled points are required.')
    for key, value in settings['supervision'].items():
        if not 0 <= float(value) <= 1:
            raise ValueError(f'Invalid supervision fraction: {key}')
    if not 0 < settings['inference']['mask_threshold'] < 1:
        raise ValueError('Invalid membership threshold.')


def selected_plots(settings, pilot):
    return [(split, plot) for split, plots in settings['splits'].items()
            for plot in plots if not pilot or plot in settings['pilot_plots']]


def manifest_path(settings, pilot):
    return Path(settings['output_root']) / ('pilot_manifest.json' if pilot else 'manifest.json')


def read_manifest(settings, pilot):
    manifest = json.loads(manifest_path(settings, pilot).read_text(encoding='utf-8'))
    actual = [(row['split'], row['plot']) for row in manifest['forests']]
    if actual != selected_plots(settings, pilot) or manifest['pilot'] != pilot:
        raise ValueError('Manifest does not match the requested forest split.')
    if manifest['regions'] != settings['regions']:
        raise ValueError('Region configuration differs from cached geometry.')
    if manifest['baseline_request'] != {
            key: settings[key] for key in ('checkpoint', 'pipeline_template')}:
        raise ValueError('Baseline request changed; run prepare with a new output_root.')
    for entry in manifest['forests']:
        directory = Path(settings['output_root']) / entry['cache']
        _, metadata = load_cache(directory)
        if metadata['signature'] != entry['signature']:
            raise ValueError(f'Cache provenance changed: {entry["plot"]}')
        region = json.loads((directory / 'regions.json').read_text(encoding='utf-8'))
        if region['settings_digest'] != digest(settings['regions']):
            raise ValueError('Cached region settings differ.')
    return manifest


def prepare(settings, pilot):
    # The existing pipeline imports spconv; it is only loaded in this command.
    from tree_learn.util import get_config, munch_to_dict
    output = Path(settings['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(settings['checkpoint']).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Official checkpoint missing: {checkpoint}')
    base = munch_to_dict(get_config(settings['pipeline_template']))
    base['pretrain'] = str(checkpoint)
    assert_original_baseline(base)
    baseline_files = (
        'tools/pipeline/pipeline.py', 'tree_learn/model/tree_learn.py',
        'tree_learn/model/blocks.py', 'tree_learn/dataset/dataset.py',
        'tree_learn/util/pipeline.py', 'tree_learn/util/data_preparation.py')
    baseline_signature = digest({
        'checkpoint_sha256': file_digest(checkpoint), 'pipeline': base,
        'source_code': {path: file_digest(path) for path in baseline_files}})
    entries = []
    for split, plot in selected_plots(settings, pilot):
        matches = sorted(path for path in Path(settings['source_root']).glob(f'{plot}.*')
                         if path.is_file() and path.suffix.lower() in
                         {'.las', '.laz', '.npy', '.npz', '.txt'})
        if len(matches) != 1:
            raise ValueError(f'Expected one source forest for {plot}: {matches}')
        source = matches[0].resolve()
        signature = digest({'baseline': baseline_signature,
                            'source_sha256': file_digest(source), 'plot': plot})
        directory = output / split / plot
        metadata = dict(signature=signature, baseline_signature=baseline_signature,
                        plot=plot, split=split, source=str(source),
                        checkpoint=str(checkpoint))
        if (directory / 'cache.json').is_file():
            _, previous = load_cache(directory, with_labels=True)
            if previous['signature'] != signature:
                raise ValueError(f'Stale cache for {plot}; choose a new output_root.')
            print(f'REUSE {plot}: provenance checked.', flush=True)
        else:
            payload = copy.deepcopy(base)
            payload.update(forest_path=str(source),
                           pipeline_base_dir=str((output / 'runtime' / plot).resolve()),
                           mask_refinement_cache_dir=str(directory.resolve()),
                           mask_refinement_cache_metadata=metadata)
            config_path = output / 'resolved_configs' / f'{plot}.yaml'
            config_path.parent.mkdir(exist_ok=True)
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
            log_path = output / 'logs' / f'{plot}.log'
            log_path.parent.mkdir(exist_ok=True)
            print(f'RUN {plot}: {log_path}', flush=True)
            with log_path.open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', 'tools/pipeline/pipeline.py',
                                '--config', str(config_path)], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            load_cache(directory, with_labels=True)
        prepare_regions(directory, settings['regions'])
        entries.append(dict(plot=plot, split=split,
                            cache=f'{split}/{plot}', signature=signature))
        print(f'READY {plot}', flush=True)
    manifest = dict(schema=1, pilot=pilot, baseline_signature=baseline_signature,
                    baseline_request={key: settings[key] for key in ('checkpoint', 'pipeline_template')},
                    regions=settings['regions'], forests=entries)
    write_json(manifest_path(settings, pilot), manifest)
    print(f'SAVED {manifest_path(settings, pilot)}', flush=True)


def assert_original_baseline(base):
    model, grouping = base['model'], base['grouping']
    if any(model.get(key, False) for key in (
            'use_upper_anchor', 'use_axis_branch', 'use_height_context_adapter')):
        raise ValueError('A1 requires original heads only.')
    if any(model.get(key, {}).get('enabled', False) for key in (
            'unet_enhancement', 'unet_finetune')):
        raise ValueError('A1 requires an unmodified frozen backbone.')
    if (base['dataset_test']['base_anchor_mode'] != 'legacy' or
            any(grouping.get(key, False) for key in (
                'use_upper_seeds', 'use_axis_fusion', 'use_seed_confidence_filter')) or
            grouping['upper_anchor_weight'] != 0 or grouping['axis_height_weight'] != 0):
        raise ValueError('A1 requires original single-anchor grouping.')
    if model['offset_loss_type'] != 'l2' or model['channels'] != 32:
        raise ValueError('A1 baseline requires original L2 loss setting and 32 channels.')
    if base.get('quality_filter', {}).get('enabled', False):
        raise ValueError('Quality filtering must be disabled.')
    if base['shape_cfg']['outer_remove'] is not None:
        raise ValueError('A1 aligned-cache evaluation requires no outer crop.')
    if not base['save_cfg'].get('mask_refinement_export_only', False):
        raise ValueError('A1 requires the cache-only export path.')


def audit_path(settings, pilot):
    return Path(settings['output_root']) / ('pilot_audit.json' if pilot else 'audit.json')


def audit_signature(settings, manifest):
    return digest({'manifest': manifest, 'supervision': settings['supervision'],
                   'gate': settings['audit_gate']})


def audit(settings, pilot):
    manifest = read_manifest(settings, pilot)
    records = []
    for entry in manifest['forests']:
        row = audit_forest(Path(settings['output_root']) / entry['cache'],
                           settings['supervision'])
        records.append(dict(row, plot=entry['plot'], split=entry['split']))
        print(f'{entry["plot"]}: trainable={row["num_trainable"]}, '
              f'oracle Coverage gain={100 * row["oracle_coverage_gain"]:.3f} pp', flush=True)
    train = [row for row in records if row['split'] == 'train']
    validation = [row for row in records if row['split'] == 'validation']
    gate = settings['audit_gate']
    checks = dict(
        training_positives=sum(row['num_positive'] for row in train) >=
        (1 if pilot else gate['min_train_positive']),
        validation_positives=sum(row['num_positive'] for row in validation) >=
        (1 if pilot else gate['min_validation_positive']))
    if not pilot:
        checks['oracle_headroom'] = float(np.mean([
            row['oracle_coverage_gain'] for row in validation])) >= gate['min_macro_oracle_coverage_gain']
    result = dict(signature=audit_signature(settings, manifest), pilot=pilot,
                  forests=records, checks=checks, passed=all(checks.values()))
    write_json(audit_path(settings, pilot), result)
    print(f'AUDIT passed={result["passed"]}: {audit_path(settings, pilot)}', flush=True)
    return result['passed']


class RegionDataset:
    def __init__(self, records, size, seed):
        self.records, self.size, self.seed = records, size, seed
        self.epoch = 0
        self.forests = ForestLRU()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, row, target, plot = self.records[index]
        arrays, candidates = self.forests.get(path)
        candidate_id = int(candidates['ids'][row])
        region = np.load(Path(path) / 'regions' / f'{candidate_id}.npy',
                         mmap_mode='r', allow_pickle=False)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        selected, weights = stratified_sample(region, arrays['labels'], target, self.size, rng)
        labels = ((arrays['labels'][selected] == target) & (target > 0)).astype(np.float32)
        return (point_inputs(arrays, candidates, row, selected),
                candidates['descriptors'][row], labels, weights, plot)


def training_records(settings, manifest, smoke):
    records = {'train': [], 'validation': []}
    for entry in manifest['forests']:
        path = Path(settings['output_root']) / entry['cache']
        forest_audit = json.loads((path / 'audit.json').read_text(encoding='utf-8'))
        if forest_audit['supervision_settings_digest'] != digest(settings['supervision']):
            raise ValueError('Per-forest supervision changed; rerun the requested audit.')
        candidates = load_candidates(path)
        with np.load(path / 'supervision.npz', allow_pickle=False) as supervision:
            if not np.array_equal(candidates['ids'], supervision['ids']):
                raise ValueError('Supervision candidate IDs do not align.')
            indices = np.flatnonzero(supervision['valid'])
            if smoke:
                indices = indices[:16]
            records[entry['split']].extend((str(path), int(row),
                                            int(supervision['target_ids'][row]), entry['plot'])
                                           for row in indices)
    if any(not value for value in records.values()):
        raise ValueError('Training and validation need supervised candidates.')
    return records


def training_statistics(settings, manifest):
    rng = np.random.default_rng(settings['seed'])
    samples = []
    for entry in manifest['forests']:
        if entry['split'] != 'train':
            continue
        arrays, _ = load_cache(Path(settings['output_root']) / entry['cache'])
        count = min(len(arrays['features']), settings['training']['statistics_points_per_forest'])
        indices = rng.choice(len(arrays['features']), count, replace=False)
        samples.append(arrays['features'][indices])
    values = np.concatenate(samples).astype(np.float64)
    return values.mean(axis=0).astype(np.float32), np.maximum(values.std(axis=0), 1e-3).astype(np.float32)


def experiment_signature(settings, manifest):
    return digest(dict(audit=audit_signature(settings, manifest),
                       training=settings['training'], seed=settings['seed'],
                       threshold=settings['inference']['mask_threshold'],
                       evaluation_gate=settings['evaluation_gate']))


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def train(settings, pilot, smoke, resume):
    import torch
    from torch.utils.data import DataLoader
    from tree_learn.mask_refinement.model import CandidateMaskMLP, mask_loss
    if pilot and not smoke:
        raise ValueError('Pilot data are for --smoke only; full training requires 18 forests.')
    manifest = read_manifest(settings, pilot)
    report = json.loads(audit_path(settings, pilot).read_text(encoding='utf-8'))
    if report['signature'] != audit_signature(settings, manifest) or not report['passed']:
        raise ValueError('A current passing audit is required before training.')
    cfg = settings['training']
    signature = experiment_signature(settings, manifest)
    work = Path(settings['work_dir']) / 'smoke' if smoke else Path(settings['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'last.pth').exists() and not resume:
        raise FileExistsError(f'{work}/last.pth exists; use --resume or a new work_dir.')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use the server training environment.')
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(settings['seed'])
        torch.cuda.reset_peak_memory_stats(device)
    model = CandidateMaskMLP(cfg['hidden_dim']).to(device)
    mean, scale = training_statistics(settings, manifest)
    model.feature_mean.copy_(torch.from_numpy(mean).to(device))
    model.feature_scale.copy_(torch.from_numpy(scale).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'])
    records = training_records(settings, manifest, smoke)
    train_data = RegionDataset(records['train'], cfg['points_per_candidate'], settings['seed'])
    validation_data = RegionDataset(records['validation'], cfg['points_per_candidate'], 1907)
    start, best, history = 0, float('inf'), []
    if resume:
        last = torch.load(work / 'last.pth', map_location=device, weights_only=False)
        if last['signature'] != signature or last['smoke'] != smoke:
            raise ValueError('Resume configuration differs from checkpoint.')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        start, best, history = last['epoch'], last['best_validation_loss'], last['history']
    epochs = 1 if smoke else int(cfg['epochs'])
    for epoch in range(start, epochs):
        train_data.epoch = epoch
        generator = torch.Generator().manual_seed(settings['seed'] + epoch)
        loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True,
                            num_workers=0, generator=generator)
        model.train()
        train_total = 0.0
        for points, descriptors, targets, weights, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = mask_loss(model(points.to(device), descriptors.to(device)),
                                   targets.to(device), weights.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            train_total += float(loss.detach()) * len(points)
        model.eval()
        forest_losses = {}
        # Fixed validation sampling; forests receive equal checkpoint-selection weight.
        with torch.no_grad():
            for points, descriptors, targets, weights, plots in DataLoader(validation_data, batch_size=1, num_workers=0):
                loss, _, _ = mask_loss(model(points.to(device), descriptors.to(device)),
                                       targets.to(device), weights.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite validation loss.')
                forest_losses.setdefault(plots[0], []).append(float(loss))
        validation_loss = float(np.mean([np.mean(value) for value in forest_losses.values()]))
        row = dict(epoch=epoch + 1, train_loss=train_total / len(train_data),
                   validation_macro_loss=validation_loss)
        history.append(row)
        improved = validation_loss < best
        best = min(best, validation_loss)
        checkpoint = dict(schema=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          hidden_dim=cfg['hidden_dim'], epoch=epoch + 1,
                          best_validation_loss=best, history=history,
                          signature=signature, manifest_digest=digest(manifest),
                          smoke=smoke, settings=settings)
        if improved:
            save_checkpoint(work / 'best.pth', checkpoint)
        save_checkpoint(work / 'last.pth', checkpoint)
        write_json(work / 'history.json', history)
        print(json.dumps(row), flush=True)
    summary = dict(smoke=smoke, parameter_count=sum(p.numel() for p in model.parameters()),
                   train_candidates=len(train_data), validation_candidates=len(validation_data),
                   best_validation_loss=best,
                   peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0,
                   checkpoint=str(work / 'best.pth'))
    write_json(work / 'training_summary.json', summary)
    print(json.dumps(summary), flush=True)


def evaluate(settings, pilot, smoke, export):
    import torch
    from tree_learn.mask_refinement.model import CandidateMaskMLP
    from tree_learn.mask_refinement.inference import predict_forest
    if pilot and not smoke:
        raise ValueError('Pilot evaluation requires --smoke.')
    manifest = read_manifest(settings, pilot)
    work = Path(settings['work_dir']) / 'smoke' if smoke else Path(settings['work_dir'])
    checkpoint = torch.load(work / 'best.pth', map_location='cpu', weights_only=False)
    if (checkpoint['signature'] != experiment_signature(settings, manifest) or
            checkpoint['manifest_digest'] != digest(manifest) or checkpoint['smoke'] != smoke):
        raise ValueError('Evaluation data/config differ from training checkpoint.')
    device = torch.device(settings['training']['device'])
    model = CandidateMaskMLP(checkpoint['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['model'])
    records = []
    for entry in manifest['forests']:
        if entry['split'] != 'validation':
            continue
        directory = Path(settings['output_root']) / entry['cache']
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        predictions, abstained, stats = predict_forest(
            model, directory, device, **dict(
                chunk_size=settings['inference']['chunk_size'],
                threshold=settings['inference']['mask_threshold']))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        stats['inference_seconds'] = time.perf_counter() - started
        stats['peak_cuda_memory_bytes'] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0)
        # Read GT only AFTER prediction has returned.
        arrays, metadata = load_cache(directory, with_labels=True)
        baseline = evaluate_labels(arrays['labels'], arrays['baseline'])
        refined = evaluate_labels(arrays['labels'], predictions)
        out = work / 'evaluation' / entry['plot']
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'refined_labels.npy', predictions, allow_pickle=False)
        np.save(out / 'unassigned_tree_mask.npy', abstained, allow_pickle=False)
        if export:
            points = arrays['coords'].astype(np.float64) + np.asarray(metadata['xyz_mean'])
            np.savez_compressed(out / 'baseline.npz', points=points, labels=arrays['baseline'])
            np.savez_compressed(out / 'refined.npz', points=points, labels=predictions)
        record = dict(plot=entry['plot'], baseline=baseline, refined=refined,
                      delta=compare_metrics(baseline, refined), assignment=stats,
                      lost_gt_ids=sorted(set(baseline['matched_gt_ids']) - set(refined['matched_gt_ids'])),
                      recovered_gt_ids=sorted(set(refined['matched_gt_ids']) - set(baseline['matched_gt_ids'])))
        write_json(out / 'metrics.json', record)
        records.append(record)
        print(f'{entry["plot"]}: {json.dumps(record["delta"])}', flush=True)
    baseline = macro_metrics([record['baseline'] for record in records])
    refined = macro_metrics([record['refined'] for record in records])
    gate = settings['evaluation_gate']
    checks = dict(
        coverage_gain=refined['coverage'] - baseline['coverage'] >= gate['min_macro_coverage_gain'],
        f1_gain=refined['f1'] - baseline['f1'] >= gate['min_macro_f1_gain'],
        completeness=baseline['completeness'] - refined['completeness'] <= gate['max_macro_completeness_drop'],
        commission=refined['commission'] - baseline['commission'] <= gate['max_macro_commission_increase'],
        forests=sum(record['refined']['coverage'] >= record['baseline']['coverage'] for record in records)
        >= gate['min_nonnegative_coverage_forests'])
    result = dict(smoke=smoke, protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5',
                  baseline_macro=baseline, refined_macro=refined,
                  delta_pp=compare_metrics(baseline, refined), forests=records,
                  checks=None if smoke else checks,
                  passed=None if smoke else all(checks.values()))
    write_json(work / 'evaluation' / 'summary.json', result)
    table = []
    for record in records:
        for method in ('baseline', 'refined'):
            row = {'plot': record['plot'], 'method': method}
            row.update({key: value for key, value in record[method].items()
                        if not isinstance(value, list)})
            table.append(row)
    with (work / 'evaluation' / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f'SAVED {work}/evaluation/summary.json; passed={result["passed"]}', flush=True)
    return True if smoke else result['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'audit', 'train', 'evaluate'])
    parser.add_argument('--config', default='configs/experiments/candidate_mask_mlp/step1.yaml')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export', action='store_true', help='Also write global-coordinate NPZ point clouds.')
    args = parser.parse_args()
    if args.smoke and args.command not in ('train', 'evaluate'):
        parser.error('--smoke only applies to train/evaluate.')
    if args.resume and args.command != 'train':
        parser.error('--resume only applies to train.')
    if args.export and args.command != 'evaluate':
        parser.error('--export only applies to evaluate.')
    if Path.cwd().resolve() != REPO_ROOT:
        parser.error('Run this command from the TreeLearn repository root.')
    with Path(args.config).open(encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    validate_settings(settings)
    if args.command == 'prepare':
        prepare(settings, args.pilot)
    elif args.command == 'audit':
        if not audit(settings, args.pilot):
            raise SystemExit(2)
    elif args.command == 'train':
        train(settings, args.pilot, args.smoke, args.resume)
    elif not evaluate(settings, args.pilot, args.smoke, args.export):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
