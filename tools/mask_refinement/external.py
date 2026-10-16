"""Prepare/evaluate an external forest with the already selected A1 checkpoint.

Run on the server. Training settings, normalization and inference policy remain
bound to the source-domain checkpoint; a failed source gate is reported, not
used to prohibit a user-requested exploratory cross-domain evaluation.
"""

import argparse
import copy
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from tree_learn.mask_refinement.data import digest, file_digest, load_cache, write_json
from tree_learn.mask_refinement.evaluation import compare_metrics, evaluate_labels

_SPEC = importlib.util.spec_from_file_location('candidate_mask_workflow', Path(__file__).with_name('run.py'))
WORKFLOW = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(WORKFLOW)


def overlaps(first, second):
    first, second = Path(first).resolve(), Path(second).resolve()
    return first == second or first in second.parents or second in first.parents


def external_settings(config, training):
    """Reuse the exact frozen-baseline preparation with separate external data."""
    source = Path(config['forest_path']).resolve()
    if (not source.is_file() or source.suffix.lower() not in {'.las', '.laz', '.npy', '.npz', '.txt'} or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', source.stem)):
        raise ValueError(f'Expected an existing external forest with a safe file stem: {source}')
    output, work = Path(config['output_root']).resolve(), Path(config['work_dir']).resolve()
    if overlaps(output, work):
        raise ValueError('External cache and work directories must be separate.')
    for destination in (output, work):
        for protected in (training['output_root'], training['work_dir'], source):
            if overlaps(destination, protected):
                raise ValueError('External outputs must not overlap source artifacts or the input forest.')
    if source.parent == Path(training['source_root']).resolve():
        raise ValueError('External input must be separate from the training forest directory.')
    if source.stem in training['splits']['train'] + training['splits']['validation']:
        raise ValueError('External input repeats a source-domain plot name.')
    settings = copy.deepcopy(training)
    settings.update(source_root=str(source.parent), output_root=str(output), work_dir=str(work),
                    splits={'external': [source.stem]}, pilot_plots=[])
    return settings


def check_checkpoint(checkpoint, training, manifest):
    if checkpoint['smoke']:
        raise ValueError('External evaluation requires a full-training checkpoint, not smoke.')
    if (checkpoint['signature'] != WORKFLOW.experiment_signature(training, manifest) or
            checkpoint['manifest_digest'] != digest(manifest)):
        raise ValueError('Training configuration or manifest differs from the selected checkpoint.')


def load_context(config):
    import torch
    with Path(config['training_config']).open(encoding='utf-8') as stream:
        training = yaml.safe_load(stream)
    WORKFLOW.validate_settings(training)
    manifest = WORKFLOW.read_manifest(training, pilot=False)
    settings = external_settings(config, training)
    selected = Path(config['checkpoint']).resolve()
    if selected != (Path(training['work_dir']) / 'best.pth').resolve():
        raise ValueError('Use the source-validation-selected best.pth for this fixed-checkpoint evaluation.')
    checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
    check_checkpoint(checkpoint, training, manifest)
    source_report_path = Path(training['work_dir']) / 'evaluation' / 'summary.json'
    source_report = json.loads(source_report_path.read_text(encoding='utf-8'))
    if (source_report['smoke'] or
            [row['plot'] for row in source_report['forests']] != training['splits']['validation']):
        raise ValueError('Keep the complete source-domain evaluation alongside the external evaluation.')
    request = dict(
        checkpoint=str(selected), checkpoint_sha256=file_digest(selected),
        checkpoint_epoch=int(checkpoint['epoch']),
        training_signature=checkpoint['signature'], training_manifest_digest=digest(manifest),
        source_evaluation_sha256=file_digest(source_report_path),
        source_validation_passed=source_report['passed'],
        forest_path=str(Path(config['forest_path']).resolve()),
        output_root=settings['output_root'], work_dir=settings['work_dir'])
    return training, manifest, settings, checkpoint, request


def lock_request(path, request, create=False):
    path = Path(path)
    if path.exists():
        lock = json.loads(path.read_text(encoding='utf-8'))
        if lock['request'] != request:
            raise ValueError('External experiment lock changed; use separate output_root/work_dir for a new experiment.')
    elif create:
        lock = dict(schema=1, request=request)
        write_json(path, lock)
    else:
        raise FileNotFoundError('External preparation lock missing; run external.py prepare first.')
    return lock


def preparation_policy(config):
    enabled = config.get('stream_tiles', False)
    if not isinstance(enabled, bool):
        raise ValueError('stream_tiles must be a boolean.')
    if not enabled:
        return None
    files = ('tools/mask_refinement/pipeline_streaming.py', 'tree_learn/mask_refinement/tiling.py')
    return dict(mode='streaming_tiles_v1', source_code={name: file_digest(REPO_ROOT / name) for name in files})


def check_preparation(lock, policy, settings, create=False):
    if lock.get('preparation') == policy:
        return
    output = Path(settings['output_root'])
    completed = ((output / 'manifest.json').exists() or 'external_manifest_digest' in lock or
                 any((output / split / plot / 'cache.json').exists()
                     for split, plots in settings['splits'].items() for plot in plots))
    # A failed pre-cache attempt may adopt streaming while keeping its weight/data lock.
    # Completed caches and already bound implementations cannot silently change.
    if create and policy is not None and lock.get('preparation') is None and not completed:
        lock['preparation'] = policy
        return
    raise ValueError('External preparation implementation changed; use separate output_root/work_dir.')


def prepare(config):
    _, training_manifest, settings, _, request = load_context(config)
    path = Path(settings['output_root']) / 'external_lock.json'
    lock = lock_request(path, request, create=True)
    policy = preparation_policy(config)
    check_preparation(lock, policy, settings, create=True)
    write_json(path, lock)
    source = Path(config['forest_path']).resolve()
    WORKFLOW.prepare(settings, pilot=False, source_paths={source.stem: source}, preparation=policy)
    external_manifest = WORKFLOW.read_manifest(settings, pilot=False)
    if external_manifest['baseline_signature'] != training_manifest['baseline_signature']:
        raise ValueError('External baseline preprocessing/weights differ from the training cache.')
    lock['external_manifest_digest'] = digest(external_manifest)
    write_json(path, lock)
    print(f'LOCKED external checkpoint epoch {request["checkpoint_epoch"]}; '
          f'source validation passed={request["source_validation_passed"]}', flush=True)


def evaluate(config, export=False):
    import torch
    from tree_learn.mask_refinement.inference import predict_forest
    from tree_learn.mask_refinement.model import CandidateMaskMLP
    training, training_manifest, settings, checkpoint, request = load_context(config)
    lock = lock_request(Path(settings['output_root']) / 'external_lock.json', request)
    policy = preparation_policy(config)
    check_preparation(lock, policy, settings)
    manifest = WORKFLOW.read_manifest(settings, pilot=False)
    if (lock.get('external_manifest_digest') != digest(manifest) or
            manifest.get('preparation') != policy or
            manifest['baseline_signature'] != training_manifest['baseline_signature']):
        raise ValueError('External cache differs from the prepared, locked baseline.')
    device = torch.device(training['training']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run on the server training environment.')
    model = CandidateMaskMLP(checkpoint['hidden_dim']).to(device)
    # Includes source-training feature_mean/feature_scale; never refit on Wytham.
    model.load_state_dict(checkpoint['model'])
    entry = manifest['forests'][0]
    directory = Path(settings['output_root']) / entry['cache']
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    predictions, abstained, stats = predict_forest(
        model, directory, device, chunk_size=training['inference']['chunk_size'],
        threshold=training['inference']['mask_threshold'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    stats['inference_seconds'] = time.perf_counter() - started
    stats['peak_cuda_memory_bytes'] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0)
    # External GT is read for scoring only after prediction has returned.
    arrays, metadata = load_cache(directory, with_labels=True)
    baseline = evaluate_labels(arrays['labels'], arrays['baseline'])
    refined = evaluate_labels(arrays['labels'], predictions)
    out = Path(settings['work_dir']) / 'evaluation'
    forest_out = out / entry['plot']
    forest_out.mkdir(parents=True, exist_ok=True)
    np.save(forest_out / 'refined_labels.npy', predictions, allow_pickle=False)
    np.save(forest_out / 'unassigned_tree_mask.npy', abstained, allow_pickle=False)
    if export:
        points = arrays['coords'].astype(np.float64) + np.asarray(metadata['xyz_mean'])
        np.savez_compressed(forest_out / 'baseline.npz', points=points, labels=arrays['baseline'])
        np.savez_compressed(forest_out / 'refined.npz', points=points, labels=predictions)
    record = dict(plot=entry['plot'], baseline=baseline, refined=refined,
                  delta=compare_metrics(baseline, refined), assignment=stats,
                  lost_gt_ids=sorted(set(baseline['matched_gt_ids']) - set(refined['matched_gt_ids'])),
                  recovered_gt_ids=sorted(set(refined['matched_gt_ids']) - set(baseline['matched_gt_ids'])))
    write_json(forest_out / 'metrics.json', record)
    result = dict(
        schema=1, external=True, smoke=False, evaluation_role='exploratory_cross_domain',
        protocol='aligned_voxels_ignore_negative_gt_iou_gt_0.5',
        preparation=policy,
        locked_training=request, external_manifest_digest=digest(manifest),
        baseline_macro=WORKFLOW.macro_metrics([baseline]),
        refined_macro=WORKFLOW.macro_metrics([refined]),
        delta_pp=record['delta'], forests=[record], checks=None, passed=None)
    write_json(out / 'summary.json', result)
    print(f'{entry["plot"]}: {json.dumps(record["delta"])}', flush=True)
    print(f'{entry["plot"]} scores: {json.dumps(stats["score_diagnostics"])}', flush=True)
    diagnostics = stats['score_diagnostics']
    if diagnostics['empty_prediction_reason'] is not None or diagnostics['num_foreground_points_not_scored']:
        print('WARNING: empty prediction or unscored foreground; inspect score diagnostics. '
              'Zero commission with zero predictions is not an improvement.', flush=True)
    print(f'SAVED {out}/summary.json; exploratory cross-domain comparison, source result retained.', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'evaluate'])
    parser.add_argument('--config', default='configs/experiments/candidate_mask_mlp/wytham.yaml')
    parser.add_argument('--export', action='store_true', help='Also save global-coordinate NPZ clouds.')
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO_ROOT:
        parser.error('Run from the TreeLearn repository root.')
    if args.export and args.command != 'evaluate':
        parser.error('--export only applies to evaluate.')
    with Path(args.config).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if args.command == 'prepare':
        prepare(config)
    else:
        evaluate(config, args.export)


if __name__ == '__main__':
    main()
