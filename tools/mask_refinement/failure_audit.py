"""Post-hoc CPU diagnostics on existing A1 caches and saved predictions only."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tree_learn.mask_refinement.data import digest, file_digest, load_cache, write_json
from tree_learn.mask_refinement.diagnostics import diagnose_assignments
from tree_learn.mask_refinement.failure_audit import (
    COUNT_KEYS, audit_supervision_support, describe_changed_gt, summarize_population,
)

SPEC = importlib.util.spec_from_file_location('failure_audit_external', Path(__file__).with_name('external.py'))
EXTERNAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTERNAL)
WORKFLOW = EXTERNAL.WORKFLOW


def saved_context(config):
    """Verify saved provenance without deserializing a checkpoint or creating a model."""
    with Path(config['training_config']).open(encoding='utf-8') as stream:
        training = yaml.safe_load(stream)
    WORKFLOW.validate_settings(training)
    source_manifest = WORKFLOW.read_manifest(training, pilot=False)
    external = EXTERNAL.external_settings(config, training)
    manifest = WORKFLOW.read_manifest(external, pilot=False)
    lock_path = Path(external['output_root']) / 'external_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    selected = Path(config['checkpoint']).resolve()
    source_evaluation = Path(training['work_dir']) / 'evaluation/summary.json'
    expected = dict(checkpoint=str(selected), checkpoint_sha256=file_digest(selected),
                    training_manifest_digest=digest(source_manifest),
                    training_signature=WORKFLOW.experiment_signature(training, source_manifest),
                    source_evaluation_sha256=file_digest(source_evaluation),
                    forest_path=str(Path(config['forest_path']).resolve()),
                    output_root=external['output_root'], work_dir=external['work_dir'])
    if (selected != (Path(training['work_dir']) / 'best.pth').resolve() or
            any(lock['request'].get(key) != value for key, value in expected.items())):
        raise ValueError('Saved external request disagrees with source config/checkpoint/artifacts.')
    policy = EXTERNAL.preparation_policy(config)
    EXTERNAL.check_preparation(lock, policy, external)
    path = Path(external['work_dir']) / 'evaluation/summary.json'
    source = json.loads(path.read_text(encoding='utf-8'))
    if (not source.get('external') or source['smoke'] or
            source['protocol'] != 'aligned_voxels_ignore_negative_gt_iou_gt_0.5' or
            source['locked_training'] != lock['request'] or
            source.get('preparation') != policy or manifest.get('preparation') != policy or
            source['external_manifest_digest'] != digest(manifest) or
            lock.get('external_manifest_digest') != digest(manifest) or
            manifest['baseline_signature'] != source_manifest['baseline_signature'] or
            [row['plot'] for row in source['forests']] != [row['plot'] for row in manifest['forests']]):
        raise ValueError('Saved external evaluation differs from the locked manifest/policy.')
    return training, source_manifest, external, manifest, source, path


def run(config, scope='all'):
    if scope not in ('all', 'external', 'source'):
        raise ValueError('Unknown failure-audit scope.')
    training, source_manifest, external, manifest, primary, primary_path = saved_context(config)
    out = Path(external['work_dir']) / 'diagnostics/failure_audit'
    result = dict(schema=1, posthoc=True, eligible_for_advancement=False, scope=scope,
                  protocol=primary['protocol'], locked_training=primary['locked_training'],
                  source_summary_sha256=file_digest(primary_path),
                  source_manifest_digest=digest(source_manifest), external_manifest_digest=digest(manifest),
                  population_units='region references may repeat points; unique-point counts are separate',
                  external=None, source=None)
    result['diagnostic_source_code'] = {name: file_digest(ROOT / name) for name in (
        'tools/mask_refinement/failure_audit.py', 'tree_learn/mask_refinement/failure_audit.py',
        'tree_learn/mask_refinement/diagnostics.py', 'tree_learn/mask_refinement/evaluation.py')}
    progress = lambda message: print(message, flush=True)
    if scope in ('all', 'external'):
        entry, saved = manifest['forests'][0], primary['forests'][0]
        directory = Path(external['output_root']) / entry['cache']
        print(f'EXTERNAL {entry["plot"]}: CPU controls and hypothetical supervision audit.', flush=True)
        support = audit_supervision_support(directory, training['supervision'], frozen=False, progress=progress)
        write_json(out / 'external/supervision.json', support)
        arrays, _ = load_cache(directory, with_labels=True)
        prediction_path = Path(external['work_dir']) / 'evaluation' / entry['plot'] / 'refined_labels.npy'
        prediction = np.load(prediction_path, mmap_mode='r', allow_pickle=False)
        report = diagnose_assignments(arrays['labels'], arrays['baseline'], prediction, expected=saved)
        report['gt_match_changes'] = describe_changed_gt(
            arrays['labels'], arrays['baseline'], prediction, saved, support['target_groups'])
        report['source_hashes'] = dict(refined_labels=file_digest(prediction_path),
                                     baseline=file_digest(directory / 'baseline.npy'),
                                     labels=file_digest(directory / 'labels.npy'))
        write_json(out / 'external/assignments.json', report)
        compact = dict(
            plot=entry['plot'], supervision=support['summary'],
            supervision_origin=support['supervision_origin'],
            delta_vs_baseline=report['delta_vs_baseline'], point_changes=report['point_changes'],
            candidate_state_transitions=report['candidate_state_transitions'],
            new_counted_fp_count=len(report['new_counted_fp_ids']),
            removed_counted_fp_count=len(report['removed_counted_fp_ids']),
            lost_gt_count=len(saved['lost_gt_ids']), recovered_gt_count=len(saved['recovered_gt_ids']))
        result['external'] = compact
        print(f'EXTERNAL controls: {json.dumps(report["delta_vs_baseline"])}', flush=True)
        del arrays, prediction, report, support
    if scope in ('all', 'source'):
        totals = {split: {key: 0 for key in COUNT_KEYS} for split in ('train', 'validation')}
        forests = []
        for entry in source_manifest['forests']:
            print(f'SOURCE {entry["plot"]}: checking frozen supervision and scanning regions.', flush=True)
            directory = Path(training['output_root']) / entry['cache']
            support = audit_supervision_support(directory, training['supervision'], frozen=True, progress=progress)
            write_json(out / 'source' / entry['plot'] / 'supervision.json', support)
            for key in COUNT_KEYS:
                totals[entry['split']][key] += support['counts'][key]
            forests.append(dict(plot=entry['plot'], split=entry['split'], summary=support['summary']))
            print(f'SOURCE {entry["plot"]}: {json.dumps(support["summary"])}', flush=True)
            del support
        result['source'] = dict(by_split={split: summarize_population(value) for split, value in totals.items()},
                                forests=forests)
        training_summary = Path(training['work_dir']) / 'training_summary.json'
        if training_summary.is_file():
            saved_training = json.loads(training_summary.read_text(encoding='utf-8'))
            if (saved_training['smoke'] or
                    saved_training['train_candidates'] != totals['train']['eligible_candidates'] or
                    saved_training['validation_candidates'] != totals['validation']['eligible_candidates']):
                raise ValueError('Frozen eligible counts disagree with the original full training summary.')
            result['source']['training_summary_sha256'] = file_digest(training_summary)
    path = out / f'summary_{scope}.json'
    write_json(path, result)
    print(f'SAVED {path}; post-hoc only, no training or prediction.', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/candidate_mask_mlp/wytham.yaml')
    parser.add_argument('--scope', choices=['all', 'external', 'source'], default='all')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the TreeLearn repository root.')
    with Path(args.config).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    # Share the launcher's lock, so diagnostics do not race a cache/evaluation write.
    import fcntl
    lock_path = Path('logs/candidate_mask_mlp/background/external.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another external run/diagnostic holds the lock; wait for it to finish.')
        run(config, args.scope)


if __name__ == '__main__':
    main()
