"""Materialize quality-times-retention filtering from saved labels on CPU."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from tree_learn.mask_refinement.data import file_digest, write_json
from tree_learn.mask_refinement.evaluation import compare_metrics, evaluate_labels, macro_metrics
from tree_learn.query_segmentation.allocation import filter_allocation


def materialize(settings):
    work = Path(settings['work_dir'])
    source_dir, output = work / 'evaluation', work / 'evaluation_allocation_filtered'
    source_path = source_dir / 'summary.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source.get('partial', True) or 'checkpoint_step' not in source or not source.get('forests'):
        raise ValueError('A completed, full query-decoder source evaluation is required.')
    if source['policy'].get('allocation_filter', 'none') != 'none':
        raise ValueError('Read the original unfiltered evaluation; do not filter twice.')
    source_config = yaml.safe_load(Path(settings['source_config']).read_text(encoding='utf-8'))
    cache_root = Path(source_config['output_root'])
    manifest = json.loads((cache_root / 'manifest.json').read_text(encoding='utf-8'))
    caches = {r['plot']: cache_root / r['cache'] for r in manifest['forests'] if r['split'] == 'validation'}
    if set(caches) != {f['plot'] for f in source['forests']}:
        raise ValueError('Source evaluation and validation caches disagree.')
    threshold = source['policy']['quality_threshold']
    reference_path = work / 'failure_diagnosis/summary.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8')) if reference_path.exists() else None
    verify_reference = bool(reference and reference.get('source_evaluation_sha256') == file_digest(source_path))
    expected = {f['plot']: f['policies']['quality_times_retention']['metrics']
                for f in reference['forests']} if verify_reference else {}
    reports = []
    for forest in source['forests']:
        plot = forest['plot']
        folder = output / plot
        folder.mkdir(parents=True, exist_ok=True)
        print(f'FILTER {plot}: saved predictions, CPU only, threshold={threshold}', flush=True)
        started = time.monotonic()
        labels = np.load(source_dir / plot / 'refined_labels.npy', mmap_mode='r', allow_pickle=False)
        records = json.loads((source_dir / plot / 'instances.json').read_text(encoding='utf-8'))
        if len(records) != forest['prediction']['final_instances']:
            raise ValueError(f'{plot}: instance metadata and evaluation disagree.')
        destination = np.lib.format.open_memmap(folder / 'refined_labels.npy', mode='w+',
                                               dtype=labels.dtype, shape=labels.shape)
        kept, rejected, allocation = filter_allocation(labels, records, destination, threshold)
        destination.flush()
        write_json(folder / 'instances.json', kept)
        write_json(folder / 'rejected_instances.json', rejected)
        write_json(folder / 'allocation_filter.json', allocation)
        filter_seconds = time.monotonic() - started
        # GT and original instance IDs open only AFTER output filtering finishes.
        truth = np.load(caches[plot] / 'labels.npy', mmap_mode='r', allow_pickle=False)
        baseline_labels = np.load(caches[plot] / 'baseline.npy', mmap_mode='r', allow_pickle=False)
        refined = evaluate_labels(truth, destination)
        if verify_reference:
            for key in ('tp', 'fp', 'fn', 'f1', 'coverage', 'completeness', 'commission',
                        'point_precision', 'point_recall', 'num_predictions'):
                if not np.isclose(refined[key], expected[plot][key], rtol=0, atol=1e-9):
                    raise ValueError(f'{plot}: materialized output differs from saved diagnosis: {key}')
        old_fg = baseline_labels > 0
        statistics = dict(forest['prediction'])
        source_seconds = statistics.pop('inference_seconds', None)
        statistics.update(final_instances=len(kept), allocation_filter=allocation, reused_predictions=True,
                          source_inference_seconds=source_seconds, allocation_filter_seconds=filter_seconds,
                          empty_prediction_reason='all_allocated_masks_filtered' if not kept else None,
                          original_foreground_points=int(old_fg.sum()),
                          abstained_original_foreground=int(np.count_nonzero(old_fg & (destination == 0))),
                          assigned_original_background_points=int(np.count_nonzero((~old_fg) & (destination > 0))),
                          recovered_gt_tree_points_from_original_background=int(np.count_nonzero(
                              (~old_fg) & (destination > 0) & (truth > 0))),
                          abstained_gt_tree_points=int(np.count_nonzero((destination == 0) & (truth > 0))))
        report = dict(plot=plot, baseline=forest['baseline'], refined=refined,
                      delta=compare_metrics(forest['baseline'], refined), prediction=statistics,
                      lost_gt_ids=sorted(set(forest['baseline']['matched_gt_ids']) - set(refined['matched_gt_ids'])),
                      recovered_gt_ids=sorted(set(refined['matched_gt_ids']) - set(forest['baseline']['matched_gt_ids'])))
        write_json(folder / 'metrics.json', report)
        write_json(folder / 'prediction.json', statistics)
        reports.append(report)
        print(json.dumps(dict(plot=plot, allocation=allocation, delta=report['delta'])), flush=True)
        del labels, destination, truth, baseline_labels, old_fg, records
    macro = macro_metrics([r['refined'] for r in reports])
    result = dict(schema=1, protocol=source['protocol'], partial=False, posthoc=True,
                  baseline_macro=source['baseline_macro'], refined_macro=macro,
                  delta_pp=compare_metrics(source['baseline_macro'], macro), forests=reports,
                  policy=dict(source['policy'], allocation_filter='quality_times_retention'),
                  checkpoint=source['checkpoint'], checkpoint_step=source['checkpoint_step'],
                  checkpoint_sha256=source.get('checkpoint_sha256'), exploratory_external=False,
                  allocation_filter_selected_on_development_data=True, reused_predictions=True,
                  diagnostic_reference_verified=verify_reference,
                  source_evaluation_sha256=file_digest(source_path),
                  source_code={name: file_digest(ROOT / name) for name in
                  ('tools/query_segmentation/apply_allocation_filter.py', 'tree_learn/query_segmentation/allocation.py')})
    write_json(output / 'summary.json', result)
    write_json(output / 'brief.json', {key: result[key] for key in
               ('checkpoint_step', 'refined_macro', 'delta_pp', 'diagnostic_reference_verified')})
    print(f'SAVED {output}/summary.json and brief.json', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/query_instance_decoder/v1.yaml')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the repository root.')
    materialize(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
