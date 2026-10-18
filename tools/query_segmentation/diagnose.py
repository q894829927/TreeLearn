"""Read existing masks on CPU to diagnose quality and allocation fragments."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from tree_learn.mask_refinement.data import file_digest, write_json
from tree_learn.mask_refinement.evaluation import compare_metrics, contingency, macro_metrics
from tree_learn.query_segmentation.diagnosis import analyze_instances


def diagnose(settings):
    work = Path(settings['work_dir'])
    evaluation = work / 'evaluation'
    source_path = evaluation / 'summary.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if (source.get('partial', True) or 'checkpoint_step' not in source or
            any('prediction' not in f for f in source.get('forests', [])) or not source.get('forests')):
        raise ValueError('Expected a completed, full query-decoder evaluation, not A1/A1b or smoke.')
    config = yaml.safe_load(Path(settings['source_config']).read_text(encoding='utf-8'))
    cache_root = Path(config['output_root'])
    manifest = json.loads((cache_root / 'manifest.json').read_text(encoding='utf-8'))
    caches = {r['plot']: cache_root / r['cache'] for r in manifest['forests'] if r['split'] == 'validation'}
    if {f['plot'] for f in source['forests']} != set(caches):
        raise ValueError('Evaluation forests and source validation manifest disagree.')
    output = work / 'failure_diagnosis'
    reports = []
    for forest in source['forests']:
        plot = forest['plot']
        print(f'DIAGNOSE {plot}: CPU, reading existing masks; no model or inference.', flush=True)
        truth_path, prediction_path = caches[plot] / 'labels.npy', evaluation / plot / 'refined_labels.npy'
        truth = np.load(truth_path, mmap_mode='r', allow_pickle=False)
        prediction = np.load(prediction_path, mmap_mode='r', allow_pickle=False)
        records = json.loads((evaluation / plot / 'instances.json').read_text(encoding='utf-8'))
        if len(records) != forest['prediction']['final_instances']:
            raise ValueError(f'{plot}: saved instance metadata belongs to another evaluation.')
        table = contingency(truth, prediction)
        report, details = analyze_instances(table, records, source['policy']['quality_threshold'])
        for key in ('tp', 'fp', 'fn', 'num_predictions', 'num_gt', 'f1', 'coverage',
                    'completeness', 'commission', 'point_precision', 'point_recall'):
            if not np.isclose(report['original'][key], forest['refined'][key], rtol=0, atol=1e-9):
                raise ValueError(f'{plot}: saved masks do not reproduce source metrics: {key}')
        report.update(plot=plot, baseline=forest['baseline'], prediction=forest['prediction'],
                      source_files={name: dict(path=str(path), size=path.stat().st_size,
                      mtime_ns=path.stat().st_mtime_ns) for name, path in
                      (('truth', truth_path), ('prediction', prediction_path))})
        reports.append(report)
        folder = output / plot
        write_json(folder / 'summary.json', report)
        if details:
            with (folder / 'instances.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(details[0]))
                writer.writeheader()
                writer.writerows(details)
        print(json.dumps(dict(plot=plot, tp=report['original']['tp'], fp=report['original']['fp'],
                              fp_retained_below_half=report['fp_retained_below_half'],
                              fp_quality_at_least_0_7=report['fp_quality_at_least_0_7'],
                              groups=report['groups'])), flush=True)
        del table, truth, prediction, records, details
    comparisons = {}
    for name in reports[0]['policies']:
        macro = macro_metrics([r['policies'][name]['metrics'] for r in reports])
        comparisons[name] = dict(macro=macro, delta_from_baseline_pp=compare_metrics(source['baseline_macro'], macro),
                                 delta_from_v1_pp=compare_metrics(source['refined_macro'], macro),
                                 totals={key: sum(r['policies'][name]['metrics'][key] for r in reports)
                                         for key in ('tp', 'fp', 'fn', 'num_predictions')})
    result = dict(schema=1, posthoc=True, analysis_only=True, scope='filter_final_masks_without_reassignment',
                  checkpoint=source['checkpoint'], checkpoint_step=source['checkpoint_step'],
                  source_evaluation_sha256=file_digest(source_path), baseline_macro=source['baseline_macro'],
                  v1_macro=source['refined_macro'], comparisons=comparisons,
                  totals={key: sum(r[key] for r in reports) for key in
                          ('fp_retained_below_half', 'fp_quality_at_least_0_7', 'fp_best_gt_already_matched')},
                  forests=reports, source_code={name: file_digest(ROOT / name) for name in
                  ('tools/query_segmentation/diagnose.py', 'tree_learn/query_segmentation/diagnosis.py')})
    write_json(output / 'summary.json', result)
    compact = {key: result[key] for key in ('checkpoint_step', 'v1_macro', 'totals', 'comparisons')}
    write_json(output / 'brief.json', compact)
    print(json.dumps(compact, indent=2), flush=True)
    print(f'SAVED {output}/brief.json and summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/query_instance_decoder/v1.yaml')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the repository root.')
    diagnose(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
