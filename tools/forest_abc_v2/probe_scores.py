"""Bounded, posthoc score-gate diagnostic of an existing V2 checkpoint. No training."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.forest_abc import run as common
from tools.forest_abc_v2.run import BlockSampler, signature
from tree_learn.forest_abc.data import entries
from tree_learn.forest_abc_v2.data import CenterForest
from tree_learn.forest_abc_v2.model import CenterForestModel, tensor_block
from tree_learn.forest_abc_v2.scoring import proposal_scores
from tree_learn.forest_abc_v2.loss import balanced_mass, candidate_quality_targets, match_instances
from tree_learn.mask_refinement.data import file_digest, write_json


@torch.no_grad()
def gate_report(layer, infer):
    logits = layer['masks'][:-2]
    quality_logits = layer['scores'][:-2]
    if not bool(torch.isfinite(layer['masks']).all() & torch.isfinite(layer['scores']).all()):
        raise FloatingPointError('Non-finite probe predictions.')
    result = {}
    background = layer['masks'][-2] - layer['masks'][-1]
    for revision, name in ((1, 'legacy_product'), (2, 'direct_quality')):
        score, membership, quality, confidence = proposal_scores(logits, quality_logits,
                                                               dict(infer, score_revision=revision))
        passing = (score >= infer['quality_threshold']) & membership.any(-1)
        evidence = (logits - background[None] + infer['quality_log_weight'] * score.clamp_min(1e-8).log()[:, None]).sigmoid()
        surviving = membership & (evidence >= infer['ownership_threshold']) & passing[:, None]
        result[name] = dict(mean_score=float(score.mean()), maximum_score=float(score.max()),
            passing_queries=int(passing.sum()), ownership_nonempty_queries=int(surviving.any(-1).sum()),
            voxel_covered_fraction=float(surviving.any(0).float().mean()))
    result['raw'] = dict(mean_quality=float(quality.mean()), maximum_quality=float(quality.max()),
        quality_quantiles=torch.quantile(quality, quality.new_tensor([0., .5, .9, .99, 1.])).cpu().tolist(),
        mean_mask_confidence=float(confidence.mean()), nonempty_queries=int(membership.any(-1).sum()),
        mean_background_log_odds=float(background.mean()))
    return result


@torch.no_grad()
def training_mask_report(prediction, block, device, loss_config):
    """Training-crop diagnostic using full-tree source targets; not forest metrics."""
    batch = tensor_block(block, device, supervision=True)
    known = batch['labels'] >= 0
    labels, weights = batch['labels'][known], batch['weights'][known]
    ids = labels[labels > 0].unique(sorted=True)
    targets = (ids[:, None] == labels[None]).to(weights)
    logits = prediction['layers'][-1]['masks'][:-2, known]
    full_sizes = torch.stack([batch['whole_sizes'][known][labels == identifier].max() for identifier in ids])
    overlaps = candidate_quality_targets(logits, targets, weights, full_sizes, return_all=True)
    candidate = overlaps.max(-1).values
    selected = match_instances(logits, targets, balanced_mass(labels, weights), loss_config['dice_weight'])
    matched_raw = overlaps[selected, torch.arange(len(ids), device=device)]
    background = prediction['layers'][-1]['masks'][-2, known] - prediction['layers'][-1]['masks'][-1, known]
    winner = torch.cat((background[None], logits)).argmax(0)
    predicted_mass = weights.new_zeros(len(logits) + 1).index_add(0, winner, weights)[1:]
    owned = []
    for column, query in enumerate(selected):
        intersection = weights[(winner == query + 1) & (labels == ids[column])].sum()
        owned.append(intersection / (full_sizes[column] + predicted_mass[query] - intersection).clamp_min(1))
    owned = torch.stack(owned)
    predicted_quality = prediction['layers'][-1]['scores'][:-2].sigmoid()
    return dict(training_crop_only=True, visible_gt=len(ids),
        raw_good_proposals=int((candidate > .5).sum()), matched_raw_iou_mean=float(matched_raw.mean()),
        matched_owner_iou_mean=float(owned.mean()), matched_predicted_quality_mean=float(predicted_quality[selected].mean()),
        good_proposals_with_quality_below_threshold=int(((candidate > .5) & (predicted_quality < .3)).sum()),
        good_matched_raw_but_bad_owned=int(((matched_raw > .5) & (owned <= .1)).sum()))


def run(checkpoint, plots=('G4N', 'O1W'), train_crops=3):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('schema') != 2 or saved.get('check_only', True):
        raise ValueError('Probe requires a full-training V2 checkpoint.')
    config = saved['config']
    output = Path(config['work_dir']) / 'diagnostics' / f'score_probe_step_{saved["step"]:06d}'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('brief.json', 'summary.json'):
        (output / name).unlink(missing_ok=True)
    source_rows = entries(config)
    if signature(config, source_rows) != saved['signature']:
        raise ValueError('Probe data/configuration differ from the checkpoint.')
    # Scoring/loss diagnostics may evolve, but the actual learned model and
    # coordinate/augmentation implementation must still match this checkpoint.
    diagnostic_changes = {'tools/forest_abc_v2/run.py', 'tree_learn/forest_abc_v2/loss.py',
                          'tree_learn/forest_abc_v2/inference.py', 'tree_learn/forest_abc_v2/scoring.py'}
    for name, expected in saved['code'].items():
        if name in diagnostic_changes:
            continue
        if expected != file_digest(ROOT / name):
            raise ValueError(f'Probe model/data code differs from checkpoint: {name}')
    device = common.device_for(config)
    common.seed_everything(config['seed'])
    model = CenterForestModel(config['model']).to(device).eval()
    model.load_state_dict(saved['model'])
    records = []

    def inspect(name, block, supervision=False):
        trace = []
        original = model.decoder.head

        def observe(queries, features, cost):
            layer, attention = original(queries, features, cost)
            trace.append(gate_report(layer, config['inference']))
            return layer, attention

        model.decoder.head = observe
        try:
            with torch.no_grad():
                prediction = model(tensor_block(block, device, supervision=False))
                report = dict(case=name, input_voxels=len(block['xyz']), heads=trace,
                              training_crop_only=supervision)
                if supervision:
                    report['mask_quality'] = training_mask_report(prediction, block, device, config['loss'])
        finally:
            model.decoder.head = original
        records.append(report)
        print('PROBE ' + json.dumps(report), flush=True)

    validation = {r['plot']: r for r in source_rows if r['split'] == 'validation'}
    for name in plots:
        if name not in validation:
            raise ValueError(f'Unknown validation plot: {name}')
        row = validation[name]
        forest = CenterForest(row['cache'], row['index'])
        block_id = int(forest.counts.argmax())
        block = forest.make_block(forest.centers[block_id], config['blocks'], config['seed'] + block_id)
        if block is None:
            raise ValueError(f'Empty selected source validation block: {name}')
        print(f'PROBE_BEGIN validation/{name} original_block={block_id + 1}/{len(forest.centers)}', flush=True)
        inspect(f'validation/{name}', block)
        del block, forest
    sampler = BlockSampler([r for r in source_rows if r['split'] == 'train'], config['blocks'], config['seed'])
    for position in range(train_crops):
        plot, block = sampler.draw()
        print(f'PROBE_BEGIN training_crop={position + 1}/{train_crops} {plot}', flush=True)
        inspect(f'training/{plot}/{position + 1}', block, supervision=True)
    summary = dict(schema=1, posthoc=True, partial=True, eligible_for_advancement=False,
        checkpoint=str(checkpoint), checkpoint_step=saved['step'], checkpoint_sha256=file_digest(checkpoint),
        scope='score_gate_probe_only', cases=records,
        notes=['No training or complete forest prediction is performed; existing predictions/checkpoints are unchanged.',
               'Only two source validation blocks and deterministic augmented training crops are inspected by default.',
               'GT is used only for post-forward training-crop measurements, not model inputs.',
               'Comparing score gates is a diagnostic, not a replacement benchmark or proof of recovery.',
               'Gate counts precede the top-k cap, local fusion, fine-point minimum size and global allocation.',
               'Old allocated IoU targets are compared with independent candidate IoU on training crops.'])
    write_json(output / 'summary.json', summary)
    brief = {key: value for key, value in summary.items() if key not in ('notes', 'cases')}
    brief['cases'] = [{**{key: value for key, value in case.items() if key != 'heads'},
                       'initial_head': case['heads'][0], 'final_head': case['heads'][-1]} for case in records]
    write_json(output / 'brief.json', brief)
    print('SAVED ' + str(output / 'brief.json') + '\n' + json.dumps(brief, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='work_dirs/forest_abc_v2_s42/step_010000.pth')
    args = parser.parse_args()
    os.chdir(ROOT)
    run(Path(args.checkpoint))
