"""Replay at most 24 original blocks to audit proposals, layers and local NMS."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree
import yaml

from tools.forest_abc.diagnose import write_csv, write_json
from tools.forest_abc.stage_audit_core import (
    best_for_gt, cell_counts, classify_target, nms_overlap, proposal_metrics, select_targets)
from tree_learn.mask_refinement.data import file_digest
from tree_learn.forest_abc.data import RawForest, entries


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def selected_points(coords, labels, targets, chunk_points=1_000_000):
    identifiers = np.asarray(sorted(int(row['gt_id']) for row in targets), dtype=np.int64)
    pieces = {int(identifier): [] for identifier in identifiers}
    for start in range(0, len(labels), chunk_points):
        stop = min(start + chunk_points, len(labels))
        local = labels[start:stop]
        positions = np.flatnonzero(np.isin(local, identifiers))
        for identifier in identifiers:
            values = positions[local[positions] == identifier]
            if len(values):
                pieces[int(identifier)].append(values + start)
        if start == 0 or stop == len(labels) or (start // chunk_points + 1) % 10 == 0:
            print(f'PLAN_SCAN points={stop}/{len(labels)}', flush=True)
    result = {}
    for row in targets:
        identifier = int(row['gt_id'])
        indices = np.concatenate(pieces[identifier]) if pieces[identifier] else np.empty(0, dtype=np.int64)
        if len(indices) != int(row['gt_points']):
            raise ValueError('Selected GT point counts differ from the saved posthoc audit.')
        result[identifier] = dict(indices=indices, xy=coords[indices, :2].mean(axis=0, dtype=np.float64))
    return result


def capture_heads(model, block, device, tensorize=None):
    """Observe eval heads; return original tensors unchanged to the decoder."""
    import torch
    if tensorize is None:
        from tree_learn.forest_abc.model import tensor_block
        tensorize = tensor_block

    captured = []
    original = model.decoder._forward_head

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        scores, masks, _ = result
        logits, quality = masks[0][:-2], scores[0][:-2].squeeze(-1)
        if not bool(torch.isfinite(logits).all() & torch.isfinite(quality).all()):
            raise FloatingPointError('Nonfinite decoder head during stage audit.')
        probability = logits.sigmoid()
        membership = probability >= model.audit_mask_threshold
        confidence = (probability * membership).sum(-1) / membership.sum(-1).clamp_min(1)
        captured.append(dict(masks=membership.cpu().numpy(), scores=(quality.sigmoid() * confidence).cpu().numpy()))
        return result

    model.decoder._forward_head = observe
    try:
        with torch.no_grad():
            batch = tensorize(block, device)
            prediction = model(batch)
            probability = prediction['semantic'].softmax(-1)[:, 1]
            candidates = torch.where(batch['core'])[0]
            if not len(candidates):
                candidates = torch.arange(len(probability), device=device)
            foreground = candidates[probability[candidates] >= .5]
            minimum = min(model.config['num_queries'], len(candidates))
            if len(foreground) < minimum:
                foreground = candidates[probability[candidates].argsort(descending=True)[:minimum]]
            if len(foreground) > model.config['isa_candidates']:
                positions = torch.linspace(0, len(foreground) - 1, model.config['isa_candidates'], device=device).long()
                foreground = foreground[positions]
            auxiliary = dict(anchors=prediction['anchors'].cpu().numpy(),
                             semantic_foreground=(probability >= .5).cpu().numpy(),
                             query_candidates=foreground.cpu().numpy())
    finally:
        model.decoder._forward_head = original
    if len(captured) != model.config['layers'] + 1:
        raise ValueError('Unexpected number of decoder heads; cannot label layer traces.')
    return captured, auxiliary


def replay_block(model, forest, labels, config, block_id, gt_ids, gt_sizes, targets,
                 global_records, output, device, remaining_members):
    from tree_learn.query_segmentation.inference import BlockMaskStore, suppress_masks

    block = forest.make_block(forest.centers[block_id], config['blocks'], config['seed'] + block_id)
    if block is None:
        raise ValueError(f'Selected original block {block_id} is empty.')
    heads, auxiliary = capture_heads(model, block, device)
    nearest = cKDTree(block['source_xyz']).query(forest.arrays['coords'][block['indices']], k=1, workers=1)[1]
    cells = cell_counts(nearest, labels[block['indices']], len(block['xyz']), gt_ids)
    visible = [(row, int(np.searchsorted(gt_ids, int(row['gt_id'])))) for row in targets
               if cells['local_gt'][np.searchsorted(gt_ids, int(row['gt_id']))] > 0]
    layer_rows = []
    for layer_id, head in enumerate(heads):
        metrics = proposal_metrics(head['masks'], cells, gt_sizes)
        for target, col in visible:
            best = best_for_gt(metrics, np.arange(len(head['scores'])), col, head['scores'])
            layer_rows.append(dict(block=block_id, gt_id=int(target['gt_id']), head=layer_id,
                                   kind='initial' if layer_id == 0 else
                                   ('cross_attention_decoder_layer' if layer_id == 1 or not config['model']['use_mfa']
                                    else 'mfa_decoder_layer'),
                                   **{key: value for key, value in best.items() if key != 'good_queries'},
                                   num_good_queries=len(best['good_queries'])))
        if layer_id != len(heads) - 1:
            head.clear()
    head, infer = heads[-1], config['inference']
    masks, scores = head['masks'], head['scores']
    quality = np.flatnonzero((scores >= infer['quality_threshold']) & masks.any(axis=1))
    sized = quality[metrics['total'][quality] >= infer['minimum_instance_points']]
    records, mask_records = [], {}
    folder = output / f'block_{block_id:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    dtype = np.int32 if forest.metadata['num_points'] < 2 ** 31 else np.int64
    with BlockMaskStore(folder, len(sized), dtype, infer['max_block_memberships']) as store:
        for local, query in enumerate(sized):
            indices = block['indices'][masks[query, nearest]]
            store.append(local, indices)
            xyz = forest.arrays['coords'][indices]
            records.append(dict(score=float(scores[query]), count=len(indices), query=int(query), block=block_id,
                                bbox=np.concatenate([xyz.min(0), xyz.max(0)]).tolist()))
        store.finalize()
        kept = suppress_masks(records, store.read, infer['nms_iou'], infer['nms_containment'])
        local_queries = np.asarray([records[row]['query'] for row in kept], dtype=np.int64)
        for row in kept:
            record = records[row]
            if record['count'] > remaining_members:
                raise ValueError('Stage audit mask budget exceeded; no masks were silently truncated.')
            remaining_members -= record['count']
            path = folder / f'query_{record["query"]:03d}.npy'
            np.save(path, store.read(row), allow_pickle=False)
            saved = global_records.get((block_id, record['query']))
            replay_matches = bool(saved is not None and saved['count'] == record['count'] and
                                  np.isclose(saved['score'], record['score'], rtol=1e-5, atol=1e-6) and
                                  np.allclose(saved['bbox'], record['bbox'], rtol=0, atol=1e-5))
            mask_records[(block_id, record['query'])] = dict(record, path=str(path),
                global_id=int(saved['instance_id']) if saved else 0,
                replay_matches_saved_metadata=replay_matches,
                target_iou={int(target['gt_id']): float(metrics['iou'][record['query'], col]) for target, col in visible})
    voxel_tree_max = np.asarray(cells['gt'].max(axis=1).toarray()).ravel()
    voxel_tree_best = np.asarray(cells['gt'].argmax(axis=1)).ravel()
    background = cells['known'] - np.asarray(cells['gt'].sum(axis=1)).ravel()
    majority = np.where(voxel_tree_max > background, gt_ids[voxel_tree_best], 0)
    majority[cells['known'] == 0] = -1
    query_rows = []
    for query in range(len(scores)):
        col = int(metrics['iou'][query].argmax())
        query_rows.append(dict(query=query, score=float(scores[query]), total_points=int(metrics['total'][query]),
                               known_points=int(metrics['known'][query]), anchor_cell_majority_gt=int(majority[auxiliary['anchors'][query]]),
                               best_gt_id=int(gt_ids[col]) if metrics['iou'][query, col] > 0 else 0,
                               best_gt_iou=float(metrics['iou'][query, col]),
                               gt_at_half_recall=int((metrics['recall'][query] >= .5).sum()),
                               passes_quality=bool(query in quality), passes_size=bool(query in sized),
                               survives_local_nms=bool(query in local_queries)))
    target_rows = []
    for target, col in visible:
        identifier = int(target['gt_id'])
        fraction = float(cells['local_gt'][col] / gt_sizes[col])
        stage_sets = dict(raw=np.arange(len(scores)), quality=quality, size=sized, local_nms=local_queries)
        containing = np.asarray(cells['gt'].getcol(col).toarray()).ravel() > 0
        target_rows.append(dict(block=block_id, gt_id=identifier, crop_fraction=fraction, full_context=fraction >= .95,
            input_cells_with_gt_points=int(containing.sum()),
            core_cells_with_gt_points=int((containing & block['core']).sum()),
            foreground_cells_with_gt_points=int((containing & auxiliary['semantic_foreground']).sum()),
            isa_candidate_cells_with_gt_points=int(containing[auxiliary['query_candidates']].sum()),
            query_cells_with_gt_points=int(containing[auxiliary['anchors']].sum()),
            query_cells_majority_gt=int((majority[auxiliary['anchors']] == identifier).sum()),
            stages={name: best_for_gt(metrics, queries, col, scores) for name, queries in stage_sets.items()}))
    write_csv(folder / 'queries.csv', query_rows)
    return dict(block=block_id, input_voxels=len(block['xyz']), available_voxels=block['available_voxels'],
                raw_points=block['raw_points'], quality_candidates=len(quality), size_candidates=len(sized),
                local_kept=len(local_queries)), target_rows, layer_rows, mask_records, remaining_members


def global_witnesses(target, block_rows, masks, final_predictions, final_counts, gt_points, config):
    identifier = int(target['gt_id'])
    good = {(row['block'], query) for row in block_rows for query in row['stages']['local_nms']['good_queries']}
    accepted, conflicts = [], []
    for key in sorted(good):
        candidate = masks[key]
        if candidate['global_id']:
            instance_id = candidate['global_id']
            common = int(np.count_nonzero(final_predictions[gt_points[identifier]['indices']] == instance_id))
            final_iou = common / max(int(target['gt_points']) + final_counts[instance_id] - common, 1)
            accepted.append(dict(block=key[0], query=key[1], global_id=instance_id,
                                 raw_iou=candidate['target_iou'][identifier], allocated_iou=final_iou,
                                 replay_matches_saved_metadata=candidate['replay_matches_saved_metadata']))
            continue
        first = np.load(candidate['path'], mmap_mode='r', allow_pickle=False)
        try:
            for other_key, other in masks.items():
                if not other['global_id'] or not other['replay_matches_saved_metadata'] or other['score'] <= candidate['score'] + 1e-6:
                    continue
                if not (np.all(np.asarray(candidate['bbox'][:3]) <= other['bbox'][3:]) and
                        np.all(np.asarray(other['bbox'][:3]) <= candidate['bbox'][3:])):
                    continue
                second = np.load(other['path'], mmap_mode='r', allow_pickle=False)
                try:
                    overlap = nms_overlap(first, second, config['inference']['nms_iou'], config['inference']['nms_containment'])
                finally:
                    second._mmap.close()
                if overlap['conflicts']:
                    conflicts.append(dict(candidate_block=key[0], candidate_query=key[1],
                        candidate_iou=candidate['target_iou'][identifier], candidate_score=candidate['score'],
                        accepted_block=other_key[0], accepted_query=other_key[1], accepted_global_id=other['global_id'],
                        accepted_score=other['score'], accepted_iou=other['target_iou'].get(identifier, 0.), **overlap))
        finally:
            first._mmap.close()
    return dict(accepted_good_proposals=accepted, sampled_higher_score_nms_conflicts=conflicts)


def run(args):
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    work = Path(config['work_dir'])
    evaluation = Path(args.evaluation_dir or work / 'external_evaluation')
    posthoc = Path(args.posthoc_dir or work / 'diagnostics' / f'external_step_{args.expect_step:06d}' / args.plot)
    source = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
    audit = json.loads((posthoc / 'summary.json').read_text(encoding='utf-8'))
    if source['checkpoint_step'] != args.expect_step or audit['checkpoint_step'] != args.expect_step:
        raise ValueError('Evaluation/posthoc checkpoint step changed.')
    rows = entries(config, external=True)
    chosen = [row for row in rows if row['plot'] == args.plot]
    if len(chosen) != 1 or audit['plot'] != args.plot or audit.get('scope') != 'final_allocated_masks_only':
        raise ValueError('Plot or posthoc audit does not match the configured external forest.')
    cache = Path(chosen[0]['cache'])
    expected_paths = dict(coords=cache / 'coords.npy', labels=cache / 'labels.npy', baseline=cache / 'baseline.npy',
                          refined=evaluation / args.plot / 'refined_labels.npy')
    if set(audit['inputs']) != set(expected_paths):
        raise ValueError('Unexpected posthoc input manifest.')
    for name, stamp in audit['inputs'].items():
        path = Path(stamp['path'])
        if path.resolve() != expected_paths[name].resolve():
            raise ValueError('Posthoc inputs belong to another cache/evaluation directory.')
        if (path.stat().st_size, path.stat().st_mtime_ns) != (stamp['size'], stamp['mtime_ns']):
            raise ValueError('Posthoc input files changed; do not mix evaluations.')
    if source['inference'] != config['inference']:
        raise ValueError('Use the unchanged inference configuration for replay.')
    forest = RawForest(chosen[0]['cache'], chosen[0]['index'])
    labels = np.load(Path(chosen[0]['cache']) / 'labels.npy', mmap_mode='r', allow_pickle=False)
    final_predictions = np.load(evaluation / args.plot / 'refined_labels.npy', mmap_mode='r', allow_pickle=False)
    instances = json.loads((evaluation / args.plot / 'instances.json').read_text(encoding='utf-8'))
    global_records = {(int(row['block']), int(row['query'])): row for row in instances}
    instances_by_id = {int(row['instance_id']): row for row in instances}
    if len(global_records) != audit['models']['refined']['final_instance_count'] or len(instances_by_id) != len(global_records):
        raise ValueError('Saved global instance records changed.')
    tree_rows = read_csv(posthoc / 'trees.csv')
    tree_rows.sort(key=lambda row: int(row['gt_id']))
    gt_ids = np.asarray([int(row['gt_id']) for row in tree_rows], dtype=np.int64)
    gt_sizes = np.asarray([int(row['gt_points']) for row in tree_rows], dtype=np.int64)
    if (len(gt_ids) != audit['models']['refined']['metrics']['num_gt'] or
            np.any(np.diff(gt_ids) <= 0) or np.any(gt_sizes <= 0)):
        raise ValueError('Malformed or incomplete GT table in trees.csv.')
    targets = select_targets(tree_rows, config['seed'])
    points = selected_points(forest.arrays['coords'], labels, targets)
    block_ids = set()
    for target in targets:
        identifier = int(target['gt_id'])
        primary = int(np.square(forest.centers - points[identifier]['xy']).sum(axis=1).argmin())
        target['primary_block'] = primary
        block_ids.add(primary)
        culprit = instances_by_id.get(int(target['refined_best_pred_id']))
        if culprit is not None:
            target['saved_best_origin_block'] = int(culprit['block'])
            block_ids.add(int(culprit['block']))
    if len(block_ids) > 24 or any(block < 0 or block >= len(forest.centers) for block in block_ids):
        raise ValueError('Diagnostic block plan is invalid or exceeds 24 blocks.')
    output = Path(args.output or work / 'stage_audit' / f'step_{args.expect_step:06d}' / args.plot)
    if any(output.resolve().is_relative_to(path.resolve()) for path in (evaluation, cache, posthoc)):
        raise ValueError('Stage-audit output must be separate from source evaluations/caches.')
    output.mkdir(parents=True, exist_ok=True)
    for marker in ('brief.json', 'summary.json'):
        (output / marker).unlink(missing_ok=True)
    plan = dict(schema=1, posthoc=True, partial=True, eligible_for_advancement=False,
                scope='gt_selected_block_replay', checkpoint_step=args.expect_step, plot=args.plot,
                blocks=sorted(block_ids), targets=targets)
    write_json(output / 'plan.json', plan)
    write_json(output / 'progress.json', dict(completed=0, total=len(block_ids), phase='planned'))
    print(f'PLAN targets={len(targets)} unique_blocks={len(block_ids)}; error-enriched diagnostic, not forest metrics.', flush=True)
    if args.plan_only:
        return
    import torch
    from tools.forest_abc.run import configuration, device_for, ForestABC, seed_everything, signature

    config = configuration(args.config)
    checkpoint = Path(args.checkpoint or work / 'best_step_020000.pth')
    if source.get('checkpoint_sha256') != file_digest(checkpoint):
        raise ValueError('Checkpoint differs from the saved full-forest evaluation.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('check_only', True) or saved['step'] != args.expect_step or saved['signature'] != signature(config, entries(config)):
        raise ValueError('Require the compatible full-training checkpoint used for evaluation.')
    for relative, digest in saved['code'].items():
        if file_digest(ROOT / relative) != digest:
            raise ValueError(f'Model/pipeline source changed since training: {relative}.')
    seed_everything(config['seed'])
    device = device_for(config)
    model = ForestABC(config['model']).to(device).eval()
    model.load_state_dict(saved['model'])
    model.audit_mask_threshold = config['inference']['mask_threshold']
    del saved
    block_reports, all_rows, layers, masks = [], [], [], {}
    remaining = args.max_saved_memberships
    started = time.monotonic()
    for position, block_id in enumerate(sorted(block_ids), 1):
        print(f'REPLAY {position}/{len(block_ids)} original_block={block_id + 1}/{len(forest.centers)}', flush=True)
        report, rows, trace, records, remaining = replay_block(model, forest, labels, config, block_id,
            gt_ids, gt_sizes, targets, global_records, output, device, remaining)
        block_reports.append(report)
        all_rows.extend(rows)
        layers.extend(trace)
        masks.update(records)
        print(json.dumps(dict(event='block_done', completed=position, total=len(block_ids),
                              elapsed_seconds=time.monotonic() - started, **report)), flush=True)
        write_json(output / 'progress.json', dict(completed=position, total=len(block_ids), last_block=block_id))
    final_counts = {int(row['pred_id']): int(row['known_points']) for row in read_csv(posthoc / 'refined_predictions.csv')}
    findings = []
    for target in targets:
        identifier = int(target['gt_id'])
        rows = [row for row in all_rows if row['gt_id'] == identifier]
        witnesses = global_witnesses(target, rows, masks, final_predictions, final_counts, points, config)
        findings.append(dict(gt_id=identifier, stratum=target['stratum'], z_span_m=float(target['z_span_m']),
            sampled_blocks=len(rows), full_context_blocks=sum(row['full_context'] for row in rows),
            max_query_cells_with_gt_points=max((row['query_cells_with_gt_points'] for row in rows), default=0),
            max_query_cells_majority_gt=max((row['query_cells_majority_gt'] for row in rows), default=0),
            max_core_cells_with_gt_points=max((row['core_cells_with_gt_points'] for row in rows), default=0),
            max_isa_candidate_cells_with_gt_points=max((row['isa_candidate_cells_with_gt_points'] for row in rows), default=0),
            saved_detected=target['refined_detected'] == 'True', saved_best_iou=float(target['refined_best_iou']),
            category=classify_target(rows, target['refined_detected'] == 'True', float(target['refined_best_iou'])),
            sampled_best_iou={name: max((row['stages'][name]['iou'] for row in rows), default=0.)
                              for name in ('raw', 'quality', 'size', 'local_nms')},
            heads_best_iou={str(head): max((row['iou'] for row in layers if row['gt_id'] == identifier and
                                           row['head'] == head), default=0.)
                            for head in range(config['model']['layers'] + 1)},
            num_sampled_nms_conflicts=len(witnesses['sampled_higher_score_nms_conflicts']),
            num_accepted_good_masks_degraded=sum(row['replay_matches_saved_metadata'] and row['allocated_iou'] <= .5
                                                for row in witnesses['accepted_good_proposals']), **witnesses))
    replay_mismatches = [dict(block=block, query=query) for (block, query), record in global_records.items()
                         if block in block_ids and ((block, query) not in masks or
                         not masks[(block, query)]['replay_matches_saved_metadata'])]
    for stamp in audit['inputs'].values():
        path = Path(stamp['path'])
        if (path.stat().st_size, path.stat().st_mtime_ns) != (stamp['size'], stamp['mtime_ns']):
            raise ValueError('An input changed during replay; do not publish a mixed report.')
    brief = dict(schema=1, posthoc=True, partial=True, eligible_for_advancement=False,
        scope='gt_selected_block_replay', checkpoint_step=args.expect_step, plot=args.plot,
        sampled_blocks=len(block_ids), sampled_targets=len(targets),
        categories=dict(Counter(row['category'] for row in findings)),
        replay_metadata_mismatches=replay_mismatches,
        targets=[{key: value for key, value in row.items() if key not in
                  ('accepted_good_proposals', 'sampled_higher_score_nms_conflicts')} for row in findings])
    flat_rows = []
    for row in all_rows:
        flat = {key: value for key, value in row.items() if key != 'stages'}
        for stage, values in row['stages'].items():
            flat.update({f'{stage}_{key}': json.dumps(value) if isinstance(value, list) else value
                         for key, value in values.items()})
        flat_rows.append(flat)
    write_csv(output / 'block_targets.csv', flat_rows)
    write_csv(output / 'layer_targets.csv', layers)
    write_json(output / 'summary.json', dict(brief, checkpoint=str(checkpoint), plan=plan,
        blocks=block_reports, block_targets=all_rows, findings=findings,
        saved_memberships=args.max_saved_memberships - remaining, replay_seconds=time.monotonic() - started,
        notes=['GT selects diagnostic cases and measures masks; model inputs remain coords/grid/neighbors/core only.',
               'Full-tree GT denominators are used; negative GT points are ignored. Good IoU is strictly > 0.5.',
               'Negative raw-mask findings require at least one sampled block containing >=95% of the target GT.',
               'Query cell counts use NN cell GT membership/majority, not exact representative-point or training-voxel labels.',
               'Only selected original blocks are replayed; absence of a witness does not clear global NMS or the decoder.',
               'Higher-score NMS conflicts are sampled witnesses, not an exact replay of all 500-block decisions.',
               'Saved accepted metadata checks score/count/bbox; deleted original masks prevent bitwise membership verification.',
               'Layer traces locate changes during this forward pass; they do not establish module-ablation causality.',
               'No predictions, training parameters, thresholds or checkpoints are updated.']))
    write_json(output / 'brief.json', brief)
    print(json.dumps(brief, indent=2), flush=True)
    print(f'SAVED {output / "brief.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/forest_abc/full.yaml')
    parser.add_argument('--plot', default='wytham_vox0.1')
    parser.add_argument('--checkpoint')
    parser.add_argument('--expect-step', type=int, default=20000)
    parser.add_argument('--evaluation-dir')
    parser.add_argument('--posthoc-dir')
    parser.add_argument('--output')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--max-saved-memberships', type=int, default=100_000_000)
    args = parser.parse_args()
    if args.max_saved_memberships <= 0:
        parser.error('--max-saved-memberships must be positive.')
    os.chdir(ROOT)
    run(args)


if __name__ == '__main__':
    main()
