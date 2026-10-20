"""Bounded fine-point projection, duplicate fusion, and two-pass ownership."""

import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
import torch

from tree_learn.mask_refinement.data import write_json
from .merge import compete, group_proposals
from .model import tensor_block


@torch.no_grad()
def predict_forest(model, forest, config, output, device, max_blocks=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('prediction.json', 'instances.json', 'metrics.json'):
        (output / name).unlink(missing_ok=True)
    infer, blocks = config['inference'], config['blocks']
    count = forest.metadata['num_points']
    dtype = np.int32 if count < 2 ** 31 else np.int64
    members_path, evidence_path = output / 'proposal_members.bin', output / 'proposal_evidence.bin'
    records, total_members, scored, score_sum = [], 0, 0, 0.
    skipped, subsampled, topk_pruned, maximum_voxels, maximum_points = 0, 0, 0, 0, 0
    started = time.monotonic()
    centers = forest.centers if max_blocks is None else forest.centers[:max_blocks]
    model.eval()
    with members_path.open('wb') as members_stream, evidence_path.open('wb') as evidence_stream:
        for block_id, center in enumerate(centers):
            block = forest.make_block(center, blocks, config['seed'] + block_id)
            if block is None:
                skipped += 1
                continue
            maximum_voxels = max(maximum_voxels, len(block['xyz']))
            maximum_points = max(maximum_points, block['raw_points'])
            subsampled += int(block['available_voxels'] > len(block['xyz']))
            prediction = model(tensor_block(block, device))
            layer = prediction['layers'][-1]
            logits = layer['masks'][:-2]
            quality = layer['scores'][:-2].sigmoid()
            if not all(bool(torch.isfinite(value).all()) for value in
                       (layer['masks'], layer['scores'], prediction['centers'])):
                raise FloatingPointError(f'Non-finite V2 prediction in block {block_id + 1}.')
            probability = logits.sigmoid()
            membership = probability >= infer['mask_threshold']
            mask_confidence = (probability * membership).sum(-1) / membership.sum(-1).clamp_min(1)
            scores = (quality * mask_confidence).cpu().numpy()
            passing = np.flatnonzero((scores >= infer['quality_threshold']) & membership.any(-1).cpu().numpy())
            passing = np.asarray(sorted(passing, key=lambda q: (-scores[q], q)), dtype=np.int64)
            topk_pruned += max(0, len(passing) - infer['max_proposals_per_block'])
            passing = passing[:infer['max_proposals_per_block']]
            raw = logits[passing].cpu().numpy()
            background = (layer['masks'][-2] - layer['masks'][-1]).cpu().numpy()
            query_centers = prediction['centers'][passing].cpu().numpy() + block['center'][:2]
            scored += len(scores)
            score_sum += float(scores.sum())
            del prediction, layer, logits, quality, probability, membership, mask_confidence
            if len(passing):
                # Pairwise ownership vs background uses the same final logits as
                # training; query quality supplies a fixed log prior at inference.
                evidence = expit(raw - background[None] +
                                 infer['quality_log_weight'] * np.log(scores[passing, None].clip(1e-8)))
                binary = ((expit(raw) >= infer['mask_threshold']) &
                          (evidence >= infer['ownership_threshold']))
                coarse = [np.flatnonzero(mask) for mask in binary]
                local_records = []
                for local, query in enumerate(passing):
                    local_records.append(dict(score=float(scores[query]), count=len(coarse[local]),
                        center_xy=query_centers[local].tolist(), bbox=[-float('inf')] * 3 + [float('inf')] * 3))
                local_groups, _ = group_proposals(local_records, lambda i: (coarse[i], None),
                    infer['center_merge_distance'], infer['mutual_overlap'])
                nearest = np.empty(len(block['indices']), dtype=np.int32)
                tree = cKDTree(block['source_xyz'])
                for start in range(0, len(nearest), infer['chunk_size']):
                    indices = block['indices'][start:start + infer['chunk_size']]
                    nearest[start:start + len(indices)] = tree.query(forest.arrays['coords'][indices],
                                                                      k=1, workers=1)[1]
                block_members = 0
                for group in local_groups:
                    best = np.zeros(len(block['xyz']), dtype=np.float32)
                    for local in group['members']:
                        np.maximum(best, np.where(binary[local], evidence[local], 0.), out=best)
                    selected = np.flatnonzero(best[nearest] >= infer['ownership_threshold'])
                    if len(selected) < infer['minimum_instance_points']:
                        continue
                    if (block_members + len(selected) > infer['max_block_memberships'] or
                            total_members + len(selected) > infer['max_forest_memberships']):
                        raise RuntimeError('V2 proposal disk budget exceeded; inspect proposal statistics. '
                                           'No memberships were silently truncated.')
                    indices = block['indices'][selected]
                    order = np.argsort(indices, kind='stable')
                    indices = indices[order].astype(dtype)
                    values = best[nearest[selected]][order].astype(np.float32)
                    positions = forest.arrays['coords'][indices]
                    local = group['leader']
                    records.append(dict(offset=total_members, count=len(indices), score=float(group['score']),
                        center_xy=group['center_xy'], block=block_id, query=int(passing[local]),
                        local_observations=len(group['members']),
                        bbox=np.concatenate((positions.min(0), positions.max(0))).tolist()))
                    indices.tofile(members_stream)
                    values.tofile(evidence_stream)
                    total_members += len(indices)
                    block_members += len(indices)
            if block_id == 0 or (block_id + 1) % infer['log_every_blocks'] == 0 or block_id + 1 == len(centers):
                print(f'PREDICT block={block_id + 1}/{len(centers)} proposals={len(records)} '
                      f'memberships={total_members} mean_score={score_sum / max(scored, 1):.4f} '
                      f'seconds={time.monotonic() - started:.1f}', flush=True)
    labels_path = output / 'refined_labels.npy'
    labels = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.int32, shape=(count,))
    confidence_path, stamp_path = output / 'allocation_confidence.npy', output / 'union_stamp.npy'
    confidence = np.lib.format.open_memmap(confidence_path, mode='w+', dtype=np.float32, shape=(count,))
    stamp = np.lib.format.open_memmap(stamp_path, mode='w+', dtype=np.int32, shape=(count,))
    groups, instances, rejected = [], [], 0
    try:
        if total_members:
            members = np.memmap(members_path, mode='r', dtype=dtype, shape=(total_members,))
            values = np.memmap(evidence_path, mode='r', dtype=np.float32, shape=(total_members,))

            def read(row):
                record = records[row]
                start, stop = record['offset'], record['offset'] + record['count']
                return members[start:stop], values[start:stop]

            try:
                print(f'GLOBAL_FUSION proposals={len(records)} memberships={total_members}', flush=True)
                groups, assignment = group_proposals(records, read, infer['center_merge_distance'], infer['mutual_overlap'])
                instances, rejected = compete(records, read, groups, assignment, labels, confidence, stamp, infer)
            finally:
                members._mmap.close()
                values._mmap.close()
        else:
            labels[:] = 0
        labels.flush()
    finally:
        labels._mmap.close()
        confidence._mmap.close()
        stamp._mmap.close()
    confidence_path.unlink()
    stamp_path.unlink()
    if not infer['keep_proposals']:
        members_path.unlink()
        evidence_path.unlink()
    stats = dict(partial=max_blocks is not None, blocks=len(centers), skipped_blocks=skipped,
                 subsampled_blocks=subsampled, max_input_voxels=maximum_voxels,
                 max_block_points=maximum_points, num_queries_scored=scored,
                 mean_score=score_sum / max(scored, 1), topk_pruned=topk_pruned,
                 local_proposals=len(records), duplicate_groups=len(groups), allocation_rejected=rejected,
                 final_instances=len(instances), proposal_memberships=total_members,
                 inference_seconds=time.monotonic() - started,
                 empty_prediction_reason=None if instances else 'no_groups_survived_ownership')
    write_json(output / 'instances.json', instances)
    write_json(output / 'prediction.json', stats)
    return np.load(labels_path, mmap_mode='r', allow_pickle=False), stats
