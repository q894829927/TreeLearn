"""Streaming fine-point masks and disk-backed mask NMS; no labels are read."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from tree_learn.mask_refinement.data import write_json


def boxes_overlap(first, second):
    return np.all(np.asarray(first[:3]) <= second[3:]) and np.all(np.asarray(second[:3]) <= first[3:])


def suppress_masks(records, read_mask, iou_threshold=.5, containment_threshold=.8):
    """Score order, exact point overlap, and an XY grid to avoid all-pairs NMS."""
    kept, buckets = [], {}
    for row in sorted(range(len(records)), key=lambda i: (-records[i]['score'], i)):
        record = records[row]
        box = record['bbox']
        lo = np.floor(np.asarray(box[:2]) / 8).astype(int)
        hi = np.floor(np.asarray(box[3:5]) / 8).astype(int)
        cells = [(x, y) for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1)]
        neighbors = set()
        for cell in cells:
            neighbors.update(buckets.get(cell, ()))
        indices, rejected = None, False
        for other in sorted(neighbors):
            previous = records[other]
            if not boxes_overlap(box, previous['bbox']):
                continue
            if indices is None:
                indices = read_mask(row)
            intersection = len(np.intersect1d(indices, read_mask(other), assume_unique=True))
            union = record['count'] + previous['count'] - intersection
            if (intersection / max(union, 1) >= iou_threshold or
                    intersection / max(min(record['count'], previous['count']), 1) >= containment_threshold):
                rejected = True
                break
        if not rejected:
            kept.append(row)
            for cell in cells:
                buckets.setdefault(cell, []).append(row)
    return kept


@torch.no_grad()
def predict_forest(model, forest, settings, output, device, progress=1., max_blocks=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    block_cfg, infer = settings['blocks'], settings['inference']
    points_count = forest.metadata['num_points']
    dtype = np.int32 if points_count < 2 ** 31 else np.int64
    flat_path = output / 'proposal_members.bin'
    records, total_members = [], 0
    started = time.monotonic()
    score_count, score_sum, score_min, score_max = 0, 0., 1., 0.
    centers = forest.centers if max_blocks is None else forest.centers[:max_blocks]
    model.eval()
    with flat_path.open('wb') as stream:
        for block_id, center in enumerate(centers):
            try:
                block = forest.make_block(center, block_cfg, settings['seed'] + block_id)
            except ValueError as error:
                if str(error) not in ('Empty spatial block.', 'No query anchors in the central region.'):
                    raise
                continue
            tokens = torch.from_numpy(block['tokens']).to(device)
            core = torch.from_numpy(block['core']).to(device)
            decoded = model.decode(tokens, core, progress)
            scores = decoded['quality_logits'].sigmoid().cpu().numpy()
            score_count += len(scores)
            score_sum += float(scores.sum())
            score_min, score_max = min(score_min, float(scores.min())), max(score_max, float(scores.max()))
            passing = np.flatnonzero(scores >= infer['quality_threshold'])
            if len(passing):
                query = decoded['queries'][-1][torch.from_numpy(passing).to(device)]
                geometry = {k: v[torch.from_numpy(passing).to(device)] for k, v in decoded['geometry'][-1].items()}
                parts = [[] for _ in passing]
                memberships = 0
                for start in range(0, len(block['indices']), infer['chunk_size']):
                    indices = block['indices'][start:start + infer['chunk_size']]
                    values = torch.from_numpy(forest.inputs(indices, block['center'], block_cfg['radius'])).to(device)
                    probabilities = model.point_predictions(values, [query], [geometry])['masks'][0].sigmoid()
                    masks = (probabilities >= infer['mask_threshold']).cpu().numpy()
                    memberships += int(masks.sum())
                    if memberships > infer['max_block_memberships']:
                        raise RuntimeError('Too many dense proposal memberships in one block; inspect mask/score learning.')
                    for row, mask in enumerate(masks):
                        if mask.any():
                            parts[row].append(indices[mask].astype(dtype))
                local_records, local_masks = [], []
                for row, pieces in enumerate(parts):
                    if not pieces:
                        continue
                    indices = np.sort(np.concatenate(pieces))
                    xyz = forest.arrays['coords'][indices]
                    local_masks.append(indices)
                    local_records.append(dict(score=float(scores[passing[row]]), count=len(indices),
                                              bbox=np.concatenate([xyz.min(axis=0), xyz.max(axis=0)]).tolist(),
                                              block=block_id, query=int(passing[row])))
                kept = suppress_masks(local_records, lambda i: local_masks[i], infer['nms_iou'], infer['nms_containment'])
                for row in kept:
                    record, indices = local_records[row], local_masks[row]
                    if total_members + len(indices) > infer['max_forest_memberships']:
                        raise RuntimeError('Proposal disk budget exceeded; inspect quality diagnostics and block stride.')
                    record['offset'] = total_members
                    indices.tofile(stream)
                    total_members += len(indices)
                    records.append(record)
                del local_masks, local_records, parts
            if block_id == 0 or (block_id + 1) % infer['log_every_blocks'] == 0:
                print(json.dumps(dict(event='predict', block=block_id + 1, blocks=len(centers),
                                      retained_local_masks=len(records), mean_quality=score_sum / max(score_count, 1),
                                      elapsed_seconds=time.monotonic() - started)), flush=True)
    labels_path = output / 'refined_labels.npy'
    labels = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.int32, shape=(points_count,))
    labels[:] = 0
    kept_rows, assigned_records = [], []
    if total_members:
        members = np.memmap(flat_path, mode='r', dtype=dtype, shape=(total_members,))

        def read(row):
            record = records[row]
            return members[record['offset']:record['offset'] + record['count']]

        print(f'GLOBAL_NMS proposals={len(records)} members={total_members}', flush=True)
        kept_rows = suppress_masks(records, read, infer['nms_iou'], infer['nms_containment'])
        for row in kept_rows:
            indices = read(row)
            unclaimed = labels[indices] == 0
            count = int(unclaimed.sum())
            if not count:
                continue
            identity = len(assigned_records) + 1
            labels[indices[unclaimed]] = identity
            record = {k: v for k, v in records[row].items() if k != 'offset'}
            assigned_records.append(dict(record, instance_id=identity, final_count=count))
        del indices
        members._mmap.close()
        del members
    labels.flush()
    del labels
    if not infer.get('keep_proposals', False):
        flat_path.unlink()
    write_json(output / 'instances.json', assigned_records)
    stats = dict(num_queries_scored=score_count, quality_min=score_min if score_count else None,
                 quality_max=score_max if score_count else None,
                 quality_mean=score_sum / max(score_count, 1),
                 local_proposals=len(records), global_kept=len(kept_rows),
                 final_instances=len(assigned_records), proposal_memberships=total_members,
                 inference_seconds=time.monotonic() - started,
                 empty_prediction_reason='no_masks_survived_quality_and_nms' if not assigned_records else None,
                 partial=max_blocks is not None)
    write_json(output / 'prediction.json', stats)
    return np.load(labels_path, mmap_mode='r', allow_pickle=False), stats
