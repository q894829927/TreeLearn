"""Whole-forest prediction from coordinates, bounded blocks and disk-backed NMS."""

import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from tree_learn.mask_refinement.data import write_json
from tree_learn.query_segmentation.inference import BlockMaskStore, suppress_masks
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
    flat_path = output / 'proposal_members.bin'
    records, total_members = [], 0
    started = time.monotonic()
    centers = forest.centers if max_blocks is None else forest.centers[:max_blocks]
    scored, score_sum, skipped, subsampled = 0, 0., 0, 0
    maximum_voxels, maximum_points = 0, 0
    model.eval()
    with flat_path.open('wb') as stream:
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
            mask_logits = layer['masks'][:-2]
            quality_logits = layer['scores'][:-2]
            if not bool(torch.isfinite(mask_logits).all() & torch.isfinite(quality_logits).all()):
                raise FloatingPointError(f'Non-finite prediction in block {block_id + 1}.')
            quality = quality_logits.sigmoid()
            probability = mask_logits.sigmoid()
            membership = probability >= infer['mask_threshold']
            mask_confidence = (probability * membership).sum(-1) / membership.sum(-1).clamp_min(1)
            scores = (quality * mask_confidence).cpu().numpy()
            voxel_masks = membership.cpu().numpy()
            scored += len(scores)
            score_sum += float(scores.sum())
            passing = np.flatnonzero((scores >= infer['quality_threshold']) & voxel_masks.any(-1))
            del prediction, layer, mask_logits, quality_logits, quality, probability, membership, mask_confidence
            if len(passing):
                tree_index = cKDTree(block['source_xyz'])
                with BlockMaskStore(output, len(passing), dtype,
                                    infer['max_block_memberships']) as store:
                    for start in range(0, len(block['indices']), infer['chunk_size']):
                        indices = block['indices'][start:start + infer['chunk_size']]
                        _, nearest = tree_index.query(forest.arrays['coords'][indices], k=1, workers=1)
                        for local, query in enumerate(passing):
                            selected = voxel_masks[query, nearest]
                            if selected.any():
                                store.append(local, indices[selected])
                    store.finalize()
                    candidates, local_rows = [], []
                    for local, query in enumerate(passing):
                        if store.counts[local] < infer['minimum_instance_points']:
                            continue
                        indices = store.read(local)
                        xyz = forest.arrays['coords'][indices]
                        candidates.append(dict(score=float(scores[query]), count=int(len(indices)),
                                               bbox=np.concatenate([xyz.min(0), xyz.max(0)]).tolist(),
                                               block=block_id, query=int(query)))
                        local_rows.append(local)
                    kept = suppress_masks(candidates, lambda row: store.read(local_rows[row]),
                                          infer['nms_iou'], infer['nms_containment'])
                    for row in kept:
                        record = candidates[row]
                        indices = store.read(local_rows[row])
                        if total_members + len(indices) > infer['max_forest_memberships']:
                            raise RuntimeError('Forest proposal disk budget exceeded. See prediction progress.')
                        records.append(dict(record, offset=total_members))
                        indices.tofile(stream)
                        total_members += len(indices)
                del tree_index
            del block, voxel_masks
            if block_id == 0 or (block_id + 1) % infer['log_every_blocks'] == 0:
                print(f'PREDICT block={block_id + 1}/{len(centers)} proposals={len(records)} '
                      f'mean_score={score_sum / max(scored, 1):.4f} '
                      f'seconds={time.monotonic() - started:.1f}', flush=True)
    labels_path = output / 'refined_labels.npy'
    labels = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.int32, shape=(count,))
    labels[:] = 0
    assigned, rejected = [], 0
    kept = []
    if total_members:
        members = np.memmap(flat_path, mode='r', dtype=dtype, shape=(total_members,))

        def read(row):
            record = records[row]
            return members[record['offset']:record['offset'] + record['count']]

        print(f'GLOBAL_NMS proposals={len(records)} memberships={total_members}', flush=True)
        def allocate(row):
            nonlocal rejected
            record = records[row]
            indices = read(row)
            available = labels[indices] == 0
            retained = int(available.sum())
            ratio = retained / len(indices)
            # A rejected fragment must neither claim points nor suppress any
            # later proposal that survived local proposal pruning.
            if (retained < infer['minimum_instance_points'] or
                    ratio < infer['minimum_retention'] or
                    record['score'] * ratio < infer['allocation_score_threshold']):
                rejected += 1
                return False
            identity = len(assigned) + 1
            labels[indices[available]] = identity
            assigned.append(dict(instance_id=identity, final_count=retained,
                                 retained_fraction=ratio,
                                 **{k: v for k, v in record.items() if k != 'offset'}))
            return True

        try:
            kept = suppress_masks(records, read, infer['nms_iou'], infer['nms_containment'], accept=allocate)
        finally:
            members._mmap.close()
    labels.flush()
    del labels
    if not infer['keep_proposals']:
        flat_path.unlink()
    stats = dict(partial=max_blocks is not None, blocks=len(centers), skipped_blocks=skipped,
                 subsampled_blocks=subsampled, max_input_voxels=maximum_voxels,
                 max_block_points=maximum_points, num_queries_scored=scored,
                 mean_score=score_sum / max(scored, 1), local_proposals=len(records),
                 global_kept=len(kept), allocation_rejected=rejected,
                 final_instances=len(assigned), proposal_memberships=total_members,
                 inference_seconds=time.monotonic() - started,
                 empty_prediction_reason=None if assigned else 'no_complete_masks_survived')
    write_json(output / 'instances.json', assigned)
    write_json(output / 'prediction.json', stats)
    return np.load(labels_path, mmap_mode='r', allow_pickle=False), stats
