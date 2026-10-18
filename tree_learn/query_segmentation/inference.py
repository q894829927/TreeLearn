"""Streaming fine-point masks and disk-backed mask NMS; no labels are read."""

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tree_learn.mask_refinement.data import write_json
from tree_learn.query_segmentation.allocation import filter_allocation


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


class BlockMaskStore:
    """Keep small blocks in RAM and spill dense blocks without dropping masks."""
    def __init__(self, directory, queries, dtype, memory_memberships):
        self.temporary = TemporaryDirectory(prefix='block_masks_', dir=directory)
        self.root = Path(self.temporary.name)
        self.dtype = np.dtype(dtype)
        self.memory_memberships = int(memory_memberships)
        if self.memory_memberships <= 0:
            self.temporary.cleanup()
            raise ValueError('Block mask memory budget must be positive.')
        self.parts = [[] for _ in range(queries)]
        self.counts = np.zeros(queries, dtype=np.int64)
        self.buffered = 0
        self.spilled = False
        self.ready = False
        self.arrays = {}

    def __enter__(self):
        return self

    def append(self, row, indices):
        if self.ready:
            raise RuntimeError('Cannot append to finalized block masks.')
        if not len(indices):
            return
        values = np.asarray(indices, dtype=self.dtype).copy()
        # Flush before an append could push buffered membership past the budget.
        if self.buffered + len(values) > self.memory_memberships:
            self.flush()
        self.counts[row] += len(values)
        if len(values) > self.memory_memberships:
            self._write(row, [values])
            self.spilled = True
        else:
            self.parts[row].append(values)
            self.buffered += len(values)

    def _write(self, row, pieces):
        with (self.root / f'{row}.bin').open('ab') as stream:
            for values in pieces:
                values.tofile(stream)

    def flush(self):
        self.spilled = True
        for row, pieces in enumerate(self.parts):
            if pieces:
                self._write(row, pieces)
                pieces.clear()
        self.buffered = 0

    def finalize(self):
        if self.ready:
            return
        if self.spilled:
            self.flush()
        else:
            for row, pieces in enumerate(self.parts):
                self.arrays[row] = (np.concatenate(pieces) if pieces else np.empty(0, dtype=self.dtype))
                self.arrays[row].sort()
                pieces.clear()
            self.buffered = 0
        self.ready = True

    def read(self, row):
        if not self.ready:
            raise RuntimeError('Finalize block masks before reading.')
        if row not in self.arrays:
            if self.counts[row]:
                values = np.memmap(self.root / f'{row}.bin', mode='r+', dtype=self.dtype,
                                   shape=(int(self.counts[row]),))
                values.sort()
                values.flush()
                self.arrays[row] = values
            else:
                self.arrays[row] = np.empty(0, dtype=self.dtype)
        return self.arrays[row]

    def __exit__(self, *error):
        for values in self.arrays.values():
            if isinstance(values, np.memmap):
                values._mmap.close()
        self.arrays.clear()
        self.parts.clear()
        self.temporary.cleanup()


@torch.no_grad()
def predict_forest(model, forest, settings, output, device, progress=1., max_blocks=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    block_cfg, infer = settings['blocks'], settings['inference']
    points_count = forest.metadata['num_points']
    dtype = np.int32 if points_count < 2 ** 31 else np.int64
    flat_path = output / 'proposal_members.bin'
    records, total_members = [], 0
    spilled_blocks, max_block_memberships = 0, 0
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
                with BlockMaskStore(output, len(passing), dtype, infer['max_block_memberships']) as store:
                    for start in range(0, len(block['indices']), infer['chunk_size']):
                        indices = block['indices'][start:start + infer['chunk_size']]
                        values = torch.from_numpy(forest.inputs(indices, block['center'], block_cfg['radius'])).to(device)
                        probabilities = model.point_predictions(values, [query], [geometry])['masks'][0].sigmoid()
                        masks = (probabilities >= infer['mask_threshold']).cpu().numpy()
                        for row, mask in enumerate(masks):
                            if mask.any():
                                store.append(row, indices[mask])
                    store.finalize()
                    memberships = int(store.counts.sum())
                    max_block_memberships = max(max_block_memberships, memberships)
                    if store.spilled:
                        spilled_blocks += 1
                        print(json.dumps(dict(event='block_masks_spilled', block=block_id + 1,
                                              memberships=memberships,
                                              memory_memberships=infer['max_block_memberships'])), flush=True)
                    local_records, local_rows = [], []
                    for row, count in enumerate(store.counts):
                        if not count:
                            continue
                        indices = store.read(row)
                        xyz = forest.arrays['coords'][indices]
                        local_rows.append(row)
                        local_records.append(dict(score=float(scores[passing[row]]), count=int(count),
                                                  bbox=np.concatenate([xyz.min(axis=0), xyz.max(axis=0)]).tolist(),
                                                  block=block_id, query=int(passing[row])))
                    kept = suppress_masks(local_records, lambda i: store.read(local_rows[i]),
                                          infer['nms_iou'], infer['nms_containment'])
                    for row in kept:
                        record, indices = local_records[row], store.read(local_rows[row])
                        if total_members + len(indices) > infer['max_forest_memberships']:
                            raise RuntimeError('Proposal disk budget exceeded; inspect quality diagnostics and block stride.')
                        record['offset'] = total_members
                        indices.tofile(stream)
                        total_members += len(indices)
                        records.append(record)
                    del indices
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
    allocation_statistics = None
    if infer.get('allocation_filter', 'none') == 'quality_times_retention':
        assigned_records, rejected, allocation_statistics = filter_allocation(
            labels, assigned_records, labels, infer['quality_threshold'])
        write_json(output / 'allocation_filter.json', allocation_statistics)
        write_json(output / 'rejected_instances.json', rejected)
    elif infer.get('allocation_filter', 'none') != 'none':
        raise ValueError('Unsupported allocation_filter.')
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
                 spilled_blocks=spilled_blocks, max_block_memberships=max_block_memberships,
                 allocation_filter=allocation_statistics,
                 inference_seconds=time.monotonic() - started,
                 empty_prediction_reason=('all_allocated_masks_filtered' if allocation_statistics and
                                          allocation_statistics['input_instances'] else
                                          'no_masks_survived_quality_and_nms') if not assigned_records else None,
                 partial=max_blocks is not None)
    write_json(output / 'prediction.json', stats)
    return np.load(labels_path, mmap_mode='r', allow_pickle=False), stats
