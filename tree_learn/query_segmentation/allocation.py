"""GT-free validity filtering AFTER all masks have claimed their points."""

import numpy as np


def filter_allocation(source, records, destination, threshold=.4, chunk_size=1_000_000):
    """Drop whole allocated masks without reassigning their points; compact IDs.

    Delaying deletion until allocation is complete reproduces the saved-mask
    diagnosis. Rejecting a mask while claiming points would change later masks.
    """
    if (source.shape != destination.shape or source.ndim != 1 or
            not np.issubdtype(source.dtype, np.integer) or
            not np.issubdtype(destination.dtype, np.integer) or
            chunk_size < 1 or not 0 <= threshold <= 1):
        raise ValueError('Invalid label arrays, threshold or chunk size.')
    ordered = sorted(records, key=lambda record: record['instance_id'])
    if [r['instance_id'] for r in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError('Expected consecutive allocated instance IDs starting at one.')
    if len(ordered) > np.iinfo(destination.dtype).max:
        raise ValueError('Destination label dtype cannot hold the instance IDs.')
    lookup = np.zeros(len(ordered) + 1, dtype=destination.dtype)
    kept, rejected = [], []
    expected = np.zeros(len(ordered) + 1, dtype=np.int64)
    for record in ordered:
        identity = int(record['instance_id'])
        raw, final, score = record['count'], record['final_count'], record['score']
        if (not np.isfinite([raw, final, score]).all() or not 0 < final <= raw or
                int(raw) != raw or int(final) != final or not 0 <= score <= 1):
            raise ValueError('Invalid allocated mask counts or score.')
        expected[identity] = final
        retention = float(final / raw)
        allocation_score = float(score * retention)
        updated = dict(record, retained_fraction=retention, allocation_score=allocation_score)
        if allocation_score >= threshold:
            lookup[identity] = len(kept) + 1
            updated.update(source_instance_id=record.get('source_instance_id', identity), instance_id=len(kept) + 1)
            kept.append(updated)
        else:
            rejected.append(updated)
    seen = np.zeros(len(ordered) + 1, dtype=np.int64)
    for start in range(0, len(source), chunk_size):
        values = source[start:start + chunk_size]
        if len(values) and (values.min() < 0 or values.max() > len(ordered)):
            raise ValueError('Saved labels contain an ID absent from instance metadata.')
        values = values.astype(np.int64, copy=False)
        seen += np.bincount(values, minlength=len(seen))
        destination[start:start + chunk_size] = lookup[values]
    if not np.array_equal(seen[1:], expected[1:]):
        raise ValueError('Allocated label counts disagree with instance metadata.')
    statistics = dict(method='quality_times_retention', threshold=float(threshold),
                      formula='raw_quality * final_count / raw_mask_count',
                      reassignment=False, input_instances=len(ordered), output_instances=len(kept),
                      removed_instances=len(rejected),
                      removed_points=sum(int(r['final_count']) for r in rejected),
                      retained_points=sum(int(r['final_count']) for r in kept))
    return kept, rejected, statistics
