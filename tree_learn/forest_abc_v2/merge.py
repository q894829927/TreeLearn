"""Center-aware duplicate fusion and point competition. No GT is read here."""

from collections import defaultdict
import numpy as np


def boxes_overlap(first, second):
    return np.all(np.asarray(first[:3]) <= second[3:]) and np.all(np.asarray(second[:3]) <= first[3:])


def group_proposals(records, read, distance, mutual_overlap):
    """Keep every observation; merge only with a fixed, overlapping leader.

    Requiring both masks to overlap substantially avoids treating a small mask
    contained in a large mixed mask as a duplicate. Fixed leaders avoid chains
    of pairwise-close centers merging arbitrarily distant trees.
    """
    buckets, groups, assignment = defaultdict(list), [], np.full(len(records), -1, dtype=np.int32)
    for position, row in enumerate(sorted(range(len(records)), key=lambda i: (-records[i]['score'], i)), 1):
        record = records[row]
        center = np.asarray(record['center_xy'], dtype=np.float64)
        key = tuple(np.floor(center / distance).astype(np.int64))
        candidates = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.update(buckets[(key[0] + dx, key[1] + dy)])
        chosen, mask = None, None
        for group in sorted(candidates):
            leader = records[groups[group]['leader']]
            if (np.linalg.norm(center - np.asarray(leader['center_xy'])) > distance or
                    not boxes_overlap(record['bbox'], leader['bbox'])):
                continue
            if mask is None:
                mask = read(row)[0]
            other = read(groups[group]['leader'])[0]
            intersection = len(np.intersect1d(mask, other, assume_unique=True))
            if intersection / max(record['count'], leader['count'], 1) >= mutual_overlap:
                chosen = group
                break
        if chosen is None:
            chosen = len(groups)
            groups.append(dict(leader=row, members=[], score=record['score'], center_xy=record['center_xy']))
            buckets[key].append(chosen)
        groups[chosen]['members'].append(row)
        assignment[row] = chosen
        if position == 1 or position % 1000 == 0 or position == len(records):
            print(f'FUSE proposals={position}/{len(records)} groups={len(groups)}', flush=True)
    return groups, assignment


def compete(records, read, groups, assignment, labels, confidence, union_stamp, config):
    """Two passes: compete, reject fragments, then release their points.

    Remaining groups can only gain points on pass two. Acceptance uses monotone
    counts/retention, so no iteration or unbounded reassignment is necessary.
    Evidence is the maximum of observations, never a sum over overlapping tiles.
    """
    chunk = config['chunk_size']
    unions = np.zeros(len(groups), dtype=np.int64)
    union_stamp[:] = -1
    for group, info in enumerate(groups):
        for row in info['members']:
            indices, _ = read(row)
            for start in range(0, len(indices), chunk):
                points = indices[start:start + chunk]
                unions[group] += np.count_nonzero(union_stamp[points] != group)
                union_stamp[points] = group
        if group == 0 or (group + 1) % 100 == 0 or group + 1 == len(groups):
            print(f'UNION groups={group + 1}/{len(groups)}', flush=True)
    counts = None
    active = np.ones(len(groups), dtype=bool)
    for iteration in (1, 2):
        labels[:] = 0
        confidence[:] = 0
        for position, row in enumerate(range(len(records)), 1):
            group = int(assignment[row])
            if not active[group]:
                continue
            indices, evidence = read(row)
            identity = group + 1
            for start in range(0, len(indices), chunk):
                points = indices[start:start + chunk]
                values = evidence[start:start + chunk].astype(np.float32)
                previous = confidence[points]
                # Stable ties prefer the earlier (higher leader-score) group.
                better = (values > previous) | ((values == previous) &
                         ((labels[points] == 0) | (identity < labels[points])))
                confidence[points[better]] = values[better]
                labels[points[better]] = identity
            if position % 1000 == 0 or position == len(records):
                print(f'ALLOCATE pass={iteration}/2 proposals={position}/{len(records)}', flush=True)
        counts = np.zeros(len(groups) + 1, dtype=np.int64)
        for start in range(0, len(labels), max(chunk, 1_000_000)):
            counts += np.bincount(labels[start:start + max(chunk, 1_000_000)], minlength=len(counts))
        if iteration == 1:
            ratio = counts[1:] / np.maximum(unions, 1)
            active = ((counts[1:] >= config['minimum_instance_points']) &
                      (ratio >= config['minimum_retention']) &
                      (np.asarray([g['score'] for g in groups]) * ratio >= config['allocation_score_threshold']))
            print(f'ALLOCATE accepted_groups={int(active.sum())}/{len(groups)}; reclaiming rejected groups', flush=True)
    # Compact identities after the final pass, in bounded point chunks.
    mapping = np.zeros(len(groups) + 1, dtype=np.int32)
    mapping[np.flatnonzero(active) + 1] = np.arange(1, int(active.sum()) + 1)
    for start in range(0, len(labels), max(chunk, 1_000_000)):
        stop = start + max(chunk, 1_000_000)
        labels[start:stop] = mapping[labels[start:stop]]
    instances = []
    for group in np.flatnonzero(active):
        info = groups[group]
        leader = records[info['leader']]
        bounds = np.asarray([records[row]['bbox'] for row in info['members']])
        instances.append(dict(instance_id=int(mapping[group + 1]), final_count=int(counts[group + 1]),
            count=int(unions[group]), retained_fraction=float(counts[group + 1] / max(unions[group], 1)),
            score=float(info['score']), center_xy=info['center_xy'], block=leader['block'], query=leader['query'],
            observations=len(info['members']), bbox=np.r_[bounds[:, :3].min(0), bounds[:, 3:].max(0)].tolist()))
    return instances, int((~active).sum())
