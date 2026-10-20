"""CPU statistics for a diagnostic replay; GT never changes predicted masks."""

import numpy as np
from scipy.sparse import coo_matrix


def select_targets(rows, seed=42):
    """Fixed diagnostic strata; deliberately error-enriched, not a test split."""
    rng = np.random.default_rng(seed)
    selected, used = [], set()
    strata = (
        ('lost_impure_10_to_20m', 4, lambda r: r['transition'] == 'lost' and
         r['refined_category'] == 'high_recall_low_purity' and 10 <= float(r['z_span_m']) < 20),
        ('lost_impure_ge_20m', 2, lambda r: r['transition'] == 'lost' and
         r['refined_category'] == 'high_recall_low_purity' and float(r['z_span_m']) >= 20),
        ('lost_impure_lt_10m', 2, lambda r: r['transition'] == 'lost' and
         r['refined_category'] == 'high_recall_low_purity' and float(r['z_span_m']) < 10),
        ('detected_control_10_to_20m', 2, lambda r: r['refined_detected'] == 'True' and
         10 <= float(r['z_span_m']) < 20),
        ('detected_control_ge_20m', 2, lambda r: r['refined_detected'] == 'True' and
         float(r['z_span_m']) >= 20),
    )
    for name, count, predicate in strata:
        pool = sorted((r for r in rows if int(r['gt_id']) not in used and predicate(r)),
                      key=lambda r: int(r['gt_id']))
        for position in rng.permutation(len(pool))[:count]:
            row = dict(pool[int(position)], stratum=name)
            selected.append(row)
            used.add(int(row['gt_id']))
    if not selected:
        raise ValueError('No diagnostic target or control trees in trees.csv.')
    return selected


def cell_counts(nearest, labels, num_voxels, gt_ids):
    """Count all/known/GT fine points in the actual NN output cells."""
    nearest, labels, gt_ids = [np.asarray(x) for x in (nearest, labels, gt_ids)]
    if nearest.shape != labels.shape or labels.ndim != 1 or np.any(nearest < 0) or np.any(nearest >= num_voxels):
        raise ValueError('Bad fine-point to voxel mapping.')
    if not len(gt_ids) or np.any(np.diff(gt_ids) <= 0):
        raise ValueError('GT IDs must be sorted and unique.')
    tree = labels > 0
    columns = np.searchsorted(gt_ids, labels[tree])
    if np.any(columns >= len(gt_ids)) or not np.array_equal(gt_ids[columns], labels[tree]):
        raise ValueError('GT IDs differ from the saved posthoc audit.')
    counts = coo_matrix((np.ones(int(tree.sum()), dtype=np.int64), (nearest[tree], columns)),
                        shape=(num_voxels, len(gt_ids))).tocsr()
    all_counts = np.bincount(nearest, minlength=num_voxels)
    known = np.bincount(nearest[labels >= 0], minlength=num_voxels)
    return dict(all=all_counts, known=known, gt=counts,
                local_gt=np.asarray(counts.sum(axis=0)).ravel())


def proposal_metrics(masks, cells, full_gt_counts, query_chunk=16):
    """Exact fine-point statistics without expanding Q x raw-point masks."""
    masks = np.asarray(masks, dtype=bool)
    full_gt_counts = np.asarray(full_gt_counts)
    if (query_chunk <= 0 or masks.ndim != 2 or masks.shape[1] != len(cells['known']) or
            full_gt_counts.shape != cells['local_gt'].shape or np.any(full_gt_counts <= 0) or
            np.any(cells['local_gt'] > full_gt_counts)):
        raise ValueError('Masks, cells and full-tree counts do not align.')
    total, known = np.zeros((2, len(masks)), dtype=np.int64)
    intersection = np.zeros((len(masks), len(full_gt_counts)), dtype=np.int64)
    for start in range(0, len(masks), query_chunk):
        stop = start + query_chunk
        values = masks[start:stop].astype(np.int64)
        total[start:stop] = values @ cells['all']
        known[start:stop] = values @ cells['known']
        intersection[start:stop] = cells['gt'].T.dot(values.T).T
    union = known[:, None] + full_gt_counts[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    precision = np.divide(intersection, known[:, None], out=np.zeros_like(iou), where=known[:, None] > 0)
    recall = intersection / full_gt_counts[None, :]
    return dict(total=total, known=known, intersection=intersection, iou=iou, precision=precision, recall=recall)


def best_for_gt(metrics, queries, column, scores):
    queries = np.asarray(queries, dtype=np.int64)
    if not len(queries):
        return dict(query=-1, iou=0., precision=0., recall=0., score=0.,
                    other_tree_fraction=0., background_fraction=0., good_queries=[])
    query = int(queries[np.argmax(metrics['iou'][queries, column])])
    return dict(query=query, iou=float(metrics['iou'][query, column]),
                precision=float(metrics['precision'][query, column]),
                recall=float(metrics['recall'][query, column]), score=float(scores[query]),
                other_tree_fraction=float((metrics['intersection'][query].sum() -
                                            metrics['intersection'][query, column]) / max(metrics['known'][query], 1)),
                background_fraction=float((metrics['known'][query] - metrics['intersection'][query].sum()) /
                                          max(metrics['known'][query], 1)),
                good_queries=queries[metrics['iou'][queries, column] > .5].astype(int).tolist())


def nms_overlap(first, second, iou_threshold, containment_threshold):
    """Masks must be sorted unique global point IDs, as in normal inference."""
    common = len(np.intersect1d(first, second, assume_unique=True))
    iou = common / max(len(first) + len(second) - common, 1)
    containment = common / max(min(len(first), len(second)), 1)
    return dict(iou=iou, containment=containment,
                conflicts=bool(iou >= iou_threshold or containment >= containment_threshold))


def classify_target(block_rows, saved_detected, saved_iou):
    """Explain sampled proposal availability, never infer an unobserved cause."""
    eligible = [row for row in block_rows if row['full_context']]
    if saved_detected:
        return 'detected_control'
    for stage, category in (
        ('raw', 'no_good_raw_mask_in_sampled_blocks'),
        ('quality', 'good_raw_masks_lost_at_quality_filter'),
        ('size', 'good_masks_lost_at_minimum_size_filter'),
        ('local_nms', 'good_masks_lost_at_local_nms'),
    ):
        if not any(row['stages'][stage]['good_queries'] for row in block_rows):
            if stage == 'raw' and not eligible:
                return 'no_sampled_full_context_block'
            return category
    return ('good_local_masks_but_saved_final_bad' if saved_iou <= .5 else
            'saved_final_good_iou_without_detection')
