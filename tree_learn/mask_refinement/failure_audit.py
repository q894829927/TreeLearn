"""Read-only population audits of frozen A1 supervision and saved predictions."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import digest, file_digest, load_cache, load_candidates
from .evaluation import contingency


def derive_supervision(arrays, candidates, settings):
    """Mirror the original audit's target selection, without writing supervision."""
    gt, _, pred, pc, intersection, iou = contingency(arrays['labels'], arrays['baseline'])
    if not len(gt):
        raise ValueError('Supervision diagnostics require labeled GT trees.')
    all_ids, all_counts = np.unique(arrays['baseline'][arrays['baseline'] > 0], return_counts=True)
    if not np.array_equal(all_ids, candidates['ids']):
        raise ValueError('Cached candidates do not match baseline instance IDs.')
    count = len(all_ids)
    target_ids = np.zeros(count, dtype=np.int64)
    valid = np.zeros(count, dtype=bool)
    baseline_iou = np.zeros(count, dtype=np.float32)
    labeled_fraction = np.zeros(count, dtype=np.float32)
    for row, candidate_id in enumerate(all_ids):
        location = int(np.searchsorted(pred, candidate_id))
        present = location < len(pred) and pred[location] == candidate_id
        labeled_count = int(pc[location]) if present else 0
        labeled_fraction[row] = labeled_count / int(all_counts[row])
        if not labeled_count:
            continue
        best = int(np.argmax(iou[location]))
        baseline_iou[row] = iou[location, best]
        if intersection[location].sum() == 0:
            valid[row] = labeled_fraction[row] >= settings['min_labeled_fraction']
        else:
            target_ids[row] = gt[best]
            valid[row] = (labeled_fraction[row] >= settings['min_labeled_fraction'] and
                          baseline_iou[row] >= settings['min_match_iou'] and
                          intersection[location, best] / labeled_count >= settings['min_match_purity'])
    return dict(ids=all_ids, target_ids=target_ids, valid=valid,
                baseline_iou=baseline_iou, labeled_fraction=labeled_fraction)


def summarize_population(counts):
    result = dict(counts)

    def fraction(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    result.update(
        excluded_candidate_fraction=fraction('excluded_candidates', 'candidates'),
        duplicate_gt_fraction_among_eligible_targets=fraction('duplicate_target_gt', 'eligible_target_gt'),
        positive_excluded_by_semantics_fraction=fraction('positive_excluded_refs', 'positive_refs'),
        labeled_excluded_by_semantics_fraction=fraction('labeled_excluded_refs', 'labeled_refs'),
        shared_positive_unique_point_fraction=fraction('shared_positive_unique_points', 'positive_unique_points'),
        shared_positive_foreground_point_fraction=fraction('shared_positive_foreground_points',
                                                          'positive_unique_foreground_points'),
        mean_candidate_positive_excluded_fraction=fraction('candidate_positive_excluded_ratio_sum',
                                                          'candidates_with_positive_support'),
        mean_candidate_labeled_excluded_fraction=fraction('candidate_labeled_excluded_ratio_sum',
                                                         'candidates_with_labeled_support'))
    return result


COUNT_KEYS = (
    'candidates', 'eligible_candidates', 'eligible_positive_candidates', 'eligible_negative_candidates',
    'excluded_candidates', 'excluded_low_labeled_fraction', 'excluded_labeled_low_iou', 'excluded_labeled_other',
    'eligible_target_gt', 'duplicate_target_gt', 'eligible_candidates_in_duplicate_groups', 'excess_positive_candidates',
    'region_refs', 'ignored_refs', 'labeled_refs', 'labeled_excluded_refs', 'positive_refs', 'positive_excluded_refs',
    'other_tree_refs', 'background_refs', 'positive_unique_points', 'positive_unique_foreground_points',
    'shared_positive_unique_points',
    'shared_positive_foreground_points', 'candidates_with_positive_support', 'candidates_with_labeled_support',
    'candidate_positive_excluded_ratio_sum', 'candidate_labeled_excluded_ratio_sum')


def audit_supervision_support(directory, settings, frozen=True, chunk_size=131072, progress=None):
    """Scan one region at a time; count references and unique shared positives separately.

    On source plots read, validate and retain the original frozen supervision.
    On external plots derive a hypothetical policy only for post-hoc diagnostics.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')
    directory = Path(directory)
    arrays, metadata = load_cache(directory, with_labels=True)
    candidates = load_candidates(directory)
    expected = derive_supervision(arrays, candidates, settings)
    if frozen:
        audit = json.loads((directory / 'audit.json').read_text(encoding='utf-8'))
        if audit['supervision_settings_digest'] != digest(settings):
            raise ValueError('Frozen supervision settings differ from requested settings.')
        with np.load(directory / 'supervision.npz', allow_pickle=False) as archive:
            supervision = {key: archive[key] for key in expected}
        for key, value in expected.items():
            if not np.array_equal(supervision[key], value):
                raise ValueError(f'Frozen supervision disagrees with cache/policy: {key}')
    else:
        supervision = expected
    ids, targets, valid = (supervision[key] for key in ('ids', 'target_ids', 'valid'))
    counts = {key: 0 for key in COUNT_KEYS}
    counts.update(candidates=len(ids), eligible_candidates=int(valid.sum()),
                  eligible_positive_candidates=int(np.count_nonzero(valid & (targets > 0))),
                  eligible_negative_candidates=int(np.count_nonzero(valid & (targets == 0))),
                  excluded_candidates=int(np.count_nonzero(~valid)))
    # Use scalar comparisons just as the original audit does (including NumPy's
    # scalar precision at a threshold); do not regroup with a different cast.
    insufficient = np.asarray([value < settings['min_labeled_fraction']
                               for value in supervision['labeled_fraction']], dtype=bool)
    low_iou = np.asarray([value < settings['min_match_iou']
                         for value in supervision['baseline_iou']], dtype=bool)
    counts['excluded_low_labeled_fraction'] = int(np.count_nonzero(~valid & insufficient))
    counts['excluded_labeled_low_iou'] = int(np.count_nonzero(~valid & ~insufficient & low_iou))
    counts['excluded_labeled_other'] = int(np.count_nonzero(~valid & ~insufficient & ~low_iou))
    target_counts = Counter(int(target) for target in targets[valid & (targets > 0)])
    counts['eligible_target_gt'] = len(target_counts)
    counts['duplicate_target_gt'] = sum(number > 1 for number in target_counts.values())
    counts['eligible_candidates_in_duplicate_groups'] = sum(number for number in target_counts.values() if number > 1)
    counts['excess_positive_candidates'] = sum(max(number - 1, 0) for number in target_counts.values())
    positive_support = np.zeros(len(arrays['labels']), dtype=np.int32)
    rows = []
    region_hash = hashlib.sha256()
    for row, candidate_id in enumerate(ids):
        if progress is not None and row % 200 == 0:
            progress(f'  region scan {row}/{len(ids)} candidates')
        record = dict(candidate_id=int(candidate_id), target_gt_id=int(targets[row]), eligible=bool(valid[row]))
        rows.append(record)
        if not valid[row]:
            continue
        path = directory / 'regions' / f'{int(candidate_id)}.npy'
        region_hash.update(f'{int(candidate_id)}:{file_digest(path)}\n'.encode())
        indices = np.load(path, mmap_mode='r', allow_pickle=False)
        if not len(indices):
            raise ValueError('Eligible candidate has an empty cached region.')
        local_counts = dict(region_refs=len(indices), ignored_refs=0, labeled_refs=0, labeled_excluded_refs=0,
                            positive_refs=0, positive_excluded_refs=0, other_tree_refs=0, background_refs=0)
        previous = -1
        for start in range(0, len(indices), chunk_size):
            selected = indices[start:start + chunk_size]
            if (not np.issubdtype(selected.dtype, np.integer) or selected.ndim != 1 or selected[0] <= previous or
                    selected[-1] >= len(positive_support) or np.any(selected[1:] <= selected[:-1])):
                raise ValueError('Region indices must be sorted, unique and in bounds.')
            previous = int(selected[-1])
            labels, owner = arrays['labels'][selected], arrays['baseline'][selected]
            labeled, excluded = labels >= 0, owner == 0
            positive = (labels == targets[row]) & (targets[row] > 0)
            for key, mask in (
                    ('ignored_refs', ~labeled), ('labeled_refs', labeled),
                    ('labeled_excluded_refs', labeled & excluded), ('positive_refs', positive),
                    ('positive_excluded_refs', positive & excluded),
                    ('other_tree_refs', (labels > 0) & ~positive), ('background_refs', labels == 0)):
                local_counts[key] += int(np.count_nonzero(mask))
            positive_support[selected[positive]] += 1
        if targets[row] > 0 and not local_counts['positive_refs']:
            raise ValueError('Eligible positive candidate has no target points in its cached region.')
        for key, value in local_counts.items():
            counts[key] += value
        for group in ('positive', 'labeled'):
            if local_counts[f'{group}_refs']:
                counts[f'candidates_with_{group}_support'] += 1
                counts[f'candidate_{group}_excluded_ratio_sum'] += (
                    local_counts[f'{group}_excluded_refs'] / local_counts[f'{group}_refs'])
        record.update(local_counts)
    shared_by_gt = Counter()
    for start in range(0, len(positive_support), chunk_size):
        support = positive_support[start:start + chunk_size]
        shared = support > 1
        counts['positive_unique_points'] += int(np.count_nonzero(support))
        foreground = arrays['baseline'][start:start + chunk_size] > 0
        counts['positive_unique_foreground_points'] += int(np.count_nonzero((support > 0) & foreground))
        counts['shared_positive_unique_points'] += int(np.count_nonzero(shared))
        counts['shared_positive_foreground_points'] += int(np.count_nonzero(
            shared & foreground))
        gt, number = np.unique(arrays['labels'][start:start + chunk_size][shared], return_counts=True)
        shared_by_gt.update({int(key): int(value) for key, value in zip(gt, number)})
    if (int(positive_support.sum(dtype=np.int64)) != counts['positive_refs'] or
            counts['labeled_refs'] != counts['positive_refs'] + counts['other_tree_refs'] + counts['background_refs']):
        raise RuntimeError('Population reference counts do not reconcile.')
    files = ['baseline.npy', 'labels.npy', 'candidates.npz', 'regions.json']
    if frozen:
        files += ['audit.json', 'supervision.npz']
    return dict(
        posthoc=True, eligible_for_advancement=False,
        supervision_origin='frozen_source' if frozen else 'hypothetical_external_policy_not_training',
        cache_signature=metadata['signature'], supervision_settings_digest=digest(settings),
        counts=counts, summary=summarize_population(counts), candidates=rows,
        target_groups=[dict(gt_id=key, eligible_candidates=number, shared_positive_unique_points=shared_by_gt[key])
                       for key, number in sorted(target_counts.items())],
        source_hashes={name: file_digest(directory / name) for name in files},
        eligible_region_files_digest=region_hash.hexdigest())


def describe_changed_gt(labels, baseline, refined, expected, target_groups):
    """Record GT-specific best IoUs before/after, not just each candidate's best GT."""
    before_ids = set(expected['baseline']['matched_gt_ids'])
    after_ids = set(expected['refined']['matched_gt_ids'])
    changes = {int(key): dict(gt_id=int(key), change='lost' if key in before_ids else 'recovered')
               for key in before_ids ^ after_ids}
    for method, prediction in (('baseline', baseline), ('refined', refined)):
        gt, _, pred, _, _, iou = contingency(labels, prediction)
        for key, row in changes.items():
            column = int(np.searchsorted(gt, key))
            if not len(pred):
                row[f'{method}_best_iou'] = 0.0
                row[f'{method}_best_candidate'] = None
            else:
                best = int(np.argmax(iou[:, column]))
                row[f'{method}_best_iou'] = float(iou[best, column])
                row[f'{method}_best_candidate'] = int(pred[best]) if iou[best, column] > 0 else None
        del iou
    groups = {row['gt_id']: row for row in target_groups}
    for key, row in changes.items():
        row['eligible_candidates_targeting_gt'] = groups.get(key, {}).get('eligible_candidates', 0)
        row['shared_positive_unique_points'] = groups.get(key, {}).get('shared_positive_unique_points', 0)
    return [changes[key] for key in sorted(changes)]
