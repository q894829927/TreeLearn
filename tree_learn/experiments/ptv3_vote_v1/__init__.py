"""Versioned local vote diagnostics; no training targets or pipeline are changed."""
import numpy as np


def quantiles(values):
    return np.quantile(values, [0, .1, .5, .9, .99, 1]).tolist() if len(values) else None


def vector_metrics(prediction, target, mask):
    prediction, target = prediction[mask].astype(np.float64), target[mask].astype(np.float64)
    if not len(target):
        return dict(points=0)
    error = prediction-target
    xy = np.linalg.norm(error[:, :2], axis=1)
    zero = np.linalg.norm(target[:, :2], axis=1)
    constant = np.linalg.norm(target[:, :2]-target[:, :2].mean(0), axis=1)
    norm_product = np.linalg.norm(prediction[:, :2], axis=1)*np.linalg.norm(target[:, :2], axis=1)
    valid = norm_product > 1e-10
    return dict(points=len(target), xy_mean_m=float(xy.mean()), xyz_mean_m=float(np.linalg.norm(error, axis=1).mean()),
                abs_z_mean_m=float(np.abs(error[:, 2]).mean()), xy_quantiles_m=quantiles(xy),
                zero_xy_mean_m=float(zero.mean()), mean_oracle_xy_mean_m=float(constant.mean()),
                xy_improvement_over_zero=1-float(xy.mean()/zero.mean()) if zero.mean() else None,
                prediction_mean=prediction.mean(0).tolist(), prediction_std=prediction.std(0).tolist(),
                target_mean=target.mean(0).tolist(), target_std=target.std(0).tolist(),
                cosine_xy_mean=float(((prediction[valid,:2]*target[valid,:2]).sum(1)/norm_product[valid]).mean()) if valid.any() else None,
                cosine_defined_fraction=float(valid.mean()),
                xyz_output_gradient_axis_rms=np.sqrt(np.mean(np.square(error/np.maximum(np.linalg.norm(error, axis=1, keepdims=True), 1e-12)), axis=0)).tolist())


def votes(coords, target, prediction, labels, mask, seeds):
    rows = []
    for label in np.unique(labels[mask & (labels > 0)]):
        index = mask & (labels == label)
        p = coords[index, :2].astype(np.float64)
        q = p+prediction[index, :2]
        anchors = p+target[index, :2]
        anchor = anchors.mean(0)
        before = float(np.sqrt(np.mean(np.square(p-p.mean(0)).sum(1))))
        after = float(np.sqrt(np.mean(np.square(q-q.mean(0)).sum(1))))
        rows.append(dict(instance=int(label), points=int(index.sum()), seed_points=int((seeds & index).sum()),
                         anchor_xy=anchor.tolist(), vote_centroid_xy=q.mean(0).tolist(),
                         gt_anchor_max_deviation_m=float(np.linalg.norm(anchors-anchor, axis=1).max()),
                         zero_vote_scatter_rms_m=before, vote_scatter_rms_m=after,
                         scatter_reduction=1-after/before if before else None,
                         centroid_anchor_error_m=float(np.linalg.norm(q.mean(0)-anchor))))
    return dict(instances=rows, trees_with_seed=sum(row['seed_points'] > 0 for row in rows),
                mean_vote_scatter_rms_m=float(np.mean([r['vote_scatter_rms_m'] for r in rows])) if rows else None,
                mean_zero_vote_scatter_rms_m=float(np.mean([r['zero_vote_scatter_rms_m'] for r in rows])) if rows else None)


def histograms(groups, labels, size, classes, weights=None):
    """Sparse full label counts; minority trees survive every pooling stage."""
    known = labels >= 0
    keys, inverse = np.unique(groups[known].astype(np.int64)*classes+labels[known].astype(np.int64), return_inverse=True)
    counts = np.bincount(inverse, weights=None if weights is None else weights[known]).astype(np.int64)
    return keys//classes, keys % classes, counts


def histogram_report(histogram, size):
    groups, labels, weights = histogram
    trees = labels > 0
    tree_kinds = np.bincount(groups[trees], minlength=size)
    order = np.lexsort((labels, -weights, groups))
    first = np.r_[True, np.diff(groups[order]) != 0]
    chosen = order[first]
    dominant = np.full(size, -1, dtype=np.int64); dominant[groups[chosen]] = labels[chosen]
    mixed = tree_kinds > 1
    return dominant, dict(groups=size, known_groups=int(len(np.unique(groups))),
                          multi_tree_groups=int(mixed.sum()), multi_tree_fraction=float(mixed.mean()),
                          known_points_in_multi_tree_groups=int(weights[mixed[groups]].sum()))


def oracle_arrays(logits, offsets, semantic, target, valid_sem, valid_off, kind):
    """GT replacements are separate diagnostic arrays, never in-place edits."""
    logits, offsets = logits.copy(), offsets.copy()
    if kind == 'semantic':
        logits[valid_sem] = -20.
        index = np.flatnonzero(valid_sem)
        logits[index, semantic[index].astype(int)] = 20.
    elif kind == 'xy':
        offsets[valid_off, :2] = target[valid_off, :2]
    elif kind == 'z':
        offsets[valid_off, 2] = target[valid_off, 2]
    else:
        raise ValueError(kind)
    return logits, offsets


def feature_summary(features, labels):
    features = features.astype(np.float64)
    mean = features.mean(0); centered = features-mean
    covariance = centered.T @ centered/max(len(centered), 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    within = 0.; between = 0.
    for label in np.unique(labels):
        x = features[labels == label]; centroid = x.mean(0)
        within += np.square(x-centroid).sum()
        between += len(x)*np.square(centroid-mean).sum()
    return dict(points=len(features), channels=features.shape[1],
                common_mean_rms=float(np.sqrt(np.square(mean).mean())),
                centered_channel_rms=float(np.sqrt(np.trace(covariance)/features.shape[1])),
                effective_covariance_rank=float(eigenvalues.sum()**2/max(np.square(eigenvalues).sum(), 1e-30)),
                between_to_within_variance=float(between/max(within, 1e-30)))
