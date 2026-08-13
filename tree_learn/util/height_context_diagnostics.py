"""Streaming statistics for height-context prediction-drift diagnostics."""

import numpy as np


HISTOGRAMS = {
    'abs_probability_delta': np.linspace(0.0, 1.0, 1001),
    'offset_xy_residual': np.linspace(0.0, 5.0, 1001),
    'offset_z_residual': np.linspace(0.0, 10.0, 1001),
    'gate': np.linspace(0.0, 1.0, 1001),
}


def histogram_quantile(counts, edges, quantile):
    """Return a deterministic histogram approximation to a quantile."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        return None
    target = float(quantile) * (int(counts.sum()) - 1)
    index = int(np.searchsorted(np.cumsum(counts), target, side='right'))
    index = min(index, len(edges) - 2)
    return float((edges[index] + edges[index + 1]) / 2)


class DriftAccumulator:
    """Accumulate drift counts, means and robust distribution summaries."""

    def __init__(self):
        self.counts = {
            key: 0 for key in (
                'points', 'base_tree', 'adapted_tree', 'tree_to_non_tree',
                'non_tree_to_tree', 'base_seed', 'adapted_seed',
                'seed_removed', 'seed_added',
                'seed_removed_semantic', 'seed_removed_offset')
        }
        self.sums = {
            key: 0.0 for key in (
                'probability_delta', 'abs_probability_delta',
                'offset_xy_residual', 'offset_z_residual', 'gate')
        }
        self.histograms = {
            key: np.zeros(len(edges) - 1, dtype=np.int64)
            for key, edges in HISTOGRAMS.items()
        }

    def update(self, metrics, mask=None):
        size = len(metrics['base_tree'])
        if mask is None:
            mask = np.ones(size, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        count = int(mask.sum())
        if count == 0:
            return

        self.counts['points'] += count
        for key in self.counts:
            if key == 'points':
                continue
            self.counts[key] += int(np.asarray(metrics[key])[mask].sum())
        for key in self.sums:
            values = np.asarray(metrics[key], dtype=np.float64)[mask]
            self.sums[key] += float(values.sum())
            if key in self.histograms:
                edges = HISTOGRAMS[key]
                clipped = np.clip(values, edges[0], np.nextafter(edges[-1], 0))
                self.histograms[key] += np.histogram(clipped, bins=edges)[0]

    def summary(self):
        points = self.counts['points']
        if points == 0:
            return {'points': 0}
        result = dict(self.counts)
        for key, value in self.counts.items():
            if key != 'points':
                result[f'{key}_rate'] = value / points
        for key, value in self.sums.items():
            result[f'{key}_mean'] = value / points
        for key, counts in self.histograms.items():
            edges = HISTOGRAMS[key]
            result[f'{key}_median'] = histogram_quantile(counts, edges, 0.5)
            result[f'{key}_p90'] = histogram_quantile(counts, edges, 0.9)
        result['tree_to_non_tree_rate_among_base_tree'] = (
            self.counts['tree_to_non_tree'] /
            max(self.counts['base_tree'], 1))
        result['seed_removed_rate_among_base_seed'] = (
            self.counts['seed_removed'] /
            max(self.counts['base_seed'], 1))
        result['seed_removed_semantic_rate_among_base_seed'] = (
            self.counts['seed_removed_semantic'] /
            max(self.counts['base_seed'], 1))
        result['seed_removed_offset_rate_among_base_seed'] = (
            self.counts['seed_removed_offset'] /
            max(self.counts['base_seed'], 1))
        return result


def make_point_metrics(base_probability, adapted_probability,
                       offset_residual, gate, verticality,
                       base_offset, adapted_offset,
                       tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0):
    """Construct pointwise arrays using the exact TreeLearn seed rule."""
    base_probability = np.asarray(base_probability, dtype=np.float64)
    adapted_probability = np.asarray(adapted_probability, dtype=np.float64)
    offset_residual = np.asarray(offset_residual, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    verticality = np.asarray(verticality, dtype=np.float64)
    base_offset = np.asarray(base_offset, dtype=np.float64)
    adapted_offset = np.asarray(adapted_offset, dtype=np.float64)

    base_tree = base_probability >= tree_conf_thresh
    adapted_tree = adapted_probability >= tree_conf_thresh
    common_geometry = verticality > tau_vert
    base_seed = (
        base_tree & common_geometry & (np.abs(base_offset[:, 2]) < tau_off))
    adapted_seed = (
        adapted_tree & common_geometry &
        (np.abs(adapted_offset[:, 2]) < tau_off))
    probability_delta = adapted_probability - base_probability
    return {
        'base_tree': base_tree,
        'adapted_tree': adapted_tree,
        'tree_to_non_tree': base_tree & ~adapted_tree,
        'non_tree_to_tree': ~base_tree & adapted_tree,
        'base_seed': base_seed,
        'adapted_seed': adapted_seed,
        'seed_removed': base_seed & ~adapted_seed,
        'seed_added': ~base_seed & adapted_seed,
        'seed_removed_semantic': base_seed & ~adapted_tree,
        'seed_removed_offset': (
            base_seed & adapted_tree &
            (np.abs(adapted_offset[:, 2]) >= tau_off)),
        'probability_delta': probability_delta,
        'abs_probability_delta': np.abs(probability_delta),
        'offset_xy_residual': np.linalg.norm(offset_residual[:, :2], axis=1),
        'offset_z_residual': np.abs(offset_residual[:, 2]),
        'gate': gate.mean(axis=1) if gate.ndim == 2 else gate,
    }
