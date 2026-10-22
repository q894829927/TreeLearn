"""Inspect input-only XY seed sections; GT relation appears only in captions."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from tree_learn.local_split_v1 import data
from tree_learn.mask_refinement.data import file_digest, load_cache, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/full.yaml')
    parser.add_argument('--assets', default='docs/experiments/assets/local_split_failure_mechanisms_20261022')
    args = parser.parse_args()
    config = data.load_config(args.config)
    output = Path(args.assets)
    records = data.read_json(output / 'seed_surfaces.json')['records']
    examples = [('A1N', 251), ('G4N', 63), ('G2W', 252), ('LG1', 437)]
    fig, axes = plt.subplots(4, 4, figsize=(13, 12), layout='constrained')
    for col, (plot, parent) in enumerate(examples):
        report, = [r for r in records if (r['plot'], r['parent_id']) == (plot, parent)]
        row, = [r for r in data.rows(config) if r['plot'] == plot]
        record, = [r for r in data.load_prepared(row, config)['records'] if r['parent_id'] == parent]
        arrays, _ = load_cache(row['cache'], with_labels=False)
        try:
            payload = data.load_parent(row, record, arrays)
            xyz = arrays['coords'][payload['indices']].astype(float)
            z0 = np.quantile(xyz[:, 2], config['candidate']['lower_quantile'])
            levels = np.floor((xyz[:, 2] - z0) / config['candidate']['slab_m']).astype(int)
            for number, layer in enumerate(report['layers']):
                axis = axes[number, col]
                groups = [xyz[at[levels[at] == layer['level']], :2] for at in [payload['seed0'], payload['seed1']]]
                origin = np.concatenate(groups).mean(0)
                for k, group in enumerate(groups):
                    color = ['tab:blue', 'tab:orange'][k]
                    axis.scatter(*(group - origin).T, s=7, c=color)
                    axis.plot(*(group.mean(0) - origin), '+', color=color, markersize=12)
                    fitted = layer['fits'][k]
                    if fitted['valid']:
                        center = np.array(fitted['center']) - origin
                        axis.add_patch(Circle(center, fitted['radius_m'], fill=False, linestyle='--', color=color))
                        axis.plot(*center, 'x', color=color, markersize=8)
                axis.set_aspect('equal', adjustable='datalim')
                axis.autoscale_view()
                axis.set(xlabel='X (m, centered)', ylabel='Y (m, centered)')
                axis.set_title(f'{plot}/{parent} {report["relation"]}\nslab {layer["level"]}, N={layer["counts"]}, ratio={layer.get("center_to_centroid_separation_ratio", 0):.2f}', fontsize=9)
        finally:
            data.close_arrays(arrays)
    fig.suptitle('Full XY points in the four original seed slabs | +=centroid, x=circle center | circles are diagnostic fits')
    fig.savefig(output / 'seed_xy_sections.png', dpi=140)
    plt.close(fig)
    write_json(output / 'seed_section_manifest.json', dict(examples=[dict(plot=p, parent_id=i) for p, i in examples],
        code_sha256=file_digest(__file__), source_signature=data.signature(config),
        input_sha256=file_digest(output / 'seed_surfaces.json'), labels_read=False))
    print('SEED_SECTION_FIGURE ' + str(output / 'seed_xy_sections.png'), flush=True)


if __name__ == '__main__':
    main()
