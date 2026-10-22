"""CPU check: sampled-edge fit versus full-edge gate, plus a threshold bound."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tree_learn.local_split_v1 import data
from tree_learn.local_split_v1.model import EdgeHead
from tree_learn.mask_refinement.data import file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/full.yaml')
    parser.add_argument('--assets', default='docs/experiments/assets/local_split_v1_20261022')
    args = parser.parse_args()
    config = data.load_config(args.config)
    assets = Path(args.assets)
    work = Path(config['work_dir'])
    e1 = data.read_json(work / 'e1/summary.json')
    if e1['signature'] != data.signature(config):
        raise ValueError('Stale E1.')
    checkpoint = e1['selected']['checkpoint']
    if file_digest(checkpoint) != e1['selected']['checkpoint_sha256']:
        raise ValueError('Changed checkpoint.')
    saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
    model = EdgeHead(saved['normalization']['mean'], saved['normalization']['scale'])
    model.load_state_dict(saved['state_dict']); model.eval()
    torch.set_num_threads(4)
    selected = {(r['plot'], tuple(r['parents'])) for r in e1['parent_records']}
    counts = dict(same=0, same_cut=0, different=0, different_cut=0)
    for record in data.read_json(work / 'banks/bank.json')['records']:
        if (record['plot'], tuple(record['parents'])) not in selected:
            continue
        if file_digest(record['path']) != record['sha256']:
            raise ValueError('Changed bank.')
        with np.load(record['path'], allow_pickle=False) as z:
            with torch.no_grad():
                cut = model(torch.as_tensor(z['raw'])).sigmoid().numpy() < .5
            for name, label in [('same', 1), ('different', 0)]:
                at = z['target'] == label
                counts[name] += int(at.sum())
                counts[name + '_cut'] += int(np.count_nonzero(at & cut))
    counts.update(same_false_cut_rate=counts['same_cut'] / counts['same'],
                  different_detection_rate=counts['different_cut'] / counts['different'])
    histogram_path = assets / 'e1_diagnosis.json'
    histogram = data.read_json(histogram_path)['score_histogram']
    same_cdf = np.cumsum(histogram['same']) / sum(histogram['same'])
    different_cdf = np.cumsum(histogram['different']) / sum(histogram['different'])
    at = int(np.flatnonzero(same_cdf > .05)[0])
    bound = dict(max_same_false_cut=.05, threshold_must_be_below=histogram['bins'][at + 1],
                 different_detection_upper_bound=float(different_cdf[at]),
                 histogram_sha256=file_digest(histogram_path),
                 note='Monotonicity bound from 0.025-wide full-edge score bins; no threshold is selected or changed.')
    result = dict(signature=e1['signature'], posthoc=True, gate_changed=False, optimizer_steps=0,
                  sampled_bank=counts, full_edge_threshold_bound=bound, code_sha256=file_digest(__file__))
    write_json(assets / 'e1_score_audit.json', result)
    print('SCORE_AUDIT ' + str(result), flush=True)


if __name__ == '__main__':
    main()
