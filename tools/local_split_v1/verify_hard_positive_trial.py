"""Check paired training inputs and export deterministic diagnostic child IDs."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tree_learn.local_split_v1 import data
from tree_learn.mask_refinement.data import file_digest, write_json


def write_or_verify(path, value):
    if path.exists():
        if data.read_json(path) != value:
            raise ValueError(f'Existing diagnostic differs: {path}')
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='work_dirs/treelearn_local_split_e1_r1_s42')
    parser.add_argument('--assets', default='docs/experiments/assets/local_split_e1_r1_20261022')
    args = parser.parse_args()
    work, assets = Path(args.work), Path(args.assets)
    manifest = data.read_json(work / 'manifest.json')
    config = data.load_config(manifest['trial']['source_config'])
    if data.signature(config) != manifest['sources']['source_signature']:
        raise ValueError('Original source signature changed.')
    result = data.read_json(work / 'audit/summary.json')
    old_path = result['baseline_e1']['selected']['checkpoint']
    new_path = result['checkpoint']['checkpoint']
    old = torch.load(old_path, map_location='cpu', weights_only=False)
    new = torch.load(new_path, map_location='cpu', weights_only=False)
    checks = dict(optimizer_steps=0, old_checkpoint_sha256=file_digest(old_path),
        new_checkpoint_sha256=file_digest(new_path),
        normalization_exactly_equal=old['normalization'] == new['normalization'],
        numpy_sampler_final_state_exactly_equal=old['rng_numpy'] == new['rng_numpy'],
        torch_final_rng_state_exactly_equal=torch.equal(old['rng_torch'], new['rng_torch']),
        old_step=old['step'], new_step=new['step'])
    if (checks['old_checkpoint_sha256'] != manifest['sources']['baseline_checkpoint_sha256'] or
            checks['new_checkpoint_sha256'] != result['checkpoint']['checkpoint_sha256'] or
            not all(checks[k] for k in checks if k.endswith('_equal'))):
        raise ValueError('Paired training inputs or weights changed.')
    write_or_verify(assets / 'training_input_equivalence.json', checks)
    print('TRAINING_INPUT_EQUIVALENCE ' + str(checks), flush=True)
    for method in ('learned', 'geometry'):
        threshold = result[method]['selected']['threshold']
        forests = []
        for plot, reports in result[method]['partition_reports'].items():
            selected = [r for r in reports if r['reason'] == 'proposal' and threshold is not None
                        and r['mean_cut_score'] <= threshold]
            if not selected:
                continue
            row, = [r for r in data.rows(config) if r['plot'] == plot]
            baseline = np.load(Path(row['cache']) / 'baseline.npy', mmap_mode='r', allow_pickle=False)
            next_id = int(baseline.max()) + 1
            baseline._mmap.close()
            parents = []
            for record in sorted(selected, key=lambda r: r['parent_id']):
                if file_digest(record['payload']) != record['payload_sha256']:
                    raise ValueError('Changed diagnostic split payload.')
                parents.append(dict(parent_id=record['parent_id'], new_id=next_id,
                    entrance=record['entrance'], payload=record['payload'], payload_sha256=record['payload_sha256']))
                next_id += 1
            forests.append(dict(plot=plot, parents=parents))
        value = dict(signature=result['signature'], scope=result['scope'], approved_for_deployment=False,
                     threshold=threshold, gt_arrays_opened=False, forests=forests)
        write_or_verify(work / 'deployment' / method / 'selected_policy.json', value)
        write_or_verify(assets / f'{method}_selected_policy.json', value)
        print(f'POLICY_IDS {method} ' + str(forests), flush=True)


if __name__ == '__main__':
    main()
