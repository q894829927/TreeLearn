"""Foreground-only local split V1 stages. Stop at the first failed gate."""

import argparse
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tree_learn.local_split_v1 import data
from tree_learn.mask_refinement.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/local_split_v1/full.yaml')
    parser.add_argument('--stage', choices=['prepare', 'audit', 'overfit', 'train', 'validate', 'external', 'all'], required=True)
    args = parser.parse_args()
    config = data.load_config(args.config)
    data.checkpoint_md5(config)
    print(f'STAGE {args.stage} signature={data.signature(config)} protocol={config["protocol"]}', flush=True)
    if args.stage in ('prepare', 'all'):
        for row in data.rows(config):
            data.prepare_forest(row, config)
        print('PREPARED source-only; no torch imported and no GT opened.', flush=True)
    if args.stage in ('audit', 'all'):
        from tree_learn.local_split_v1.audit import audit_all
        result = audit_all(config)
        if not result['gate']['passed']:
            print('STOP E0 failed: no training, validation selection, or Wytham execution.', flush=True)
            return
    if args.stage in ('overfit', 'train', 'validate', 'external', 'all'):
        from tree_learn.local_split_v1 import learning
        if args.stage in ('overfit', 'all'):
            result = learning.fit(config, overfit=True)
            if not result['gate']['passed']:
                print('STOP E1 failed: full training and Wytham are forbidden.', flush=True)
                return
        if args.stage in ('train', 'all'):
            result = learning.fit(config)
            if not result['gate']['passed']:
                print('STOP E2 failed: Wytham execution is forbidden.', flush=True)
                return
        if args.stage in ('validate', 'all'):
            learning.validate_saved(config)
        if args.stage in ('external', 'all'):
            learning.external(config)
    if args.stage == 'all':
        write_json(Path(config['work_dir']) / 'completed.json', dict(signature=data.signature(config), completed=True))


if __name__ == '__main__':
    main()
