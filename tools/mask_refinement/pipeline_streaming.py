"""Run the original cache-export pipeline with streaming tile storage only."""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tree_learn.util import get_config
    from tree_learn.util.data_preparation import SampleGenerator
    from tree_learn.mask_refinement.tiling import tile_generate_and_save_streaming

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    if not config.tile_generation or not config.save_cfg.mask_refinement_export_only:
        raise ValueError('Streaming entry is restricted to A1 cache-export preparation.')
    spec = importlib.util.spec_from_file_location('original_treelearn_pipeline', ROOT / 'tools/pipeline/pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    original = SampleGenerator.tile_generate_and_save
    SampleGenerator.tile_generate_and_save = tile_generate_and_save_streaming
    try:
        pipeline.run_treelearn_pipeline(config, args.config)
    finally:
        SampleGenerator.tile_generate_and_save = original


if __name__ == '__main__':
    main()
