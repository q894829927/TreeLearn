import argparse
import json
from pathlib import Path

import numpy as np

from summarize_e6_quality_filter import (
    exact_metrics, load_results, read_pipeline_seconds)


def build_ratio_gate(baseline, filtered, metadata, config, runtime_fraction):
    f1_gain = filtered['f1_score'] - baseline['f1_score']
    commission_reduction = (
        baseline['commission_error_rate'] -
        filtered['commission_error_rate'])
    completeness_drop = (
        baseline['completeness'] - filtered['completeness'])
    completeness_passed = (
        completeness_drop <=
        float(config.gate.max_completeness_drop_pp) + 1e-12)
    effect_mode = str(config.gate.effect_mode)
    if effect_mode == 'f1_not_below':
        effect_passed = (
            f1_gain >= float(config.gate.min_f1_gain_pp) - 1e-12)
    elif effect_mode == 'strong':
        effect_passed = (
            f1_gain >= float(config.gate.min_f1_gain_pp) - 1e-12 or
            (
                commission_reduction >=
                float(config.gate.min_commission_reduction_pp) - 1e-12 and
                completeness_passed
            )
        )
    else:
        raise ValueError(f'Unknown effect mode: {effect_mode}')
    gate = {
        'checkpoint_seed_locked': (
            int(metadata['checkpoint_seed']) ==
            int(config.locked.checkpoint_seed)),
        'filter_enabled': bool(metadata.get('filter_enabled', False)),
        'filter_mode_locked': (
            str(metadata.get('filter_mode')) ==
            str(config.locked.filter_mode)),
        'keep_ratio_locked': (
            abs(float(metadata.get('keep_ratio', -1.0)) -
                float(config.locked.keep_ratio)) <= 1e-12),
        'kept_count_correct': (
            int(metadata['num_kept']) ==
            int(np.ceil(
                float(config.locked.keep_ratio) *
                int(metadata['num_instances'])))),
        'completeness_drop_passed': completeness_passed,
        'effect_size_passed': effect_passed,
        'runtime_overhead_passed': (
            runtime_fraction <=
            float(config.gate.max_quality_runtime_fraction) + 1e-12),
    }
    gate['passed'] = bool(all(gate.values()))
    return gate, {
        'f1_gain_pp': f1_gain,
        'commission_reduction_pp': commission_reduction,
        'completeness_drop_pp': completeness_drop,
    }


def format_markdown(report):
    rows = [
        f"# E7b 域稳健比例过滤复核：{report['dataset_name']}",
        '',
        f"- 数据角色：**{report['dataset_role']}**",
        f"- 锁定 keep ratio：{report['keep_ratio']:.4f}",
        f"- 保留实例：{report['metadata']['num_kept']} / "
        f"{report['metadata']['num_instances']}",
        f"- 质量评分耗时占完整 pipeline："
        f"{100.0 * report['quality_runtime_fraction']:.3f}%",
        '',
        '| Run | Completeness | Commission | F1 | Precision | Recall | Coverage |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('baseline', 'filtered'):
        item = report[name]
        rows.append(
            f"| {name} | {item['completeness']:.6f}% | "
            f"{item['commission_error_rate']:.6f}% | "
            f"{item['f1_score']:.6f}% | "
            f"{item['precision']:.6f}% | "
            f"{item['recall']:.6f}% | "
            f"{item['coverage']:.6f}% |")
    rows.extend([
        '',
        '## Delta',
        '',
        f"- F1：{report['delta']['f1_gain_pp']:+.6f} 个百分点",
        f"- Commission 降低："
        f"{report['delta']['commission_reduction_pp']:+.6f} 个百分点",
        f"- Completeness 下降："
        f"{report['delta']['completeness_drop_pp']:+.6f} 个百分点",
        '',
        '## Gate',
        '',
    ])
    for name, passed in report['gate'].items():
        rows.append(f"- {name}: **{passed}**")
    rows.extend([
        '',
        'PASS' if report['gate']['passed'] else 'STOP',
        '',
    ])
    return '\n'.join(rows)


def main():
    from tree_learn.util import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    paths = [
        Path(config.evaluations.baseline),
        Path(config.evaluations.filtered),
        Path(config.metadata),
        Path(config.pipeline_log),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline = exact_metrics(load_results(paths[0]))
    filtered = exact_metrics(load_results(paths[1]))
    with paths[2].open(encoding='utf-8') as file:
        metadata = json.load(file)
    pipeline_seconds = read_pipeline_seconds(paths[3])
    runtime_fraction = metadata['scoring_seconds'] / pipeline_seconds
    gate, delta = build_ratio_gate(
        baseline, filtered, metadata, config, runtime_fraction)
    report = {
        'dataset_name': str(config.dataset_name),
        'dataset_role': str(config.dataset_role),
        'keep_ratio': float(config.locked.keep_ratio),
        'baseline': baseline,
        'filtered': filtered,
        'delta': delta,
        'metadata': metadata,
        'pipeline_seconds': pipeline_seconds,
        'quality_runtime_fraction': runtime_fraction,
        'gate': gate,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = format_markdown(report)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    if not gate['passed']:
        raise RuntimeError('E7b top-ratio gate failed.')


if __name__ == '__main__':
    main()