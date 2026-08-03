import argparse
import json
from pathlib import Path

from summarize_e6_quality_filter import (
    exact_metrics, load_results, read_pipeline_seconds)


def build_gate(baseline, filtered, metadata, config, runtime_fraction):
    f1_gain = filtered['f1_score'] - baseline['f1_score']
    commission_reduction = (
        baseline['commission_error_rate'] -
        filtered['commission_error_rate'])
    completeness_drop = (
        baseline['completeness'] - filtered['completeness'])
    completeness_passed = (
        completeness_drop <=
        float(config.gate.max_completeness_drop_pp) + 1e-12)
    effect_passed = (
        f1_gain >= float(config.gate.min_f1_gain_pp) - 1e-12 or
        (
            commission_reduction >=
            float(config.gate.min_commission_reduction_pp) - 1e-12 and
            completeness_passed
        )
    )
    gate = {
        'checkpoint_seed_locked': (
            int(metadata['checkpoint_seed']) ==
            int(config.locked.checkpoint_seed)),
        'threshold_locked': (
            abs(float(metadata['threshold']) -
                float(config.locked.threshold)) <= 1e-12),
        'filter_enabled': bool(metadata['filter_enabled']),
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
        f"# E7 锁定复核：{report['dataset_name']}",
        '',
        f"- 数据角色：**{report['dataset_role']}**",
        '- 本结果不得用于重新选择 checkpoint 或阈值。',
        f"- checkpoint：{report['checkpoint']}",
        f"- threshold：{report['threshold']:.4f}",
        f"- 质量评分耗时：{report['metadata']['scoring_seconds']:.3f}s",
        f"- 评分耗时占完整 pipeline："
        f"{100.0 * report['quality_runtime_fraction']:.3f}%",
        f"- 保留实例：{report['metadata']['num_kept']} / "
        f"{report['metadata']['num_instances']}",
        '',
        '| Run | Completeness | Commission | F1 | Precision | Recall | Coverage |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('baseline', 'filtered'):
        metrics = report[name]
        rows.append(
            f"| {name} | {metrics['completeness']:.6f}% | "
            f"{metrics['commission_error_rate']:.6f}% | "
            f"{metrics['f1_score']:.6f}% | "
            f"{metrics['precision']:.6f}% | "
            f"{metrics['recall']:.6f}% | "
            f"{metrics['coverage']:.6f}% |")
    rows.extend([
        '',
        '## Delta: filtered minus baseline',
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
        'PASS：锁定方法可进入独立外部测试。'
        if report['gate']['passed']
        else 'STOP：记录跨域失败，不得在 Wytham 上修改阈值。',
        '',
    ])
    return '\n'.join(rows)


def main():
    from tree_learn.util import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = get_config(args.config)

    baseline_path = Path(config.evaluations.baseline)
    filtered_path = Path(config.evaluations.filtered)
    for path in (baseline_path, filtered_path, Path(config.metadata)):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline = exact_metrics(load_results(baseline_path))
    filtered = exact_metrics(load_results(filtered_path))
    with open(config.metadata, encoding='utf-8') as file:
        metadata = json.load(file)
    pipeline_seconds = read_pipeline_seconds(config.pipeline_log)
    runtime_fraction = metadata['scoring_seconds'] / pipeline_seconds
    gate, delta = build_gate(
        baseline, filtered, metadata, config, runtime_fraction)
    report = {
        'dataset_name': str(config.dataset_name),
        'dataset_role': str(config.dataset_role),
        'checkpoint': str(config.locked.checkpoint),
        'threshold': float(config.locked.threshold),
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
        raise RuntimeError(
            'E7 locked Wytham gate failed; do not tune on Wytham.')


if __name__ == '__main__':
    main()
