"""Apply immutable L1W safety and Wytham final-effect gates."""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=('l1w', 'wytham'), required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


def parse_metrics(text):
    result = {}
    names = (
        'Completeness', 'Commission Error Rate', 'F1 Score',
        'Precision', 'Recall', 'Coverage',
    )
    for name in names:
        values = re.findall(
            rf'{re.escape(name)}:\s*([0-9.]+)%', text)
        if not values:
            raise ValueError(f'Missing metric: {name}')
        result[name.lower().replace(' ', '_')] = float(values[-1])
    return result


def evaluate_gate(domain, metrics):
    if domain == 'l1w':
        gate = {
            'f1_drop_passed': 98.4 - metrics['f1_score'] <= 0.5,
            'completeness_passed': metrics['completeness'] >= 100.0,
            'coverage_drop_passed': 98.0 - metrics['coverage'] <= 0.3,
        }
    else:
        f1_effect = metrics['f1_score'] - 72.0 >= 0.5
        coverage_effect = (
            metrics['coverage'] - 57.7 >= 1.0 and
            62.5 - metrics['precision'] <= 1.0)
        gate = {
            'f1_effect_passed': f1_effect,
            'coverage_effect_passed': coverage_effect,
            'effect_size_passed': f1_effect or coverage_effect,
            'final_evaluation_no_tuning': True,
        }
    gate['passed'] = all(
        value for key, value in gate.items()
        if key not in ('f1_effect_passed', 'coverage_effect_passed')
    )
    return gate


def main():
    args = parse_args()
    metrics = parse_metrics(
        Path(args.log).read_text(encoding='utf-8', errors='replace'))
    gate = evaluate_gate(args.domain, metrics)
    report = {
        'domain': args.domain,
        'metrics_percent': metrics,
        'gate': gate,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    lines = [
        f'# Sparse U-Net locked {args.domain.upper()} evaluation',
        '',
        *[f'- {key}: {value:.3f}%'
          for key, value in metrics.items()],
        '',
        '## Gate',
        '',
        *[f'- {key}: **{value}**' for key, value in gate.items()],
        '',
    ]
    summary = '\n'.join(lines)
    (output / 'summary.md').write_text(summary, encoding='utf-8')
    print(summary)
    if not gate['passed']:
        raise RuntimeError(
            f'Locked {args.domain} gate failed; do not tune on this domain.')


if __name__ == '__main__':
    main()
