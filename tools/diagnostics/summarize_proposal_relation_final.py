"""Summarize locked P3 L1W/Wytham evaluations without tuning."""

import argparse
import json
import re
from pathlib import Path

import yaml


PATTERNS = {
    'completeness': r'Completeness:\s*([0-9.]+)%',
    'commission': r'Commission Error Rate:\s*([0-9.]+)%',
    'f1': r'F1 Score:\s*([0-9.]+)%',
    'precision': r'Precision:\s*([0-9.]+)%',
    'recall': r'Recall:\s*([0-9.]+)%',
    'coverage': r'Coverage:\s*([0-9.]+)%',
}


def _metrics(path):
    text = Path(path).read_text(encoding='utf-8')
    result = {}
    for name, pattern in PATTERNS.items():
        values = re.findall(pattern, text)
        if not values:
            raise ValueError(f'Missing {name} in {path}.')
        result[name] = float(values[-1])
    return result


def run(config_path):
    settings = yaml.safe_load(
        Path(config_path).read_text(encoding='utf-8'))['proposal_relation_final']
    p2 = json.loads(Path(settings['p2_summary']).read_text(encoding='utf-8'))
    if not p2['gate']['passed']:
        raise RuntimeError('P2 gate did not pass; P3 is forbidden.')
    l1w = _metrics(settings['l1w_log'])
    b1 = _metrics(settings['wytham_b1_log'])
    wytham = _metrics(settings['wytham_relation_log'])
    official = settings['official']
    gate_cfg = settings['gate']
    l1w_gate = {
        'f1_safe': official['l1w_f1']-l1w['f1'] <=
            float(gate_cfg['maximum_l1w_f1_drop_pp']),
        'completeness_preserved': l1w['completeness'] >= 100.,
        'coverage_safe': official['l1w_coverage']-l1w['coverage'] <=
            float(gate_cfg['maximum_l1w_coverage_drop_pp']),
    }
    f1_gain = wytham['f1']-float(official['wytham_f1'])
    coverage_gain = wytham['coverage']-float(official['wytham_coverage'])
    precision_drop = float(official['wytham_precision'])-wytham['precision']
    wytham_gate = {
        'beats_official_effect': (
            f1_gain >= float(gate_cfg['minimum_wytham_f1_gain_pp']) or
            (coverage_gain >= float(gate_cfg['minimum_wytham_coverage_gain_pp'])
             and precision_drop <=
             float(gate_cfg['maximum_wytham_precision_drop_pp']))),
        'beats_locked_b1': wytham['f1']-b1['f1'] >=
            float(gate_cfg['minimum_gain_over_b1_pp']),
    }
    result = {
        'locked_seed': p2['locked_seed'],
        'locked_threshold': p2['locked_threshold'],
        'locked_checkpoint': p2['locked_checkpoint'],
        'l1w': l1w, 'wytham_b1_hdbscan': b1, 'wytham_relation': wytham,
        'l1w_gate': l1w_gate, 'wytham_gate': wytham_gate,
        'f1_gain_over_official_pp': f1_gain,
        'f1_gain_over_b1_pp': wytham['f1']-b1['f1'],
        'coverage_gain_over_official_pp': coverage_gain,
    }
    result['passed'] = all(l1w_gate.values()) and all(wytham_gate.values())
    output = Path(settings['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# P3 垂直关系注意力锁定复核', '',
        f'- checkpoint: `{result["locked_checkpoint"]}`',
        f'- seed: {result["locked_seed"]}',
        f'- threshold: {result["locked_threshold"]:.4f}', '',
        '| Run | Completeness | Commission | F1 | Precision | Recall | Coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in (
            ('L1W relation', l1w), ('Wytham B1-HDBSCAN', b1),
            ('Wytham relation', wytham)):
        lines.append(
            f'| {name} | {row["completeness"]:.3f}% | '
            f'{row["commission"]:.3f}% | {row["f1"]:.3f}% | '
            f'{row["precision"]:.3f}% | {row["recall"]:.3f}% | '
            f'{row["coverage"]:.3f}% |')
    lines += ['', '## Wytham delta', '',
              f'- F1 vs official: {f1_gain:+.3f} pp',
              f'- F1 vs locked B1: {result["f1_gain_over_b1_pp"]:+.3f} pp',
              f'- Coverage vs official: {coverage_gain:+.3f} pp',
              '', '## Gate', '']
    lines += [f'- L1W {k}: **{v}**' for k, v in l1w_gate.items()]
    lines += [f'- Wytham {k}: **{v}**' for k, v in wytham_gate.items()]
    lines += ['', f'- passed: **{result["passed"]}**', '',
              'PASS' if result['passed']
              else 'STOP：记录锁定跨域结果，不得在 Wytham 上调参。', '']
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    if not result['passed']:
        raise RuntimeError('P3 locked final gate failed; do not tune on Wytham.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
