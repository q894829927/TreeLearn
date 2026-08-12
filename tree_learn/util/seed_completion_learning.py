"""Learning artifacts for GT-free seed-completion proposal activation."""

import csv
import json
from pathlib import Path

import numpy as np

from .omission_diagnostics import compute_gt_omission_diagnostics
from .seed_completion_proposals import (
    PROPOSAL_FEATURE_NAMES,
    build_multiscale_seed_completion_proposals,
    proposal_rows_for_csv,
    select_oracle_seed_completion_proposals,
)


ARTIFACT_VERSION = 1
PROPOSAL_CSV_NAME = 'proposals.csv'
METADATA_NAME = 'metadata.json'
LABEL_COLUMNS = (
    'oracle_activated',
    'oracle_target_proposal',
    'oracle_target_added_points',
    'oracle_selected_purity',
    'oracle_target_tree_ids',
)


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered not in {'true', 'false'}:
        raise ValueError(f'Invalid boolean value: {value!r}')
    return lowered == 'true'


def parse_target_ids(value):
    text = str(value).strip()
    if not text:
        return ()
    return tuple(sorted(set(
        int(item) for item in text.split(';') if item.strip())))


def proposal_center(row):
    scale = float(row['scale'])
    return np.asarray([
        scale * (float(row['cell_x']) + float(row['shift_x']) + 0.5),
        scale * (float(row['cell_y']) + float(row['shift_y']) + 0.5),
    ], dtype=np.float32)


def validate_proposal_rows(rows, expected_features=PROPOSAL_FEATURE_NAMES):
    if not rows:
        raise ValueError('Seed-completion learning artifact has no proposals.')
    required = {
        'proposal_id', 'cell_x', 'cell_y', *expected_features,
        *LABEL_COLUMNS,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f'Seed-completion proposal rows miss fields: {sorted(missing)}')
    proposal_ids = np.asarray([
        int(row['proposal_id']) for row in rows], dtype=np.int64)
    if len(np.unique(proposal_ids)) != len(proposal_ids):
        raise ValueError('Seed-completion proposal IDs are not unique.')
    features = np.asarray([
        [float(row[name]) for name in expected_features]
        for row in rows], dtype=np.float32)
    centers = np.asarray([
        proposal_center(row) for row in rows], dtype=np.float32)
    if not np.isfinite(features).all() or not np.isfinite(centers).all():
        raise ValueError('Seed-completion proposal features are non-finite.')
    activated = np.asarray([
        parse_bool(row['oracle_activated']) for row in rows], dtype=bool)
    coverage = np.asarray([
        parse_bool(row['oracle_target_proposal']) for row in rows], dtype=bool)
    if np.any(activated & ~coverage):
        raise ValueError('Every activated proposal must cover a target tree.')
    target_ids = [parse_target_ids(
        row['oracle_target_tree_ids']) for row in rows]
    if any(bool(ids) != bool(flag) for ids, flag in zip(
            target_ids, coverage)):
        raise ValueError('Proposal target IDs do not align with coverage labels.')
    return {
        'features': features,
        'centers': centers,
        'proposal_ids': proposal_ids,
        'activation_target': activated,
        'coverage_target': coverage,
        'target_tree_ids': target_ids,
    }


def build_seed_completion_learning_artifact(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, baseline_predictions,
        baseline_initial_predictions, source_plot, split,
        tree_conf_thresh=0.5, tau_vert=0.6, tau_off=4.0,
        tau_min=50, tree_class_index=0, scales=(0.6, 1.2),
        shift_fractions=(0.0, 0.5), match_iou_threshold=0.5,
        min_precision_for_counted_fp=0.5,
        min_recall_for_undersegmentation=0.5,
        min_fragment_overlap_fraction=0.05,
        min_fragment_overlap_points=20,
        expected_target_tree_ids=None):
    """Create proposal labels without rerunning proposal clustering."""
    if str(split) not in {'train', 'validation'}:
        raise ValueError('Seed-completion split must be train or validation.')
    q3 = compute_gt_omission_diagnostics(
        coords, semantic_logits, offset_predictions, verticality,
        instance_labels, baseline_predictions,
        baseline_initial_predictions,
        tree_conf_thresh=tree_conf_thresh, tau_vert=tau_vert,
        tau_off=tau_off, tau_min=tau_min,
        tree_class_index=tree_class_index,
        match_iou_threshold=match_iou_threshold,
        min_precision_for_counted_fp=min_precision_for_counted_fp,
        min_recall_for_undersegmentation=min_recall_for_undersegmentation,
        min_fragment_overlap_fraction=min_fragment_overlap_fraction,
        min_fragment_overlap_points=min_fragment_overlap_points)
    target_ids = sorted(
        int(row['gt_tree_id']) for row in q3['rows']
        if row['category'] == 'base_seed_support_failure')
    if expected_target_tree_ids is not None:
        expected = sorted(set(map(int, expected_target_tree_ids)))
        if target_ids != expected:
            raise ValueError(
                'Seed-completion learning targets differ from the locked '
                f'validation Q3 artifact: observed={target_ids}, '
                f'expected={expected}.')
    generated = build_multiscale_seed_completion_proposals(
        coords, semantic_logits, offset_predictions, verticality,
        tree_conf_thresh=tree_conf_thresh, tau_vert=tau_vert,
        tau_off=tau_off, tau_min=tau_min,
        tree_class_index=tree_class_index, scales=scales,
        shift_fractions=shift_fractions)
    selected = select_oracle_seed_completion_proposals(
        generated['proposals'], generated['baseline_mask'],
        instance_labels, target_ids, tau_min=tau_min)
    rows = proposal_rows_for_csv(selected['proposal_rows'])
    arrays = validate_proposal_rows(rows)
    target_set = set(target_ids)
    covered = set()
    for ids in arrays['target_tree_ids']:
        covered.update(target_set.intersection(ids))
    metadata = {
        'artifact_version': ARTIFACT_VERSION,
        'source_plot': str(source_plot),
        'split': str(split),
        'feature_names': list(PROPOSAL_FEATURE_NAMES),
        'num_proposals': int(len(rows)),
        'num_activation_positive': int(
            arrays['activation_target'].sum()),
        'num_coverage_positive': int(
            arrays['coverage_target'].sum()),
        'num_target_trees': int(len(target_ids)),
        'num_proposal_covered_target_trees': int(len(covered)),
        'target_tree_ids': target_ids,
        'activated_proposal_ids': list(map(
            int, selected['activated_proposal_ids'])),
        'baseline': q3['baseline'],
        'label_parameters': {
            'match_iou_threshold': float(match_iou_threshold),
            'min_precision_for_counted_fp': float(
                min_precision_for_counted_fp),
            'min_recall_for_undersegmentation': float(
                min_recall_for_undersegmentation),
        },
        'proposal_parameters': {
            'scales': list(map(float, scales)),
            'shift_fractions': list(map(float, shift_fractions)),
            'tau_min': int(tau_min),
        },
    }
    return {'rows': rows, 'metadata': metadata}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_learning_artifact(artifact, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / PROPOSAL_CSV_NAME, artifact['rows'])
    (output / METADATA_NAME).write_text(
        json.dumps(
            artifact['metadata'], indent=2, ensure_ascii=False),
        encoding='utf-8')
    return output / PROPOSAL_CSV_NAME, output / METADATA_NAME


def validate_learning_artifact(csv_path, metadata_path):
    csv_path = Path(csv_path)
    metadata_path = Path(metadata_path)
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f'Incomplete seed-completion learning artifact: {csv_path}')
    with metadata_path.open(encoding='utf-8') as file:
        metadata = json.load(file)
    with csv_path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    arrays = validate_proposal_rows(
        rows, tuple(metadata['feature_names']))
    if int(metadata.get('artifact_version', 0)) != ARTIFACT_VERSION:
        raise ValueError('Seed-completion learning artifact version is stale.')
    if int(metadata['num_proposals']) != len(rows):
        raise ValueError('Proposal count differs between CSV and metadata.')
    if int(metadata['num_activation_positive']) != int(
            arrays['activation_target'].sum()):
        raise ValueError('Activation-positive count differs.')
    if int(metadata['num_coverage_positive']) != int(
            arrays['coverage_target'].sum()):
        raise ValueError('Coverage-positive count differs.')
    if len(set(metadata['target_tree_ids'])) != int(
            metadata['num_target_trees']):
        raise ValueError('Target-tree metadata is inconsistent.')
    expected_targets = set(map(int, metadata['target_tree_ids']))
    covered_targets = set()
    for target_ids in arrays['target_tree_ids']:
        covered_targets.update(target_ids)
    if not covered_targets.issubset(expected_targets):
        raise ValueError('Proposal labels contain an unexpected target tree.')
    if len(covered_targets) != int(
            metadata['num_proposal_covered_target_trees']):
        raise ValueError('Covered-target count differs.')
    if 'activated_proposal_ids' in metadata:
        observed_activated = sorted(map(
            int, arrays['proposal_ids'][arrays['activation_target']]))
        expected_activated = sorted(set(map(
            int, metadata['activated_proposal_ids'])))
        if observed_activated != expected_activated:
            raise ValueError('Activated proposal IDs differ.')
    return metadata, rows, arrays
