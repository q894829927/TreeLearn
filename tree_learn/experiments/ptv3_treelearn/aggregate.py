"""Aggregate completed formal runs; short smoke scores are never paper scores."""
import json
from pathlib import Path

import numpy as np

from .provenance import sha256, write_json, canonical_hash

METRICS = ("detection_precision", "detection_recall", "detection_f1", "completeness",
           "omission", "commission", "coverage", "segmentation_precision", "segmentation_recall")


def evaluation_identity(row, evaluation, pipeline):
    """Bind metrics to this pipeline's actual output and common evaluation rules."""
    prediction = Path(evaluation["prediction"]).resolve()
    expected = (Path(row["pipeline_manifest"]).resolve().parent / "results/full_forest").resolve()
    if prediction.parent != expected:
        raise ValueError("Evaluation prediction is not an output of the indexed pipeline")
    if not prediction.is_file() or sha256(prediction) != evaluation["prediction_sha256"]:
        raise ValueError("Evaluation prediction file/hash changed")
    if not pipeline.get("code_sha256") or evaluation.get("code_sha256") != pipeline["code_sha256"]:
        raise ValueError("Inference/evaluation code provenance missing or inconsistent")
    cfg = pipeline["pipeline"]
    sample = {k: v for k, v in cfg["sample_generation"].items() if k != "sample_generator"}
    sample["sample_generator"] = {k: v for k, v in cfg["sample_generation"]["sample_generator"].items()
                                  if k not in ("plot_path", "features_path", "save_dir")}
    return canonical_hash({"forest": pipeline["forest_sha256"], "code": pipeline["code_sha256"],
        "grouping": cfg["grouping"], "sample_generation": sample, "shape": cfg["shape_cfg"],
        "dataset_test": {k: v for k, v in cfg["dataset_test"].items() if k != "data_root"},
        "dataloader": cfg["dataloader"], "fp16": cfg["fp16"], "return_type": cfg["save_cfg"]["return_type"],
        "streamed_tiles": pipeline["streamed_tiles"], "saved_backbone_features": pipeline["saved_backbone_features"],
        "evaluation": {k: v for k, v in evaluation["config"].items() if k != "paths"}})


def aggregate(index_path, output):
    """Index rows: group, domain, seed, metrics, pipeline_manifest, training_run.

    E2 rows omit training_run and use seed=null. E0/E1 require a completed formal
    final checkpoint whose bytes match the inference manifest. Domains must use
    the same GT SHA. Groups are aggregated separately, never pooled with E2.
    """
    index = json.loads(Path(index_path).read_text())
    domains, groups, identities, evaluation_protocols, checkpoints = {}, {}, {}, {}, {}
    for row in index:
        metrics = json.loads(Path(row["metrics"]).read_text())
        evaluation = json.loads((Path(row["metrics"]).parent / "evaluation_manifest.json").read_text())
        pipeline = json.loads(Path(row["pipeline_manifest"]).read_text())
        domain = row["domain"]
        checkpoint_key = (row["group"], row["seed"])
        if checkpoint_key in checkpoints and checkpoints[checkpoint_key] != pipeline["checkpoint_sha256"]:
            raise ValueError("Different checkpoints within one group/seed across domains")
        checkpoints[checkpoint_key] = pipeline["checkpoint_sha256"]
        if domain in domains and domains[domain] != evaluation["gt_sha256"]:
            raise ValueError("Different GT point sets within one evaluation domain")
        domains[domain] = evaluation["gt_sha256"]
        if not row["group"].startswith("E2"):
            root = Path(row["training_run"])
            run = json.loads((root / "experiment_manifest.json").read_text())
            provenance = json.loads((root / "checkpoint_provenance.json").read_text())
            if not provenance["is_formal_final"] or provenance["global_update"] != 250000 or run["phase"] != "formal":
                raise ValueError("Only complete fixed-budget formal runs can be aggregated")
            if provenance.get("final_sha256") != pipeline["checkpoint_sha256"]:
                raise ValueError("Evaluation is not based on the formal final checkpoint")
            if run["seed"] != row["seed"] or run["config"]["experiment"]["group"] != row["group"]:
                raise ValueError("Run/group/seed identity mismatch")
            if pipeline["seed"] != row["seed"]:
                raise ValueError("Inference RNG seed must match the formal run seed")
            if pipeline["config_sha256"] != run["config_sha256"]:
                raise ValueError("Inference configuration differs from the formal training run")
            identities.setdefault(domain, []).append({"group": row["group"], "seed": row["seed"],
                "data": run["data_sha256"], "plan": run["plan_sha256"],
                "fairness": run["fairness_sha256"], "code": run["code_sha256"], "config": run["config_sha256"],
                "variant": run["config"]["experiment"]["variant"]})
        protocol = evaluation_identity(row, evaluation, pipeline)
        if domain in evaluation_protocols and evaluation_protocols[domain] != protocol:
            raise ValueError("Different inference/evaluation protocols within one domain")
        evaluation_protocols[domain] = protocol
        key = (row["group"], domain)
        groups.setdefault(key, []).append({"seed": row["seed"], "metrics": metrics,
            "metrics_sha256": sha256(row["metrics"]), "pipeline_manifest_sha256": sha256(row["pipeline_manifest"])})
    # All trained controls share data/protocol; only initialization/backbone vary.
    for domain, rows in identities.items():
        paired = {}
        for row in rows:
            if row["group"] not in ("E0-scratch", "E0-main-hais", "E1-ptv3"):
                continue
            identity = (row["data"], row["plan"], row["variant"], row["fairness"], row["code"])
            if row["seed"] in paired and paired[row["seed"]] != identity:
                raise ValueError("E0/E1 fairness mismatch")
            paired[row["seed"]] = identity
    results = []
    for (group, domain), rows in sorted(groups.items()):
        variants = {row["variant"] for row in identities.get(domain, []) if row["group"] == group}
        if len(variants) > 1:
            raise ValueError("Do not pool different resource protocols across seeds")
        protocols = {(row["data"], row["fairness"], row["code"], row["config"])
                     for row in identities.get(domain, []) if row["group"] == group}
        if len(protocols) > 1:
            raise ValueError("Training data/configuration/code changed across seeds")
        seeds = [row["seed"] for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seeds would bias the aggregate")
        if not group.startswith("E2") and set(seeds) != {42, 43, 44}:
            raise ValueError("Formal report requires all three seeds: 42, 43, 44")
        summary = {}
        for name in METRICS:
            values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
            summary[name] = {"mean": float(np.mean(values)) if values else None,
                             "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             "n": len(values)}
        variant = next(iter(variants), None)
        label = "PTv3-resource-limited" if group == "E1-ptv3" and variant and "resource-limited" in variant else group
        results.append({"group": group, "display_label": label, "training_protocol_variant": variant,
                        "domain": domain, "per_seed": rows, "summary": summary})
    write_json(output, {"results": results, "gt_sha256_by_domain": domains,
               "inference_evaluation_protocol_sha256_by_domain": evaluation_protocols,
               "std_definition": "sample standard deviation, ddof=1", "metric_units": "fractions 0..1",
               "source_index_sha256": sha256(index_path), "E2_role": "mature model reference; not architecture fairness control"})
