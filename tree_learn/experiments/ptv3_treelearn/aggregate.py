"""Aggregate completed formal runs; short smoke scores are never paper scores."""
import json
from pathlib import Path

import numpy as np

from .provenance import sha256, write_json

METRICS = ("detection_precision", "detection_recall", "detection_f1", "completeness",
           "omission", "commission", "coverage", "segmentation_precision", "segmentation_recall")


def aggregate(index_path, output):
    """Index rows: group, domain, seed, metrics, pipeline_manifest, training_run.

    E2 rows omit training_run and use seed=null. E0/E1 require a completed formal
    final checkpoint whose bytes match the inference manifest. Domains must use
    the same GT SHA. Groups are aggregated separately, never pooled with E2.
    """
    index = json.loads(Path(index_path).read_text())
    domains, groups, identities = {}, {}, {}
    for row in index:
        metrics = json.loads(Path(row["metrics"]).read_text())
        evaluation = json.loads((Path(row["metrics"]).parent / "evaluation_manifest.json").read_text())
        pipeline = json.loads(Path(row["pipeline_manifest"]).read_text())
        domain = row["domain"]
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
            identities.setdefault(domain, []).append({"group": row["group"], "seed": row["seed"],
                "data": run["data_sha256"], "plan": run["plan_sha256"],
                "fairness": run["fairness_sha256"], "code": run["code_sha256"],
                "variant": run["config"]["experiment"]["variant"]})
        key = (row["group"], domain)
        groups.setdefault(key, []).append({"seed": row["seed"], "metrics": metrics,
            "metrics_sha256": sha256(row["metrics"]), "pipeline_manifest_sha256": sha256(row["pipeline_manifest"])})
    # Strict E0-scratch/E1 pairing requires the same data, variant and per-seed plan.
    for domain, rows in identities.items():
        paired = {}
        for row in rows:
            if row["group"] not in ("E0-scratch", "E1-ptv3"):
                continue
            identity = (row["data"], row["plan"], row["variant"], row["fairness"], row["code"])
            if row["seed"] in paired and paired[row["seed"]] != identity:
                raise ValueError("E0-scratch/E1 fairness mismatch")
            paired[row["seed"]] = identity
    results = []
    for (group, domain), rows in sorted(groups.items()):
        variants = {row["variant"] for row in identities.get(domain, []) if row["group"] == group}
        if len(variants) > 1:
            raise ValueError("Do not pool different resource protocols across seeds")
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
               "std_definition": "sample standard deviation, ddof=1", "metric_units": "fractions 0..1",
               "source_index_sha256": sha256(index_path), "E2_role": "mature model reference; not architecture fairness control"})
