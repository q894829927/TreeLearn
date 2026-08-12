"""Train and compare Q5b1 seed-completion proposal activation heads."""

import argparse
import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np

from tree_learn.util.seed_completion_learning import (
    validate_learning_artifact,
)

EXPECTED_SPLITS = {
    "train": [
        "A1N", "A1W", "G1N", "G1W", "G2N", "G2W", "G3N",
        "G3W", "L2N", "L2W", "LG1", "LG2", "LG3",
    ],
    "validation": ["G4N", "G4W", "L1N", "O1N", "O1W"],
}

def validate_settings(settings):
    if "wytham" in json.dumps(settings).lower():
        raise ValueError("Q5b1 training must not reference Wytham.")
    if settings.get("splits") != EXPECTED_SPLITS:
        raise ValueError("Q5b1 must use the preregistered 13/5 plot split.")
    if list(settings.get("seeds", ())) != [42, 43, 44]:
        raise ValueError("Q5b1 seeds must remain [42, 43, 44].")
    if int(settings.get("locked_seed", -1)) != 42:
        raise ValueError("Q5b1 locked seed must remain 42.")
    expected_models = [
        "multitask_mlp", "logistic_regression", "fixed_rule"]
    if list(settings["selection"].get("model_priority", ())) != expected_models:
        raise ValueError("Q5b1 model priority differs from preregistration.")
    ratios = list(map(float, settings["selection"]["keep_ratios"]))
    if ratios != sorted(set(ratios)) or not ratios or (
            ratios[0] <= 0.0 or ratios[-1] > 0.02):
        raise ValueError("Q5b1 keep-ratio grid is invalid.")
    selection = settings["selection"]
    gate = settings["gate"]
    for name in (
            "min_activation_precision", "min_activation_recall",
            "min_target_tree_recall"):
        if float(selection[name]) != float(gate[name]):
            raise ValueError(
                f"Q5b1 selection and deployment gates differ for {name}.")
    if int(gate["min_mlp_seed_passes"]) < 2:
        raise ValueError("Q5b1 must pass on at least two MLP seeds.")


def load_settings(config_path):
    import yaml
    with Path(config_path).open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    validate_settings(settings)
    return settings

def binary_metrics(targets, scores):
    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if len(targets) != len(scores) or len(np.unique(targets)) != 2:
        raise ValueError("Binary metrics need aligned examples of both classes.")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    ascending = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[ascending]
    _, starts, counts = np.unique(
        sorted_scores, return_index=True, return_counts=True)
    sorted_ranks = np.repeat(starts + 0.5 * (counts + 1), counts)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[ascending] = sorted_ranks
    auc = (
        ranks[targets].sum() - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    descending = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[descending].astype(np.int64)
    sorted_scores = scores[descending]
    tp = np.cumsum(sorted_targets)
    fp = np.cumsum(1 - sorted_targets)
    ends = np.flatnonzero(np.r_[
        sorted_scores[1:] != sorted_scores[:-1], True])
    recall = tp[ends] / positives
    precision = tp[ends] / (tp[ends] + fp[ends])
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {
        "roc_auc": float(auc),
        "average_precision": ap,
        "prevalence": float(targets.mean()),
        "ap_lift": float(ap / max(targets.mean(), 1e-12)),
    }


def robust_fit(values):
    values = np.asarray(values, dtype=np.float32)
    center = np.median(values, axis=0).astype(np.float32)
    scale = (
        np.quantile(values, 0.75, axis=0) -
        np.quantile(values, 0.25, axis=0)).astype(np.float32)
    scale = np.maximum(scale, 1e-3)
    return center, scale


def normalize(values, center, scale):
    output = np.clip(
        (np.asarray(values, dtype=np.float32) - center) / scale,
        -10.0, 10.0).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Q5b1 normalized features are non-finite.")
    return output


def load_dataset(settings):
    root = Path(settings["data_root"])
    summary_path = root / "generation_summary.json"
    manifest_path = root / "manifest.csv"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Run Q5b1 data generation before training.")
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    if not summary["gate"]["passed"]:
        raise ValueError("Q5b1 data gate did not pass.")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    requested = {
        (split, plot)
        for split in ("train", "validation")
        for plot in settings["splits"][split]}
    observed = {
        (row["split"], row["source_plot"]) for row in manifest}
    if requested != observed:
        raise ValueError("Q5b1 training split differs from generation.")
    if set(settings["splits"]["train"]) & set(
            settings["splits"]["validation"]):
        raise ValueError("Q5b1 train/validation plots overlap.")

    records = []
    feature_names = None
    for row in manifest:
        plot = row["source_plot"]
        split = row["split"]
        directory = root / split / plot
        metadata, _, arrays = validate_learning_artifact(
            directory / "proposals.csv",
            directory / "metadata.json")
        names = list(metadata["feature_names"])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("Q5b1 proposal feature schema differs.")
        records.append({
            "plot": plot,
            "split": split,
            "features": arrays["features"],
            "centers": arrays["centers"],
            "proposal_ids": arrays["proposal_ids"],
            "activation_target": arrays["activation_target"],
            "coverage_target": arrays["coverage_target"],
            "target_tree_ids": arrays["target_tree_ids"],
            "all_target_tree_ids": tuple(map(
                int, metadata["target_tree_ids"])),
        })
    train_values = np.concatenate([
        row["features"] for row in records if row["split"] == "train"])
    center, scale = robust_fit(train_values)
    for row in records:
        row["x"] = normalize(row["features"], center, scale)
    return {
        "records": records,
        "feature_names": feature_names,
        "center": center,
        "scale": scale,
    }


def sampled_train_data(dataset, settings, seed):
    generator = np.random.default_rng(int(seed))
    rows = []
    ratio = int(settings["training"]["negative_to_positive_ratio"])
    minimum = int(settings["training"]["min_negatives_per_plot"])
    for record in dataset["records"]:
        if record["split"] != "train":
            continue
        positive = record["coverage_target"] | record["activation_target"]
        positive_indices = np.flatnonzero(positive)
        negative_indices = np.flatnonzero(~positive)
        negative_count = min(
            len(negative_indices),
            max(minimum, ratio * max(len(positive_indices), 1)))
        chosen_negative = generator.choice(
            negative_indices, size=negative_count, replace=False)
        selected = np.concatenate([positive_indices, chosen_negative])
        generator.shuffle(selected)
        rows.append((
            record["x"][selected],
            record["activation_target"][selected],
            record["coverage_target"][selected]))
    if not rows:
        raise ValueError("Q5b1 has no training proposal rows.")
    return {
        "x": np.concatenate([row[0] for row in rows]),
        "activation": np.concatenate([row[1] for row in rows]),
        "coverage": np.concatenate([row[2] for row in rows]),
    }


def probability_product(first, second):
    return np.sqrt(np.clip(
        np.asarray(first, dtype=np.float64) *
        np.asarray(second, dtype=np.float64), 0.0, 1.0))


def rule_scores(records, feature_names):
    indices = {name: index for index, name in enumerate(feature_names)}
    output = {}
    for record in records:
        values = record["features"]
        score = (
            2.0 * values[:, indices["tree_probability_mean"]] +
            1.5 * values[:, indices["verticality_mean"]] +
            0.02 * values[:, indices["added_seed_count"]] -
            1.0 * values[:, indices["vote_radius_rms"]] -
            0.5 * values[:, indices["tree_probability_std"]] -
            0.25 * values[:, indices["verticality_std"]])
        output[record["plot"]] = score.astype(np.float64)
    return output


def deterministic_select(
        record, scores, keep_ratio, nms_radius, max_selected):
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != len(record["proposal_ids"]):
        raise ValueError("Q5b1 selection scores are not aligned.")
    budget = min(
        int(math.ceil(float(keep_ratio) * len(values))),
        int(max_selected))
    budget = max(budget, 1)
    order = np.lexsort((record["proposal_ids"], -values))
    accepted = []
    accepted_centers = []
    radius_squared = float(nms_radius) ** 2
    for index in order:
        center = record["centers"][index]
        if accepted_centers:
            existing = np.asarray(accepted_centers)
            if np.any(np.sum((existing - center) ** 2, axis=1) <
                      radius_squared):
                continue
        accepted.append(int(index))
        accepted_centers.append(center)
        if len(accepted) >= budget:
            break
    mask = np.zeros(len(values), dtype=bool)
    mask[accepted] = True
    return mask


def selection_metrics(records, scores_by_plot, selection):
    accepted_count = 0
    activation_positive = 0
    activation_selected = 0
    coverage_selected = 0
    target_total = 0
    target_covered = 0
    per_plot = []
    for record in records:
        if record["split"] != "validation":
            continue
        selected = deterministic_select(
            record, scores_by_plot[record["plot"]],
            selection["keep_ratio"], selection["nms_radius"],
            selection["max_selected_per_plot"])
        accepted_count += int(selected.sum())
        activation_positive += int(record["activation_target"].sum())
        activation_selected += int(
            (selected & record["activation_target"]).sum())
        coverage_selected += int(
            (selected & record["coverage_target"]).sum())
        covered_ids = set()
        for index in np.flatnonzero(selected):
            covered_ids.update(record["target_tree_ids"][index])
        expected = set(record["all_target_tree_ids"])
        recovered = len(covered_ids & expected)
        target_total += len(expected)
        target_covered += recovered
        per_plot.append({
            "source_plot": record["plot"],
            "selected": int(selected.sum()),
            "activation_recall": float(
                (selected & record["activation_target"]).sum() /
                max(record["activation_target"].sum(), 1)),
            "target_tree_recall": float(
                recovered / max(len(expected), 1)),
        })
    return {
        "selected": accepted_count,
        "activation_precision": float(
            activation_selected / max(accepted_count, 1)),
        "activation_recall": float(
            activation_selected / max(activation_positive, 1)),
        "coverage_precision": float(
            coverage_selected / max(accepted_count, 1)),
        "target_tree_recall": float(
            target_covered / max(target_total, 1)),
        "covered_target_trees": int(target_covered),
        "target_trees": int(target_total),
        "per_plot": per_plot,
    }


def evaluate_scores(records, scores_by_plot, settings):
    activation_targets = np.concatenate([
        row["activation_target"] for row in records
        if row["split"] == "validation"])
    coverage_targets = np.concatenate([
        row["coverage_target"] for row in records
        if row["split"] == "validation"])
    scores = np.concatenate([
        scores_by_plot[row["plot"]] for row in records
        if row["split"] == "validation"])
    binary = {
        "activation": binary_metrics(activation_targets, scores),
        "coverage": binary_metrics(coverage_targets, scores),
    }
    candidates = []
    selection_settings = settings["selection"]
    for ratio in selection_settings["keep_ratios"]:
        row = selection_metrics(
            records, scores_by_plot, {
                "keep_ratio": float(ratio),
                "nms_radius": float(selection_settings["nms_radius"]),
                "max_selected_per_plot": int(
                    selection_settings["max_selected_per_plot"]),
            })
        row["keep_ratio"] = float(ratio)
        row["eligible"] = (
            row["activation_precision"] >= float(
                selection_settings["min_activation_precision"]) and
            row["activation_recall"] >= float(
                selection_settings["min_activation_recall"]) and
            row["target_tree_recall"] >= float(
                selection_settings["min_target_tree_recall"]))
        candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible if eligible else candidates
    selected = max(
        pool,
        key=lambda row: (
            row["target_tree_recall"],
            row["activation_recall"],
            row["activation_precision"],
            -row["selected"],
            -row["keep_ratio"]))
    return {
        "binary": binary,
        "selection_candidates": candidates,
        "selected": selected,
        "passed": bool(eligible),
    }


def set_seed(seed):
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_logistic(train, records, settings):
    from sklearn.linear_model import LogisticRegression

    models = {}
    scores = {}
    for name in ("activation", "coverage"):
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=int(settings["logistic"]["max_iter"]),
            C=float(settings["logistic"]["c"]),
            random_state=int(settings["locked_seed"]))
        model.fit(train["x"], train[name])
        models[name] = model
    for record in records:
        first = models["activation"].predict_proba(record["x"])[:, 1]
        second = models["coverage"].predict_proba(record["x"])[:, 1]
        scores[record["plot"]] = probability_product(first, second)
    return models, scores


def build_mlp(input_dim, model_settings):
    import torch
    from torch import nn

    hidden = int(model_settings["hidden_dim"])
    dropout = float(model_settings["dropout"])

    class ProposalActivationMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU())
            self.activation_head = nn.Linear(hidden, 1)
            self.coverage_head = nn.Linear(hidden, 1)

        def forward(self, values):
            embedding = self.encoder(values)
            return (
                self.activation_head(embedding).squeeze(-1),
                self.coverage_head(embedding).squeeze(-1))

    return ProposalActivationMLP()


def predict_mlp(model, records, device, chunk_size):
    import torch

    scores = {}
    model.eval()
    with torch.no_grad():
        for record in records:
            chunks = []
            for start in range(0, len(record["x"]), int(chunk_size)):
                values = torch.from_numpy(
                    record["x"][start:start + int(chunk_size)]).to(device)
                activation, coverage = model(values)
                chunks.append(probability_product(
                    torch.sigmoid(activation).cpu().numpy(),
                    torch.sigmoid(coverage).cpu().numpy()))
            scores[record["plot"]] = np.concatenate(chunks)
    return scores


def train_mlp(train, records, settings, seed):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp(train["x"].shape[1], settings["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["optimizer"]["lr"]),
        weight_decay=float(settings["optimizer"]["weight_decay"]))
    activation_positive = float(train["activation"].sum())
    coverage_positive = float(train["coverage"].sum())
    activation_weight = min(
        (len(train["activation"]) - activation_positive) /
        max(activation_positive, 1.0),
        float(settings["loss"]["max_pos_weight"]))
    coverage_weight = min(
        (len(train["coverage"]) - coverage_positive) /
        max(coverage_positive, 1.0),
        float(settings["loss"]["max_pos_weight"]))
    dataset = TensorDataset(
        torch.from_numpy(train["x"]),
        torch.from_numpy(train["activation"].astype(np.float32)),
        torch.from_numpy(train["coverage"].astype(np.float32)))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(settings["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0)
    best = None
    patience = 0
    training_settings = settings["training"]
    for epoch in range(1, int(training_settings["epochs"]) + 1):
        model.train()
        for values, activation_target, coverage_target in loader:
            values = values.to(device)
            activation_target = activation_target.to(device)
            coverage_target = coverage_target.to(device)
            activation, coverage = model(values)
            loss = (
                F.binary_cross_entropy_with_logits(
                    activation, activation_target,
                    pos_weight=torch.tensor(
                        activation_weight, device=device)) +
                float(settings["loss"]["coverage_weight"]) *
                F.binary_cross_entropy_with_logits(
                    coverage, coverage_target,
                    pos_weight=torch.tensor(
                        coverage_weight, device=device)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training_settings["grad_clip"]))
            optimizer.step()
        if epoch % int(training_settings["eval_frequency"]) != 0:
            continue
        scores = predict_mlp(
            model, records, device,
            training_settings["inference_chunk_size"])
        evaluated = evaluate_scores(records, scores, settings)
        objective = (
            evaluated["binary"]["activation"]["average_precision"] +
            evaluated["binary"]["coverage"]["average_precision"])
        if best is None or objective > best["objective"] + 1e-12:
            best = {
                "epoch": epoch,
                "objective": objective,
                "state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
        if patience >= int(training_settings["patience_evaluations"]):
            break
    model.load_state_dict(best["state_dict"])
    scores = predict_mlp(
        model, records, device,
        training_settings["inference_chunk_size"])
    return model, scores, best["epoch"], str(device)


def qualification(evaluated, gate):
    selected = evaluated["selected"]
    return {
        "activation_ap_lift_passed": (
            evaluated["binary"]["activation"]["ap_lift"] >=
            float(gate["min_activation_ap_lift"])),
        "coverage_ap_lift_passed": (
            evaluated["binary"]["coverage"]["ap_lift"] >=
            float(gate["min_coverage_ap_lift"])),
        "activation_precision_passed": (
            selected["activation_precision"] >=
            float(gate["min_activation_precision"])),
        "activation_recall_passed": (
            selected["activation_recall"] >=
            float(gate["min_activation_recall"])),
        "target_tree_recall_passed": (
            selected["target_tree_recall"] >=
            float(gate["min_target_tree_recall"])),
        "selection_found": evaluated["passed"],
    }


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8")


def run(config_path):
    settings = load_settings(config_path)
    dataset = load_dataset(settings)
    records = dataset["records"]
    output = Path(settings["output_dir"])
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    rule = evaluate_scores(
        records,
        rule_scores(records, dataset["feature_names"]),
        settings)
    rule_gate = qualification(rule, settings["gate"])
    rule_gate["passed"] = all(rule_gate.values())
    rows.append({
        "model": "fixed_rule",
        "seed": int(settings["locked_seed"]),
        "epoch": None,
        "device": "cpu",
        "evaluation": rule,
        "gate": rule_gate,
    })

    train = sampled_train_data(
        dataset, settings, int(settings["locked_seed"]))
    logistic_models, logistic_scores = train_logistic(
        train, records, settings)
    logistic = evaluate_scores(records, logistic_scores, settings)
    logistic_gate = qualification(logistic, settings["gate"])
    logistic_gate["passed"] = all(logistic_gate.values())
    rows.append({
        "model": "logistic_regression",
        "seed": int(settings["locked_seed"]),
        "epoch": None,
        "device": "cpu",
        "evaluation": logistic,
        "gate": logistic_gate,
    })
    with (checkpoint_dir / "logistic_regression.pkl").open("wb") as file:
        pickle.dump({
            "models": logistic_models,
            "feature_names": dataset["feature_names"],
            "center": dataset["center"],
            "scale": dataset["scale"],
        }, file)

    for seed in settings["seeds"]:
        train = sampled_train_data(dataset, settings, int(seed))
        model, scores, epoch, device = train_mlp(
            train, records, settings, int(seed))
        evaluated = evaluate_scores(records, scores, settings)
        gate = qualification(evaluated, settings["gate"])
        gate["passed"] = all(gate.values())
        rows.append({
            "model": "multitask_mlp",
            "seed": int(seed),
            "epoch": int(epoch),
            "device": device,
            "evaluation": evaluated,
            "gate": gate,
        })
        import torch
        torch.save({
            "net": model.state_dict(),
            "seed": int(seed),
            "epoch": int(epoch),
            "feature_names": dataset["feature_names"],
            "feature_center": dataset["center"],
            "feature_scale": dataset["scale"],
            "model": settings["model"],
            "selection": evaluated["selected"],
        }, checkpoint_dir / f"multitask_mlp_seed{seed}.pth")
        print(
            f"DONE MLP seed {seed}: epoch={epoch}, "
            f"activation AP={evaluated['binary']['activation']['average_precision']:.6f}, "
            f"target recall={evaluated['selected']['target_tree_recall']:.3%}",
            flush=True)

    models = {}
    for name in ("fixed_rule", "logistic_regression", "multitask_mlp"):
        values = [row for row in rows if row["model"] == name]
        pass_count = sum(row["gate"]["passed"] for row in values)
        required = (
            int(settings["gate"]["min_mlp_seed_passes"])
            if name == "multitask_mlp" else 1)
        models[name] = {
            "runs": len(values),
            "pass_count": int(pass_count),
            "required_passes": int(required),
            "eligible": pass_count >= required,
            "mean_activation_ap": float(np.mean([
                row["evaluation"]["binary"]["activation"][
                    "average_precision"] for row in values])),
            "mean_target_tree_recall": float(np.mean([
                row["evaluation"]["selected"]["target_tree_recall"]
                for row in values])),
        }
    priority = list(settings["selection"]["model_priority"])
    eligible = [name for name in priority if models[name]["eligible"]]
    recommended = eligible[0] if eligible else None
    locked_rows = [
        row for row in rows
        if row["model"] == recommended and
        int(row["seed"]) == int(settings["locked_seed"])]
    locked_passed = bool(
        locked_rows and locked_rows[0]["gate"]["passed"])
    gate = {
        "data_gate_passed": True,
        "at_least_one_eligible_model": recommended is not None,
        "locked_seed_passed": locked_passed,
    }
    gate["passed"] = all(gate.values())
    result = {
        "recommended_model": recommended,
        "locked_seed": int(settings["locked_seed"]),
        "models": models,
        "runs": rows,
        "gate": gate,
    }
    save_json(output / "summary.json", result)
    with (output / "per_run_metrics.csv").open(
            "w", newline="", encoding="utf-8") as file:
        fields = [
            "model", "seed", "epoch", "activation_ap",
            "activation_ap_lift", "coverage_ap", "coverage_ap_lift",
            "keep_ratio", "selected", "activation_precision",
            "activation_recall", "target_tree_recall", "passed"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            evaluated = row["evaluation"]
            selected = evaluated["selected"]
            writer.writerow({
                "model": row["model"],
                "seed": row["seed"],
                "epoch": row["epoch"],
                "activation_ap": evaluated["binary"]["activation"][
                    "average_precision"],
                "activation_ap_lift": evaluated["binary"]["activation"][
                    "ap_lift"],
                "coverage_ap": evaluated["binary"]["coverage"][
                    "average_precision"],
                "coverage_ap_lift": evaluated["binary"]["coverage"][
                    "ap_lift"],
                "keep_ratio": selected["keep_ratio"],
                "selected": selected["selected"],
                "activation_precision": selected["activation_precision"],
                "activation_recall": selected["activation_recall"],
                "target_tree_recall": selected["target_tree_recall"],
                "passed": row["gate"]["passed"],
            })
    lines = [
        "# Q5b1 Seed-Completion proposal activation", "",
        "- Uses only the fixed train/validation forests; Wytham is forbidden.",
        f"- Recommended model: **{recommended}**",
        f"- Locked seed: {settings['locked_seed']}", "",
        "| Model | Runs | Pass | Activation AP | Target recall | Eligible |",
        "|---|---:|---:|---:|---:|---|"]
    for name in priority:
        row = models[name]
        lines.append(
            f"| {name} | {row['runs']} | "
            f"{row['pass_count']}/{row['required_passes']} | "
            f"{row['mean_activation_ap']:.6f} | "
            f"{row['mean_target_tree_recall']:.3%} | "
            f"{row['eligible']} |")
    lines.extend(["", "## Gate", ""])
    lines.extend(
        f"- {name}: **{passed}**" for name, passed in gate.items())
    lines.extend(["", (
        "PASS: proceed to Q5b2 pipeline integration on validation forests."
        if gate["passed"] else
        "STOP: proposal activation is not deployable; close Seed-Completion.")])
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if not gate["passed"]:
        raise RuntimeError(
            "Q5b1 activation-head gate failed; do not integrate pipeline.")


def main():
    parser = argparse.ArgumentParser(
        description="Train Q5b1 proposal activation heads.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
