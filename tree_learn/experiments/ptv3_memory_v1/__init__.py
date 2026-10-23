"""Versioned full-cloud memory audit; baseline and vendored sources stay intact."""

BASE_COMMIT = "855f831c01089261b93658fa2d9a4507bc43be27"


def resolved_config(config, variant):
    """Only select recomputation policy; retain every model/data parameter."""
    import copy
    options = copy.deepcopy(config)
    variants = {"reference", "serialization", "block", "stage", "block_nested", "stage_nested", "offload"}
    if variant not in variants:
        raise ValueError(variant)
    if options["model"].get("backbone") != "ptv3":
        raise ValueError("This adapter requires the PTv3 baseline configuration")
    options["model"]["activation_checkpointing"] = variant not in {"block", "stage"}
    return options


def build_optimized_model(config, variant="block_nested"):
    """Independent opt-in factory; upstream and baseline factories stay intact."""
    from tree_learn.experiments.ptv3_treelearn.model import build_model
    from .adapters import apply_optimization
    model = build_model(resolved_config(config, variant))
    return model if variant == "reference" else apply_optimization(model, variant)
