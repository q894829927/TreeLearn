"""Numerical A/B evidence, including all gradients and all persistent buffers."""
import math
import copy
import json
from pathlib import Path

import torch

from tree_learn.experiments.ptv3_treelearn.provenance import sha256, write_json, state_equal


def tensor_errors(reference, actual):
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError("Shape/dtype changed")
    a, b = reference.double(), actual.double()
    difference = a-b
    return dict(max_abs=float(difference.abs().max()) if a.numel() else 0.,
                squared_error=float(difference.square().sum()), squared_reference=float(a.square().sum()),
                elements=a.numel(), bitwise_equal=torch.equal(reference, actual))


def compare(reference, actual, output):
    reference, actual = Path(reference), Path(actual)
    manifests = [json.loads((folder/"manifest.json").read_text()) for folder in (reference, actual)]
    for key in ("input_sha256", "initial_state_sha256"):
        identities = [manifest["input"].get(key) for manifest in manifests]
        if None in identities or identities[0] != identities[1]:
            raise ValueError("Paired comparison requires identical frozen inputs and initial weights")
    configs = [copy.deepcopy(manifest["config"]) for manifest in manifests]
    for config in configs:
        config["model"].pop("activation_checkpointing")
    if configs[0] != configs[1]:
        raise ValueError("Non-checkpoint configuration changed")
    left, right = torch.load(reference/"comparison.pth", map_location="cpu"), torch.load(actual/"comparison.pth", map_location="cpu")
    result = dict(reference=str(reference), actual=str(actual), reference_sha256=sha256(reference/"comparison.pth"),
                  actual_sha256=sha256(actual/"comparison.pth"), prediction_errors={}, gradient_errors={}, buffer_errors={})
    result["identical_input_and_initial_weights"] = True
    result["identical_non_checkpoint_config"] = True
    for group, target in (("predictions", "prediction_errors"), ("gradients", "gradient_errors"), ("buffers", "buffer_errors")):
        if left[group].keys() != right[group].keys(): raise ValueError("Different state/gradient coverage")
        for name in left[group]:
            result[target][name] = tensor_errors(left[group][name], right[group][name])
    gradients = list(result["gradient_errors"].values())
    result["gradient_summary"] = dict(max_abs=max(x["max_abs"] for x in gradients),
        relative_l2=math.sqrt(sum(x["squared_error"] for x in gradients)/max(sum(x["squared_reference"] for x in gradients),1e-300)),
        tensors=len(gradients), all_finite=all(math.isfinite(x["max_abs"]) for x in gradients))
    result["bn_and_all_buffers_exact"] = all(x["bitwise_equal"] for x in result["buffer_errors"].values())
    result["rng_after_backward_exact"] = state_equal(left["rng"],right["rng"])
    result["loss_abs_diff"] = abs(left["loss"]-right["loss"])
    # Acceptance is intentionally reported alongside raw per-tensor errors.
    # Gradient relative L2 tolerates CUDA atomic/Flash backward reduction order;
    # the unscaled maximum is exposed and is never described as bitwise replay.
    result["acceptance"] = dict(prediction_atol=1e-4,prediction_rtol=1e-3,gradient_relative_l2=1e-3,
                                buffers="exact",rng="exact")
    result["status"] = "pass" if all(torch.allclose(left["predictions"][k],right["predictions"][k],atol=1e-4,rtol=1e-3)
        for k in left["predictions"]) and result["gradient_summary"]["relative_l2"]<=1e-3 and result["gradient_summary"]["all_finite"] and result["bn_and_all_buffers_exact"] and result["rng_after_backward_exact"] else "fail"
    write_json(output, result)
    return {k:result[k] for k in ("status","gradient_summary","bn_and_all_buffers_exact","rng_after_backward_exact","loss_abs_diff")}
