"""Isolate the captured Linear overflow; FP32 is a diagnostic counterfactual.

This never edits or resumes training and never exports an optimized model.
"""
import argparse
import copy
import functools
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    base, replay, out = map(lambda x: Path(x).resolve(), [args.run_root, args.replay, args.output])
    if out.exists():
        p.error("Use a new output directory")
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/"diagnostic_source.py")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    frozen = base/"source"
    sys.path[:0] = [str(frozen/"work_dirs/ptv3_treelearn/runtime/packages"), str(frozen)]
    os.chdir(frozen)
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from tree_learn.model.blocks import MLP
    from tree_learn.experiments.ptv3_treelearn.data import PlannedDataset, sample_digest
    from tree_learn.experiments.ptv3_treelearn.provenance import sha256

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    case = torch.load(replay/"first_bad_operator.pt", map_location="cpu")
    if case["module"] != "semantic_linear.0":
        raise ValueError("This probe requires the captured first semantic Linear")
    state = torch.load(replay/"state_before_target.pth", map_location="cpu")["net"]
    for k, v in case["state"].items():
        assert torch.equal(v, state["semantic_linear.0."+k])
    x = case["inputs"][0].cuda()
    weight, bias = case["state"]["weight"].cuda(), case["state"]["bias"].cuda()

    def stats(t):
        finite = torch.isfinite(t)
        result = dict(dtype=str(t.dtype), shape=list(t.shape), finite=bool(finite.all()),
                      nan=int(torch.isnan(t).sum()), posinf=int(torch.isposinf(t).sum()),
                      neginf=int(torch.isneginf(t).sum()))
        if result["finite"]:
            result.update(min=float(t.min()), max=float(t.max()))
        return result

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            half = F.linear(x, weight, bias)
        with torch.cuda.amp.autocast():
            half_repeat = F.linear(x, weight, bias)
        full = F.linear(x, weight, bias)
        quantized = F.linear(x.half().float(), weight.half().float(), bias.half().float())
        bad = ~torch.isfinite(half)
        channels = torch.where(bad.any(0))[0].tolist()
        exceed = full.abs() > torch.finfo(torch.float16).max
        above_round_threshold = full.abs() >= 65520
        rows = torch.unique(torch.cat([torch.tensor([0, len(x)//2, len(x)-1], device=x.device),
                                      torch.where(bad.any(1))[0][:8]]))
        double = F.linear(x[rows].double(), weight.double(), bias.double())
        op = dict(input=stats(x), weight=stats(weight), bias=stats(bias), amp=stats(half),
                  fp32=stats(full), quantized_operands_fp32=stats(quantized),
                  bad_channels=channels, bad_count_per_channel=bad.sum(0).tolist(),
                  fp16_max=torch.finfo(torch.float16).max,
                  fp32_values_above_fp16_max=int(exceed.sum()),
                  fp32_values_above_round_to_inf_threshold=int(above_round_threshold.sum()),
                  fp32_bad_channel_ranges={str(c):[float(full[:,c].min()),float(full[:,c].max())] for c in channels},
                  fp16_repeat_same_nonfinite_mask=bool(torch.equal(torch.isfinite(half),torch.isfinite(half_repeat))),
                  fp16_repeat_equal_finite_values=bool(torch.equal(half[~bad],half_repeat[~bad])),
                  fp32_vs_float64_sample_max_abs_error=float((full[rows].double()-double).abs().max()),
                  finite_amp_vs_fp32_max_abs_error=float((half[~bad].float()-full[~bad]).abs().max()),
                  fp32_input_channel_mean_min=float(x.mean(0).min()),
                  fp32_input_channel_mean_max=float(x.mean(0).max()),
                  fp32_input_channel_std_max=float(x.std(0).max()))
    del half, half_repeat, full, quantized, bad, exceed, above_round_threshold
    target = json.loads((replay/"target_input.json").read_text())
    audit = target["sampling_audit"][0]
    cfg = json.loads((replay/"manifest.json").read_text())["config"]
    opt = dict(cfg["dataset_train"])
    opt.pop("data_root",None)
    dataset = PlannedDataset([audit["path"]], maximum_points=None, **opt)
    batch = dataset.collate_fn([dataset[(0,audit["augmentation_seed"])]])
    if sample_digest(batch) != target["digest"]:
        raise ValueError("Regenerated target input differs")
    labels, mask = batch["semantic_labels"].cuda(), batch["masks_sem"].cuda()
    native = MLP(64,2,norm_fn=functools.partial(nn.BatchNorm1d,eps=1e-4,momentum=.1),num_layers=2).cuda()
    native.load_state_dict({k[len("semantic_linear."):]:v for k,v in state.items() if k.startswith("semantic_linear.")})
    stable = copy.deepcopy(native)
    native.train(); stable.train()
    with torch.cuda.amp.autocast():
        logits_amp = native(x)
        loss_amp = 50*F.cross_entropy(logits_amp.float()[mask],labels[mask],reduction="sum")/int(mask.sum())
    x.requires_grad_()
    with torch.cuda.amp.autocast(enabled=False):
        logits_full = stable(x)
        loss_full = 50*F.cross_entropy(logits_full[mask],labels[mask],reduction="sum")/int(mask.sum())
    loss_full.backward()
    head = dict(amp_logits=stats(logits_amp), amp_loss=stats(loss_amp),
                fp32_logits=stats(logits_full), fp32_loss=stats(loss_full),
                fp32_loss_value=float(loss_full), fp32_input_gradient=stats(x.grad),
                fp32_parameter_gradients={k:stats(v.grad) for k,v in stable.named_parameters()},
                amp_bn_running_mean=stats(native[1].running_mean),
                amp_bn_running_var=stats(native[1].running_var),
                fp32_bn_running_mean=stats(stable[1].running_mean),
                fp32_bn_running_var=stats(stable[1].running_var),
                bn_updates_each=int(stable[1].num_batches_tracked)-int(state["semantic_linear.1.num_batches_tracked"]),
                semantic_supervised_points=int(mask.sum()))
    result = dict(status="complete", purpose="diagnostic counterfactual only; no formal configuration changes",
                  case_sha256=sha256(replay/"first_bad_operator.pt"),
                  pre_target_state_sha256=sha256(replay/"state_before_target.pth"),
                  target_digest=target["digest"], input_crop_sha256=sha256(audit["path"]),
                  linear=op, semantic_head=head,
                  scope="Same captured full input, same weights and pre-forward BN state. FP32 head backward only; no backbone backward or optimizer step.")
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
