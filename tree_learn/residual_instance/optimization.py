"""Finite optimizer updates, with bounded AMP retries on the same crop.

The paired experiment freezes all BatchNorm running statistics. Retrying must
not advance its crop index, learning-rate schedule, optimizer or model buffers.
"""

import json

import torch


def nonfinite_gradients(model):
    parameters = [(name, value.grad) for name, value in model.named_parameters()
                  if value.grad is not None]
    if not parameters:
        raise RuntimeError('Objective produced no parameter gradients.')
    finite = torch.stack([torch.isfinite(gradient).all() for _, gradient in parameters])
    if bool(finite.all()):
        return []
    return [dict(parameter=name, nan=int(torch.isnan(gradient).sum()),
                 inf=int(torch.isinf(gradient).sum()), elements=gradient.numel())
            for (name, gradient), valid in zip(parameters, finite.tolist()) if not valid]


def optimizer_update(model, optimizer, scaler, closure, gradient_clip,
                     fp16_retries=2, on_event=None, before_step=None):
    """Return loss statistics and update statistics after exactly ONE update.

    closure returns (loss, scalar_statistics). Bad AMP gradients are never
    clipped or used by the optimizer. GradScaler skips that attempt and lowers
    its scale; the same closure/crop is then retried with restored Torch RNG.
    After the retry budget, use an unscaled FP32 attempt. Invalid FP32 results
    still fail, with diagnostics, instead of being masked or counted as steps.
    """
    if type(fp16_retries) is not int or fp16_retries < 0:
        raise ValueError('fp16_retries must be a nonnegative integer.')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    use_amp = scaler.is_enabled()
    amp_failures = 0
    history = []
    for attempt in range(1, fp16_retries + 3 if use_amp else 2):
        mixed = use_amp and amp_failures <= fp16_retries
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
        scale = float(scaler.get_scale())
        failure = None
        values = {}
        loss = None
        try:
            with torch.cuda.amp.autocast(enabled=mixed):
                loss, values = closure()
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError('Non-finite objective.')
        except FloatingPointError as error:
            failure = dict(reason='nonfinite_objective', detail=str(error))
        if failure is None:
            if mixed:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            bad = nonfinite_gradients(model)
            if bad:
                failure = dict(reason='nonfinite_gradient', parameters=bad)
                if mixed:
                    # Public GradScaler API: its recorded found_inf guarantees
                    # the underlying optimizer.step is skipped on this attempt.
                    scaler.step(optimizer)
                    scaler.update()
        if failure is not None:
            event = dict(event='numerical_retry' if mixed else 'numerical_failure',
                         attempt=attempt, precision='fp16' if mixed else 'fp32',
                         loss_scale=scale, next_loss_scale=float(scaler.get_scale()),
                         values=values, **failure)
            history.append(event)
            if on_event is not None:
                on_event(event)
            optimizer.zero_grad(set_to_none=True)
            if not mixed:
                raise FloatingPointError('FP32 optimizer update rejected: ' + json.dumps(event))
            if failure['reason'] == 'nonfinite_objective':
                # Forward overflow cannot be fixed by lowering a backward scale.
                amp_failures = fp16_retries + 1
            else:
                amp_failures += 1
            del loss
            continue
        try:
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip, error_if_nonfinite=True)
        except RuntimeError as error:
            event = dict(event='numerical_failure', reason='nonfinite_gradient_norm',
                         attempt=attempt, precision='fp16' if mixed else 'fp32',
                         loss_scale=scale, detail=str(error), values=values)
            if on_event is not None:
                on_event(event)
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(json.dumps(event)) from error
        if before_step is not None:
            before_step()
        if mixed:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        update = dict(gradient_norm=float(gradient), loss_scale=float(scaler.get_scale()),
                      update_precision='fp16' if mixed else 'fp32',
                      update_attempts=attempt, amp_retries=len(history),
                      fp32_fallback=bool(use_amp and not mixed))
        if history and on_event is not None:
            on_event(dict(event='numerical_recovered', values=values, **update))
        return values, update
    raise RuntimeError('AMP retry loop exhausted without an FP32 attempt.')
