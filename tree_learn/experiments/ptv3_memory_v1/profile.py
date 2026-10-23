"""Tensor-free event logging. Instrumentation overhead is measured separately."""
import contextlib
import json
import time
import traceback
import weakref
from contextvars import ContextVar

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_ACTIVE = ContextVar("ptv3_memory_audit", default=None)


def _memory():
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def memory():
    values = _memory()
    audit = _ACTIVE.get()
    if audit is not None:
        audit.observe(values)
        values.update(audit.peaks)
    return values


class OperatorMode(TorchDispatchMode):
    def __init__(self, audit):
        super().__init__()
        self.audit = audit

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        audit = self.audit
        name = str(func)
        before = audit.memory()
        # A local operator peak includes workspace already released on return.
        # Accumulate it in every enclosing scope and retain a separate run peak.
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        node = torch._C._current_autograd_node()
        backward_module = node.metadata.get("ptv3_memory_v1_module") if node else None
        try:
            result = func(*args, **(kwargs or {}))
            # Operator times include GPU completion, not merely CPU launch.
            torch.cuda.synchronize()
        except BaseException:
            audit.emit("operator_error", operator=name, before=before, after=audit.memory(),
                       autograd_node=node.name() if node else None, backward_module=backward_module,
                       seconds=time.perf_counter()-started, stack=traceback.format_exc())
            raise
        audit.last_operator = name
        audit.emit("operator", operator=name, autograd_node=node.name() if node else None,
                   backward_module=backward_module,
                   before=before, after=audit.memory(), seconds=time.perf_counter()-started)
        return result


class MemoryAudit:
    def __init__(self, folder, operators=False):
        self.stream = (folder/"events.jsonl").open("w", buffering=1)
        self.operators = operators
        self.phase = "setup"
        self.active = []
        self.last_operator = self.last_module = None
        self.handles, self.points = [], {}
        self.created_points = self.released_points = 0
        self.peaks = dict(peak_allocated=0, peak_reserved=0)
        self.scope_peaks = []

    def observe(self, values):
        for key in self.peaks:
            self.peaks[key] = max(self.peaks[key], values[key])
            for scope in self.scope_peaks:
                scope[key] = max(scope[key], values[key])

    def memory(self):
        # CUDA autograd worker threads do not inherit Python ContextVars.
        # Operator/hooks carry this explicit audit reference across backward.
        values = _memory()
        self.observe(values)
        return values | self.peaks

    def emit(self, event, **fields):
        self.stream.write(json.dumps(dict(event=event, phase=self.phase, active=list(self.active),
                                         time=time.perf_counter(), **fields), allow_nan=False)+"\n")

    @contextlib.contextmanager
    def scope(self, name):
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.active.append(name)
        start_memory = self.memory()
        torch.cuda.reset_peak_memory_stats()
        local_peak = dict(peak_allocated=start_memory["allocated"], peak_reserved=start_memory["reserved"])
        self.scope_peaks.append(local_peak)
        self.emit("module_enter", module=name, **self.memory())
        try:
            yield
        except BaseException:
            current = self.memory()
            self.emit("module_error", module=name, seconds=time.perf_counter()-started,
                      window_peak_allocated=local_peak["peak_allocated"],
                      window_peak_reserved=local_peak["peak_reserved"], **current)
            raise
        else:
            torch.cuda.synchronize()
            self.last_module = name
            current = self.memory()
            self.emit("module_exit", module=name, seconds=time.perf_counter()-started,
                      window_peak_allocated=local_peak["peak_allocated"],
                      window_peak_reserved=local_peak["peak_reserved"], **current)
        finally:
            self.active.pop()
            self.scope_peaks.pop()

    @contextlib.contextmanager
    def attach(self, model):
        from tree_learn.experiments.ptv3_treelearn.vendor.model import Point
        contexts, pending = {}, []
        token = _ACTIVE.set(self)
        for name, module in model.named_modules():
            def enter(mod, args, label=name):
                ctx = self.scope(label or "model")
                contexts.setdefault(id(mod), []).append(ctx)
                pending.append(ctx)
                ctx.__enter__()
            def leave(mod, args, output, label=name):
                # Tag nodes with strings, never retain their output tensors.
                # This identifies dense backward allocations outside module
                # forward scopes, including during reentrant recomputation.
                tensors = [output] if isinstance(output, torch.Tensor) else []
                if isinstance(output, dict) and isinstance(output.get("feat"), torch.Tensor):
                    tensors.append(output["feat"])
                for tensor in tensors:
                    if tensor.grad_fn is not None:
                        tensor.grad_fn.metadata.setdefault("ptv3_memory_v1_module", label)
                ctx = contexts[id(mod)].pop()
                pending.remove(ctx)
                ctx.__exit__(None, None, None)
            self.handles.extend([module.register_forward_pre_hook(enter), module.register_forward_hook(leave)])
        originals = {name: getattr(Point, name) for name in ("serialization", "sparsify", "__init__")}
        for name in ("serialization", "sparsify"):
            def wrapped(point, *args, _name=name, **kwargs):
                with self.scope("Point."+_name):
                    return originals[_name](point, *args, **kwargs)
            setattr(Point, name, wrapped)
        def init(point, *args, **kwargs):
            originals["__init__"](point, *args, **kwargs)
            identity = id(point)
            self.created_points += 1
            def released(ref):
                self.points.pop(identity, None)
                self.released_points += 1
            self.points[identity] = weakref.ref(point, released)
        Point.__init__ = init
        try:
            with OperatorMode(self) if self.operators else contextlib.nullcontext():
                yield
        finally:
            # Failed modules have no post-hook; unwind their profiler scopes.
            for ctx in reversed(pending):
                ctx.__exit__(RuntimeError, RuntimeError("aborted"), None)
            for handle in self.handles:
                handle.remove()
            for name, original in originals.items():
                setattr(Point, name, original)
            _ACTIVE.reset(token)

    def finish(self):
        self.emit("point_lifetimes", created=self.created_points, released=self.released_points,
                  alive=len(self.points), **self.memory())
        self.stream.close()
