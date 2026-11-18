"""Profile one production training step for cross-backend PI0.5 alignment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from fluxvla.engines.runners.base_train_runner import BaseTrainRunner
from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner


OUTPUT_ROOT = Path(os.environ.get("FLUXVLA_ALIGNMENT_DIR", "work_dirs/mcore_alignment"))
CAPTURE_VALUES = os.environ.get("FLUXVLA_CAPTURE_VALUES", "0") == "1"
CAPTURE_STATS = os.environ.get("FLUXVLA_CAPTURE_STATS", "1") == "1"
VALUE_PATTERN = re.compile(
    os.environ.get(
        "FLUXVLA_CAPTURE_VALUE_PATTERN",
        r"^(batch|flow\.|embed_(prefix|suffix)|.*layers\.0\.(input_layernorm|mlp)(\.|$)|output\.)",
    )
)
MODULE_PATTERN = re.compile(
    os.environ.get(
        "FLUXVLA_CAPTURE_MODULE_PATTERN",
        r"^(llm_(backbone|expert)\.layers\.(0|8|17)\."
        r"(input_layernorm|self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"post_attention_layernorm|mlp(\.(gate_proj|up_proj|down_proj))?)|"
        r"action_out_proj)",
    )
)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _canonical_name(name: str) -> str:
    name = name.removeprefix("module.").removeprefix("model.")
    return name.replace(".layer.", ".")


class AlignmentCapture:
    """Collect ordered tensor metadata and optional values for one train step."""

    def __init__(self) -> None:
        self.rank = _rank()
        self.rank_dir = OUTPUT_ROOT / f"rank{self.rank}"
        self.tensor_dir = self.rank_dir / "tensors"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        if CAPTURE_VALUES:
            self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.handles = []
        self.active = False
        self.finalized = False
        self.profiler = None
        self._range_contexts: dict[int, list] = {}
        self._calls: dict[str, int] = {}

    @staticmethod
    def _tensors(value, prefix=""):
        if isinstance(value, torch.Tensor):
            yield prefix or "0", value
        elif isinstance(value, Mapping):
            for key, child in value.items():
                yield from AlignmentCapture._tensors(child, f"{prefix}.{key}".strip("."))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                yield from AlignmentCapture._tensors(child, f"{prefix}.{index}".strip("."))

    def record(self, name: str, value, stage: str) -> None:
        if not self.active:
            return
        canonical = _canonical_name(name)
        call_key = f"{stage}:{canonical}"
        call = self._calls.get(call_key, 0)
        self._calls[call_key] = call + 1
        for leaf, tensor in self._tensors(value):
            detached = tensor.detach()
            float_tensor = detached.float()
            event = {
                "sequence": len(self.events),
                "name": canonical,
                "stage": stage,
                "call": call,
                "leaf": leaf,
                "shape": list(detached.shape),
                "stride": list(detached.stride()),
                "dtype": str(detached.dtype),
                "device": str(detached.device),
                "numel": detached.numel(),
            }
            if CAPTURE_STATS and detached.numel():
                event.update(
                    finite=bool(torch.isfinite(float_tensor).all().item()),
                    min=float(float_tensor.min().item()),
                    max=float(float_tensor.max().item()),
                    mean=float(float_tensor.mean().item()),
                    std=float(float_tensor.std(unbiased=False).item()),
                    norm=float(float_tensor.norm().item()),
                )
            if CAPTURE_VALUES and VALUE_PATTERN.search(canonical):
                filename = f"{event['sequence']:05d}_{stage}_{canonical}_{leaf}.pt"
                filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
                cpu_tensor = detached.cpu().contiguous()
                torch.save(cpu_tensor, self.tensor_dir / filename)
                event["tensor_file"] = f"tensors/{filename}"
                event["sha256"] = hashlib.sha256(
                    cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            self.events.append(event)

    def _pre_hook(self, name: str, module: torch.nn.Module, args) -> None:
        if not self.active:
            return
        context = record_function(f"PI05/{_canonical_name(name)}")
        context.__enter__()
        self._range_contexts.setdefault(id(module), []).append(context)
        self.record(name, args, "forward_input")

    def _post_hook(self, name: str, module: torch.nn.Module, _args, output) -> None:
        if not self.active:
            return
        self.record(name, output, "forward_output")
        contexts = self._range_contexts.get(id(module), [])
        if contexts:
            contexts.pop().__exit__(None, None, None)

    def attach(self, wrapped_model: torch.nn.Module) -> None:
        model = wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model
        if hasattr(model, "model"):
            model = model.model
        for name, module in model.named_modules():
            if name and MODULE_PATTERN.search(name):
                self.handles.append(
                    module.register_forward_pre_hook(
                        lambda mod, args, module_name=name: self._pre_hook(module_name, mod, args)
                    )
                )
                self.handles.append(
                    module.register_forward_hook(
                        lambda mod, args, output, module_name=name: self._post_hook(
                            module_name, mod, args, output
                        )
                    )
                )

        for method_name in ("embed_prefix", "embed_suffix"):
            original = getattr(model, method_name)

            def wrapped(instance, *args, __name=method_name, __original=original, **kwargs):
                with record_function(f"PI05/{__name}"):
                    output = __original(*args, **kwargs)
                self.record(__name, output, "forward_output")
                return output

            setattr(model, method_name, MethodType(wrapped, model))

        for method_name, record_name in (("sample_noise", "flow.noise"), ("sample_time", "flow.time")):
            original = getattr(model, method_name)

            def wrapped(instance, *args, __record=record_name, __original=original, **kwargs):
                output = __original(*args, **kwargs)
                self.record(__record, output, "forward_output")
                return output

            setattr(model, method_name, MethodType(wrapped, model))

    def start_profiler(self) -> None:
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.rank_dir / "trace")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        self.profiler.start()

    def finish_step(self) -> None:
        if self.profiler is not None:
            self.profiler.step()
        self.active = False

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        if self.profiler is not None:
            self.profiler.stop()
            try:
                summary = self.profiler.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=200
                )
                (self.rank_dir / "operator_summary.txt").write_text(
                    summary, encoding="utf-8"
                )
                self.profiler.export_stacks(
                    str(self.rank_dir / "cpu_stacks.txt"), metric="self_cpu_time_total"
                )
            except Exception as error:  # PyTorch 2.12 can fail decoding stack source text.
                (self.rank_dir / "operator_summary_error.txt").write_text(
                    f"{type(error).__name__}: {error}\n", encoding="utf-8"
                )
        manifest = {
            "rank": self.rank,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "capture_values": CAPTURE_VALUES,
            "capture_stats": CAPTURE_STATS,
            "module_pattern": MODULE_PATTERN.pattern,
            "value_pattern": VALUE_PATTERN.pattern,
            "events": self.events,
        }
        (self.rank_dir / "tensor_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        for handle in self.handles:
            handle.remove()


capture: AlignmentCapture | None = None
original_setup = DDPTrainRunner.run_setup
original_prepare_batch = BaseTrainRunner._prepare_batch
original_training_step = BaseTrainRunner._training_step
original_run = BaseTrainRunner.run


def profiled_setup(self, *args, **kwargs):
    result = original_setup(self, *args, **kwargs)
    global capture
    capture = AlignmentCapture()
    capture.attach(self.vla)
    capture.start_profiler()
    return result


def captured_prepare_batch(self, *args, **kwargs):
    result = original_prepare_batch(self, *args, **kwargs)
    if capture is not None:
        capture.record("batch", result, "model_ready_input")
    return result


def profiled_training_step(self, batch, should_step=True):
    if capture is not None:
        capture.active = True
    with record_function("PI05/TrainingStep"):
        result = original_training_step(self, batch, should_step=should_step)
    if capture is not None:
        capture.record("output.loss", result, "loss")
        capture.finish_step()
    return result


def profiled_run(self, *args, **kwargs):
    try:
        return original_run(self, *args, **kwargs)
    finally:
        if capture is not None:
            capture.finalize()


DDPTrainRunner.run_setup = profiled_setup
BaseTrainRunner._prepare_batch = captured_prepare_batch
BaseTrainRunner._training_step = profiled_training_step
BaseTrainRunner.run = profiled_run

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
