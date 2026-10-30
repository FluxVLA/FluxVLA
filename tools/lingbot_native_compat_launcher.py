"""Run the read-only LingBot trainer with its supported LeRobot v3 API.

LingBot-VLA-v2 imports ``hf_transform_to_torch`` from its pre-refactor
location.  Current LeRobot keeps the function in ``datasets.io_utils``.
Install the compatibility alias before executing the native entrypoint so
parity tests do not need to modify the native repository.
"""

from __future__ import annotations

import runpy
import sys
import os
import inspect
import builtins
import importlib
import json
import re
import types
from contextlib import contextmanager
from pathlib import Path


def _install_transformers_compat() -> None:
    import transformers
    import transformers.cache_utils as cache_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen25vl
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl
    import transformers.utils.import_utils as import_utils

    # Write directly to the lazy module dictionary.  Attribute assignment may
    # be routed through its resolver and disappear before a later from-import.
    transformers.__dict__["AutoModelForVision2Seq"] = transformers.AutoModel
    # Transformers' _LazyModule resolves imports through __getattr__ and can
    # ignore a dynamically assigned compatibility alias.  Intercept only this
    # removed symbol so ``from transformers import AutoModelForVision2Seq``
    # remains reliable for the read-only native loader.
    lazy_type = type(transformers)
    if not getattr(lazy_type, "_fluxvla_vision2seq_compat", False):
        original_getattr = lazy_type.__getattr__
        vision_alias = transformers.AutoModel

        def lazy_getattr(self, name):
            if name == "AutoModelForVision2Seq":
                return vision_alias
            return original_getattr(self, name)

        lazy_type.__getattr__ = lazy_getattr
        lazy_type._fluxvla_vision2seq_compat = True
    if not hasattr(modeling_utils, "no_init_weights"):

        @contextmanager
        def no_init_weights(_enable: bool = True):
            yield

        modeling_utils.no_init_weights = no_init_weights
    if (
        not getattr(modeling_utils.PreTrainedModel, "_fluxvla_lingbot_tie_compat", False)
        and hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys")
    ):
        original_tie_keys = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys

        def get_expanded_tied_weights_keys_compat(self, all_submodels=False):
            if isinstance(getattr(self, "_tied_weights_keys", None), list):
                return {}
            return original_tie_keys(self, all_submodels=all_submodels)

        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = (
            get_expanded_tied_weights_keys_compat
        )
        modeling_utils.PreTrainedModel._fluxvla_lingbot_tie_compat = True
    if not hasattr(import_utils, "is_safetensors_available"):
        import_utils.is_safetensors_available = lambda: True
    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = cache_utils.DynamicCache
    if not hasattr(qwen25vl, "Qwen2RMSNorm"):
        qwen25vl.Qwen2RMSNorm = qwen25vl.Qwen2_5_VLRMSNorm
    qwen_cls = getattr(qwen3vl, "Qwen3VLForConditionalGeneration", None)
    if qwen_cls is not None and not hasattr(qwen_cls, "visual"):
        qwen_cls.visual = property(lambda self: self.model.visual)
    qwen_model_cls = getattr(qwen3vl, "Qwen3VLModel", None)
    if qwen_model_cls is not None and not getattr(qwen_model_cls, "_fluxvla_rope_compat", False):
        original_rope = qwen_model_cls.get_rope_index

        def get_rope_index_compat(self, *args, **kwargs):
            if "mm_token_type_ids" not in kwargs:
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                if input_ids is None:
                    raise TypeError("get_rope_index requires input_ids")
                mm = torch.zeros_like(input_ids, dtype=torch.int32)
                image_token_id = getattr(self.config, "image_token_id", None)
                if image_token_id is not None:
                    mm[input_ids == int(image_token_id)] = 1
                kwargs["mm_token_type_ids"] = mm
            return original_rope(self, *args, **kwargs)

        import torch

        qwen_model_cls.get_rope_index = get_rope_index_compat
        qwen_model_cls._fluxvla_rope_compat = True


def _install_fixed_flow_replay() -> None:
    """Inject deterministic flow inputs for native/adapter parity replays.

    The reference model samples ``noise`` and ``time`` inside ``forward``.
    Passing the values through a launcher keeps the read-only native checkout
    untouched while making the exact same replay batch usable by FluxVLA.
    This is opt-in because normal training must retain native sampling.
    """
    if os.environ.get("LINGBOT_FIXED_FLOW_REPLAY") != "1":
        return
    module_name = "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2"
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    patching = False

    def patch_loaded_module():
        loaded = sys.modules.get(module_name)
        if loaded is None or not hasattr(loaded, "FlowMatchingV2") or getattr(
            loaded.FlowMatchingV2, "_fluxvla_fixed_flow_replay", False
        ):
            return
        try:
            from transformers import AutoModel

            AutoModel.register(loaded.LingbotVLAV2Config, loaded.LingbotVlaV2Policy, exist_ok=True)
        except (AttributeError, ValueError, TypeError):
            pass
        cls = loaded.FlowMatchingV2
        original = cls.forward
        signature = inspect.signature(original)

        def forward_with_fixed_flow(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            actions = bound.arguments["actions"]
            if os.environ.get("LINGBOT_FIXED_NOISE", "zeros") != "zeros":
                raise ValueError("Only LINGBOT_FIXED_NOISE=zeros is supported")
            bound.arguments["noise"] = actions.new_zeros(actions.shape)
            bound.arguments["time"] = actions.new_full((actions.shape[0],), 0.5)
            return original(*bound.args, **bound.kwargs)

        cls.forward = forward_with_fixed_flow
        cls._fluxvla_fixed_flow_replay = True
        _install_native_batch_dump(loaded)
        if os.environ.get("LINGBOT_FIXED_FLOW_DEBUG") == "1":
            print("LINGBOT_FIXED_FLOW_REPLAY_INSTALLED", flush=True)

    def import_and_patch(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        loaded = sys.modules.get(module_name)
        if os.environ.get("LINGBOT_FIXED_FLOW_DEBUG") == "2" and "lingbotvla.models.vla.lingbot_vla" in name:
            print("FIXED_IMPORT", name, bool(loaded), bool(loaded and hasattr(loaded, "FlowMatchingV2")), flush=True)
        if loaded is not None and hasattr(loaded, "FlowMatchingV2") and not getattr(
            loaded.FlowMatchingV2, "_fluxvla_fixed_flow_replay", False
        ):
            # Transformers 5.x may fall back to AutoModel before the native
            # registry sees this dynamically imported VLA class. Register the
            # native pair locally so fixed replay does not change model choice.
            try:
                from transformers import AutoModel

                config_cls = loaded.LingbotVLAV2Config
                model_cls = loaded.LingbotVlaV2Policy
                AutoModel.register(config_cls, model_cls, exist_ok=True)
            except (AttributeError, ValueError, TypeError):
                pass
            cls = loaded.FlowMatchingV2
            original = cls.forward
            signature = inspect.signature(original)

            def forward_with_fixed_flow(self, *args, **kwargs):
                bound = signature.bind(self, *args, **kwargs)
                actions = bound.arguments["actions"]
                if os.environ.get("LINGBOT_FIXED_NOISE", "zeros") != "zeros":
                    raise ValueError("Only LINGBOT_FIXED_NOISE=zeros is supported")
                bound.arguments["noise"] = actions.new_zeros(actions.shape)
                bound.arguments["time"] = actions.new_full((actions.shape[0],), 0.5)
                return original(*bound.args, **bound.kwargs)

            cls.forward = forward_with_fixed_flow
            cls._fluxvla_fixed_flow_replay = True
            _install_native_batch_dump(loaded)
            if os.environ.get("LINGBOT_FIXED_FLOW_DEBUG") == "1":
                print("LINGBOT_FIXED_FLOW_REPLAY_INSTALLED", flush=True)
        return module

    def import_module_and_patch(name, package=None):
        nonlocal patching
        module = original_import_module(name, package)
        if patching:
            return module
        patching = True
        if os.environ.get("LINGBOT_FIXED_FLOW_DEBUG") == "2" and "lingbotvla.models.vla.lingbot_vla" in name:
            loaded = sys.modules.get(module_name)
            print("FIXED_IMPORTLIB", name, bool(loaded), bool(loaded and hasattr(loaded, "FlowMatchingV2")), flush=True)
        try:
            patch_loaded_module()
        finally:
            patching = False
        return module

    importlib.import_module = import_module_and_patch


def _install_native_batch_dump(loaded) -> None:
    """Optionally persist native model inputs and outputs for parity diagnostics."""
    output = os.environ.get("LINGBOT_DUMP_BATCH")
    output_values = os.environ.get("LINGBOT_DUMP_OUTPUT")
    activation_output = os.environ.get("LINGBOT_ACTIVATION_TRACE_DIR")
    policy = getattr(loaded, "LingbotVlaV2Policy", None)
    if (
        (not output and not output_values and not activation_output)
        or policy is None
        or getattr(policy, "_fluxvla_batch_dump", False)
    ):
        return
    original = policy.forward
    call_index = 0

    def activation_hooks(instance):
        trace_root = os.environ.get("LINGBOT_ACTIVATION_TRACE_DIR")
        if not trace_root:
            return [], {}
        selected = {
            "model.state_proj",
            "model.action_in_proj",
            "model.action_out_proj",
        }
        selected.update(
            f"model.qwenvl_with_expert.qwen_expert.model.layers.{index}"
            for index in range(36)
        )
        layer0 = "model.qwenvl_with_expert.qwen_expert.model.layers.0"
        selected.update(
            {
                f"{layer0}.input_layernorm",
                f"{layer0}.self_attn.q_proj",
                f"{layer0}.self_attn.k_proj",
                f"{layer0}.self_attn.v_proj",
                f"{layer0}.self_attn.o_proj",
                f"{layer0}.post_attention_layernorm",
                f"{layer0}.mlp",
            }
        )
        trace = {}
        handles = []

        def first_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (tuple, list)):
                return next(
                    (item for item in value if isinstance(item, torch.Tensor)),
                    None,
                )
            return None

        for name, module in instance.named_modules():
            if name not in selected:
                continue

            def save(_module, inputs, result, *, module_name=name):
                output_tensor = first_tensor(result)
                for index, input_tensor in enumerate(inputs):
                    if isinstance(input_tensor, torch.Tensor):
                        trace[f"{module_name}.input.{index}"] = (
                            input_tensor.detach().float().cpu()
                        )
                if output_tensor is not None:
                    trace[f"{module_name}.output"] = output_tensor.detach().float().cpu()

            handles.append(module.register_forward_hook(save))
        parameter_names = {
            "model.action_time_mlp_in.weight",
            "model.action_time_mlp_in.bias",
            "model.action_time_mlp_out.weight",
            "model.action_time_mlp_out.bias",
        }
        for name, parameter in instance.named_parameters():
            if name in parameter_names:
                trace[f"parameter.{name}"] = parameter.detach().float().cpu()
        return handles, trace

    def rank_path(pattern, step):
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        path = pattern.format(rank=rank, step=step)
        if path == pattern and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            stem, suffix = os.path.splitext(pattern)
            path = f"{stem}.rank{rank}{suffix}"
        return path

    def compact_output(result):
        names = (
            "loss",
            "loss_vla",
            "loss_depth",
            "loss_future_depth",
            "loss_future_video",
            "loss_seq_wise",
        )
        output_row = {
            name: value.detach().cpu()
            for name, value in zip(names, result[: len(names)])
            if hasattr(value, "detach")
        }
        loss_log = result[6] if len(result) > 6 else None
        if isinstance(loss_log, dict):
            output_row["loss_log"] = {
                key: value.detach().cpu()
                for key, value in loss_log.items()
                if hasattr(value, "detach") and value.numel() == 1
            }
        return output_row

    def forward_with_dump(self, *args, **kwargs):
        nonlocal call_index
        call_index += 1
        dump_all_batches = output and "{step" in output
        dump_all_outputs = output_values and "{step" in output_values
        if output and (call_index == 1 or dump_all_batches):
            names = (
                "images", "img_masks", "state", "lang_tokens", "lang_masks",
                "actions", "joint_mask", "action_is_pad", "noise", "time",
                "image_grid_thw",
            )
            values = dict(zip(names, args))
            values.update(kwargs)
            torch.save(
                {key: value.detach().cpu() for key, value in values.items()
                 if hasattr(value, "detach")},
                rank_path(output, call_index),
            )
        handles, activation_trace = activation_hooks(self)
        try:
            result = original(self, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        trace_root = os.environ.get("LINGBOT_ACTIVATION_TRACE_DIR")
        if trace_root and activation_trace and call_index == 1:
            trace_path = Path(trace_root)
            trace_path.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            torch.save(activation_trace, trace_path / f"native_rank{rank}.pt")
            if os.environ.get("LINGBOT_TRACE_FORWARD_ONLY") == "1":
                raise RuntimeError("LingBot native forward-only trace complete")
        if output_values and (call_index == 1 or dump_all_outputs):
            value = compact_output(result) if dump_all_outputs else [
                item.detach().cpu() if hasattr(item, "detach") else item
                for item in result
            ]
            torch.save(value, rank_path(output_values, call_index))
        return result

    import torch

    policy.forward = forward_with_dump
    policy._fluxvla_batch_dump = True


def _install_checkpoint_save_skip() -> None:
    """Skip large native checkpoints in opt-in numerical certification runs."""
    if os.environ.get("LINGBOT_SKIP_CHECKPOINT_SAVE") != "1":
        return
    checkpoint_module = importlib.import_module("lingbotvla.checkpoint")
    original_builder = checkpoint_module.build_checkpointer

    def build_checkpointer_without_save(*args, **kwargs):
        checkpointer = original_builder(*args, **kwargs)

        class CheckpointerProxy:
            @staticmethod
            def load(*load_args, **load_kwargs):
                return checkpointer.load(*load_args, **load_kwargs)

            @staticmethod
            def save(*_args, **_kwargs):
                return None

        return CheckpointerProxy

    checkpoint_module.build_checkpointer = build_checkpointer_without_save


def _install_native_optimizer_trace() -> None:
    """Trace native AdamW updates without changing the LingBot worktree.

    FSDP2 exposes DTensor parameters to the optimizer.  ``full_tensor`` is a
    collective, so every rank materializes each selected tensor before rank 0
    writes it.  Keep the regex narrow when tracing large models.
    """
    trace_dir_value = os.environ.get("LINGBOT_OPTIMIZER_TRACE_DIR")
    if not trace_dir_value:
        return

    import torch
    import torch.distributed as dist
    import lingbotvla.optim as optim_module

    trace_dir = Path(trace_dir_value)
    pattern = os.environ.get(
        "LINGBOT_OPTIMIZER_TRACE_PATTERN",
        r"action_model\.state_proj\.(weight|bias)$",
    )
    matcher = re.compile(pattern)
    original_builder = optim_module.build_optimizer

    def canonical_name(name: str) -> str:
        return name.removeprefix("module.").replace("._fsdp_wrapped_module.", ".")

    def full_cpu(value):
        if value is None:
            return None
        detached = value.detach()
        if hasattr(detached, "full_tensor"):
            detached = detached.full_tensor()
        return detached.float().cpu().clone()

    def build_optimizer_with_trace(model, *args, **kwargs):
        optimizer = original_builder(model, *args, **kwargs)
        names_by_id = {
            id(parameter): canonical_name(name)
            for name, parameter in model.named_parameters()
        }
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            trace_dir.mkdir(parents=True, exist_ok=True)
            groups = []
            for group_index, group in enumerate(optimizer.param_groups):
                names = [names_by_id.get(id(parameter), "<unnamed>") for parameter in group["params"]]
                groups.append(
                    {
                        "group": group_index,
                        "lr": float(group["lr"]),
                        "weight_decay": float(group["weight_decay"]),
                        "betas": list(group["betas"]),
                        "eps": float(group["eps"]),
                        "foreach": group.get("foreach"),
                        "fused": group.get("fused"),
                        "parameter_count": len(names),
                        "parameter_names": names,
                    }
                )
            with (trace_dir / "optimizer_groups.json").open("w", encoding="utf-8") as stream:
                json.dump(groups, stream, indent=2, sort_keys=True)

        original_step = optimizer.step
        step_index = 0

        def traced_step(_optimizer, *step_args, **step_kwargs):
            nonlocal step_index
            step_index += 1
            snapshots = []
            for group_index, group in enumerate(optimizer.param_groups):
                for parameter in group["params"]:
                    name = names_by_id.get(id(parameter), "<unnamed>")
                    if not matcher.search(name):
                        continue
                    snapshots.append(
                        {
                            "name": name,
                            "group": group_index,
                            "lr": float(group["lr"]),
                            "parameter_before": full_cpu(parameter),
                            "post_clip_grad": full_cpu(parameter.grad),
                        }
                    )
            result = original_step(*step_args, **step_kwargs)
            for row in snapshots:
                parameter = next(
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if names_by_id.get(id(parameter), "<unnamed>") == row["name"]
                )
                state = optimizer.state.get(parameter, {})
                row.update(
                    {
                        "parameter_after": full_cpu(parameter),
                        "exp_avg": full_cpu(state.get("exp_avg")),
                        "exp_avg_sq": full_cpu(state.get("exp_avg_sq")),
                    }
                )
            if rank == 0:
                trace_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"backend": "native", "step": step_index, "parameters": snapshots},
                    trace_dir / f"native_step_{step_index:06d}.pt",
                )
            return result

        optimizer.step = types.MethodType(traced_step, optimizer)
        return optimizer

    optim_module.build_optimizer = build_optimizer_with_trace


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: lingbot_native_compat_launcher.py ENTRYPOINT [ARGS...]")

    # Do not import FluxVLA here: its package initialization owns a separate
    # distributed runtime and would initialize the default process group.
    _install_transformers_compat()

    from lerobot.datasets import io_utils, utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not hasattr(utils, "hf_transform_to_torch"):
        utils.hf_transform_to_torch = io_utils.hf_transform_to_torch

    # LingBot's reader subclass still uses the pre-refactor dataset members.
    # LeRobot v3 moved them behind ``dataset.reader`` without changing their
    # behavior, so expose forwarding properties/methods for the native run.
    reader_members = (
        "delta_indices",
        "_absolute_to_relative_idx",
        "_get_query_indices",
        "_get_query_timestamps",
    )
    for name in reader_members:
        if not hasattr(LeRobotDataset, name):
            setattr(
                LeRobotDataset,
                name,
                property(lambda self, member=name: getattr(self.reader, member)),
            )
    if not hasattr(LeRobotDataset, "video_backend"):
        LeRobotDataset.video_backend = property(lambda self: self._video_backend)

    _install_fixed_flow_replay()
    _install_checkpoint_save_skip()
    _install_native_optimizer_trace()

    entrypoint = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(entrypoint, run_name="__main__")


if __name__ == "__main__":
    main()
