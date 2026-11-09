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
import torch
from contextlib import contextmanager
from pathlib import Path


def _expand_fused_checkpoint_item(key: str, tensor: torch.Tensor):
    """Yield eager expert entries for a native fused MoE checkpoint tensor.

    The native checkpoint stores each projection as ``[expert, out, in]``
    under ``experts.{projection}``, while the eager module exposes one Linear
    per expert.  Keeping this conversion as a streaming iterator avoids
    materializing a second full checkpoint in host memory.
    """
    markers = ("gate_proj", "up_proj", "down_proj")
    if tensor.ndim != 3 or ".experts." not in key:
        yield key, tensor
        return
    prefix, projection = key.rsplit(".experts.", 1)
    if projection not in markers:
        yield key, tensor
        return
    for expert_index in range(tensor.shape[0]):
        yield (
            f"{prefix}.experts.{expert_index}.{projection}.weight",
            tensor[expert_index].contiguous(),
        )


def _install_eager_checkpoint_compat() -> None:
    """Patch the read-only native iterator for fused-to-eager loading.

    This is opt-in because fused training remains the production default.
    The patch affects only the compatibility launcher process and can be
    enabled with ``LINGBOT_EAGER_CHECKPOINT_CONVERSION=1``.
    """
    if os.environ.get("LINGBOT_EAGER_CHECKPOINT_CONVERSION") != "1":
        return
    from lingbotvla.models import module_utils

    iterator_cls = module_utils.StateDictIterator
    if getattr(iterator_cls, "_fluxvla_eager_conversion", False):
        return
    original_iter = iterator_cls.__iter__

    def iter_with_eager_conversion(self):
        for key, tensor in original_iter(self):
            yield from _expand_fused_checkpoint_item(key, tensor)

    iterator_cls.__iter__ = iter_with_eager_conversion
    iterator_cls._fluxvla_eager_conversion = True
    print("LINGBOT_EAGER_CHECKPOINT_CONVERSION_INSTALLED", flush=True)


def _install_optional_alignment_checkpoint_compat() -> None:
    """Ignore optional teacher-alignment tensors in a VLA-only smoke run."""
    if os.environ.get("LINGBOT_IGNORE_OPTIONAL_ALIGNMENT") != "1":
        return
    from lingbotvla.models.loader import LingBotVLAWeightLoader

    cls = LingBotVLAWeightLoader
    if getattr(cls, "_fluxvla_optional_alignment", False):
        return
    original_map = cls.map_ckpt_key
    optional_markers = (
        "current_video_align_",
        "depth_align_",
        "future_depth_align_",
        "future_video_align_",
        "current_shared_task_proj",
        "future_shared_task_proj",
    )

    def map_optional(self, key, load_vlm_only, post_training):
        if any(marker in key for marker in optional_markers):
            return None
        return original_map(self, key, load_vlm_only, post_training)

    cls.map_ckpt_key = map_optional
    cls._fluxvla_optional_alignment = True
    print("LINGBOT_IGNORE_OPTIONAL_ALIGNMENT_INSTALLED", flush=True)


def _install_null_vlm_repo_compat() -> None:
    """Treat serialized YAML null sentinels as ``None`` in the read-only loader.

    The native mmengine YAML export can serialize a missing ``vlm_repo_id`` as
    ``''`` or ``'null'``.  The loader uses ``is not None`` to select its
    VLM-only checkpoint path, so either string silently rewrites every full
    checkpoint key with an extra ``qwenvl`` prefix.  Normalize only in this
    compatibility process; production source remains untouched.
    """
    from lingbotvla.models.loader import CustomizedModelingLoader

    cls = CustomizedModelingLoader
    if getattr(cls, "_fluxvla_null_vlm_repo", False):
        return
    original_load_model = cls.load_model

    def load_model(self, init_kwargs, **kwargs):
        if kwargs.get("vlm_repo_id") in ("", "null", "None"):
            kwargs["vlm_repo_id"] = None
        return original_load_model(self, init_kwargs, **kwargs)

    cls.load_model = load_model
    cls._fluxvla_null_vlm_repo = True
    print("LINGBOT_NULL_VLM_REPO_COMPAT_INSTALLED", flush=True)


def _install_null_cli_compat() -> None:
    """Restore resolved YAML values that the native CLI cannot round-trip.

    Native ``parse_args`` serializes YAML ``null`` with ``json.dumps`` before
    handing it to argparse.  Optional string fields consequently become the
    literal string ``"null"``, which is observably different from their
    dataclass default of ``None``.  It also expands an empty list to only the
    option name even though list arguments require at least one value.  The two
    affected fields have empty-list dataclass defaults, so omit only those
    exact resolved values while parsing and restore normal YAML handling
    immediately afterwards.
    """
    import lingbotvla.utils.arguments as arguments_module
    import yaml as yaml_module

    if getattr(arguments_module, "_fluxvla_null_cli_compat", False):
        return
    original_parse_args = arguments_module.parse_args

    def parse_args_with_nulls(rootclass):
        original_safe_load = yaml_module.safe_load

        def safe_load_resolved_config(stream):
            data = original_safe_load(stream)
            if isinstance(data, dict):
                for group_name, field_name in (
                    ("model", "basic_modules"),
                    ("train", "muon_exclude_name_patterns"),
                ):
                    group = data.get(group_name)
                    if isinstance(group, dict) and group.get(field_name) == []:
                        group.pop(field_name)
            return data

        yaml_module.safe_load = safe_load_resolved_config
        try:
            parsed = original_parse_args(rootclass)
        finally:
            yaml_module.safe_load = original_safe_load
        for group in vars(parsed).values():
            if not hasattr(group, "__dict__"):
                continue
            for name, value in vars(group).items():
                if isinstance(value, str) and value in ("null", "None"):
                    setattr(group, name, None)
        return parsed

    arguments_module.parse_args = parse_args_with_nulls
    arguments_module._fluxvla_null_cli_compat = True
    print("LINGBOT_NULL_CLI_COMPAT_INSTALLED", flush=True)


def _install_lerobot_reader_compat(dataset_cls: type) -> None:
    """Expose reader-owned APIs expected by LingBot's LeRobot subclass.

    LeRobot 0.6 moved query state and helpers to ``DatasetReader`` while the
    LingBot dataset subclass still consumes the older dataset-level API.  Add
    only missing members so this launcher also remains valid with LeRobot
    releases that expose them natively.
    """

    def reader_property(reader_name: str):
        def get_value(instance):
            reader = instance.__dict__.get("reader")
            if reader is not None and hasattr(reader, reader_name):
                return getattr(reader, reader_name)
            return instance.__dict__.get(reader_name)

        def set_value(instance, value):
            reader = instance.__dict__.get("reader")
            if reader is not None and hasattr(reader, reader_name):
                setattr(reader, reader_name, value)
            else:
                instance.__dict__[reader_name] = value

        return property(get_value, set_value)

    def reader_method(reader_name: str):
        def call_reader(instance, *args, **kwargs):
            reader = instance.__dict__.get("reader")
            if reader is None:
                raise RuntimeError(f"LeRobot dataset reader is unavailable for {reader_name}")
            return getattr(reader, reader_name)(*args, **kwargs)

        return call_reader

    properties = {
        "delta_indices": "delta_indices",
        "_absolute_to_relative_idx": "_absolute_to_relative_idx",
        "video_backend": "_video_backend",
    }
    methods = ("_get_query_indices", "_get_query_timestamps")

    for dataset_name, reader_name in properties.items():
        if inspect.getattr_static(dataset_cls, dataset_name, None) is None:
            setattr(dataset_cls, dataset_name, reader_property(reader_name))
    for method_name in methods:
        if inspect.getattr_static(dataset_cls, method_name, None) is None:
            setattr(dataset_cls, method_name, reader_method(method_name))


def _install_transformers_compat() -> None:
    import transformers
    import transformers.cache_utils as cache_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen25vl
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl
    import transformers.utils.import_utils as import_utils
    import transformers.modeling_flash_attention_utils as flash_utils
    import transformers.modeling_utils as modeling_utils_module

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
    if os.environ.get("LINGBOT_DISABLE_FLASH_ATTN") == "1":
        # The container can have a flash-attn wheel built for a different
        # PyTorch ABI.  Native model modules consult these functions during
        # import, before the configured eager attention path is selected.
        import_utils.is_flash_attn_2_available = lambda: False
        import_utils.is_flash_attn_3_available = lambda: False
        flash_utils.is_flash_attn_2_available = lambda: False
        flash_utils.is_flash_attn_3_available = lambda: False
        flash_utils.is_flash_attn_available = lambda: False
        base_model = modeling_utils_module.PreTrainedModel
        if not getattr(base_model, "_fluxvla_eager_attn_check", False):
            original_attn_check = base_model._check_and_adjust_attn_implementation

            def check_eager(self, attn_implementation, is_init_check=False, *args, **kwargs):
                if attn_implementation in ("flash_attention_2", "flash_attention_3"):
                    attn_implementation = "eager"
                return original_attn_check(
                    self,
                    attn_implementation,
                    is_init_check,
                    *args,
                    **kwargs,
                )

            base_model._check_and_adjust_attn_implementation = check_eager
            base_model._fluxvla_eager_attn_check = True
        # LingBot's read-only policy currently hard-codes flash_attention_2
        # on the Qwen3 config before constructing the HF submodel. Rewrite the
        # config at that boundary so eager attention is actually selected.
        for class_name in ("Qwen3VLForConditionalGeneration", "Qwen3VLModel"):
            cls = getattr(qwen3vl, class_name, None)
            if cls is None or getattr(cls, "_fluxvla_eager_attn", False):
                continue
            original_init = cls.__init__

            def init_eager(self, config, *args, _original=original_init, **kwargs):
                for value in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
                    if value is not None and hasattr(value, "_attn_implementation"):
                        value._attn_implementation = "eager"
                return _original(self, config, *args, **kwargs)

            cls.__init__ = init_eager
            cls._fluxvla_eager_attn = True
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

        import inspect
        rope_accepts_mm = "mm_token_type_ids" in inspect.signature(original_rope).parameters

        def get_rope_index_compat(self, *args, **kwargs):
            if rope_accepts_mm and "mm_token_type_ids" not in kwargs:
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                if input_ids is None:
                    raise TypeError("get_rope_index requires input_ids")
                mm = torch.zeros_like(input_ids, dtype=torch.int32)
                image_token_id = getattr(self.config, "image_token_id", None)
                if image_token_id is not None:
                    mm[input_ids == int(image_token_id)] = 1
                kwargs["mm_token_type_ids"] = mm
            elif not rope_accepts_mm:
                # New Qwen3-VL derives multimodal positions directly from the
                # token/grid inputs and rejects the legacy keyword.
                kwargs.pop("mm_token_type_ids", None)
            return original_rope(self, *args, **kwargs)

        import torch

        qwen_model_cls.get_rope_index = get_rope_index_compat
        qwen_model_cls._fluxvla_rope_compat = True


def _resolve_native_flow_inputs(
    *,
    actions: torch.Tensor,
    sample_time,
    call_index: int,
    capture_flow: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one native flow sample with replay files taking precedence."""
    flow_input = os.environ.get("LINGBOT_FLOW_INPUT")
    if flow_input:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        flow = torch.load(
            flow_input.format(rank=rank, step=call_index),
            map_location="cpu",
            weights_only=False,
        )
        return (
            flow["noise"].to(actions.device, actions.dtype),
            flow["time"].to(actions.device, actions.dtype),
        )
    if capture_flow:
        # Match native stochastic order: randn(actions.shape), then sample_time().
        return (
            torch.randn(actions.shape, device=actions.device, dtype=actions.dtype),
            sample_time(actions.size(0), actions.device).to(actions.dtype),
        )
    if os.environ.get("LINGBOT_FIXED_NOISE", "zeros") != "zeros":
        raise ValueError("Only LINGBOT_FIXED_NOISE=zeros is supported")
    return (
        actions.new_zeros(actions.shape),
        actions.new_full((actions.shape[0],), 0.5),
    )


def _install_fixed_flow_replay() -> None:
    """Inject deterministic flow inputs for native/adapter parity replays.

    The reference model samples ``noise`` and ``time`` inside ``forward``.
    Passing the values through a launcher keeps the read-only native checkout
    untouched while making the exact same replay batch usable by FluxVLA.
    This is opt-in because normal training must retain native sampling.
    """
    fixed_replay = os.environ.get("LINGBOT_FIXED_FLOW_REPLAY") == "1"
    capture_flow = os.environ.get("LINGBOT_CAPTURE_FLOW") == "1"
    if not fixed_replay and not capture_flow:
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
            call_index = _FLOW_CALL_INDEX[0]
            noise, time = _resolve_native_flow_inputs(
                actions=actions,
                sample_time=self.sample_time,
                call_index=call_index,
                capture_flow=capture_flow,
            )
            bound.arguments["noise"] = noise
            bound.arguments["time"] = time
            dump_flow = os.environ.get("LINGBOT_DUMP_FLOW")
            if dump_flow:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                torch.save(
                    {"noise": bound.arguments["noise"].detach().cpu(), "time": bound.arguments["time"].detach().cpu()},
                    dump_flow.format(rank=rank, step=call_index),
                )
            _FLOW_CALL_INDEX[0] += 1
            return original(*bound.args, **bound.kwargs)

        _FLOW_CALL_INDEX = [0]
        cls.forward = forward_with_fixed_flow
        cls._fluxvla_fixed_flow_replay = True
        _install_native_batch_dump(loaded)
        if os.environ.get("LINGBOT_FIXED_FLOW_DEBUG") == "1":
            print("LINGBOT_FIXED_FLOW_REPLAY_INSTALLED", flush=True)

    def import_and_patch(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal patching
        if patching:
            return original_import(name, globals, locals, fromlist, level)
        patching = True
        try:
            module = original_import(name, globals, locals, fromlist, level)
        finally:
            patching = False
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
                call_index = _FLOW_CALL_INDEX[0]
                noise, time = _resolve_native_flow_inputs(
                    actions=actions,
                    sample_time=self.sample_time,
                    call_index=call_index,
                    capture_flow=capture_flow,
                )
                bound.arguments["noise"] = noise
                bound.arguments["time"] = time
                dump_flow = os.environ.get("LINGBOT_DUMP_FLOW")
                if dump_flow:
                    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                    flow_path = dump_flow.format(rank=rank, step=call_index)
                    torch.save(
                        {
                            "noise": bound.arguments["noise"].detach().cpu(),
                            "time": bound.arguments["time"].detach().cpu(),
                        },
                        flow_path,
                    )
                _FLOW_CALL_INDEX[0] += 1
                return original(*bound.args, **bound.kwargs)

            _FLOW_CALL_INDEX = [0]
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
    builtins.__import__ = import_and_patch


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
            return [], {}, []
        selected = {
            "model.state_proj",
            "model.action_in_proj",
            "model.action_out_proj",
        }
        selected.update(
            f"model.qwenvl_with_expert.qwen_expert.model.layers.{index}"
            for index in range(36)
        )
        trace_layer = int(os.environ.get("LINGBOT_ACTIVATION_TRACE_LAYER", "0"))
        if not 0 <= trace_layer < 36:
            raise ValueError(
                "LINGBOT_ACTIVATION_TRACE_LAYER must be in [0, 35], "
                f"got {trace_layer}"
            )
        traced_layer = (
            "model.qwenvl_with_expert.qwen_expert.model.layers."
            f"{trace_layer}"
        )
        selected.update(
            {
                f"{traced_layer}.input_layernorm",
                f"{traced_layer}.input_layernorm.gamma",
                f"{traced_layer}.input_layernorm.beta",
                f"{traced_layer}.self_attn.q_proj",
                f"{traced_layer}.self_attn.k_proj",
                f"{traced_layer}.self_attn.v_proj",
                f"{traced_layer}.self_attn.o_proj",
                f"{traced_layer}.post_attention_layernorm",
                f"{traced_layer}.mlp",
            }
        )
        vlm_layer0 = "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0"
        selected.update(
            f"{vlm_layer0}.self_attn.{projection}"
            for projection in ("q_proj", "k_proj", "v_proj")
        )
        if os.environ.get("LINGBOT_TRACE_MOE") == "1":
            selected.update(
                {
                    f"{traced_layer}.mlp.gate",
                    f"{traced_layer}.mlp.experts",
                    f"{traced_layer}.mlp.shared_expert",
                    f"{traced_layer}.mlp.shared_expert_gate",
                }
            )
        trace = {"__metadata__": {}}
        handles = []
        grad_targets = []

        def record(key, value):
            detached = value.detach()
            trace[key] = detached.float().cpu()
            placements = getattr(detached, "placements", None)
            trace["__metadata__"][key] = {
                "dtype": str(detached.dtype),
                "shape": tuple(detached.shape),
                "stride": tuple(detached.stride()),
                "contiguous": detached.is_contiguous(),
                "placements": tuple(str(item) for item in placements) if placements else (),
            }

        if os.environ.get("LINGBOT_TRACE_ATTENTION") == "1":
            # RoPE and attention dispatch are owned by the native
            # QwenvlWithExpertV2Model wrapper, not its nested Qwen2 model.
            expert = instance.model.qwenvl_with_expert
            original_mrope = expert.apply_mrope
            original_attention = expert.attention_interface
            core_module = sys.modules.get(type(expert).__module__)
            original_flex_attention = getattr(core_module, "flex_attention_with_block_mask", None)

            def trace_mrope(query, key, position_ids):
                if "attention.query_pre_rope" in trace:
                    return original_mrope(query, key, position_ids)
                trace["attention.position_ids"] = position_ids.detach().float().cpu()
                trace["attention.query_pre_rope"] = query.detach().float().cpu()
                trace["attention.key_pre_rope"] = key.detach().float().cpu()
                query_out, key_out = original_mrope(query, key, position_ids)
                trace["attention.query_post_rope"] = query_out.detach().float().cpu()
                trace["attention.key_post_rope"] = key_out.detach().float().cpu()
                return query_out, key_out

            def trace_attention(query, key, value, mask):
                if "attention.output" in trace:
                    return original_attention(query, key, value, mask)
                trace["attention.value"] = value.detach().float().cpu()
                if mask is not None:
                    trace["attention.mask"] = mask.detach().float().cpu()
                output = original_attention(query, key, value, mask)
                trace["attention.output"] = output.detach().float().cpu()
                return output

            expert.apply_mrope = trace_mrope
            expert.attention_interface = trace_attention
            if original_flex_attention is not None:
                def trace_flex_attention(query, key, value, block_mask, query_len):
                    if "attention.output" in trace:
                        return original_flex_attention(query, key, value, block_mask, query_len)
                    trace["attention.value"] = value.detach().float().cpu()
                    output = original_flex_attention(query, key, value, block_mask, query_len)
                    trace["attention.output"] = output.detach().float().cpu()
                    return output

                core_module.flex_attention_with_block_mask = trace_flex_attention

            def restore_attention():
                expert.apply_mrope = original_mrope
                expert.attention_interface = original_attention
                if original_flex_attention is not None:
                    core_module.flex_attention_with_block_mask = original_flex_attention

            handles.append(types.SimpleNamespace(remove=restore_attention))

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
                        record(f"{module_name}.input.{index}", input_tensor)
                if output_tensor is not None:
                    record(f"{module_name}.output", output_tensor)
                    if (
                        os.environ.get("LINGBOT_ACTIVATION_GRAD_TRACE") == "1"
                        and output_tensor.requires_grad
                    ):
                        grad_targets.append((f"{module_name}.output", output_tensor))
                    if os.environ.get("LINGBOT_TRACE_MOE") == "1" and module_name.endswith("mlp.gate"):
                        block = instance.get_submodule(module_name[:-len(".gate")])
                        logits = output_tensor.reshape(-1, output_tensor.shape[-1]).float()
                        scores = (
                            logits.sigmoid()
                            if getattr(block, "_router_activation", "softmax") == "sigmoid"
                            else torch.softmax(logits, dim=-1)
                        )
                        bias = getattr(block, "e_score_correction_bias", None)
                        choice = scores + bias.detach().float().unsqueeze(0) if bias is not None else scores
                        _, selected = torch.topk(choice, block.top_k, dim=-1)
                        weights = scores.gather(1, selected)
                        if getattr(block, "norm_topk_prob", False):
                            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
                        scale = getattr(block, "routed_scaling_factor", 1.0)
                        if scale != 1.0:
                            weights = weights * scale
                        prefix = module_name[:-len(".gate")]
                        trace[f"{prefix}.router_logits"] = logits.detach().cpu()
                        trace[f"{prefix}.selected_experts"] = selected.detach().cpu()
                        trace[f"{prefix}.routing_weights"] = weights.detach().cpu()

            handles.append(module.register_forward_hook(save))

        if os.environ.get("LINGBOT_TRACE_MOE") == "1":
            for name, module in instance.named_modules():
                if not name.endswith("mlp.gate"):
                    continue
                parent_name = name[:-len(".gate")]
                parent = instance.get_submodule(parent_name)

                def trace_router(_module, inputs, *, block=parent, prefix=parent_name):
                    if not inputs:
                        return
                    hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
                    with torch.amp.autocast(hidden.device.type, enabled=False):
                        logits = torch.nn.functional.linear(
                            hidden.float(), block.gate.weight.float()
                        )
                    scores = (
                        logits.sigmoid()
                        if getattr(block, "_router_activation", "softmax") == "sigmoid"
                        else torch.softmax(logits, dim=-1)
                    )
                    bias = getattr(block, "e_score_correction_bias", None)
                    choice = scores + bias.unsqueeze(0) if bias is not None else scores
                    _, selected = torch.topk(choice, block.top_k, dim=-1)
                    weights = scores.gather(1, selected)
                    if getattr(block, "norm_topk_prob", False):
                        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
                    scale = getattr(block, "routed_scaling_factor", 1.0)
                    if scale != 1.0:
                        weights = weights * scale
                    trace[f"{prefix}.router_logits"] = logits.detach().float().cpu()
                    trace[f"{prefix}.selected_experts"] = selected.detach().cpu()
                    trace[f"{prefix}.routing_weights"] = weights.detach().float().cpu()

                handles.append(module.register_forward_pre_hook(trace_router))
        parameter_names = {
            "model.action_time_mlp_in.weight",
            "model.action_time_mlp_in.bias",
            "model.action_time_mlp_out.weight",
            "model.action_time_mlp_out.bias",
            f"{traced_layer}.input_layernorm.weight",
            f"{traced_layer}.post_attention_layernorm.weight",
            f"{traced_layer}.self_attn.q_proj.weight",
        }
        if os.environ.get("LINGBOT_TRACE_MOE") == "1":
            parameter_names.update(
                {
                    f"{traced_layer}.mlp.gate.weight",
                    f"{traced_layer}.mlp.experts.gate_proj",
                    f"{traced_layer}.mlp.experts.up_proj",
                    f"{traced_layer}.mlp.experts.down_proj",
                    f"{traced_layer}.mlp.shared_expert.gate_proj.weight",
                    f"{traced_layer}.mlp.shared_expert.up_proj.weight",
                    f"{traced_layer}.mlp.shared_expert.down_proj.weight",
                }
            )
        for name, parameter in instance.named_parameters():
            if name in parameter_names:
                trace[f"parameter.{name}"] = parameter.detach().float().cpu()
        return handles, trace, grad_targets

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
        activation_trace_step = int(
            os.environ.get("LINGBOT_ACTIVATION_TRACE_STEP", "1")
        )
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
        if call_index == activation_trace_step:
            handles, activation_trace, grad_targets = activation_hooks(self)
        else:
            handles, activation_trace, grad_targets = [], {}, []
        trace_root = os.environ.get("LINGBOT_ACTIVATION_TRACE_DIR")

        def register_activation_grad_trace() -> None:
            if not trace_root or not grad_targets:
                return
            keys, tensors = zip(*grad_targets)

            def save_gradients(grads) -> None:
                grad_trace = {"__metadata__": {}}
                for key, grad in zip(keys, grads):
                    if grad is None:
                        continue
                    detached = grad.detach()
                    grad_key = f"{key}.grad"
                    grad_trace[grad_key] = detached.float().cpu()
                    grad_trace["__metadata__"][grad_key] = {
                        "dtype": str(detached.dtype),
                        "shape": tuple(detached.shape),
                        "stride": tuple(detached.stride()),
                        "contiguous": detached.is_contiguous(),
                    }
                trace_path = Path(trace_root)
                trace_path.mkdir(parents=True, exist_ok=True)
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                torch.save(
                    grad_trace,
                    trace_path
                    / f"native_step{call_index:06d}_rank{rank}_activation_grads.pt",
                )

            torch.autograd.graph.register_multi_grad_hook(tensors, save_gradients)

        def save_activation_trace() -> None:
            if (
                not trace_root
                or not activation_trace
                or call_index != activation_trace_step
            ):
                return
            trace_path = Path(trace_root)
            trace_path.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            filename = (
                f"native_rank{rank}.pt"
                if activation_trace_step == 1
                else f"native_step{call_index:06d}_rank{rank}.pt"
            )
            torch.save(activation_trace, trace_path / filename)

        try:
            result = original(self, *args, **kwargs)
            register_activation_grad_trace()
        finally:
            for handle in handles:
                handle.remove()
            # Keep diagnostics available even when the opt-in forward-only
            # sentinel or the model itself raises after producing activations.
            save_activation_trace()
        if (
            trace_root
            and activation_trace
            and call_index == activation_trace_step
        ):
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
    if not any(
        argument.startswith("--train.save_hf_weights")
        for argument in sys.argv
    ):
        sys.argv.append("--train.save_hf_weights=false")
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


def _install_force_eager_attention() -> None:
    """Force native Qwen configs to eager attention for dependency diagnostics.

    Native production runs should keep their configured attention backend.  In
    environments without flash-attn, setting ``LINGBOT_FORCE_EAGER_ATTN=1``
    lets the launcher reach model/optimizer construction without editing the
    read-only LingBot source or its checkpoint config.
    """
    if os.environ.get("LINGBOT_FORCE_EAGER_ATTN") != "1":
        return
    import transformers

    patched = 0
    for model_cls in (transformers.PreTrainedModel,):
        if getattr(model_cls, "_fluxvla_force_eager_attention", False):
            continue
        original_check = model_cls._check_and_adjust_attn_implementation

        def check_with_eager(self, attn_implementation, *args, _original=original_check, **kwargs):
            # Transformers passes the requested implementation string here,
            # not the config object. Preserve remaining validation arguments.
            return _original(self, "eager", *args, **kwargs)

        model_cls._check_and_adjust_attn_implementation = check_with_eager
        model_cls._fluxvla_force_eager_attention = True
        patched += 1
    qwen_module = importlib.import_module("lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla")
    model_cls = qwen_module.Qwen3VLPreTrainedModel
    if not getattr(model_cls, "_fluxvla_force_eager_attention", False):
        original_check = model_cls._check_and_adjust_attn_implementation

        def check_qwen_with_eager(self, attn_implementation, *args, **kwargs):
            return original_check(self, "eager", *args, **kwargs)

        model_cls._check_and_adjust_attn_implementation = check_qwen_with_eager
        model_cls._fluxvla_force_eager_attention = True
        patched += 1
    print("LINGBOT_FORCE_EAGER_ATTN_INSTALLED", flush=True)


def _install_native_fused_moe_backward_trace() -> None:
    """Trace native fused-MoE backward outputs without editing LingBot."""
    root_value = os.environ.get("LINGBOT_FUSED_MOE_BACKWARD_TRACE_DIR")
    if not root_value:
        return

    fused_moe_module = importlib.import_module("lingbotvla.ops.fused_moe")
    function_cls = fused_moe_module.FusedMoeExpertFunction
    if getattr(function_cls, "_fluxvla_backward_trace", False):
        return

    original_backward = function_cls.backward
    calls_per_step = int(os.environ.get("LINGBOT_FUSED_MOE_CALLS_PER_STEP", "36"))
    target_step = int(os.environ.get("LINGBOT_FUSED_MOE_BACKWARD_TRACE_STEP", "1"))
    root = Path(root_value)
    call_index = 0

    def backward_with_trace(ctx, grad_output):
        nonlocal call_index
        result = original_backward(ctx, grad_output)
        call_index += 1
        step = (call_index - 1) // calls_per_step + 1
        if step != target_step:
            return result

        call_in_step = (call_index - 1) % calls_per_step + 1
        predicted_layer = calls_per_step - call_in_step
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        root.mkdir(parents=True, exist_ok=True)

        def cpu_float(tensor):
            return tensor.detach().float().cpu().clone()

        torch.save(
            {
                "backend": "native",
                "step": step,
                "rank": rank,
                "call_in_step": call_in_step,
                "predicted_layer": predicted_layer,
                "grad_output": cpu_float(
                    grad_output.reshape(-1, grad_output.shape[-1])
                ),
                "grad_gate_weight": cpu_float(result[1]),
                "grad_hidden_states": cpu_float(result[3]),
            },
            root
            / (
                f"native_step{step:06d}_rank{rank}_call{call_in_step:03d}_"
                f"layer{predicted_layer:03d}.pt"
            ),
        )
        return result

    function_cls.backward = staticmethod(backward_with_trace)
    function_cls._fluxvla_backward_trace = True

    model_module = importlib.import_module(
        "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2"
    )
    flow_cls = model_module.FlowMatchingV2
    original_moe_losses = flow_cls._moe_losses_and_metrics
    moe_loss_call_index = 0

    def moe_losses_with_router_trace(self, router_logits_list, losses):
        nonlocal moe_loss_call_index
        moe_loss_call_index += 1
        result = original_moe_losses(self, router_logits_list, losses)
        router_root_value = os.environ.get("LINGBOT_ROUTER_GRAD_TRACE_DIR")
        router_target_step = int(
            os.environ.get("LINGBOT_ROUTER_GRAD_TRACE_STEP", "1")
        )
        if not router_root_value or moe_loss_call_index != router_target_step:
            return result

        router_root = Path(router_root_value)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        for layer, logits in enumerate(router_logits_list or ()):
            if logits is None or not logits.requires_grad:
                continue

            def save_gradient(grad, *, layer_index=layer, logits_tensor=logits):
                router_root.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "backend": "native",
                        "step": moe_loss_call_index,
                        "rank": rank,
                        "layer": layer_index,
                        "router_logits": logits_tensor.detach().float().cpu().clone(),
                        "router_logits_grad": grad.detach().float().cpu().clone(),
                    },
                    router_root
                    / (
                        f"native_router_step{moe_loss_call_index:06d}_rank{rank}_"
                        f"layer{layer_index:03d}.pt"
                    ),
                )
                return grad

            logits.register_hook(save_gradient)
        return result

    flow_cls._moe_losses_and_metrics = moe_losses_with_router_trace
    print("LINGBOT_FUSED_MOE_BACKWARD_TRACE_INSTALLED", flush=True)


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
    target_step = int(os.environ.get("LINGBOT_OPTIMIZER_TRACE_STEP", "0"))
    grad_norm_trace_dir_value = os.environ.get(
        "LINGBOT_GRAD_NORM_INPUT_TRACE_DIR"
    )
    original_builder = optim_module.build_optimizer

    def canonical_name(name: str) -> str:
        return name.removeprefix("module.").replace("._fsdp_wrapped_module.", ".")

    def full_cpu(value):
        if value is None:
            return None
        detached = value.detach()
        # Do not insert an all-gather before gradient norm calculation. The
        # paired runs use the same FSDP2 topology, so matching rank-local
        # shards are sufficient for optimizer first-bad diagnostics.
        if hasattr(detached, "to_local"):
            detached = detached.to_local()
        return detached.float().cpu().clone()

    def build_optimizer_with_trace(model, *args, **kwargs):
        optimizer = original_builder(model, *args, **kwargs)
        names_by_id = {
            id(parameter): canonical_name(name)
            for name, parameter in model.named_parameters()
        }
        raw_grads = {}
        original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

        def snapshot_raw_grads(parameters):
            params = list(parameters)
            if target_step and step_index + 1 != target_step:
                return params
            grad_norm_rows = []
            grad_norms = []
            for parameter in params:
                name = names_by_id.get(id(parameter), "<unnamed>")
                grad = parameter.grad
                if matcher.search(name) and grad is not None:
                    raw_grads[name] = full_cpu(grad)
                if grad_norm_trace_dir_value and grad is not None:
                    local_grad = grad.detach()
                    if hasattr(local_grad, "to_local"):
                        local_grad = local_grad.to_local()
                    grad_norm_rows.append(
                        {
                            "index": len(grad_norm_rows),
                            "name": name,
                            "parameter_shape": list(parameter.shape),
                            "local_grad_shape": list(local_grad.shape),
                            "dtype": str(local_grad.dtype),
                        }
                    )
                    grad_norms.append(
                        torch.linalg.vector_norm(
                            local_grad, 2.0, dtype=torch.float32
                        )
                    )
            if grad_norm_trace_dir_value:
                grad_norm_trace_dir = Path(grad_norm_trace_dir_value)
                grad_norm_trace_dir.mkdir(parents=True, exist_ok=True)
                grad_norm_values = (
                    torch.stack(grad_norms).cpu()
                    if grad_norms else torch.empty(0, dtype=torch.float32)
                )
                torch.save(
                    {
                        "backend": "native",
                        "step": step_index + 1,
                        "rank": rank,
                        "parameters": grad_norm_rows,
                        "grad_norms": grad_norm_values,
                    },
                    grad_norm_trace_dir
                    / f"native_step_{step_index + 1:06d}_rank_{rank:02d}.pt",
                )
            return params

        def clip_grad_norm_with_trace(parameters, *clip_args, **clip_kwargs):
            # ``clip_grad_norm_`` materializes the iterable internally. Doing
            # it here exposes the exact ordered inputs while preserving the
            # original norm and clipping implementation.
            params = snapshot_raw_grads(parameters)
            return original_clip_grad_norm(
                params, *clip_args, **clip_kwargs
            )

        torch.nn.utils.clip_grad_norm_ = clip_grad_norm_with_trace
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
            trace_this_step = not target_step or step_index == target_step
            snapshots = []
            for group_index, group in enumerate(optimizer.param_groups) if trace_this_step else ():
                for parameter in group["params"]:
                    name = names_by_id.get(id(parameter), "<unnamed>")
                    if not matcher.search(name):
                        continue
                    state = optimizer.state.get(parameter, {})
                    snapshots.append(
                        {
                            "name": name,
                            "group": group_index,
                            "lr": float(group["lr"]),
                            "parameter_before": full_cpu(parameter),
                            "raw_grad": raw_grads.get(name),
                            "post_clip_grad": full_cpu(parameter.grad),
                            "exp_avg_before": full_cpu(state.get("exp_avg")),
                            "exp_avg_sq_before": full_cpu(state.get("exp_avg_sq")),
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
            if rank == 0 and trace_this_step:
                trace_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"backend": "native", "step": step_index, "parameters": snapshots},
                    trace_dir / f"native_step_{step_index:06d}.pt",
                )
            return result

        optimizer.step = types.MethodType(traced_step, optimizer)
        return optimizer

    optim_module.build_optimizer = build_optimizer_with_trace


def _install_native_dtype_debug() -> None:
    """Emit a compact runtime dtype report for the read-only native runner.

    The native launcher accepts ``LINGBOT_DTYPE_DEBUG=1`` for one-shot
    configuration diagnosis.  This deliberately samples metadata only: it
    never materializes DTensors and therefore does not add a model-sized
    checkpoint or alter the numerical path.
    """
    if os.environ.get("LINGBOT_DTYPE_DEBUG") != "1":
        return

    import torch
    import torch.distributed as dist
    import lingbotvla.optim as optim_module

    output_value = os.environ.get("LINGBOT_DTYPE_DEBUG_DIR", "work_dirs/native_dtype_debug")
    output_dir = Path(output_value)
    original_builders = {
        name: getattr(optim_module, name, None)
        for name in ("build_optimizer", "build_muon_optimizer")
    }

    def dtype_name(value):
        return str(getattr(value, "dtype", None)).removeprefix("torch.")

    def is_dtensor(value):
        return value.__class__.__name__ == "DTensor" or hasattr(value, "placements")

    def build_optimizer_with_dtype_debug(original_builder, model, *args, **kwargs):
        optimizer = original_builder(model, *args, **kwargs)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        rows = []
        for name, parameter in model.named_parameters():
            if len(rows) >= 32:
                break
            if any(token in name for token in (
                "state_proj", "action_in_proj", "action_out_proj", "vision", "visual",
                "language_model", "action_model", "expert",
            )) or len(rows) < 8:
                rows.append({
                    "name": name,
                    "parameter_dtype": dtype_name(parameter),
                    "grad_dtype": dtype_name(parameter.grad) if parameter.grad is not None else None,
                    "is_dtensor": is_dtensor(parameter),
                    "requires_grad": bool(parameter.requires_grad),
                    "shape": list(parameter.shape),
                })
        module_rows = []
        for name, module in model.named_modules():
            if not name or len(module_rows) >= 32:
                continue
            if any(token in name for token in (
                "vision", "visual", "language_model", "action_model", "expert", "state_proj",
            )):
                module_rows.append({
                    "name": name,
                    "class": module.__class__.__name__,
                    "parameter_dtypes": sorted({
                        dtype_name(parameter) for parameter in module.parameters(recurse=False)
                    }),
                })

        report = {
            "backend": "lingbot_native",
            "rank": rank,
            "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
            "model_class": model.__class__.__name__,
            "parameter_count": sum(1 for _ in model.parameters()),
            "parameter_dtypes": sorted({dtype_name(parameter) for parameter in model.parameters()}),
            "parameter_samples": rows,
            "module_samples": module_rows,
            "optimizer_class": optimizer.__class__.__name__,
            "optimizer_defaults": {
                key: str(value) if not isinstance(value, (int, float, bool, str, type(None))) else value
                for key, value in optimizer.defaults.items()
            },
            "optimizer_state": [],
        }

        original_step = optimizer.step
        recorded_state = False

        def debug_step(*step_args, **step_kwargs):
            nonlocal recorded_state
            result = original_step(*step_args, **step_kwargs)
            if not recorded_state:
                for parameter, state in optimizer.state.items():
                    if len(report["optimizer_state"]) >= 16:
                        break
                    report["optimizer_state"].append({
                        "parameter_shape": list(parameter.shape),
                        "parameter_dtype": dtype_name(parameter),
                        "state_dtypes": {
                            key: dtype_name(value)
                            for key, value in state.items()
                            if torch.is_tensor(value)
                        },
                    })
                recorded_state = True
                if rank == 0:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with (output_dir / "runtime_dtype.json").open("w", encoding="utf-8") as stream:
                        json.dump(report, stream, indent=2, sort_keys=True)
            return result

        optimizer.step = types.MethodType(lambda _optimizer, *step_args, **step_kwargs: debug_step(*step_args, **step_kwargs), optimizer)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "runtime_dtype_pre_step.json").open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
        return optimizer

    for builder_name, original_builder in original_builders.items():
        if original_builder is not None:
            setattr(
                optim_module,
                builder_name,
                lambda model, *args, _builder=original_builder, **kwargs: build_optimizer_with_dtype_debug(
                    _builder, model, *args, **kwargs
                ),
            )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: lingbot_native_compat_launcher.py ENTRYPOINT [ARGS...]")

    # Do not import FluxVLA here: its package initialization owns a separate
    # distributed runtime and would initialize the default process group.
    _install_transformers_compat()

    # LeRobot v2 exposed ``hf_transform_to_torch`` from ``io_utils`` while
    # newer releases folded it into ``datasets.utils``.  Keep this launcher
    # usable with either layout; the native LingBot worktree remains read-only.
    from lerobot.datasets import utils
    try:
        from lerobot.datasets import io_utils
    except ImportError:
        io_utils = utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not hasattr(utils, "hf_transform_to_torch"):
        utils.hf_transform_to_torch = io_utils.hf_transform_to_torch

    _install_lerobot_reader_compat(LeRobotDataset)

    _install_fixed_flow_replay()
    _install_native_fused_moe_backward_trace()
    _install_force_eager_attention()
    _install_checkpoint_save_skip()
    _install_native_dtype_debug()
    _install_native_optimizer_trace()
    _install_eager_checkpoint_compat()
    _install_optional_alignment_checkpoint_compat()
    _install_null_cli_compat()
    _install_null_vlm_repo_compat()

    entrypoint = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(entrypoint, run_name="__main__")


if __name__ == "__main__":
    main()
