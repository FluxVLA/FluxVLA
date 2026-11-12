"""FluxVLA adapter for the native LingBot-VLA-v2 policy.

The LingBot implementation intentionally remains an independent model core.  This
module only owns the FluxVLA registry contract, checkpoint validation, and the
loss-dictionary protocol consumed by :class:`BaseTrainRunner`.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from fluxvla.engines import VLAS
from fluxvla.engines.utils.fsdp_wrapping import build_module_wrap_policy


def _load_native_symbols(source_root: str | None):
    """Load the native core without making it a FluxVLA dependency at import time."""
    root = source_root or os.environ.get("LINGBOT_VLA_SOURCE_ROOT")
    if root:
        root = str(Path(root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    _install_transformers_compat()
    _install_native_import_shims(root)
    try:
        config_mod = importlib.import_module(
            "lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla"
        )
        # The reference training entrypoint applies this patch indirectly via
        # ``lingbotvla.models.auto``.  FluxVLA deliberately avoids importing
        # that package because it eagerly loads LingBot's legacy data stack,
        # so the model-only adapter must apply the same native patch explicitly.
        qwen_patch_mod = importlib.import_module(
            "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla"
        )
        # LingBot's logger helper is richer than the plain Transformers logger
        # available in the FluxVLA container.  The patch only uses this method
        # for rank-zero informational output, so provide the local equivalent
        # without changing the read-only reference tree.
        if not hasattr(qwen_patch_mod.logger, "info_rank0"):
            qwen_patch_mod.logger.info_rank0 = qwen_patch_mod.logger.info
        qwen_patch_mod.apply_lingbot_qwen3_vl_patch()
        model_mod = importlib.import_module(
            "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2"
        )
    except ImportError as exc:
        raise ImportError(
            "LingBot-VLA-v2 native core is unavailable. Install the vendored "
            "core or set LINGBOT_VLA_SOURCE_ROOT to its read-only source root."
        ) from exc
    return config_mod.LingbotVLAV2Config, model_mod.LingbotVlaV2Policy


def _install_native_import_shims(root: str | None) -> None:
    """Avoid importing the reference training/data package for model-only use.

    The reference package's top-level ``models`` import eagerly imports its
    dataloader and an older LeRobot API.  FluxVLA owns data loading, so those
    modules are not needed to construct the native model.  Package stubs retain
    normal relative imports while keeping this compatibility boundary explicit.
    """
    if not root:
        return
    import types

    base = Path(root) / "lingbotvla"
    package_paths = {
        "lingbotvla": base,
        "lingbotvla.models": base / "models",
        "lingbotvla.models.vla": base / "models" / "vla",
        "lingbotvla.models.vla.lingbot_vla": base / "models" / "vla" / "lingbot_vla",
        "lingbotvla.models.vla.vision_models": base / "models" / "vla" / "vision_models",
        "lingbotvla.models.vla.vision_models.align_heads": base / "models" / "vla" / "vision_models" / "align_heads",
        "lingbotvla.ops": base / "ops",
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    loader = types.ModuleType("lingbotvla.models.loader")
    loader.LingBotVLAWeightLoader = type("LingBotVLAWeightLoader", (), {})
    sys.modules.setdefault("lingbotvla.models.loader", loader)
    moe = types.ModuleType("lingbotvla.ops.robby_moe")
    moe.robby_moe_forward = None
    sys.modules.setdefault("lingbotvla.ops.robby_moe", moe)
    # The loss module is model-only and contains a vectorized PyTorch fallback
    # when Triton is unavailable.  Prefer the real implementation; the stub is
    # only a last-resort import guard for incomplete native installations.
    if "lingbotvla.ops.triton_moe_loss" not in sys.modules:
        try:
            importlib.import_module("lingbotvla.ops.triton_moe_loss")
        except Exception:
            triton = types.ModuleType("lingbotvla.ops.triton_moe_loss")
            triton.triton_sequence_wise_balance_loss = None
            sys.modules["lingbotvla.ops.triton_moe_loss"] = triton


def _install_transformers_compat() -> None:
    """Bridge only removed Transformers symbols used by the reference loader.

    This is intentionally local to the adapter import path.  It does not patch
    attention/model implementations and is covered by the import smoke test.
    """
    import transformers
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils
    import transformers.cache_utils as cache_utils
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen25vl
    from contextlib import contextmanager

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModel
    if not hasattr(modeling_utils, "no_init_weights"):
        @contextmanager
        def no_init_weights(_enable: bool = True):
            yield

        modeling_utils.no_init_weights = no_init_weights
    # Transformers 4.x accepted a list of tied keys; 5.x expects a mapping.
    # LingBot deletes both LM heads for action training, so an empty mapping is
    # the correct compatibility behavior during construction.
    if not getattr(modeling_utils.PreTrainedModel, "_fluxvla_lingbot_tie_compat", False):
        original_tie_keys = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys

        def get_expanded_tied_weights_keys_compat(self, all_submodels=False):
            if isinstance(getattr(self, "_tied_weights_keys", None), list):
                return {}
            return original_tie_keys(self, all_submodels=all_submodels)

        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys_compat
        modeling_utils.PreTrainedModel._fluxvla_lingbot_tie_compat = True
    if not hasattr(import_utils, "is_safetensors_available"):
        import_utils.is_safetensors_available = lambda: True
    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = cache_utils.DynamicCache
    if not hasattr(qwen25vl, "Qwen2RMSNorm"):
        qwen25vl.Qwen2RMSNorm = qwen25vl.Qwen2_5_VLRMSNorm


def _as_config_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return dict(value)


def _native_output_to_dict(output: Any) -> dict[str, Any]:
    """Convert the native 11-tuple to FluxVLA's stable output contract."""
    if isinstance(output, Mapping):
        if "loss" not in output:
            raise ValueError("Native LingBot output mapping must contain 'loss'.")
        return dict(output)
    if not isinstance(output, (tuple, list)) or len(output) != 11:
        raise ValueError(
            "LingBotVLAV2 expects the native 11-field output tuple, got "
            f"{type(output).__name__} with length "
            f"{len(output) if isinstance(output, (tuple, list)) else 'n/a'}."
        )
    (
        total_loss,
        loss_vla,
        loss_depth,
        loss_future_depth,
        loss_future_video,
        loss_seq_wise,
        loss_log,
        depth_preds,
        future_depth_preds,
        future_video_preds,
        current_video_preds,
    ) = output
    result = {
        "loss": total_loss,
        "loss_vla": loss_vla,
        "loss_depth": loss_depth,
        "loss_future_depth": loss_future_depth,
        "loss_future_video": loss_future_video,
        # Native field six is the sequence-wise auxiliary loss.  Router-Z and
        # other MoE diagnostics are carried in ``loss_log`` (field seven).
        # Keep the explicit name so metrics do not mislabel this component as
        # a generic MoE loss.
        "loss_seq_wise": loss_seq_wise,
        "loss_moe_aux": loss_seq_wise,
        "depth_preds": depth_preds,
        "future_depth_preds": future_depth_preds,
        "future_video_preds": future_video_preds,
        "current_video_preds": current_video_preds,
    }
    if isinstance(loss_log, Mapping):
        result.update(loss_log)
    return result


def _resolve_decoder_layers(root: nn.Module | None) -> nn.ModuleList | None:
    """Resolve decoder layers across Qwen text-only and multimodal layouts."""
    if root is None:
        return None
    candidates = (
        getattr(getattr(root, "model", None), "layers", None),
        getattr(
            getattr(getattr(root, "model", None), "language_model", None),
            "layers",
            None,
        ),
        getattr(root, "layers", None),
    )
    return next((layers for layers in candidates if isinstance(layers, nn.ModuleList)), None)


@VLAS.register_module()
class LingBotVLAV2(nn.Module):
    """Native LingBot-VLA-v2 wrapped in the FluxVLA model contract.

    ``source_root`` is deliberately explicit so production runs can point at a
    vendored FluxVLA copy while read-only development can use the reference tree.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        source_root: str | None = None,
        native_config_path: str | None = None,
        pretrained_name_or_path: str | None = None,
        tokenizer_path: str | None = None,
        freeze_vision_encoder: bool = False,
        low_cpu_mem_usage: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        config_cls, policy_cls = _load_native_symbols(source_root)
        config_kwargs = _as_config_dict(model_config)
        if native_config_path is not None:
            import yaml

            native_cfg = yaml.safe_load(Path(native_config_path).read_text())
            merged_native = {}
            merged_native.update(native_cfg.get("model", {}))
            merged_native.update(native_cfg.get("train", {}))
            merged_native.update(config_kwargs)
            config_kwargs = merged_native
        action_cfg = _as_config_dict(config_kwargs.pop("action", None))
        attention_cfg = _as_config_dict(config_kwargs.pop("attention", None))
        moe_cfg = _as_config_dict(config_kwargs.pop("moe", None))
        alignment_cfg = _as_config_dict(config_kwargs.pop("alignment", None))
        config_kwargs.update(action_cfg)
        config_kwargs.update(attention_cfg)
        if "implementation" in attention_cfg:
            config_kwargs["attention_implementation"] = attention_cfg["implementation"]
        if "vit_implementation" in attention_cfg:
            config_kwargs["vit_attn_implementation"] = attention_cfg["vit_implementation"]
        if "enabled" in moe_cfg:
            config_kwargs["use_moe"] = moe_cfg["enabled"]
        if "enabled" in alignment_cfg:
            config_kwargs["align_params"] = alignment_cfg if alignment_cfg["enabled"] else {}
        # The native training entrypoint exposes this as a train-level flag,
        # then forwards it to the model as ``action_fp32``. Preserve that
        # contract when a native YAML is used through FluxVLA.
        if "action_fp32" not in config_kwargs and "enable_fp32" in config_kwargs:
            config_kwargs["action_fp32"] = bool(config_kwargs["enable_fp32"])
        config_kwargs.update(kwargs)
        if tokenizer_path is not None:
            config_kwargs["tokenizer_path"] = tokenizer_path
        if pretrained_name_or_path is not None:
            config_kwargs.setdefault("pretrained_name_or_path", pretrained_name_or_path)
        for numeric_key in (
            "sequence_wise_loss_coeff",
            "router_z_loss_coeff",
            "depth_loss_weight",
            "future_depth_loss_weight",
        ):
            if isinstance(config_kwargs.get(numeric_key), str):
                config_kwargs[numeric_key] = float(config_kwargs[numeric_key])
        config = config_cls(**config_kwargs)
        self.config = config
        # BaseTrainRunner probes these generic BaseVLA contract attributes.
        # LingBot owns one native policy instead of separate FluxVLA backbone
        # modules, so expose explicit empty slots without re-registering the
        # native model under multiple module names.
        self.all_module_keys = ["model"]
        self.llm_backbone = None
        self.vlm_backbone = None
        self.llm_expert = None
        self.source_root = source_root
        self.pretrained_name_or_path = pretrained_name_or_path
        checkpoint_is_dir = bool(
            pretrained_name_or_path
            and Path(pretrained_name_or_path).is_dir()
            and list(Path(pretrained_name_or_path).glob("*.safetensors"))
        )
        if low_cpu_mem_usage is None:
            low_cpu_mem_usage = checkpoint_is_dir
        self._low_cpu_mem_usage = bool(low_cpu_mem_usage and checkpoint_is_dir)
        if self._low_cpu_mem_usage:
            try:
                from accelerate import init_empty_weights
            except ImportError as exc:  # pragma: no cover - minimal installations
                raise ImportError(
                    "low_cpu_mem_usage requires accelerate; install accelerate or "
                    "set low_cpu_mem_usage=False."
                ) from exc
            with init_empty_weights():
                self.model = policy_cls(config)
        else:
            self.model = policy_cls(config)
        # Some Transformers 5.x builds materialize the vision tower under
        # ``Qwen3VLForConditionalGeneration.model.visual`` even after the
        # native LingBot patch.  LingBot's model core intentionally addresses
        # the tower as ``qwenvl.visual``; expose that alias at the adapter
        # boundary so the native forward graph remains unchanged.
        native_core = getattr(self.model, "model", self.model)
        qwen = getattr(getattr(native_core, "qwenvl_with_expert", None), "qwenvl", None)
        if qwen is not None and not hasattr(qwen, "visual") and hasattr(getattr(qwen, "model", None), "visual"):
            # A direct assignment would register the same module twice and
            # change checkpoint key coverage.  Use a class-level property so
            # state_dict names remain the native ``qwenvl.model.visual.*``
            # names while old LingBot call sites can keep using ``visual``.
            qwen_type = type(qwen)
            if not hasattr(qwen_type, "_fluxvla_visual_alias"):
                qwen_type.visual = property(lambda self: self.model.visual)
                qwen_type._fluxvla_visual_alias = True
        qwen_model = getattr(qwen, "model", None)
        rope_owner = type(qwen_model) if qwen_model is not None else None
        if rope_owner is not None and hasattr(rope_owner, "get_rope_index") and not hasattr(rope_owner, "_fluxvla_rope_compat"):
            native_get_rope_index = rope_owner.get_rope_index

            def get_rope_index_compat(self, *args, **call_kwargs):
                if "mm_token_type_ids" not in call_kwargs:
                    input_ids = call_kwargs.get("input_ids", args[0] if args else None)
                    grid = call_kwargs.get("image_grid_thw")
                    if input_ids is None:
                        raise TypeError("get_rope_index requires input_ids")
                    # LingBot constructs a prefix as [image boundary/patch
                    # tokens, language tokens].  The grid volume is the raw
                    # vision-patch count; Qwen3-VL RoPE consumes the merged
                    # patch count. Boundary tokens remain text positions.
                    mm = torch.zeros_like(input_ids, dtype=torch.int32)
                    image_token_id = getattr(self.config, "image_token_id", None)
                    if image_token_id is not None:
                        # Native prefixes contain vision boundaries between
                        # views; only the repeated image token id is modality
                        # type 1. Padding-view tokens are removed by the
                        # attention mask after this classification.
                        mm[input_ids == int(image_token_id)] = 1
                    elif grid is not None and grid.numel():
                        grid = torch.as_tensor(grid, device=input_ids.device).reshape(-1, 3)
                        image_len = int(
                            (grid.prod(-1) // int(self.config.vision_config.spatial_merge_size**2))
                            .sum()
                            .item()
                        )
                        mm[:, :image_len] = 1
                    call_kwargs["mm_token_type_ids"] = mm
                return native_get_rope_index(self, *args, **call_kwargs)

            rope_owner.get_rope_index = get_rope_index_compat
            rope_owner._fluxvla_rope_compat = True
        self.freeze_vision_encoder = freeze_vision_encoder
        self._validate_model_contract()

    def _validate_model_contract(self) -> None:
        if self.config.max_action_dim != 55 or self.config.max_state_dim != 55:
            raise ValueError(
                "LingBotVLAV2 FluxVLA integration requires max_state_dim and "
                f"max_action_dim=55, got {self.config.max_state_dim}/"
                f"{self.config.max_action_dim}."
            )
        if self.config.n_action_steps != self.config.chunk_size:
            raise ValueError("n_action_steps must equal chunk_size.")
        if not hasattr(self.model, "sample_actions"):
            raise TypeError("Native LingBot policy must expose sample_actions().")

    def forward(self, **batch: Any) -> dict[str, Any]:
        self._validate_batch_contract(batch)
        batch = self._promote_action_flow_inputs(batch)
        handles, trace = self._install_activation_trace_hooks()
        try:
            output = self.model(**batch)
        finally:
            for handle in handles:
                handle.remove()
        self._save_activation_trace(trace)
        if os.environ.get("FLUXVLA_LINGBOT_TRACE_FORWARD_ONLY") == "1":
            raise RuntimeError("FluxVLA LingBot forward-only trace complete")
        return _native_output_to_dict(output)

    def _promote_action_flow_inputs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Restore LingBot's FP32 flow contract after FSDP input casting.

        Native training samples noise/time inside the policy from FP32 actions.
        Replay datasets provide them explicitly, so FSDP2 casts them to BF16
        before this adapter runs unless they are promoted together with the
        state/action tensors.
        """
        prepared = dict(batch)
        if getattr(self.config, "action_fp32", False):
            for key in ("state", "actions", "noise", "time"):
                value = prepared.get(key)
                if isinstance(value, Tensor) and value.dtype.is_floating_point:
                    prepared[key] = value.float()
        return prepared

    def _install_activation_trace_hooks(self):
        """Install one-forward hooks for opt-in native/adapter first-bad analysis."""
        trace_root = os.environ.get("FLUXVLA_LINGBOT_ACTIVATION_TRACE_DIR")
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
        trace: dict[str, Tensor] = {}
        handles = []

        def first_tensor(value):
            if isinstance(value, Tensor):
                return value
            if isinstance(value, (tuple, list)):
                return next((item for item in value if isinstance(item, Tensor)), None)
            return None

        for name, module in self.model.named_modules():
            if name not in selected:
                continue

            def save(_module, inputs, output, *, module_name=name):
                output_tensor = first_tensor(output)
                for index, input_tensor in enumerate(inputs):
                    if isinstance(input_tensor, Tensor):
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

        def save_parameters(module, _inputs):
            for name, parameter in module.named_parameters():
                if name in parameter_names:
                    trace[f"parameter.{name}"] = parameter.detach().float().cpu()

        handles.append(self.model.register_forward_pre_hook(save_parameters))
        return handles, trace

    @staticmethod
    def _save_activation_trace(trace: Mapping[str, Tensor]) -> None:
        root_value = os.environ.get("FLUXVLA_LINGBOT_ACTIVATION_TRACE_DIR")
        if not root_value or not trace:
            return
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        torch.save(dict(trace), root / f"flux_rank{rank}.pt")

    def prepare_training_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Reproduce the native trainer's per-field input precision contract.

        Native LingBot's collator emits BF16 visual and teacher features, while
        ``action_fp32`` promotes only state/action flow inputs inside the policy.
        FluxVLA's generic FP32-master path otherwise promotes every float tensor.
        """
        native_bf16_keys = (
            "images",
            "depth_targets",
            "future_depth_targets",
            "future_video_targets",
            "future_video_current_patch",
        )
        prepared = dict(batch)
        for key in native_bf16_keys:
            value = prepared.get(key)
            if isinstance(value, Tensor) and value.dtype.is_floating_point:
                prepared[key] = value.to(dtype=torch.bfloat16)
        return prepared

    def _validate_batch_contract(self, batch: Mapping[str, Any]) -> None:
        required = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "actions")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"LingBotVLAV2 batch is missing required keys: {missing}")
        state, actions = batch["state"], batch["actions"]
        if not isinstance(state, Tensor) or state.shape[-1] != 55:
            raise ValueError(f"state must end in 55, got {getattr(state, 'shape', None)}")
        if not isinstance(actions, Tensor) or actions.shape[-1] != 55:
            raise ValueError(f"actions must end in 55, got {getattr(actions, 'shape', None)}")
        if "image_grid_thw" not in batch:
            raise KeyError("LingbotVLAV2 requires image_grid_thw from Qwen3-VL preprocessing.")

    def sample_actions(self, **observation: Any) -> Tensor:
        return self.model.sample_actions(**observation)

    def from_pretrained(self) -> None:
        if self.pretrained_name_or_path is None:
            return
        checkpoint = Path(self.pretrained_name_or_path)
        if checkpoint.is_dir():
            self._load_safetensors_directory(checkpoint)
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "model" in state:
                state = state["model"]
            self._load_state_checked(state)

    def _load_safetensors_directory(self, directory: Path) -> None:
        from safetensors import safe_open

        paths = sorted(directory.glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"No safetensors files found under {directory}")
        expected = self.model.state_dict()
        assign = self._low_cpu_mem_usage and any(t.is_meta for t in expected.values())
        seen: set[str] = set()
        for path in paths:
            shard: dict[str, Tensor] = {}
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key not in expected:
                        raise RuntimeError(f"Unexpected LingBot checkpoint key: {key}")
                    tensor = handle.get_tensor(key)
                    if tuple(tensor.shape) != tuple(expected[key].shape):
                        raise RuntimeError(
                            f"Shape mismatch for {key}: checkpoint={tuple(tensor.shape)} "
                            f"model={tuple(expected[key].shape)}"
                        )
                    shard[key] = tensor
                    seen.add(key)
            # Materialize meta parameters from the current shard. This keeps
            # only one shard plus the destination parameters resident.
            self.model.load_state_dict(shard, strict=False, assign=assign)
        missing = sorted(set(expected) - seen)
        if missing:
            raise RuntimeError(
                f"LingBot checkpoint coverage failure: missing={len(missing)}, "
                f"first_missing={missing[:3]}"
            )

    def _load_state_checked(self, state: Mapping[str, Tensor]) -> None:
        # Native LingBot checkpoints are written from ``LingbotVlaV2Policy``;
        # the FluxVLA adapter itself must not add an extra ``model.`` prefix.
        expected = self.model.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatch = sorted(
            key for key in set(expected).intersection(state)
            if tuple(expected[key].shape) != tuple(state[key].shape)
        )
        if missing or mismatch:
            raise RuntimeError(
                "LingBot checkpoint coverage failure: "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"shape_mismatch={len(mismatch)}; first_missing={missing[:3]}, "
                f"first_mismatch={mismatch[:3]}"
            )
        self.model.load_state_dict(dict(state), strict=True)

    def freeze_backbones(self) -> None:
        if self.freeze_vision_encoder and hasattr(self.model, "qwenvl_with_expert"):
            vision = getattr(self.model.qwenvl_with_expert, "qwenvl", None)
            vision = getattr(vision, "visual", None)
            if vision is not None:
                vision.requires_grad_(False)

    def gradient_checkpointing_enable(self) -> None:
        enable = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise RuntimeError("Native LingBot policy does not expose gradient_checkpointing_enable")
        enable()

    def initialize_distributed_runtime(self) -> None:
        """Register FluxVLA's data-parallel topology with native LingBot ops."""
        if not torch.distributed.is_initialized():
            return
        from lingbotvla.distributed.parallel_state import (
            get_parallel_state,
            init_parallel_state,
        )

        world_size = torch.distributed.get_world_size()
        init_parallel_state(dp_size=world_size, dp_mode="fsdp1")
        parallel_state = get_parallel_state()
        if parallel_state.dp_size != world_size:
            raise RuntimeError(
                "LingBot parallel state does not match the FluxVLA process group: "
                f"dp_size={parallel_state.dp_size}, world_size={world_size}."
            )

    def get_fsdp_wrapping_policy(self) -> Callable:
        classes = self._fsdp_vlm_decoder_layer_classes()
        return build_module_wrap_policy(classes)

    def get_fsdp2_wrapping_modules(self) -> list[nn.Module]:
        """Return Qwen-VL text blocks in native ``vlm_fsdp`` order."""
        native_core = getattr(self.model, "model", self.model)
        core = getattr(native_core, "qwenvl_with_expert", None)
        layers = _resolve_decoder_layers(getattr(core, "qwenvl", None))
        if not layers:
            raise RuntimeError("Could not resolve LingBot VLM decoder layers for FSDP2.")
        return list(layers)

    def get_fsdp2_full_precision_modules(self) -> list[nn.Module]:
        """Return native ``basic_modules`` in child-first application order.

        LingBot applies its FP32 FSDP2 policy to every module named in the
        policy's ``_no_split_modules`` list before applying BF16 VLM/root
        sharding.  In particular, this keeps the action decoder and AdaNorm
        computation in FP32 when ``enable_fp32`` is active.
        """
        class_names = set(getattr(self.model, "_no_split_modules", ()))
        modules: list[nn.Module] = []

        def collect(module: nn.Module) -> None:
            if module.__class__.__name__ in class_names:
                modules.append(module)

        self.model.apply(collect)
        return modules

    def _fsdp_vlm_decoder_layer_classes(self) -> set[type[nn.Module]]:
        """Match native ``vlm_fsdp``: shard Qwen-VL text blocks only."""
        classes = set()
        native_core = getattr(self.model, "model", self.model)
        core = getattr(native_core, "qwenvl_with_expert", None)
        layers = _resolve_decoder_layers(getattr(core, "qwenvl", None))
        if layers:
            classes.update(type(layer) for layer in layers)
        if not classes:
            raise RuntimeError("Could not resolve LingBot VLM decoder layer classes for FSDP.")
        return classes

    def get_fsdp_ignored_modules(self) -> list[nn.Module]:
        return []


__all__ = ["LingBotVLAV2"]
