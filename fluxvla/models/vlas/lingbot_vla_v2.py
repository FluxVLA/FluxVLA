"""FluxVLA adapter for the native LingBot-VLA-v2 policy.

The LingBot implementation intentionally remains an independent model core.  This
module only owns the FluxVLA registry contract, checkpoint validation, and the
loss-dictionary protocol consumed by :class:`BaseTrainRunner`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from fluxvla.engines import VLAS
from fluxvla.engines.utils.fsdp_wrapping import build_module_wrap_policy


def _load_native_symbols():
    """Load the FluxVLA-owned LingBot model core."""
    _install_transformers_compat()
    from .lingbot_native_core.configuration_lingbot_vla import LingbotVLAV2Config
    from .lingbot_native_core.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from .lingbot_native_core.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

    apply_lingbot_qwen3_vl_patch()

    return LingbotVLAV2Config, LingbotVlaV2Policy


def _install_transformers_compat() -> None:
    """Bridge only removed Transformers symbols used by the reference loader.

    This is intentionally local to the adapter import path.  It does not patch
    attention/model implementations and is covered by the import smoke test.
    """
    import transformers
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils
    import transformers.cache_utils as cache_utils
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl
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
    if os.environ.get("LINGBOT_DISABLE_FLASH_ATTN") == "1":
        # The reference config selects flash_attention_2, but the wheel may be
        # unavailable or built for a different PyTorch ABI.  Keep this opt-in
        # and local to the adapter process so production environments can use
        # flash attention normally.
        import_utils.is_flash_attn_2_available = lambda: False
        import_utils.is_flash_attn_3_available = lambda: False
        # Transformers 5.3 keeps these probes in import_utils; older releases
        # additionally expose a flash_attn_utils module.  Patch whichever API
        # exists so the adapter works across both layouts.
        try:
            import transformers.utils.flash_attn_utils as flash_utils
        except ModuleNotFoundError:
            flash_utils = None
        if flash_utils is not None:
            flash_utils.is_flash_attn_2_available = lambda: False
            flash_utils.is_flash_attn_3_available = lambda: False
            flash_utils.is_flash_attn_available = lambda: False
        base_model = modeling_utils.PreTrainedModel
        if not getattr(base_model, "_fluxvla_eager_attn_check", False):
            original_attn_check = base_model._check_and_adjust_attn_implementation

            def check_eager(self, attn_implementation, is_init_check=False, *args, **kwargs):
                if attn_implementation in ("flash_attention_2", "flash_attention_3"):
                    attn_implementation = "eager"
                return original_attn_check(self, attn_implementation, is_init_check, *args, **kwargs)

            base_model._check_and_adjust_attn_implementation = check_eager
            base_model._fluxvla_eager_attn_check = True
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


def _install_chunked_dense_mlp_oracle(model: nn.Module, chunks: int) -> int:
    """Replay SP's token GEMM shape without changing Native module ownership."""
    if chunks <= 1:
        raise ValueError("LingBot chunked MLP oracle requires chunks > 1")
    native_core = getattr(model, "model", model)
    core = getattr(native_core, "qwenvl_with_expert", None)
    layers = _resolve_decoder_layers(getattr(core, "qwenvl", None))
    if layers is None:
        raise RuntimeError("Could not resolve LingBot VLM layers for chunked MLP oracle")

    wrapped = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or hasattr(mlp, "experts"):
            continue
        if getattr(mlp, "_fluxvla_chunked_oracle", False):
            continue
        # Keep the class function instead of a bound method. Capturing
        # ``mlp.forward`` would retain the module instance inside its own
        # replacement closure, which interferes with FSDP/checkpoint module
        # lifetime and can leave parameter materialization waiting on the
        # stale bound owner.
        original_forward = type(mlp).forward

        def chunked_forward(
            self,
            hidden_states: Tensor,
            *args,
            _original_forward=original_forward,
            **kwargs,
        ) -> Tensor:
            sequence_length = hidden_states.shape[-2]
            pad_length = (-sequence_length) % chunks
            if pad_length:
                hidden_states = torch.nn.functional.pad(
                    hidden_states,
                    (0, 0, 0, pad_length),
                )
            outputs = [
                _original_forward(self, part, *args, **kwargs)
                for part in hidden_states.chunk(chunks, dim=-2)
            ]
            return torch.cat(outputs, dim=-2).narrow(-2, 0, sequence_length)

        mlp.forward = types.MethodType(chunked_forward, mlp)
        mlp._fluxvla_chunked_oracle = True
        wrapped += 1
    return wrapped


@VLAS.register_module()
class LingBotVLAV2(nn.Module):
    """Native LingBot-VLA-v2 wrapped in the FluxVLA model contract.

    The model core and checkpoint contract are owned by FluxVLA.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        native_config_path: str | None = None,
        pretrained_name_or_path: str | None = None,
        tokenizer_path: str | None = None,
        freeze_vision_encoder: bool = False,
        low_cpu_mem_usage: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        config_cls, policy_cls = _load_native_symbols()
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
        # ``enable_fp32`` is a training/reduction switch, not the native
        # LingBot model's ``action_fp32`` contract.  Do not infer the latter:
        # the native YAML explicitly carries ``action_fp32`` (false for the
        # reference RobotWin recipe), and silently mapping the flags changes
        # flow-matching numerics and breaks loss parity.
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
                # The production trainer passes ``torch_dtype=float32`` when
                # mixed precision is enabled, then lets FSDP own compute
                # casting.  Keep checkpoint master values in FP32 here too.
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
        oracle_chunks = int(
            os.environ.get("FLUXVLA_LINGBOT_CHUNKED_MLP_ORACLE", "1")
        )
        if oracle_chunks > 1:
            wrapped = _install_chunked_dense_mlp_oracle(
                self.model,
                oracle_chunks,
            )
            if not wrapped:
                raise RuntimeError("LingBot chunked MLP oracle wrapped no modules")
            print(
                "LingBot Native chunked MLP oracle: "
                f"chunks={oracle_chunks}, wrapped={wrapped}",
                flush=True,
            )

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
        call_index = getattr(self, "_activation_trace_call_index", 0) + 1
        self._activation_trace_call_index = call_index
        trace_step = int(os.environ.get(
            "FLUXVLA_LINGBOT_ACTIVATION_TRACE_STEP", "1"))
        if call_index == trace_step:
            handles, trace, grad_targets = self._install_activation_trace_hooks()
        else:
            handles, trace, grad_targets = [], {}, []
        try:
            output = self.model(**batch)
            self._register_activation_grad_trace(
                grad_targets, step=call_index
            )
        finally:
            for handle in handles:
                handle.remove()
        self._save_activation_trace(trace, step=call_index)
        if (
            call_index == trace_step
            and os.environ.get("FLUXVLA_LINGBOT_TRACE_FORWARD_ONLY") == "1"
        ):
            raise RuntimeError("FluxVLA LingBot forward-only trace complete")
        output_dict = _native_output_to_dict(output)
        trace_output = dict(output_dict)
        if "sample_index" in batch:
            trace_output["sample_index"] = batch["sample_index"]
        self._write_local_loss_trace(trace_output, call_index=call_index)
        return output_dict

    @staticmethod
    def _write_local_loss_trace(
        output: Mapping[str, Any], *, call_index: int
    ) -> None:
        """Persist pre-DP scalar losses for rank-wise parity diagnosis."""
        root_value = os.environ.get("FLUXVLA_LINGBOT_LOCAL_LOSS_TRACE_DIR")
        if not root_value:
            return
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        row: dict[str, Any] = {"call_index": int(call_index), "rank": rank}
        sample_index = output.get("sample_index")
        if isinstance(sample_index, Tensor):
            row["sample_index"] = sample_index.detach().cpu().reshape(-1).tolist()
        for name, value in output.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                row[name] = float(value.detach().float().cpu().item())
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _promote_action_flow_inputs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Restore LingBot's FP32 flow contract after FSDP input casting.

        Native training casts only state/actions for the action_fp32 contract.
        Noise and time remain in the dataloader dtype (BF16 in the production
        recipe); promoting them here changes the flow-matching numerical path.
        """
        prepared = dict(batch)
        if getattr(self.config, "action_fp32", False):
            for key in ("state", "actions"):
                value = prepared.get(key)
                if isinstance(value, Tensor) and value.dtype.is_floating_point:
                    prepared[key] = value.float()
        return prepared

    def _install_activation_trace_hooks(self):
        """Install one-forward hooks for opt-in native/adapter first-bad analysis."""
        trace_root = os.environ.get("FLUXVLA_LINGBOT_ACTIVATION_TRACE_DIR")
        if not trace_root:
            return [], {}, []
        trace_ranks = os.environ.get("FLUXVLA_LINGBOT_TRACE_RANKS")
        if trace_ranks:
            enabled_ranks = {
                int(item.strip())
                for item in trace_ranks.split(",")
                if item.strip()
            }
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            if rank not in enabled_ranks:
                return [], {}, []
        selected = {
            "model.state_proj",
            "model.action_in_proj",
            "model.action_out_proj",
            "model.qwenvl_with_expert.qwenvl.model.language_model.norm",
            "model.qwenvl_with_expert.qwen_expert.model.norm",
        }
        action_head_only = (
            os.environ.get("FLUXVLA_LINGBOT_TRACE_ACTION_HEAD_ONLY") == "1"
        )
        if action_head_only:
            selected = {"model.action_out_proj"}
        else:
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
        if not action_head_only:
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
            vlm_layer0 = (
                "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0"
            )
            selected.update(
                f"{vlm_layer0}.self_attn.{projection}"
                for projection in ("q_proj", "k_proj", "v_proj")
            )
            selected.update({f"{vlm_layer0}.self_attn", vlm_layer0})
        trace_moe = os.environ.get("LINGBOT_REPLAY_TRACE_MOE") == "1"
        router_only = (
            os.environ.get("LINGBOT_REPLAY_TRACE_ROUTER_ONLY") == "1"
        )
        moe_activations_only = (
            os.environ.get("LINGBOT_REPLAY_TRACE_MOE_ACTIVATIONS_ONLY") == "1"
        )
        if trace_moe and not router_only:
            selected.update(
                {
                    f"{traced_layer}.mlp.gate",
                    f"{traced_layer}.mlp.experts",
                    f"{traced_layer}.mlp.shared_expert",
                    f"{traced_layer}.mlp.shared_expert_gate",
                }
            )
            # EP replaces the fused expert tensor with a local ModuleList.
            # Include each local expert module in the trace so checkpoint
            # slicing can be compared against the Native fused tensor.
            selected.update(
                f"{traced_layer}.mlp.experts.{expert_id}"
                for expert_id in range(32)
            )
        trace: dict[str, Any] = {"__metadata__": {}}
        handles = []
        grad_targets = []

        def record(key: str, value: Tensor) -> None:
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

        if os.environ.get("LINGBOT_REPLAY_TRACE_ATTENTION") == "1":
            # The native LingBot attention path owns RoPE and the attention
            # dispatch on QwenvlWithExpertV2Model.  The nested Qwen2 language
            # model only contains transformer layers and does not expose
            # these methods.
            expert = self.model.model.qwenvl_with_expert
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

            # ``flex_cached`` bypasses ``attention_interface`` in the native
            # forward and calls this module-level helper directly.  Wrap the
            # actual dispatch so the trace contains the real aggregation.
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

            import types

            handles.append(types.SimpleNamespace(remove=restore_attention))

        def first_tensor(value):
            if isinstance(value, Tensor):
                return value
            if isinstance(value, (tuple, list)):
                return next((item for item in value if isinstance(item, Tensor)), None)
            return None

        for name, module in self.model.named_modules():
            if name not in selected and not any(name.endswith(token) for token in selected):
                continue

            def save(_module, inputs, output, *, module_name=name):
                output_tensor = first_tensor(output)
                # Record runtime identity/storage at the actual call site.
                # A model-level pre-hook is insufficient when Float16Module,
                # FSDP, or an adapter wrapper changes parameter storage.
                runtime = {
                    "module_id": id(_module),
                    "class": f"{type(_module).__module__}.{type(_module).__qualname__}",
                    "dtype": str(next(_module.parameters(), torch.empty(0)).dtype),
                }
                is_moe_component = module_name.startswith(f"{traced_layer}.mlp.")
                if not (moe_activations_only and is_moe_component):
                    for parameter_name, parameter in _module.named_parameters(recurse=False):
                        record(
                            f"{module_name}.runtime_parameter.{parameter_name}",
                            parameter,
                        )
                for attribute in ("variance_epsilon", "eps"):
                    if hasattr(_module, attribute):
                        runtime[attribute] = float(getattr(_module, attribute))
                trace["__metadata__"][f"{module_name}.runtime"] = runtime
                for index, input_tensor in enumerate(inputs):
                    if isinstance(input_tensor, Tensor):
                        input_key = f"{module_name}.input.{index}"
                        record(input_key, input_tensor)
                        if (
                            os.environ.get(
                                "FLUXVLA_LINGBOT_ACTIVATION_GRAD_TRACE"
                            ) == "1"
                            and input_tensor.requires_grad
                        ):
                            grad_targets.append((input_key, input_tensor))
                if output_tensor is not None:
                    record(f"{module_name}.output", output_tensor)
                    if (
                        os.environ.get(
                            "FLUXVLA_LINGBOT_ACTIVATION_GRAD_TRACE"
                        ) == "1"
                        and output_tensor.requires_grad
                    ):
                        grad_targets.append(
                            (f"{module_name}.output", output_tensor)
                        )
                    if (
                        os.environ.get("LINGBOT_REPLAY_TRACE_MOE") == "1"
                        and module_name.endswith("mlp.gate")
                    ):
                        block = self.model.get_submodule(module_name[:-len(".gate")])
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
                        trace[f"{module_name[:-len('.gate')]}.router_logits"] = logits.detach().cpu()
                        trace[f"{module_name[:-len('.gate')]}.selected_experts"] = selected.detach().cpu()
                        trace[f"{module_name[:-len('.gate')]}.routing_weights"] = weights.detach().cpu()

            handles.append(module.register_forward_hook(save))

        # Capture the routing contract separately from the expert kernels.  A
        # gate pre-hook is independent of the implementation (eager, fused,
        # or EP) and lets parity tests distinguish routing from GEMM/reduction
        # differences without importing the official LingBot code.
        if trace_moe:
            for name, module in self.model.named_modules():
                if not name.endswith("mlp.gate"):
                    continue
                parent_name = name[:-len(".gate")]
                parent = self.model.get_submodule(parent_name)

                @torch.no_grad()
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

            # Fused MoE calls F.linear with ``gate.weight`` directly, so the
            # gate module hook above never fires. Trace the same routing math
            # from the selected MLP boundary as an implementation-independent
            # fallback. The key is intentionally identical to the eager path.
            parent_name = f"{traced_layer}.mlp"
            parent = self.model.get_submodule(parent_name)

            @torch.no_grad()
            def trace_fused_router(_module, inputs, *, block=parent,
                                   prefix=parent_name):
                if not inputs or f"{prefix}.selected_experts" in trace:
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
                choice = (
                    scores + bias.unsqueeze(0) if bias is not None else scores
                )
                _, selected_experts = torch.topk(
                    choice, block.top_k, dim=-1
                )
                weights = scores.gather(1, selected_experts)
                if getattr(block, "norm_topk_prob", False):
                    weights = weights / (
                        weights.sum(dim=-1, keepdim=True) + 1e-20
                    )
                scale = getattr(block, "routed_scaling_factor", 1.0)
                if scale != 1.0:
                    weights = weights * scale
                trace[f"{prefix}.router_logits"] = logits.detach().float().cpu()
                trace[f"{prefix}.selected_experts"] = selected_experts.detach().cpu()
                trace[f"{prefix}.routing_weights"] = weights.detach().float().cpu()

            handles.append(parent.register_forward_pre_hook(trace_fused_router))

        parameter_names = {
            "model.state_proj.weight",
            "model.state_proj.bias",
            "model.action_in_proj.weight",
            "model.action_in_proj.bias",
            "model.action_time_mlp_in.weight",
            "model.action_time_mlp_in.bias",
            "model.action_time_mlp_out.weight",
            "model.action_time_mlp_out.bias",
            f"{traced_layer}.input_layernorm.weight",
            f"{traced_layer}.post_attention_layernorm.weight",
            f"{traced_layer}.self_attn.q_proj.weight",
        }
        if trace_moe and not moe_activations_only:
            parameter_names.add(f"{traced_layer}.mlp.gate.weight")
            if not router_only:
                parameter_names.update({
                    f"{traced_layer}.mlp.experts.gate_proj",
                    f"{traced_layer}.mlp.experts.up_proj",
                    f"{traced_layer}.mlp.experts.down_proj",
                    f"{traced_layer}.mlp.shared_expert.gate_proj.weight",
                    f"{traced_layer}.mlp.shared_expert.up_proj.weight",
                    f"{traced_layer}.mlp.shared_expert.down_proj.weight",
                })

        def save_parameters(module, _inputs):
            trace_local_experts = (
                trace_moe and not router_only and not moe_activations_only
            )
            for name, parameter in module.named_parameters():
                is_local_expert = bool(
                    re.search(
                        r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$",
                        name,
                    )
                )
                if name in parameter_names or (
                    trace_local_experts and is_local_expert
                ):
                    trace[f"parameter.{name}"] = parameter.detach().float().cpu()

        handles.append(self.model.register_forward_pre_hook(save_parameters))
        return handles, trace, grad_targets

    @staticmethod
    def _register_activation_grad_trace(
        grad_targets: list[tuple[str, Tensor]], *, step: int
    ) -> None:
        root_value = os.environ.get("FLUXVLA_LINGBOT_ACTIVATION_TRACE_DIR")
        if not root_value or not grad_targets:
            return
        keys, tensors = zip(*grad_targets)

        def save_gradients(grads) -> None:
            trace: dict[str, Any] = {"__metadata__": {}}
            for key, grad in zip(keys, grads):
                if grad is None:
                    continue
                detached = grad.detach()
                grad_key = f"{key}.grad"
                trace[grad_key] = detached.float().cpu()
                trace["__metadata__"][grad_key] = {
                    "dtype": str(detached.dtype),
                    "shape": tuple(detached.shape),
                    "stride": tuple(detached.stride()),
                    "contiguous": detached.is_contiguous(),
                }
            root = Path(root_value)
            root.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            torch.save(
                trace,
                root
                / f"flux_step{step:06d}_rank{rank}_activation_grads.pt",
            )

        torch.autograd.graph.register_multi_grad_hook(tensors, save_gradients)

    @staticmethod
    def _save_activation_trace(
        trace: Mapping[str, Tensor], *, step: int = 1
    ) -> None:
        root_value = os.environ.get("FLUXVLA_LINGBOT_ACTIVATION_TRACE_DIR")
        if not root_value or not trace:
            return
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        trace_step = int(os.environ.get(
            "FLUXVLA_LINGBOT_ACTIVATION_TRACE_STEP", "1"))
        filename = (
            f"flux_rank{rank}.pt"
            if trace_step == 1
            else f"flux_step{step:06d}_rank{rank}.pt"
        )
        torch.save(dict(trace), root / filename)

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
        self._save_parameter_identity_trace()

    def _save_parameter_identity_trace(self) -> None:
        """Persist selected post-load parameter identities for backend bisection."""
        root_value = os.environ.get("FLUXVLA_PARAMETER_IDENTITY_TRACE_DIR")
        if not root_value:
            return
        import hashlib
        import json
        import re

        pattern = re.compile(
            os.environ.get(
                "FLUXVLA_PARAMETER_IDENTITY_PATTERN",
                r"qwen_expert\.model\.layers\.(0|35)\.input_layernorm\.(gamma|beta)\.(weight|bias)$",
            )
        )
        rank = int(os.environ.get("RANK", "0"))
        rows = []
        for name, parameter in self.model.named_parameters():
            if not pattern.search(name):
                continue
            value = parameter.detach().float().cpu().contiguous()
            rows.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(parameter.dtype),
                    "sum": float(value.sum()),
                    "abs_max": float(value.abs().max()),
                    "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                }
            )
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"rank_{rank:02d}.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
        )

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
                    tensor = handle.get_tensor(key)
                    mapped = self._expand_fused_expert_tensor(key, tensor, expected)
                    if mapped is None:
                        if key not in expected:
                            # Checkpoints are commonly produced with the
                            # optional post-training depth/video alignment
                            # heads enabled.  A VLA-only parity run disables
                            # those heads, matching native ``post_training``
                            # semantics; their tensors are intentionally
                            # ignored rather than treated as corruption.
                            optional_alignment = (
                                "model.current_video_align_",
                                "model.depth_align_",
                                "model.future_depth_align_",
                                "model.future_video_align_",
                                "model.current_shared_task_proj.",
                                "model.future_shared_task_proj.",
                            )
                            if key.startswith(optional_alignment):
                                continue
                            raise RuntimeError(f"Unexpected LingBot checkpoint key: {key}")
                        mapped = {key: tensor}
                    for target_key, target_tensor in mapped.items():
                        if tuple(target_tensor.shape) != tuple(expected[target_key].shape):
                            raise RuntimeError(
                                f"Shape mismatch for {target_key}: checkpoint={tuple(target_tensor.shape)} "
                                f"model={tuple(expected[target_key].shape)}"
                            )
                        # The official low-memory loader replaces empty
                        # parameters with checkpoint tensors and therefore
                        # preserves their source dtype.  In particular, the
                        # action expert config declares BF16 but its production
                        # checkpoint contains FP32 weights.  Casting to the
                        # meta shell dtype here irreversibly quantizes those
                        # weights before FSDP can apply its runtime policy.
                        shard[target_key] = target_tensor
                        seen.add(target_key)
            # Materialize meta parameters from the current shard. This keeps
            # only one shard plus the destination parameters resident.
            self.model.load_state_dict(shard, strict=False, assign=assign)
        missing = sorted(set(expected) - seen)
        if missing:
            raise RuntimeError(
                f"LingBot checkpoint coverage failure: missing={len(missing)}, "
                f"first_missing={missing[:3]}"
            )

    @staticmethod
    def _expand_fused_expert_tensor(
        key: str, tensor: Tensor, expected: Mapping[str, Tensor]
    ) -> dict[str, Tensor] | None:
        """Expand native fused [E, ...] expert tensors into eager parameters.

        This is a checkpoint-only representation conversion.  No numerical
        transformation is performed: each eager expert receives exactly one
        slice of the native fused tensor, which keeps native and adapter loss
        trajectories comparable when SP/EP requests the safe eager path.
        """
        markers = {
            ".experts.gate_proj": "gate_proj.weight",
            ".experts.up_proj": "up_proj.weight",
            ".experts.down_proj": "down_proj.weight",
        }
        suffix = next((suffix for marker, suffix in markers.items() if marker in key), None)
        if suffix is None or tensor.ndim != 3:
            return None
        prefix = key.split(".experts.", 1)[0]
        result: dict[str, Tensor] = {}
        for expert_index in range(tensor.shape[0]):
            target = f"{prefix}.experts.{expert_index}.{suffix}"
            if target not in expected:
                return None
            result[target] = tensor[expert_index].contiguous()
        return result

    def _load_state_checked(self, state: Mapping[str, Tensor]) -> None:
        # Native LingBot checkpoints are written from ``LingbotVlaV2Policy``;
        # the FluxVLA adapter itself must not add an extra ``model.`` prefix.
        expected = self.model.state_dict()
        expanded: dict[str, Tensor] = {}
        for key, tensor in state.items():
            mapped = self._expand_fused_expert_tensor(key, tensor, expected)
            expanded.update(mapped if mapped is not None else {key: tensor})
        state = expanded
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

    def initialize_distributed_runtime(
        self,
        *,
        dp_size: int | None = None,
        tp_size: int = 1,
        ep_size: int = 1,
        dp_mode: str = "fsdp1",
    ) -> None:
        """Validate the FluxVLA process topology for the native model."""
        if not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        dp_size = world_size if dp_size is None else int(dp_size)
        tp_size = int(tp_size)
        if dp_size * tp_size * int(ep_size) != world_size:
            raise RuntimeError(
                "FluxVLA LingBot topology does not match the process group: "
                f"dp/tp/ep={dp_size}/{tp_size}/{int(ep_size)}, world_size={world_size}."
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

    @staticmethod
    def get_fsdp2_root_output_dtype(param_dtype: torch.dtype) -> torch.dtype:
        """Match the production LingBot root FSDP mixed-precision boundary.

        The official trainer casts the policy's nested loss outputs to the
        root FSDP parameter dtype.  Keeping them in FP32 changes the loss
        values observed by the training loop even though the cast backward is
        an identity operation.
        """
        return param_dtype

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
