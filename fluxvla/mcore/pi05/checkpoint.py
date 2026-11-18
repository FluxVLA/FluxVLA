"""PI0.5-specific HF/OpenPI to native MCore checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch


def _match_source_parameter_load_dtype(name: str, source: torch.Tensor) -> torch.Tensor:
    """Reproduce source-module dtype at checkpoint assignment time.

    FluxVLA's unconditioned ``GemmaRMSNorm.weight`` parameters are constructed
    as BF16.  The FSDP runner later promotes them to FP32, but it cannot recover
    bits lost when the checkpoint was first copied into those BF16 tensors.
    Native MCore norms are constructed in FP32, so explicitly reproduce that
    assignment boundary before copying into the runtime parameter.
    """
    is_vlm_norm = (
        name == "vlm_final_norm.norm_weight"
        or (
            name.startswith("layers.")
            and (".vlm_input_norm.norm_weight" in name
                 or ".vlm_post_attn_norm.norm_weight" in name)
        )
    )
    if is_vlm_norm:
        return source.to(torch.bfloat16).float()
    return source


def _source_key(name: str) -> str | None:
    # PI0.5 suffix/action projectors are top-level tensors in the HF/OpenPI
    # checkpoint, while the native provider owns them under suffix_embedder.
    if name.startswith("suffix_embedder.action_in_proj."):
        return "action_in_proj." + name[len("suffix_embedder.action_in_proj."):]
    if name.startswith("suffix_embedder.time_mlp_in."):
        return "time_mlp_in." + name[len("suffix_embedder.time_mlp_in."):]
    if name.startswith("suffix_embedder.time_mlp_out."):
        return "time_mlp_out." + name[len("suffix_embedder.time_mlp_out."):]
    # adaRMS conditioning projections use the expert's dense conditioner in
    # the source checkpoint.  The native layer exposes it as cond_proj.
    parts = name.split(".")
    if len(parts) >= 5 and parts[0] == "layers" and parts[2] in {
        "exp_input_norm", "exp_post_attn_norm"
    } and parts[3] == "cond_proj":
        layer, norm = parts[1], parts[2]
        source_norm = (
            "input_layernorm" if norm == "exp_input_norm"
            else "post_attention_layernorm"
        )
        return (
            "paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.{source_norm}.dense.{parts[4]}"
        )
    if name == "exp_final_norm.cond_proj.weight":
        return "paligemma_with_expert.gemma_expert.model.norm.dense.weight"
    if name == "exp_final_norm.cond_proj.bias":
        return "paligemma_with_expert.gemma_expert.model.norm.dense.bias"
    if name == "vlm_final_norm.norm_weight":
        return "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    if name == "prefix_embedder._vlm_model.language_model.embed_tokens.weight":
        return "paligemma_with_expert.paligemma.lm_head.weight"
    if name.startswith("prefix_embedder._vlm_model."):
        return "paligemma_with_expert.paligemma.model." + name[len("prefix_embedder._vlm_model."):]
    parts = name.split(".")
    if len(parts) >= 5 and parts[0] == "layers" and parts[2] == "attn":
        layer, branch = parts[1], parts[3]
        branch_root = "paligemma_with_expert.paligemma" if branch.startswith("vlm_") else "paligemma_with_expert.gemma_expert"
        proj_name = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}[branch[-1]]
        return f"{branch_root}.model.{'language_model.' if branch.startswith('vlm_') else ''}layers.{layer}.self_attn.{proj_name}.weight"
    if len(parts) >= 5 and parts[0] == "layers" and parts[2] in {"vlm_mlp", "exp_mlp"}:
        layer, branch, proj = parts[1], parts[2], parts[3]
        root = "paligemma_with_expert.paligemma.model.language_model" if branch == "vlm_mlp" else "paligemma_with_expert.gemma_expert.model"
        return f"{root}.layers.{layer}.mlp.{proj}.weight"
    if len(parts) >= 4 and parts[0] == "layers" and parts[2] in {"vlm_input_norm", "vlm_post_attn_norm"}:
        layer, norm = parts[1], parts[2]
        suffix = "input_layernorm" if norm == "vlm_input_norm" else "post_attention_layernorm"
        return f"paligemma_with_expert.paligemma.model.language_model.layers.{layer}.{suffix}.weight"
    if name.startswith("action_out_proj."):
        return name
    if name.startswith("suffix_embedder."):
        return name
    return None


def load_pi05_checkpoint(model, path: str) -> dict[str, int]:
    """Load matching tensors and TP-slice MCore parameters.

    Unmatched parameters are reported rather than silently reshaped.  This is
    intentionally conservative for the first native backend milestone.
    """
    try:
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
    except ImportError:
        state = torch.load(path, map_location="cpu", weights_only=True)
    tp_rank = 0
    tp_size = 1
    try:
        from megatron.core import parallel_state
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    except Exception:
        pass
    loaded = skipped = 0
    skipped_keys = []
    skip_reasons = {}
    for name, parameter in model.named_parameters():
        key = _source_key(name)
        if key is None or key not in state:
            skipped += 1
            skipped_keys.append(name)
            skip_reasons[name] = "unmapped" if key is None else "source_missing"
            continue
        source = _match_source_parameter_load_dtype(name, state[key]).to(
            dtype=parameter.dtype
        )
        if source.shape == parameter.shape:
            value = source
        elif source.ndim == parameter.ndim == 2 and source.shape[0] == parameter.shape[0] * tp_size:
            value = source[tp_rank * parameter.shape[0]:(tp_rank + 1) * parameter.shape[0]]
        elif source.ndim == parameter.ndim == 2 and source.shape[1] == parameter.shape[1] * tp_size:
            value = source[:, tp_rank * parameter.shape[1]:(tp_rank + 1) * parameter.shape[1]]
        else:
            skipped += 1
            skipped_keys.append(name)
            skip_reasons[name] = (
                f"shape_mismatch:model={tuple(parameter.shape)},source={tuple(source.shape)}"
            )
            continue
        with torch.no_grad():
            parameter.copy_(value)
        loaded += 1
    return {
        "loaded": loaded,
        "skipped": skipped,
        "source_tensors": len(state),
        "skipped_keys": skipped_keys,
        "skip_reasons": skip_reasons,
    }
