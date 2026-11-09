"""PI0.5-specific ordering contracts used by numerical diagnostics."""

from __future__ import annotations

import re

_LAYER_PATTERN = re.compile(r"^layers\.(\d+)\.(.+)$")
_PROJECTION_ORDER = {"q": 0, "k": 1, "v": 2, "o": 3}
_MLP_ORDER = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}


def fsdp_grad_norm_order(name: str) -> tuple:
    """Return the equivalent FluxVLA FSDP parameter-registration order."""
    if name.startswith("prefix_embedder._vlm_model.vision_tower."):
        return (0,)
    if name == "prefix_embedder._vlm_model.language_model.embed_tokens.weight":
        return (1,)

    match = _LAYER_PATTERN.match(name)
    if match:
        layer = int(match.group(1))
        component = match.group(2)
        branch = 2 if component.startswith("vlm_") or ".vlm_" in component else 10
        if component.startswith("attn."):
            projection = (
                component.split(".")[1].removeprefix("vlm_").removeprefix("exp_")
            )
            return (branch, layer, 0, _PROJECTION_ORDER[projection])
        if component.startswith(("vlm_mlp.", "exp_mlp.")):
            projection = component.split(".")[1]
            return (branch, layer, 1, _MLP_ORDER[projection])
        if component.startswith(("vlm_input_norm.", "exp_input_norm.")):
            suffix = component.rsplit(".", 1)[-1]
            return (branch, layer, 2, 0 if suffix in {"norm_weight", "weight"} else 1)
        if component.startswith(("vlm_post_attn_norm.", "exp_post_attn_norm.")):
            suffix = component.rsplit(".", 1)[-1]
            return (branch, layer, 3, 0 if suffix in {"norm_weight", "weight"} else 1)

    fixed_prefixes = (
        ("vlm_final_norm.", 3),
        ("prefix_embedder._vlm_model.multi_modal_projector.linear.", 4),
        ("suffix_embedder.action_in_proj.", 5),
        ("action_out_proj.", 6),
        ("suffix_embedder.time_mlp_in.", 7),
        ("suffix_embedder.time_mlp_out.", 8),
        ("exp_final_norm.", 11),
    )
    for prefix, group in fixed_prefixes:
        if name.startswith(prefix):
            return (group,)
    raise ValueError(f"No FSDP gradient-order mapping for PI0.5 parameter: {name}")


# FluxVLA owns an unused expert embedding parameter whose gradient is zero.
# Native PI0.5 intentionally omits that tensor, but the zero participates in
# FSDP's stacked per-parameter norm reduction and therefore preserves its slot.
FSDP_GRAD_NORM_ZERO_KEYS = ((9,),)
FSDP_GRAD_REDUCE_ZERO_SEGMENTS = (((9,), 257152 * 1024),)


__all__ = [
    "FSDP_GRAD_NORM_ZERO_KEYS",
    "FSDP_GRAD_REDUCE_ZERO_SEGMENTS",
    "fsdp_grad_norm_order",
]
