"""Checkpoint contract owned by the FluxVLA-native LingBot model."""

from __future__ import annotations

from typing import Optional


class LingBotVLAWeightLoader:
    """Map official LingBot checkpoint keys to the local model namespace."""

    def get_vlm_submodule(self, model):
        return model.model.qwenvl_with_expert.qwenvl

    def get_expert_vision_submodule(self, model):
        return getattr(model.model.qwenvl_with_expert, "expert_visual", None)

    def get_submodule_prefix(self, model, submodule) -> str:
        for name, module in model.named_modules():
            if module is submodule:
                return name
        raise ValueError(f"Submodule {type(submodule).__name__} not found")

    def map_ckpt_key(
        self, key: str, load_vlm_only: bool, post_training: bool
    ) -> Optional[str]:
        if key.startswith("expert_visual.") and not post_training:
            return "model.qwenvl_with_expert." + key
        if load_vlm_only:
            return "model.qwenvl_with_expert.qwenvl." + key
        return key
