"""Opt-in tensor capture for cross-backend SigLIP alignment."""

from pathlib import Path

import torch
import torch.nn as nn


def trace_siglip_layers(vision_tower: nn.Module, trace_dir: str, prefix: str):
    """Register hooks for one first-image SigLIP forward."""
    targets = {
        "embeddings": "vision_model.embeddings",
        "layer_00_norm1": "vision_model.encoder.layers.0.layer_norm1",
        "layer_00_q_proj": "vision_model.encoder.layers.0.self_attn.q_proj",
        "layer_00_k_proj": "vision_model.encoder.layers.0.self_attn.k_proj",
        "layer_00_v_proj": "vision_model.encoder.layers.0.self_attn.v_proj",
        "layer_00_attention": "vision_model.encoder.layers.0.self_attn",
        "layer_00_norm2": "vision_model.encoder.layers.0.layer_norm2",
        "layer_00_fc1": "vision_model.encoder.layers.0.mlp.fc1",
        "layer_00_fc2": "vision_model.encoder.layers.0.mlp.fc2",
        "layer_00": "vision_model.encoder.layers.0",
        "layer_01": "vision_model.encoder.layers.1",
        "layer_13": "vision_model.encoder.layers.13",
        "layer_26": "vision_model.encoder.layers.26",
        "post_layernorm": "vision_model.post_layernorm",
    }
    modules = dict(vision_tower.named_modules())
    output_dir = Path(trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captured = set()
    handles = []

    def save_output(name):
        def hook(_module, _inputs, output):
            if name in captured:
                return
            value = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(value, torch.Tensor):
                torch.save(value[:1].detach().cpu(),
                           output_dir / f"{prefix}_{name}.pt")
                captured.add(name)

        return hook

    for label, module_name in targets.items():
        module = modules.get(module_name)
        if module is not None:
            handles.append(module.register_forward_hook(save_output(label)))
    return handles
