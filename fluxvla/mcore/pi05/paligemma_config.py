# Copyright 2025 VLASH team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Local PaliGemma configuration for PI0 and PI0.5.

The architecture values originate from the Apache-2.0 VLASH PI configuration
modules. Keeping the model-only subset here avoids loading VLASH at runtime.
"""

from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig


def build_pi0_paligemma_config() -> PaliGemmaConfig:
    """Return the PaliGemma architecture expected by PI0/PI0.5 checkpoints."""
    config = PaliGemmaConfig()
    config._vocab_size = 257152
    config.image_token_index = 257152

    text_config = config.text_config
    text_config.hidden_size = 2048
    text_config.intermediate_size = 16384
    text_config.num_attention_heads = 8
    text_config.head_dim = 256
    text_config.num_hidden_layers = 18
    text_config.num_key_value_heads = 1
    text_config.hidden_activation = "gelu_pytorch_tanh"
    text_config.torch_dtype = "float32"
    text_config.vocab_size = 257152
    text_config.use_adarms = False
    text_config.adarms_cond_dim = None

    vision_config = config.vision_config
    vision_config.intermediate_size = 4304
    vision_config.projection_dim = 2048
    # The reference PI0.5 policy configures SigLIP's own MLP activation.
    # ``projector_hidden_act`` is not consumed by the vision encoder, so using
    # it here silently left the tower on the Transformers default activation.
    vision_config.hidden_activation = "gelu_fast"
    vision_config.torch_dtype = "float32"
    return config
