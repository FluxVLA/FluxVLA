# Copyright 2026 Limx Dynamics
"""Native Hugging Face causal VLM, including Qwen image/video SFT inputs."""

from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText

from fluxvla.engines import VLM_BACKBONES
from fluxvla.engines.losses.causal_lm import causal_lm_loss
from fluxvla.engines.utils.fsdp_wrapping import \
    build_transformer_layer_wrap_policy
from fluxvla.engines.utils.name_map import str_to_dtype
from .hf_vlm import validate_attn_implementation


@VLM_BACKBONES.register_module()
class CausalVLMBackbone(nn.Module):
    """Keep the complete native generation model and its vocabulary head.

    ``model_path`` loads pretrained HF weights. ``model_config`` constructs
    an architecture without downloading weights (also useful for tiny tests).
    Native input fields, including video grids and multimodal token types,
    are forwarded unchanged. Only decoder-only generation is supported.
    """

    supports_vlm_training = True

    def __init__(self,
                 model_path=None,
                 model_config=None,
                 attn_implementation='sdpa',
                 torch_dtype='fp32',
                 freeze_vision=False,
                 trust_remote_code=False):
        super().__init__()
        attn = validate_attn_implementation(attn_implementation)
        dtype = str_to_dtype(torch_dtype)
        if model_config is not None:
            config = dict(model_config)
            config = AutoConfig.for_model(**config)
        elif model_path:
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        else:
            raise ValueError('Provide model_path or model_config.')
        if config.is_encoder_decoder:
            raise ValueError('CausalVLMBackbone requires a decoder-only VLM.')
        kwargs = dict(
            config=config,
            attn_implementation=attn,
            trust_remote_code=trust_remote_code)
        if model_path:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=dtype, **kwargs)
        else:
            self.vlm = AutoModelForImageTextToText.from_config(**kwargs)
            self.vlm.to(dtype=dtype)
        self.freeze_vision = freeze_vision

    @property
    def config(self):
        return self.vlm.config

    def forward(self, *, task='vlm', ignore_index=-100, **batch):
        if task != 'vlm':
            raise ValueError('CausalVLMBackbone currently supports task=vlm.')
        return self.forward_vlm(ignore_index=ignore_index, **batch)

    def forward_vlm(self,
                    *,
                    labels,
                    input_ids,
                    attention_mask=None,
                    ignore_index=-100,
                    **inputs):
        """Use native multimodal fusion and return causal language loss."""
        if labels.shape != input_ids.shape:
            raise ValueError('VLM labels must match input_ids.')
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask.bool(), ignore_index)
        # Compute CE only in causal_lm_loss, retaining its token-sum output.
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
            **inputs)
        return causal_lm_loss(outputs.logits, labels, ignore_index)

    def apply_trainable_policy(self):
        if self.freeze_vision:
            visual = getattr(self.vlm.model, 'visual', None)
            if visual is None:
                raise ValueError('freeze_vision requires a native visual '
                                 'module; configure this model explicitly.')
            visual.requires_grad_(False)

    def enable_gradient_checkpointing(self):
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

    def prepare_vlm_training(self):
        self.apply_trainable_policy()

    def get_fsdp_wrapping_policy(self):
        layer_names = set(self.vlm._no_split_modules or [])
        # Vision is conditional on the rank's batch. Keep it in the root
        # FSDP unit so text-only ranks do not skip vision all-gathers.
        language = getattr(self.vlm.model, 'language_model', None)
        if language is None:
            raise ValueError('Cannot locate native language layers; '
                             'use fsdp_wrap_policy="root".')
        layer_classes = {
            type(module)
            for module in language.modules()
            if type(module).__name__ in layer_names
        }
        if not layer_classes:
            raise ValueError('Native VLM does not declare FSDP layer classes; '
                             'use fsdp_wrap_policy="root".')
        return build_transformer_layer_wrap_policy(layer_classes)
