# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Type

import torch
from safetensors.torch import load_file
from torch import nn
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLPreTrainedModel,
    Qwen3VLTextModel, Qwen3VLTextRMSNorm, Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel)

from fluxvla.engines import VLM_BACKBONES, initialize_overwatch
from ....third_party_models.cosmos3.data.vfm.sequence_packing import (
    FactoredSequencePack, from_joint, get_device_and_dtype, get_gen_seq,
    get_und_seq, set_gen_seq, set_und_seq, zeros_like)
from ....third_party_models.cosmos3.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding, Nemotron3DenseVLMLP, Nemotron3DenseVLRMSNorm,
    Nemotron3DenseVLTextConfig, apply_rotary_pos_emb_partial)
from .cosmos3_mot_layer import \
    Cosmos3TextDecoderLayer as _Cosmos3TextDecoderLayer

overwatch = initialize_overwatch(__name__)


class _Cosmos3ConfigAdapter:
    """Private config adapter for the registered FluxVLA Cosmos3 backbone."""

    def __init__(
        self,
        config_dict: Mapping[str, Any],
        *,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        include_visual: bool = False,
        packed_attention_backend: str = 'flash2',
        text_config_overrides: Mapping[str, Any] | None = None,
    ):
        self.config_dict = dict(config_dict)
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.include_visual = include_visual
        self.packed_attention_backend = packed_attention_backend
        self.text_config_overrides: dict[str, Any] = dict(
            text_config_overrides) if text_config_overrides else {}

    @property
    def full_config(self) -> Qwen3VLConfig:
        return Qwen3VLConfig(**self.config_dict)

    @property
    def text_config(self) -> Qwen3VLTextConfig:
        nested = self.config_dict.get('text_config')
        text_dict = nested if isinstance(nested, dict) else self.config_dict
        text_dict = {
            **text_dict, 'packed_attention_backend':
            self.packed_attention_backend
        }
        overrides = getattr(self, 'text_config_overrides', None) or {}
        if overrides:
            text_dict = {**text_dict, **overrides}
        return Qwen3VLTextConfig(**text_dict)

    @property
    def vision_config(self) -> Qwen3VLVisionConfig | None:
        if not self.include_visual:
            return None
        vision_dict = self.config_dict.get('vision_config')
        if vision_dict is None:
            raise ValueError(
                'include_visual=True requires a vision_config sub-section in '
                'the language-model JSON config.')
        return Qwen3VLVisionConfig(**vision_dict)


class _Cosmos3TextModel(Qwen3VLTextModel):
    """Qwen3-VL text tower with Cosmos3 MoT decoder layers."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            _Cosmos3TextDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                qk_norm_for_text=qk_norm_for_text,
                qk_norm_for_diffusion=qk_norm_for_diffusion,
            ) for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward_packed(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
    ) -> tuple[FactoredSequencePack, dict[str, Any]]:
        """Forward Cosmos3 packed understanding/generation tokens."""
        device, dtype = get_device_and_dtype(pack)
        meta_tensor = torch.tensor([], dtype=dtype, device=device)
        cos, sin = self.rotary_emb(
            meta_tensor,
            position_ids=position_ids.unsqueeze(0)
            if position_ids.ndim == 1 else position_ids.unsqueeze(1),
        )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        position_embeddings = (from_joint(cos, pack), from_joint(sin, pack))

        hidden_states = pack
        for decoder_layer in self.layers:
            # Use ``__call__`` so FSDP-wrapped layers all-gather parameters.
            hidden_states = decoder_layer(
                packed_sequence=hidden_states,
                packed_position_embeddings=position_embeddings,
                packed_attention_mask=attention_mask,
            )

        hidden_states_out = zeros_like(hidden_states)
        set_und_seq(hidden_states_out, self.norm(get_und_seq(hidden_states)))
        set_gen_seq(hidden_states_out,
                    self.norm_moe_gen(get_gen_seq(hidden_states)))
        return hidden_states_out, {}

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value


class _Cosmos3NemotronTextDecoderLayer(_Cosmos3TextDecoderLayer):
    """Nemotron dense dual-pathway MoT layer selected by model config."""

    def __init__(
        self,
        config: Nemotron3DenseVLTextConfig,
        layer_idx: int,
        *,
        qk_norm_for_text: bool = False,
        qk_norm_for_diffusion: bool = True,
    ) -> None:
        super().__init__(
            config,
            layer_idx,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            mlp_cls=Nemotron3DenseVLMLP,
            rms_norm_cls=Nemotron3DenseVLRMSNorm,
            apply_rotary_pos_emb=apply_rotary_pos_emb_partial,
        )


class _Cosmos3NemotronTextModel(nn.Module):

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            _Cosmos3NemotronTextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Nemotron3DenseVLRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Nemotron3DenseVLRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MultiModalRotaryEmbedding(config)

    def forward_packed(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
    ) -> tuple[FactoredSequencePack, dict[str, Any]]:
        device, dtype = get_device_and_dtype(pack)
        meta_tensor = torch.tensor([], dtype=dtype, device=device)
        rope_ids = (
            position_ids.unsqueeze(0)
            if position_ids.ndim == 1 else position_ids.unsqueeze(1))
        cos, sin = self.rotary_emb(meta_tensor, position_ids=rope_ids)
        position_embeddings = (from_joint(cos.squeeze(0), pack),
                               from_joint(sin.squeeze(0), pack))

        hidden_states = pack
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                packed_sequence=hidden_states,
                packed_position_embeddings=position_embeddings,
                packed_attention_mask=attention_mask,
            )

        output = zeros_like(hidden_states)
        set_und_seq(output, self.norm(get_und_seq(hidden_states)))
        set_gen_seq(output, self.norm_moe_gen(get_gen_seq(hidden_states)))
        return output, {}


class _Cosmos3Model(Qwen3VLModel):
    """Qwen3-VL multimodal model with Cosmos3 MoT text layers."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        include_visual: bool,
    ):
        Qwen3VLPreTrainedModel.__init__(self, config)
        if include_visual:
            self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = _Cosmos3TextModel(
            config.text_config,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
        )
        self.rope_deltas = None
        self.post_init()


class _Cosmos3NemotronModel(nn.Module):

    architecture_family = 'edge_nemotron'
    transformer_layer_cls = _Cosmos3NemotronTextDecoderLayer

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.language_model = _Cosmos3NemotronTextModel(config)

    @staticmethod
    def matches_config(config_dict: Mapping[str, Any]) -> bool:
        nested = config_dict.get('text_config')
        text_config = nested if isinstance(nested, Mapping) else config_dict
        return (text_config.get('model_type') ==
                Nemotron3DenseVLTextConfig.model_type)

    @classmethod
    def build_backbone_components(
        cls,
        config_dict: Mapping[str, Any],
        *,
        include_visual: bool,
        packed_attention_backend: str,
        text_config_overrides: Mapping[str, Any] | None,
        vision_encoder_path: str | Path | None,
        skip_init_weights: bool,
    ) -> tuple[_Cosmos3NemotronModel, nn.Linear]:
        if include_visual:
            raise NotImplementedError(
                'Cosmos3-Edge currently supports the Nemotron generator '
                'tower without the SigLIP2 reasoner.')
        if vision_encoder_path is not None:
            raise ValueError(
                'vision_encoder_path is only valid with include_visual=True.')

        nested = config_dict.get('text_config')
        text_dict = dict(nested) if isinstance(nested,
                                               Mapping) else dict(config_dict)
        text_dict['packed_attention_backend'] = packed_attention_backend
        if text_config_overrides:
            text_dict.update(dict(text_config_overrides))
        if text_dict.get('num_hidden_layers') == 56:
            text_dict['num_hidden_layers'] = 28
        text_dict['tie_word_embeddings'] = False
        text_config = Nemotron3DenseVLTextConfig(**text_dict)

        parameter_dtype = cls._parameter_dtype(text_config.torch_dtype)
        with cls._temporary_default_dtype(parameter_dtype):
            if skip_init_weights:
                with no_init_weights():
                    return cls._build_components(text_config)
            return cls._build_components(text_config)

    @classmethod
    def _build_components(
        cls, text_config: Nemotron3DenseVLTextConfig
    ) -> tuple[_Cosmos3NemotronModel, nn.Linear]:
        model = cls(text_config)
        lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False)
        return model, lm_head

    @staticmethod
    def _parameter_dtype(value: str | torch.dtype | None) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if value is None:
            return torch.get_default_dtype()
        names = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        name = str(value).lower()
        if name not in names:
            raise ValueError(f'Unsupported parameter dtype: {value!r}.')
        return names[name]

    @staticmethod
    @contextmanager
    def _temporary_default_dtype(dtype: torch.dtype):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


@contextmanager
def _cosmos3_no_init_weights():
    original_qwen_init_weights = Qwen3VLPreTrainedModel.init_weights

    def empty_init_weights(*args, **kwargs):
        return None

    Qwen3VLPreTrainedModel.init_weights = empty_init_weights
    try:
        with no_init_weights():
            yield
    finally:
        Qwen3VLPreTrainedModel.init_weights = original_qwen_init_weights


@VLM_BACKBONES.register_module()
class Cosmos3MoTBackbone(Qwen3VLForConditionalGeneration):
    """FluxVLA Cosmos3 MoT tower selected by ``vlm_config.model_type``."""

    _tied_weights_keys = ['lm_head.weight']

    def __init__(
        self,
        vlm_config: Mapping[str, Any] | str,
        *,
        include_visual: bool = False,
        packed_attention_backend: str = 'flash2',
        text_config_overrides: Mapping[str, Any] | None = None,
        vision_encoder_path: str | Path | None = None,
        skip_init_weights: bool = False,
    ) -> None:
        config_dict = self._load_vlm_config(vlm_config)
        is_nemotron = _Cosmos3NemotronModel.matches_config(config_dict)
        if is_nemotron:
            nn.Module.__init__(self)
            self.model, self.lm_head = (
                _Cosmos3NemotronModel.build_backbone_components(
                    config_dict,
                    include_visual=include_visual,
                    packed_attention_backend=packed_attention_backend,
                    text_config_overrides=text_config_overrides,
                    vision_encoder_path=vision_encoder_path,
                    skip_init_weights=skip_init_weights,
                ))
            self.config = self.model.config
            self.architecture_family = self.model.architecture_family
            self._transformer_layer_cls = self.model.transformer_layer_cls
        else:
            backbone_config = self._build_backbone_config(
                vlm_config=config_dict,
                include_visual=include_visual,
                packed_attention_backend=packed_attention_backend,
                text_config_overrides=text_config_overrides,
            )
            if skip_init_weights:
                with _cosmos3_no_init_weights():
                    self._init_from_backbone_config(backbone_config)
            else:
                self._init_from_backbone_config(backbone_config)
            self.architecture_family = 'qwen3_vl'
            self._transformer_layer_cls = _Cosmos3TextDecoderLayer

        self.vlm_config_dict = config_dict
        self.include_visual = include_visual
        self.packed_attention_backend = packed_attention_backend
        self.text_config_overrides = self._build_text_config_overrides(
            text_config_overrides)
        self.vision_encoder_path = vision_encoder_path
        if (not is_nemotron and include_visual
                and vision_encoder_path is not None):
            self.load_visual_encoder_pretrained(vision_encoder_path)

    def load_visual_encoder_pretrained(self, path: str | Path) -> None:
        visual = getattr(self.model, 'visual', None)
        if visual is None:
            raise ValueError(
                'Cosmos3 visual encoder weights require include_visual=True.')

        path = Path(path)
        checkpoint_files = [path] if path.is_file() else sorted(
            path.glob('*.safetensors'))
        if not checkpoint_files:
            raise FileNotFoundError(
                f'No visual encoder safetensors found at {path}.')

        state_dict = {}
        for checkpoint_file in checkpoint_files:
            state_dict.update(load_file(str(checkpoint_file), device='cpu'))

        missing, unexpected = visual.load_state_dict(state_dict, strict=False)
        if missing:
            overwatch.info(f'Cosmos3 visual encoder missing keys: {missing}')
        if unexpected:
            overwatch.info(
                f'Cosmos3 visual encoder unexpected keys: {unexpected}')

    @staticmethod
    def _load_vlm_config(
            vlm_config: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(vlm_config, (str, Path)):
            with open(vlm_config, encoding='utf-8') as reader:
                return json.load(reader)
        return copy.deepcopy(dict(vlm_config))

    @staticmethod
    def _build_text_config_overrides(
        text_config_overrides: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        overrides = {'tie_word_embeddings': False}
        if text_config_overrides:
            overrides.update(dict(text_config_overrides))
        return overrides

    @classmethod
    def _build_backbone_config(
        cls,
        vlm_config: Mapping[str, Any] | str,
        *,
        include_visual: bool = False,
        packed_attention_backend: str = 'flash2',
        text_config_overrides: Mapping[str, Any] | None = None,
    ):
        return _Cosmos3ConfigAdapter(
            config_dict=cls._load_vlm_config(vlm_config),
            include_visual=include_visual,
            packed_attention_backend=packed_attention_backend,
            text_config_overrides=cls._build_text_config_overrides(
                text_config_overrides),
        )

    def _init_from_backbone_config(
            self, backbone_config: _Cosmos3ConfigAdapter) -> None:
        full_config = backbone_config.full_config
        text_config = backbone_config.text_config
        full_config.text_config = text_config
        Qwen3VLPreTrainedModel.__init__(self, full_config)

        self.model = _Cosmos3Model(
            full_config,
            qk_norm_for_text=backbone_config.qk_norm_for_text,
            qk_norm_for_diffusion=backbone_config.qk_norm_for_diffusion,
            include_visual=backbone_config.include_visual,
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    @property
    def text_config(self):
        return self.model.language_model.config

    @property
    def decoder_layers(self):
        return self.model.language_model.layers

    @property
    def token_embedding(self):
        return self.model.language_model.embed_tokens

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return self._transformer_layer_cls

    def embed_text_ids(self, text_ids):
        return self.token_embedding(text_ids)

    def forward_packed(self, pack, attention_mask, position_ids):
        """Forward Cosmos3 packed und/gen tokens."""
        return self.model.language_model.forward_packed(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def fsdp_transformer_layer_cls(self):
        return {self.transformer_layer_cls}

    def forward(self, *args, **kwargs):
        if self.architecture_family == 'edge_nemotron':
            raise NotImplementedError(
                'Edge Nemotron backbone only supports packed forward; use '
                'forward_packed().')
        return super().forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if self.architecture_family == 'edge_nemotron':
            raise NotImplementedError(
                'Autoregressive Edge reasoner generation is not supported.')
        return super().generate(*args, **kwargs)


__all__ = [
    'Cosmos3MoTBackbone',
]
