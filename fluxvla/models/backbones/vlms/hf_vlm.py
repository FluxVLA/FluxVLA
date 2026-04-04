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

import json
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configs import VLM_BACKBONE_CONFIGS

logger = logging.getLogger(__name__)


class VLMBackbone(nn.Module):

    def __init__(self,
                 vlm_backbone_id: str,
                 vlm_config: Dict,
                 vlm_path: Optional[str] = None,
                 **kwargs):
        super().__init__()
        self.vlm_backbone_id = vlm_backbone_id
        self.vla_pretrained_path = kwargs.pop('vla_pretrained_path', None)
        self.vla_artifacts_dir = self._infer_vla_artifacts_dir(
            self.vla_pretrained_path)
        self.vlm_path = vlm_path  # for saving HF checkpoint (processor, etc.)
        vlm_cls = VLM_BACKBONE_CONFIGS[vlm_backbone_id]['model_cls']
        vlm_cfg = VLM_BACKBONE_CONFIGS[vlm_backbone_id]['config']
        if vlm_config is None:
            if vlm_path is not None:
                if os.path.isdir(vlm_path):
                    cfg_file = os.path.join(vlm_path, 'config.json')
                    if not os.path.isfile(cfg_file):
                        logger.warning(
                            'VLMBackbone: vlm_path=%s has no config.json; '
                            'will try from_pretrained (HF cache or hub).',
                            vlm_path)
                logger.info(
                    'VLMBackbone: loading config and weights from vlm_path=%s',
                    vlm_path)
                vlm_config = vlm_cfg.from_pretrained(vlm_path)
                self.vlm = vlm_cls.from_pretrained(
                    vlm_path, config=vlm_config, **kwargs)
            elif self.vla_pretrained_path is not None:
                cfg_path = None
                if self.vla_artifacts_dir is not None:
                    cfg_path = (
                        self.vla_artifacts_dir / 'vlm_backbone_config.json')
                if cfg_path is None or not cfg_path.is_file():
                    raise OSError(
                        'vlm_path is None and cannot find '
                        'vlm_backbone_config.json from VLA checkpoint '
                        'artifacts.')
                with open(cfg_path, 'r') as f:
                    cfg_dict = json.load(f)
                vlm_config = vlm_cfg(**cfg_dict)
                self.vlm = vlm_cls(config=vlm_config)
                logger.info(
                    'VLMBackbone: built model from VLA config=%s '
                    'and will load VLM weights from checkpoint=%s',
                    str(cfg_path),
                    self.vla_pretrained_path,
                )
            else:
                raise ValueError(
                    'vlm_config/vlm_path cannot both be None when '
                    'vla_pretrained_path is not provided.')
        else:
            vlm_config = vlm_cfg(**vlm_config)
            if vlm_path is not None:
                logger.info('VLMBackbone: loading weights from vlm_path=%s',
                            vlm_path)
                self.vlm = vlm_cls.from_pretrained(
                    vlm_path, config=vlm_config, **kwargs)
            else:
                # Build from config only (e.g., modular vision+LLM loading).
                self.vlm = vlm_cls(config=vlm_config)

        self._load_vlm_weights_from_vla_checkpoint_if_needed()
        self.config = self.vlm.config
        # Sanity check: log config so user can confirm correct model size
        h = getattr(self.config, 'hidden_size', None)
        n = getattr(self.config, 'num_hidden_layers', None)
        if h is None and hasattr(self.config, 'text_config'):
            tc = getattr(self.config, 'text_config', None)
            if tc is not None:
                h = getattr(tc, 'hidden_size', None)
                n = getattr(tc, 'num_hidden_layers', None)
        if h is not None or n is not None:
            logger.info(
                'VLMBackbone: loaded hidden_size=%s num_hidden_layers=%s',
                h if h is not None else '?', n if n is not None else '?')

    @staticmethod
    def _infer_vla_artifacts_dir(
            vla_pretrained_path: Optional[str]) -> Optional[Path]:
        if vla_pretrained_path is None:
            return None
        path = Path(vla_pretrained_path)
        if path.is_file():
            # Typical: <run_dir>/checkpoints/step-xxxx.pt
            if path.parent.name == 'checkpoints':
                return path.parent.parent
            return path.parent
        if path.is_dir():
            if (path / 'vlm_backbone_config.json').is_file():
                return path
            if path.name == 'checkpoints':
                return path.parent
        return None

    @staticmethod
    def _unwrap_checkpoint_dict(
            checkpoint_obj) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(checkpoint_obj, dict):
            for key in ('model', 'state_dict', 'module'):
                nested = checkpoint_obj.get(key, None)
                if isinstance(nested, dict):
                    checkpoint_obj = nested
                    break
        if not isinstance(checkpoint_obj, dict):
            return None
        return checkpoint_obj

    @classmethod
    def _load_checkpoint_state_dict(
            cls, ckpt_path: str) -> Optional[Dict[str, torch.Tensor]]:
        if ckpt_path is None:
            return None
        if os.path.isdir(ckpt_path):
            state_dict = {}
            safetensor_files = sorted(
                f for f in os.listdir(ckpt_path) if f.endswith('.safetensors'))
            if safetensor_files:
                for f in safetensor_files:
                    state_dict.update(
                        load_file(os.path.join(ckpt_path, f), device='cpu'))
                return state_dict
            pt_files = sorted(
                f for f in os.listdir(ckpt_path)
                if f.endswith('.pt') or f.endswith('.pth'))
            if pt_files:
                checkpoint = torch.load(
                    os.path.join(ckpt_path, pt_files[0]), map_location='cpu')
                return cls._unwrap_checkpoint_dict(checkpoint)
            return None

        if ckpt_path.endswith('.safetensors'):
            return load_file(ckpt_path, device='cpu')
        if ckpt_path.endswith('.pt') or ckpt_path.endswith('.pth'):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            return cls._unwrap_checkpoint_dict(checkpoint)
        return None

    def _load_vlm_weights_from_vla_checkpoint_if_needed(self) -> None:
        if self.vla_pretrained_path is None:
            return
        state_dict = self._load_checkpoint_state_dict(self.vla_pretrained_path)
        if state_dict is None:
            logger.warning(
                'VLMBackbone: skip VLA load, unsupported or empty '
                'checkpoint: %s', self.vla_pretrained_path)
            return

        prefixes = (
            'vlm_backbone.vlm.',
            '_fsdp_wrapped_module.vlm_backbone.vlm.',
            'module.vlm_backbone.vlm.',
            '_fsdp_wrapped_module.module.vlm_backbone.vlm.',
        )
        vlm_state = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    vlm_state[key[len(prefix):]] = value
                    break

        if not vlm_state:
            current_keys = self.vlm.state_dict().keys()
            overlap = sum(1 for k in state_dict.keys() if k in current_keys)
            if overlap > 0:
                vlm_state = state_dict

        if not vlm_state:
            logger.warning(
                'VLMBackbone: checkpoint=%s has no VLM-prefixed weights.',
                self.vla_pretrained_path)
            return

        incompat = self.vlm.load_state_dict(vlm_state, strict=False)
        logger.info(
            'VLMBackbone: loaded VLM weights from VLA checkpoint=%s '
            '(tensors=%d, missing=%d, unexpected=%d)',
            self.vla_pretrained_path,
            len(vlm_state),
            len(incompat.missing_keys),
            len(incompat.unexpected_keys),
        )

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns a function used to determine which modules to wrap with FSDP.

        Returns:
            Callable: Wrapping policy function.
        """
        ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """
        Enables gradient checkpointing on the LLM model to save memory.
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the model.

        Args:
            input_ids (Optional[torch.LongTensor]): Token IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            position_ids (Optional[torch.LongTensor]): Position indices.
            past_key_values (Optional[List[torch.FloatTensor]]):
                Cached KV pairs.
            inputs_embeds (Optional[torch.FloatTensor]): Embedded inputs.
            labels (Optional[torch.LongTensor]): Labels for loss computation.
            use_cache (Optional[bool]): Whether to use cache.
            output_attentions (Optional[bool]): Return attention scores.
            output_hidden_states (Optional[bool]): Return hidden states.
            return_dict (Optional[bool]): Return output as dict.

        Returns:
            CausalLMOutputWithPast: Model output.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Embeds input token IDs using the model's embedding layer.

        Args:
            input_ids (torch.LongTensor): Input token IDs.

        Returns:
            torch.Tensor: Embedded token vectors.
        """
        ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> Type[nn.Module]:
        """
        Returns the class of the transformer's basic layer.

        Returns:
            Type[nn.Module]: Transformer block class.
        """
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        """
        Returns the appropriate dtype for half precision (e.g., bf16 or fp16).

        Returns:
            torch.dtype: Half precision type.
        """
        ...

    @property
    @abstractmethod
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        """
        Returns a sequence of modules used in last-layer finetuning.

        Returns:
            Sequence[nn.Module]: Modules to be fine-tuned.
        """
        ...

    @property
    def embed_dim(self) -> int:
        """
        Returns the model's hidden size (embedding dimension).

        Returns:
            int: Embedding dimension.
        """
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        """
        Returns the pad token ID used by the tokenizer.

        Returns:
            int: Pad token ID.
        """
        return self.tokenizer.pad_token_id
