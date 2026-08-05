# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""FluxVLA-native GR00T N1.7 processor.

This module mirrors the official N1.7 processor contract without importing the
official ``gr00t`` package. It is intentionally scoped to processor/collator
semantics; model assembly remains in later native port stages.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import re
from typing import Any, Dict, Optional

import numpy as np
import torch

from fluxvla.collators.qwen_vl_action_prediction_collator import (
    GrootN17DataCollator,
    build_qwen_vl_processor as _build_qwen3_processor,
)
from fluxvla.engines.utils import PROCESSORS
from fluxvla.transforms.modality_state_action import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    ModalityStateActionCodec as GrootN17StateActionProcessor,
    load_groot_n17_metadata,
    normalize_tag_value as _tag_value,
)
from fluxvla.transforms.qwen_vl_action_inputs import (
    build_n17_image_transformations as _build_image_transformations,
    build_n17_image_transformations_albumentations as
    _build_image_transformations_albumentations,
    build_qwen_vl_chat_content,
    resolve_qwen_vl_model_path,
    stack_n17_vlm_images,
)


@PROCESSORS.register_module()
class GrootN17Processor:
    """Compatibility facade for the official GR00T N1.7 processor contract.

    The current facade combines metadata loading, state/action codec calls,
    action-head target construction, image augmentation, Qwen-VL chat content,
    and collator construction. Future refactors should move these pieces into
    config-visible FluxVLA transforms/collators while keeping this facade as a
    golden-reference compatibility path.
    """
    data_collator_class = GrootN17DataCollator

    def __init__(self,
                 modality_configs: Dict[str, Any],
                 statistics: Optional[Dict[str, Any]] = None,
                 use_percentiles: bool = False,
                 clip_outliers: bool = True,
                 image_crop_size: Optional[list[int]] = None,
                 image_target_size: Optional[list[int]] = None,
                 shortest_image_edge: int = 256,
                 crop_fraction: float = 0.95,
                 random_rotation_angle: Optional[int] = None,
                 color_jitter_params: Optional[dict] = None,
                 formalize_language: bool = True,
                 model_name: str = 'nvidia/Cosmos-Reason2-2B',
                 model_type: str = 'qwen',
                 max_state_dim: int = 29,
                 max_action_dim: int = 29,
                 max_action_horizon: int = 50,
                 apply_sincos_state_encoding: bool = False,
                 use_albumentations: bool = False,
                 extra_augmentation_config: Optional[dict] = None,
                 use_relative_action: bool = False,
                 embodiment_id_mapping: Optional[dict[str, int]] = None,
                 transformers_loading_kwargs: Optional[dict] = None,
                 exclude_state: bool = False,
                 state_dropout_prob: float = 0.0,
                 use_mean_std: bool = False,
                 letter_box_transform: bool = False):
        del extra_augmentation_config, use_mean_std, letter_box_transform
        self.modality_configs = deepcopy(modality_configs)
        self.state_action_processor = GrootN17StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action)
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.exclude_state = exclude_state
        self.state_dropout_prob = state_dropout_prob
        self.formalize_language = formalize_language
        self.model_name = resolve_qwen_vl_model_path(model_name)
        self.model_type = model_type
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction
        self.use_albumentations = use_albumentations
        self.processor = _build_qwen3_processor(
            self.model_name, transformers_loading_kwargs)
        self.embodiment_id_mapping = dict(
            embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX)
        for key, value in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            self.embodiment_id_mapping.setdefault(key, value)
        if use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                _build_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction))
        else:
            self.train_image_transform, self.eval_image_transform = (
                _build_image_transformations(
                    image_target_size, image_crop_size,
                    random_rotation_angle, color_jitter_params))
        self._collator = self.data_collator_class(
            model_name=self.model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs)
        self.training = True

    @property
    def collator(self):
        return self._collator

    def train(self):
        self.training = True
        self.state_action_processor.train()

    def eval(self):
        self.training = False
        self.state_action_processor.eval()

    def set_statistics(self, statistics: Dict[str, Any],
                       override: bool = False) -> None:
        self.state_action_processor.set_statistics(statistics, override=override)

    def _apply_vlm_processing(self, images: np.ndarray,
                              language: str) -> Dict[str, Any]:
        return {
            'vlm_content':
            build_qwen_vl_chat_content(self.processor, images, language)
        }

    def _get_vlm_inputs(self, image_keys: list[str], images: Dict[str, Any],
                        masks: Optional[dict], image_transform,
                        language: str) -> Dict[str, Any]:
        del masks
        stacked = stack_n17_vlm_images(
            image_keys,
            images,
            image_transform,
            use_albumentations=self.use_albumentations)
        return self._apply_vlm_processing(stacked, language)

    def __call__(self, messages: list[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(messages) == 1
        content = messages[0]['content']
        embodiment_tag = _tag_value(content.embodiment)
        action_data = content.actions
        state_data = content.states
        # 1. Per-modality state/action normalization and representation
        # conversion. This is the codec step that later should become a
        # config-visible state/action transform.
        norm_state_dict, normalized_actions = self.state_action_processor.apply(
            state=state_data, action=action_data, embodiment_tag=embodiment_tag)
        if normalized_actions:
            # 2. Build continuous action-head targets and masks.
            action_keys = self.modality_configs[embodiment_tag]['action']['modality_keys']
            normalized_action = torch.cat(
                [torch.from_numpy(normalized_actions[key]) for key in action_keys],
                dim=-1)
            action_dim = normalized_action.shape[1]
            normalized_action = torch.cat([
                normalized_action,
                torch.zeros(normalized_action.shape[0],
                            self.max_action_dim - normalized_action.shape[1])
            ],
                                          dim=-1)
            action_horizon = normalized_action.shape[0]
            normalized_action = torch.cat([
                normalized_action,
                torch.zeros(self.max_action_horizon - normalized_action.shape[0],
                            self.max_action_dim)
            ],
                                          dim=0)
            action_mask = torch.ones_like(normalized_action)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, 'Action is required in training mode'
            normalized_action = None
            action_mask = None
        # 3. Build padded state features used by the continuous action head.
        state_keys = self.modality_configs[embodiment_tag]['state']['modality_keys']
        state_cfg = self.modality_configs[embodiment_tag]['state']
        exclude_state = self.exclude_state or bool(
            state_cfg.get('exclude_state', False))
        if exclude_state or (self.state_dropout_prob > 0
                             and random.random() < self.state_dropout_prob
                             and self.training):
            normalized_state = torch.cat(
                [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys],
                dim=-1)
        else:
            normalized_state = torch.cat(
                [torch.from_numpy(norm_state_dict[key]) for key in state_keys],
                dim=-1)
        normalized_state = torch.cat([
            normalized_state,
            torch.zeros(normalized_state.shape[0],
                        self.max_state_dim - normalized_state.shape[1])
        ],
                                     dim=-1)
        language = content.text or ''
        if self.formalize_language:
            language = re.sub(r'[^\w\s]', '', language.lower())
        # 4. Build Qwen-VL chat/image content. Tokenization is intentionally
        # delayed to GrootN17DataCollator so text and images are batched
        # together by the Hugging Face processor.
        image_transform = self.train_image_transform if self.training else self.eval_image_transform
        image_keys = self.modality_configs[embodiment_tag]['video']['modality_keys']
        transformed = {'state': normalized_state.to(torch.get_default_dtype())}
        if normalized_action is not None:
            transformed['action'] = normalized_action.to(torch.get_default_dtype())
        transformed.update(
            self._get_vlm_inputs(image_keys, content.images,
                                 getattr(content, 'masks', None),
                                 image_transform, language))
        if action_mask is not None:
            transformed['action_mask'] = action_mask
        transformed['embodiment_id'] = self.embodiment_id_mapping[embodiment_tag]
        return transformed

    def decode_action(self,
                      action: np.ndarray,
                      embodiment_tag: Any,
                      state: Optional[Dict[str, np.ndarray]] = None):
        embodiment_key = _tag_value(embodiment_tag)
        out = {}
        start_idx = 0
        action_cfg = self.modality_configs[embodiment_key]['action']
        horizon = len(action_cfg['delta_indices'])
        for key in action_cfg['modality_keys']:
            dim = int(self.state_action_processor.norm_params[embodiment_key]['action'][key]['dim'])
            out[key] = action[..., :horizon, start_idx:start_idx + dim]
            start_idx += dim
        return self.state_action_processor.unapply_action(
            out, embodiment_key, state=state)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path,
                        **kwargs):
        transformers_loading_kwargs = kwargs.pop(
            'transformers_loading_kwargs', {'trust_remote_code': True})
        processor_kwargs = load_groot_n17_metadata(
            pretrained_model_name_or_path, **kwargs)
        return cls(
            **processor_kwargs,
            transformers_loading_kwargs=transformers_loading_kwargs)
