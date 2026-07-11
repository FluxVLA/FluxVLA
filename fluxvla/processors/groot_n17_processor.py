# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""FluxVLA-native GR00T N1.7 processor and collator.

This module mirrors the official N1.7 processor contract without importing the
official ``gr00t`` package. It is intentionally scoped to processor/collator
semantics; model assembly remains in later native port stages.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, Optional

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from transformers import Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature

from fluxvla.engines.utils import COLLATORS, PROCESSORS


EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    'oxe_droid_relative_eef_relative_joint': 24,
    'xdof_relative_eef_relative_joint': 27,
    'xdof_relative_eef_relative_joint_subtask': 27,
    'real_g1_relative_eef_relative_joints': 25,
    'real_r1_pro_sharpa_relative_eef': 26,
    'real_r1_pro_sharpa_relative_eef_human': 26,
    'real_r1_pro_sharpa_relative_eef_maxinsights': 26,
    'real_r1_pro_sharpa_relative_eef_mecka': 26,
    'unitree_g1_full_body_with_waist_height_nav_cmd': 25,
    'unitree_g1_sonic': 11,
    'simpler_env_google': 0,
    'simpler_env_widowx': 1,
    'libero_sim': 2,
    'new_embodiment': 10,
    'robocasa_panda_omron': 10,
    'robocasa_gr1_tabletop': 10,
}


def _tag_value(tag: Any) -> str:
    if hasattr(tag, 'value'):
        return str(tag.value)
    return str(tag)


def _build_qwen3_processor(model_name: str,
                           transformers_loading_kwargs: Optional[dict] = None):
    kwargs = dict(transformers_loading_kwargs or {})
    processor = Qwen3VLProcessor.from_pretrained(model_name, **kwargs)
    processor.tokenizer.padding_side = 'left'
    return processor


def _normalization_dim(stats: Dict[str, Any]) -> int:
    for field in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
        if field in stats:
            return int(np.asarray(stats[field]).shape[-1])
    raise KeyError(f'No supported statistics fields in {sorted(stats)}')


def _apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.sin(values), np.cos(values)], axis=-1)


def _normalize_minmax(values: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    min_vals = params['min']
    max_vals = params['max']
    normalized = np.zeros_like(values)
    mask = ~np.isclose(max_vals, min_vals)
    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask])
    normalized[..., mask] = 2 * normalized[..., mask] - 1
    return normalized


def _unnormalize_minmax(values: np.ndarray,
                        params: Dict[str, np.ndarray]) -> np.ndarray:
    min_vals = params['min']
    max_vals = params['max']
    return (np.clip(values, -1.0, 1.0) + 1.0) / 2.0 * (
        max_vals - min_vals) + min_vals


def _normalize_meanstd(values: np.ndarray,
                       params: Dict[str, np.ndarray]) -> np.ndarray:
    return (values - params['mean']) / params['std']


def _unnormalize_meanstd(values: np.ndarray,
                         params: Dict[str, np.ndarray]) -> np.ndarray:
    return values * params['std'] + params['mean']


def _action_config_value(config: Dict[str, Any], key: str,
                         default: str) -> str:
    return str(config.get(key, default)).upper()


class FractionalCenterCrop(A.DualTransform):

    def __init__(self,
                 crop_fraction: float = 0.9,
                 p: float = 1.0,
                 always_apply: Optional[bool] = None):
        super().__init__(p=p, always_apply=always_apply)
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords, **params) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def get_params_dependent_on_data(self, params, data) -> dict:
        height, width = params['shape'][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2
        return {
            'crop_coords':
            (x_min, y_min, x_min + crop_width, y_min + crop_height)
        }

    def get_transform_init_args_names(self):
        return ('crop_fraction',)


class FractionalRandomCrop(FractionalCenterCrop):

    def get_params_dependent_on_data(self, params, data) -> dict:
        height, width = params['shape'][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return {
            'crop_coords':
            (x_min, y_min, x_min + crop_width, y_min + crop_height)
        }


class LetterBoxPad(A.DualTransform):

    def __init__(self, p: float = 1.0, always_apply: Optional[bool] = None):
        super().__init__(p=p, always_apply=always_apply)

    def apply(self,
              img: np.ndarray,
              pad_top: int = 0,
              pad_bottom: int = 0,
              pad_left: int = 0,
              pad_right: int = 0,
              **params) -> np.ndarray:
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return img
        return cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0)

    def get_params_dependent_on_data(self, params, data) -> dict:
        h, w = params['shape'][:2]
        if h == w:
            return {'pad_top': 0, 'pad_bottom': 0, 'pad_left': 0, 'pad_right': 0}
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        return {
            'pad_top': pad_h // 2,
            'pad_bottom': pad_h - pad_h // 2,
            'pad_left': pad_w // 2,
            'pad_right': pad_w - pad_w // 2,
        }

    def get_transform_init_args_names(self):
        return ()


class LetterBoxTransform:

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        *leading_dims, c, h, w = img.shape
        if h == w:
            return img
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if leading_dims:
            batch_size = torch.tensor(leading_dims).prod().item()
            reshaped = img.reshape(batch_size, c, h, w)
            padded = transforms.functional.pad(
                reshaped,
                padding=[pad_left, pad_top, pad_right, pad_bottom],
                fill=0)
            return padded.reshape(leading_dims + [c, max_dim, max_dim])
        return transforms.functional.pad(
            img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)


def _build_image_transformations(image_target_size, image_crop_size,
                                 random_rotation_angle, color_jitter_params):
    train_ops = [
        transforms.ToImage(),
        LetterBoxTransform(),
        transforms.Resize(size=image_target_size),
        transforms.RandomCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_ops.append(
            transforms.RandomRotation(
                degrees=[-random_rotation_angle, random_rotation_angle]))
    if color_jitter_params is not None:
        train_ops.append(transforms.ColorJitter(**color_jitter_params))
    eval_ops = [
        transforms.ToImage(),
        LetterBoxTransform(),
        transforms.Resize(size=image_target_size),
        transforms.CenterCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _build_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
):
    fraction = crop_fraction
    if fraction is None:
        fraction = image_crop_size[0] / image_target_size[0]
    max_size = shortest_image_edge if shortest_image_edge is not None else image_target_size[0]
    train_ops = [
        LetterBoxPad(),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=fraction),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_ops.append(A.Rotate(limit=random_rotation_angle, p=1.0))
    if color_jitter_params is not None:
        train_ops.append(A.ColorJitter(**color_jitter_params, p=1.0))
    eval_transform = A.Compose([
        LetterBoxPad(),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalCenterCrop(crop_fraction=fraction),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ])
    return A.ReplayCompose(train_ops, p=1.0), eval_transform


def _apply_albumentations(transform, images, replay=None):
    tensors = []
    current_replay = replay
    has_replay = hasattr(transform, 'replay')
    for img in images:
        img_array = np.array(img)
        if has_replay:
            if current_replay is None:
                augmented = transform(image=img_array)
                current_replay = augmented['replay']
            else:
                augmented = transform.replay(
                    image=img_array, saved_augmentations=current_replay)
        else:
            augmented = transform(image=img_array)
        img_array = augmented['image']
        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        tensors.append(torch.from_numpy(img_array).permute(2, 0, 1))
    return tensors, current_replay


class GrootN17StateActionProcessor:

    def __init__(self,
                 modality_configs: Dict[str, Any],
                 statistics: Optional[Dict[str, Any]] = None,
                 use_percentiles: bool = False,
                 clip_outliers: bool = True,
                 apply_sincos_state_encoding: bool = False,
                 use_relative_action: bool = False):
        self.modality_configs = deepcopy(modality_configs)
        self.statistics: Dict[str, Any] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.norm_params: Dict[str, Any] = {}
        self.training = True
        if statistics is not None:
            self.set_statistics(statistics)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(self,
                       statistics: Dict[str, Any],
                       override: bool = False) -> None:
        for key, value in statistics.items():
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(value)
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        self.norm_params = {}
        for embodiment_tag, emb_stats in self.statistics.items():
            self.norm_params[embodiment_tag] = {}
            for modality in ('state', 'action'):
                if modality not in emb_stats:
                    continue
                self.norm_params[embodiment_tag][modality] = {}
                for key, stats in emb_stats[modality].items():
                    low_field, high_field = (
                        ('q01', 'q99') if self.use_percentiles else ('min', 'max'))
                    params = {
                        'min': np.asarray(stats[low_field]),
                        'max': np.asarray(stats[high_field]),
                        'mean': np.asarray(stats['mean']),
                        'std': np.asarray(stats['std']),
                        'dim': np.array(_normalization_dim(stats)),
                    }
                    self.norm_params[embodiment_tag][modality][key] = params
            action_cfg = self.modality_configs.get(embodiment_tag, {}).get(
                'action', {})
            for key, cfg in zip(action_cfg.get('modality_keys') or [],
                                action_cfg.get('action_configs') or []):
                if (self.use_relative_action
                        and _action_config_value(cfg, 'rep', '') == 'RELATIVE'):
                    action_dim = self.norm_params[embodiment_tag]['action'][key]['dim']
                    rel_stats = emb_stats['relative_action'][key]
                    self.norm_params[embodiment_tag]['action'][key] = {
                        'min': np.asarray(rel_stats['min']),
                        'max': np.asarray(rel_stats['max']),
                        'mean': np.asarray(rel_stats['mean']),
                        'std': np.asarray(rel_stats['std']),
                        'dim': action_dim,
                    }

    def _use_mean_std(self, embodiment_tag: str, modality: str,
                      key: str) -> bool:
        cfg = self.modality_configs[embodiment_tag][modality]
        keys = cfg.get('mean_std_embedding_keys')
        return bool(keys and key in keys)

    def _normalize(self, values: np.ndarray, embodiment_tag: str, modality: str,
                   key: str) -> np.ndarray:
        params = self.norm_params[embodiment_tag][modality][key]
        if self._use_mean_std(embodiment_tag, modality, key):
            normalized = _normalize_meanstd(values, params)
        else:
            normalized = _normalize_minmax(values, params)
        if self.clip_outliers:
            normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

    def _unnormalize(self, values: np.ndarray, embodiment_tag: str,
                     modality: str, key: str) -> np.ndarray:
        params = self.norm_params[embodiment_tag][modality][key]
        if self._use_mean_std(embodiment_tag, modality, key):
            return _unnormalize_meanstd(values, params)
        return _unnormalize_minmax(values, params)

    def apply_state(self, state: Dict[str, np.ndarray],
                    embodiment_tag: str) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['state']
        sincos_keys = set(cfg.get('sin_cos_embedding_keys') or [])
        result = {}
        for key in cfg['modality_keys']:
            if self.apply_sincos_state_encoding and key in sincos_keys:
                result[key] = _apply_sin_cos_encoding(state[key])
            else:
                result[key] = self._normalize(
                    state[key], embodiment_tag, 'state', key)
        return result

    def _relative(self, action: np.ndarray, reference_state: np.ndarray,
                  action_config: Dict[str, Any], to_absolute: bool) -> np.ndarray:
        action_type = _action_config_value(action_config, 'type', 'NON_EEF')
        action_format = _action_config_value(action_config, 'format', 'DEFAULT')
        if action_type != 'NON_EEF' or action_format != 'DEFAULT':
            raise NotImplementedError(
                'Native N1.7 processor currently supports NON_EEF/DEFAULT '
                f'relative actions only, got {action_type}/{action_format}.')
        action = action.astype(np.float64)
        reference_state = reference_state.astype(np.float64)
        return action + reference_state if to_absolute else action - reference_state

    def _maybe_relative(self, values: np.ndarray, state: Dict[str, np.ndarray],
                        key: str, action_config: Dict[str, Any],
                        to_absolute: bool) -> np.ndarray:
        if (not self.use_relative_action
                or _action_config_value(action_config, 'rep', '') != 'RELATIVE'):
            return values
        state_key = action_config.get('state_key') or key
        reference = np.asarray(state[state_key])[-1]
        return self._relative(values, reference, action_config, to_absolute)

    def apply_action(self, action: Dict[str, np.ndarray], embodiment_tag: str,
                     state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['action']
        action_configs = cfg.get('action_configs') or [{} for _ in cfg['modality_keys']]
        result = {}
        for key, action_config in zip(cfg['modality_keys'], action_configs):
            values = deepcopy(action[key])
            values = self._maybe_relative(
                values, state, key, action_config, to_absolute=False)
            result[key] = self._normalize(
                values, embodiment_tag, 'action', key)
        return result

    def unapply_action(self, action: Dict[str, np.ndarray], embodiment_tag: str,
                       state: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['action']
        action_configs = cfg.get('action_configs') or [{} for _ in cfg['modality_keys']]
        result = {}
        for key, action_config in zip(cfg['modality_keys'], action_configs):
            values = self._unnormalize(
                action[key], embodiment_tag, 'action', key)
            if (self.use_relative_action
                    and _action_config_value(action_config, 'rep', '') == 'RELATIVE'):
                if state is None:
                    raise ValueError(f'State is required to decode {key!r}.')
                values = self._maybe_relative(
                    values, state, key, action_config, to_absolute=True)
            result[key] = values
        return result

    def apply(self, state: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
              embodiment_tag: str):
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state)
        else:
            assert not self.training, 'Action is required in training mode'
            processed_action = {}
        return processed_state, processed_action

    def get_action_dim(self, embodiment_tag: str) -> int:
        total = 0
        for key in self.modality_configs[embodiment_tag]['action']['modality_keys']:
            total += int(self.norm_params[embodiment_tag]['action'][key]['dim'])
        return total


@COLLATORS.register_module()
class GrootN17DataCollator:

    def __init__(self,
                 model_name: str,
                 model_type: str = 'qwen',
                 transformers_loading_kwargs: Optional[dict] = None):
        self.processor = _build_qwen3_processor(
            model_name, transformers_loading_kwargs)
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))
        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == 'vlm_content':
                texts = []
                images = []
                for value in values:
                    texts.append(value['text'])
                    images.extend(value['images'])
                vlm_inputs = self.processor(
                    text=texts, images=images, return_tensors='pt', padding=True)
                batch.update(vlm_inputs)
            elif key in ('pixel_values', 'image_grid_thw', 'attention_mask',
                         'input_ids'):
                raise NotImplementedError(f'Pre-tokenized {key} is not supported.')
            else:
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={'inputs': batch})


@PROCESSORS.register_module()
class GrootN17Processor:
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
        self.model_name = model_name
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
            model_name, transformers_loading_kwargs)
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
            model_name=model_name,
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
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]
        conversation = [{
            'role':
            'user',
            'content': [
                *[{
                    'type': 'image',
                    'image': img
                } for img in pil_images],
                {
                    'type': 'text',
                    'text': language
                },
            ],
        }]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False)
        return {
            'vlm_content': {
                'text': text,
                'images': pil_images,
                'conversation': conversation,
            }
        }

    def _get_vlm_inputs(self, image_keys: list[str], images: Dict[str, Any],
                        masks: Optional[dict], image_transform,
                        language: str) -> Dict[str, Any]:
        del masks
        temporal_stacked_images = {}
        if self.use_albumentations:
            replay = None
            for view in image_keys:
                transformed, replay = _apply_albumentations(
                    image_transform, images[view], replay)
                temporal_stacked_images[view] = torch.stack(transformed)
        else:
            for view in image_keys:
                temporal_stacked_images[view] = torch.stack(
                    [image_transform(img) for img in images[view]])
        stacked = (
            torch.stack([temporal_stacked_images[view] for view in image_keys],
                        dim=1).flatten(0, 1).numpy())
        return self._apply_vlm_processing(stacked, language)

    def __call__(self, messages: list[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(messages) == 1
        content = messages[0]['content']
        embodiment_tag = _tag_value(content.embodiment)
        action_data = content.actions
        state_data = content.states
        norm_state_dict, normalized_actions = self.state_action_processor.apply(
            state=state_data, action=action_data, embodiment_tag=embodiment_tag)
        if normalized_actions:
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
        root = Path(pretrained_model_name_or_path)
        with open(root / 'processor_config.json', 'r') as f:
            config = json.load(f)
        with open(root / 'statistics.json', 'r') as f:
            statistics = json.load(f)
        embodiment_file = root / 'embodiment_id.json'
        embodiment_id_mapping = None
        if os.path.exists(embodiment_file):
            with open(embodiment_file, 'r') as f:
                embodiment_id_mapping = json.load(f)
        processor_kwargs = config['processor_kwargs']
        processor_kwargs['statistics'] = statistics
        processor_kwargs['embodiment_id_mapping'] = embodiment_id_mapping
        processor_kwargs.setdefault('model_name', 'nvidia/Cosmos-Reason2-2B')
        processor_kwargs.setdefault('model_type', 'qwen')
        processor_kwargs.setdefault('clip_outliers', True)
        if kwargs:
            modality_configs = kwargs.pop('modality_configs', {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs['modality_configs'][embodiment_tag] = modality_config
            for key in (
                    'random_rotation_angle',
                    'color_jitter_params',
                    'use_relative_action',
                    'exclude_state',
                    'state_dropout_prob',
                    'model_name',
                    'model_type',
                    'max_action_horizon',
                    'max_state_dim',
                    'max_action_dim',
            ):
                if key in kwargs and kwargs[key] is not None:
                    processor_kwargs[key] = kwargs[key]
        return cls(
            **processor_kwargs,
            transformers_loading_kwargs=transformers_loading_kwargs)
