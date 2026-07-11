# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Collator that bridges FluxVLA parquet samples to native GR00T N1.7 inputs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image
import torch

from fluxvla.engines import COLLATORS
from fluxvla.processors.groot_n17_processor import GrootN17Processor


N17_EMBODIMENT_ALIASES = {
    'ROBOCASA_GR1_TABLETOP': 'robocasa_gr1_tabletop',
    'robocasa_gr1_tabletop': 'robocasa_gr1_tabletop',
    'gr1_unified': 'robocasa_gr1_tabletop',
    'LIBERO_PANDA': 'libero_sim',
    'libero_sim': 'libero_sim',
}

ROBOCASA_GR1_FLUXVLA_SLICES = {
    'left_arm': (0, 7),
    'left_hand': (7, 13),
    'right_arm': (13, 20),
    'right_hand': (20, 26),
    'waist': (26, 29),
}


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _stat_dim(stats: Dict[str, Any]) -> int:
    for key in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
        if key in stats:
            return int(np.asarray(stats[key]).shape[-1])
    raise KeyError(f'Cannot infer dimension from stats fields: {sorted(stats)}')


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    array = _to_numpy(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array).convert('RGB')


@COLLATORS.register_module()
class GrootN17NativeCollator:
    """Build ``GrootN17VLA.forward(inputs=...)`` batches from parquet samples.

    The dataset side can stay on FluxVLA's existing ``ParquetDataset`` and
    ``ProcessParquetInputs`` path. This collator splits flat state/action
    vectors into the modality dictionaries expected by the native N1.7
    processor, then delegates tokenization/image processing to
    ``GrootN17Processor.collator``.
    """

    def __init__(
        self,
        processor_path: str,
        embodiment_tag: str = 'ROBOCASA_GR1_TABLETOP',
        state_key: str = 'states',
        action_key: str = 'actions',
        action_mask_key: str = 'action_masks',
        image_key: str = 'images',
        text_key: str = 'task_description',
        flat_layout: str = 'auto',
        train_mode: bool = True,
        processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.processor_path = processor_path
        self.embodiment_key = N17_EMBODIMENT_ALIASES.get(
            embodiment_tag, N17_EMBODIMENT_ALIASES.get(
                str(embodiment_tag).lower(), str(embodiment_tag).lower()))
        self.state_key = state_key
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        self.image_key = image_key
        self.text_key = text_key
        self.flat_layout = flat_layout
        self.processor = GrootN17Processor.from_pretrained(
            processor_path, **dict(processor_kwargs or {}))
        if train_mode:
            self.processor.train()
        else:
            self.processor.eval()

        self.modality_config = self.processor.modality_configs[
            self.embodiment_key]
        self.statistics = self.processor.state_action_processor.statistics[
            self.embodiment_key]

    def _flat_slices(self, modality: str) -> Dict[str, tuple[int, int]]:
        keys = self.modality_config[modality]['modality_keys']
        if (self.flat_layout in ('auto', 'robocasa_gr1_fluxvla')
                and self.embodiment_key == 'robocasa_gr1_tabletop'):
            return {key: ROBOCASA_GR1_FLUXVLA_SLICES[key] for key in keys}

        start = 0
        slices = {}
        for key in keys:
            dim = _stat_dim(self.statistics[modality][key])
            slices[key] = (start, start + dim)
            start += dim
        return slices

    def _split_flat(self, value: Any, modality: str) -> Dict[str, np.ndarray]:
        if isinstance(value, dict):
            return {key: _to_numpy(item).astype(np.float32)
                    for key, item in value.items()}

        array = _to_numpy(value).astype(np.float32)
        if array.ndim == 1:
            array = array[None, :]
        slices = self._flat_slices(modality)
        return {
            key: array[..., start:end].copy()
            for key, (start, end) in slices.items()
        }

    def _split_images(self, value: Any) -> Dict[str, list[Image.Image]]:
        if isinstance(value, dict):
            return {
                key: [_to_pil_rgb(img) for img in images]
                for key, images in value.items()
            }

        images = list(value)
        image_keys = self.modality_config['video']['modality_keys']
        if len(image_keys) == 1:
            return {image_keys[0]: [_to_pil_rgb(img) for img in images]}

        if len(images) % len(image_keys) != 0:
            raise ValueError(
                f'Cannot split {len(images)} images across {len(image_keys)} '
                f'video keys: {image_keys}')
        per_key = len(images) // len(image_keys)
        return {
            key: [_to_pil_rgb(img) for img in images[i * per_key:(i + 1) * per_key]]
            for i, key in enumerate(image_keys)
        }

    def _apply_external_action_mask(self, processed: Dict[str, Any],
                                    sample: Dict[str, Any]) -> None:
        if self.action_mask_key not in sample or 'action_mask' not in processed:
            return
        mask = torch.as_tensor(
            _to_numpy(sample[self.action_mask_key]), dtype=processed[
                'action_mask'].dtype)
        if mask.ndim == 1:
            mask = mask[:, None]
        horizon = min(mask.shape[0], processed['action_mask'].shape[0])
        processed['action_mask'][:horizon] *= mask[:horizon]
        if horizon < processed['action_mask'].shape[0]:
            processed['action_mask'][horizon:] = 0

    def _process_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        step = SimpleNamespace(
            images=self._split_images(sample[self.image_key]),
            states=self._split_flat(sample[self.state_key], 'state'),
            actions=self._split_flat(sample[self.action_key], 'action'),
            text=sample.get(self.text_key, ''),
            embodiment=self.embodiment_key,
        )
        processed = self.processor([{'type': 'episode_step', 'content': step}])
        self._apply_external_action_mask(processed, sample)
        return processed

    def __call__(self, batch: Iterable[Dict[str, Any]]):
        processed = [self._process_one(sample) for sample in batch]
        collated = self.processor.collator(processed).data
        if 'inputs' in collated and 'action_mask' in collated['inputs']:
            collated['inputs']['action_mask'] = collated['inputs'][
                'action_mask'].to(dtype=torch.float32)
        return collated
