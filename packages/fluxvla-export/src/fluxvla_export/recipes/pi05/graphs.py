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
"""Flat tensor graph boundaries over the canonical PI0.5 computations.

The wrappers hold the already loaded model, not a second set of weights. Only
cache packing and state metadata adaptation live here; all policy tensor math
remains in the same helpers used by the ordinary training/reference policy.
"""

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicLayer

from ...errors import ContractError
from .model import PI05Model


def cache_tensor_names(num_layers: int) -> tuple[str, ...]:
    """Name the complete prefix cache in layer-interleaved key/value order."""
    if type(num_layers) is not int or num_layers < 1:
        raise ContractError('PI0.5 cache requires a positive layer count.')
    return tuple(
        name for index in range(num_layers)
        for name in (f'key_{index}', f'value_{index}'))


def _validate_model(model: PI05Model) -> None:
    if not isinstance(model, PI05Model):
        raise ContractError('PI0.5 graphs require a loaded PI05Model.')
    if any(module.training for module in model.modules()):
        raise ContractError(
            'PI0.5 graphs require all model modules in eval mode.')


class _GraphCacheLayer(DynamicLayer):
    """Use properly ranked empty tensors with HF's unchanged cache update.

    Native DynamicLayer starts with a rank-one empty tensor, which PyTorch
    permits concatenating with rank-four KV tensors. ONNX Concat requires
    matching ranks. Empty sequence-axis views represent the same empty cache
    without depending on that PyTorch-only rank exception.
    """

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states[..., :0, :]
        # Preserve native K-driven dtype/device and its V promotion on cat.
        self.values = value_states[..., :0, :].to(
            dtype=self.dtype, device=self.device)
        self.is_initialized = True


def _new_graph_cache(num_layers):
    return Cache(layers=[_GraphCacheLayer() for _ in range(num_layers)])


class PI05PrefillGraph(nn.Module):
    """Image/language tensors -> prefix padding plus the full flat KV cache."""

    name = 'prefill'
    input_names = ('images', 'img_masks', 'lang_tokens', 'lang_masks')

    def __init__(self, model: PI05Model) -> None:
        super().__init__()
        _validate_model(model)
        self.model = model
        self.num_layers = len(model.llm_backbone.layers)
        self.output_names = ('prefix_pad_masks', ) + cache_tensor_names(
            self.num_layers)
        self.autocast_enabled = model.enable_mixed_precision_training
        self.eval()

    def forward(self, images: torch.Tensor, img_masks: torch.Tensor,
                lang_tokens: torch.Tensor,
                lang_masks: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.autocast(
                device_type=images.device.type,
                dtype=torch.bfloat16,
                enabled=self.autocast_enabled):
            prefix_pad_masks, cache = self.model.prefill_prefix(
                images=images,
                lang_tokens=lang_tokens,
                img_masks=img_masks,
                lang_masks=lang_masks,
                past_key_values=_new_graph_cache(self.num_layers))
        if len(cache.layers) != self.num_layers:
            raise ContractError(
                'PI0.5 prefill returned an incomplete KV cache.')
        flat_cache = tuple(
            tensor for layer in cache.layers
            for tensor in (layer.keys, layer.values))
        return (prefix_pad_masks, ) + flat_cache


class PI05DenoiseGraph(nn.Module):
    """Noisy actions, time and flat prefix cache -> one velocity prediction.

    State values have no role in PI0.5 suffix computation. ``states_dtype`` is
    taken from the validated six-input deployment contract; its device is the
    common runtime device of noisy actions and all cache tensors. This omits
    the unused states input without adding dummy arithmetic to retain it.
    """

    name = 'denoise'
    output_names = ('velocity', )

    def __init__(self, model: PI05Model, *, states_dtype: torch.dtype) -> None:
        super().__init__()
        _validate_model(model)
        if states_dtype not in (torch.float32, torch.bfloat16):
            raise ContractError(
                'PI0.5 states_dtype must be float32 or bfloat16.')
        if (states_dtype == torch.bfloat16
                and not model.enable_mixed_precision_training):
            raise ContractError(
                'PI0.5 bfloat16 states require bfloat16 autocast.')
        self.model = model
        self.states_dtype = states_dtype
        self.num_layers = len(model.llm_backbone.layers)
        self.input_names = ('noisy_actions', 'timestep',
                            'prefix_pad_masks') + (
                                cache_tensor_names(self.num_layers))
        self.autocast_enabled = model.enable_mixed_precision_training
        self.eval()

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor,
                prefix_pad_masks: torch.Tensor, *flat_cache:
                torch.Tensor) -> torch.Tensor:
        if len(flat_cache) != 2 * self.num_layers:
            raise ContractError(
                'PI0.5 denoise requires one key/value pair per prefix layer; '
                f'expected {2 * self.num_layers} cache tensors, '
                f'got {len(flat_cache)}.')
        # Fresh HF cache containers and inherited DynamicLayer.update preserve
        # input tensors; the canonical expert reads with use_cache=False.
        cache = _new_graph_cache(self.num_layers)
        for index in range(self.num_layers):
            cache.update(flat_cache[2 * index], flat_cache[2 * index + 1],
                         index)
        with torch.autocast(
                device_type=noisy_actions.device.type,
                dtype=torch.bfloat16,
                enabled=self.autocast_enabled):
            suffix = self.model.embed_suffix_with_metadata(
                noisy_actions,
                timestep,
                state_dtype=self.states_dtype,
                state_device=noisy_actions.device)
            return self.model.denoise_suffix(prefix_pad_masks, cache,
                                             noisy_actions, suffix)
