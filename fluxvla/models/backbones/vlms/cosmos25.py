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
#
# Adapted from Mondo-Robotics/DiT4DiT:
# DiT4DiT/model/modules/vlm/Cosmos25.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import VLM_BACKBONES
from fluxvla.engines.utils.name_map import str_to_dtype


@dataclass
class Cosmos25BackboneOutput:
    hidden_states: List[torch.Tensor]
    future_video_loss: Optional[torch.Tensor] = None
    pred_future_video: Optional[torch.Tensor] = None


class _DefaultDummySafetyChecker:
    """Construct Cosmos without pulling in external guardrail dependencies."""

    def to(self, device):
        return self

    def check_text_safety(self, text):
        return True

    def check_video_safety(self, video):
        return video


@VLM_BACKBONES.register_module()
class Cosmos25Backbone(nn.Module):
    """Cosmos-Predict2.5 feature backbone for DiT4DiT-style policies.

    The backbone runs the Cosmos2.5 video transformer for one or more denoising
    steps, captures a selected transformer block output, and exposes it as
    ``hidden_states=[Tensor[B, S, D]]`` for an action head. It accepts raw text
    prompts through ``task_description``/``prompts``. Using already-tokenized
    language ids is intentionally not supported because Cosmos owns its text
    encoder and tokenizer.

    Args:
        model_id_or_path: Local path or HF id for Cosmos-Predict2.5.
        revision: HF revision. DiT4DiT uses ``diffusers/base/post-trained``.
        torch_dtype: dtype used to load Cosmos modules.
        local_files_only: Passed to ``from_pretrained``.
        extract_layer: Transformer block index to hook.
        max_sequence_length: Text prompt max sequence length.
        trainable: Whether to leave Cosmos parameters trainable.
        split_future_frames: If ``images`` contains a temporal dimension,
            use frame 0 as conditioning and the remaining frames only to infer
            the training output horizon.
        fixed_seed: Optional deterministic noise seed for latent extraction.
        fsdp_min_num_params: Auto-wrap large non-block Cosmos modules above
            this parameter count when building an FSDP policy.
    """

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        base_model: Optional[str] = None,
        revision: str = 'diffusers/base/post-trained',
        torch_dtype: Union[str, torch.dtype] = 'bf16',
        local_files_only: bool = True,
        extract_layer: int = 17,
        max_sequence_length: int = 512,
        trainable: bool = False,
        frozen_submodules: Optional[Sequence[str]] = None,
        split_future_frames: bool = True,
        fixed_seed: Optional[int] = 42,
        num_inference_steps: int = 1,
        conditional_frame_timestep: float = 0.001,
        future_loss_type: Optional[str] = None,
        detach_hidden_states: bool = False,
        flow_matching_time_distribution: str = 'logit_normal',
        flow_matching_high_sigma_ratio: Optional[float] = 0.05,
        flow_matching_high_sigma_min: Optional[float] = 0.98,
        fsdp_min_num_params: int = 10_000_000,
        device: Optional[Union[str, torch.device]] = None,
        safety_checker: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        model_id_or_path = model_id_or_path or base_model
        if not model_id_or_path:
            raise ValueError('Cosmos25Backbone requires `model_id_or_path` or '
                             '`base_model` in the config.')

        if isinstance(torch_dtype, str):
            torch_dtype = str_to_dtype(torch_dtype)

        self.model_id_or_path = model_id_or_path
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.extract_layer = int(extract_layer)
        self.max_sequence_length = int(max_sequence_length)
        self.trainable = bool(trainable)
        self.frozen_submodules = tuple(frozen_submodules or ())
        self.split_future_frames = bool(split_future_frames)
        self.fixed_seed = fixed_seed
        self.num_inference_steps = int(num_inference_steps)
        self.conditional_frame_timestep = float(conditional_frame_timestep)
        self.future_loss_type = (
            str(future_loss_type).lower()
            if future_loss_type is not None else None)
        self.detach_hidden_states = bool(detach_hidden_states)
        self.flow_matching_time_distribution = str(
            flow_matching_time_distribution)
        self.flow_matching_high_sigma_ratio = flow_matching_high_sigma_ratio
        self.flow_matching_high_sigma_min = flow_matching_high_sigma_min
        self.fsdp_min_num_params = int(fsdp_min_num_params)

        self._hook_handle = None
        self._cached_hidden: list[torch.Tensor] = []
        self._capture_hidden_enabled = True

        pipe = self._build_pipeline(safety_checker)
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.video_processor = pipe.video_processor

        self.latents_mean = getattr(pipe, 'latents_mean', None)
        self.latents_std = getattr(pipe, 'latents_std', None)
        self.vae_scale_factor_temporal = int(
            getattr(pipe, 'vae_scale_factor_temporal', 1) or 1)
        self.vae_scale_factor_spatial = int(
            getattr(pipe, 'vae_scale_factor_spatial', 1) or 1)

        self._register_hidden_hook()
        if not self.trainable:
            self.requires_grad_(False)
        else:
            self.freeze_configured_submodules()
        if device is not None:
            self.to(device)
        del pipe

    def _build_pipeline(self, safety_checker):
        try:
            from diffusers import Cosmos2_5_PredictBasePipeline
        except Exception as exc:
            try:
                import diffusers

                diffusers_version = getattr(diffusers, '__version__',
                                            'unknown')
            except Exception:
                diffusers_version = 'not installed'
            raise ImportError(
                'Cosmos25Backbone requires a diffusers build that provides '
                '`Cosmos2_5_PredictBasePipeline`. Installed diffusers: '
                f'{diffusers_version}. Install the DiT4DiT-compatible build '
                'with: pip install --upgrade '
                '"diffusers @ git+https://github.com/huggingface/'
                'diffusers.git@3996788b602eaae4da41a1d45726b62e662b73cf"'
            ) from exc

        if safety_checker is None:
            safety_checker = _DefaultDummySafetyChecker()

        return Cosmos2_5_PredictBasePipeline.from_pretrained(
            self.model_id_or_path,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
            safety_checker=safety_checker,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def embed_dim(self) -> Optional[int]:
        cfg = getattr(self.transformer, 'config', None)
        if cfg is None:
            return None
        for key in ('hidden_size', 'inner_dim', 'model_dim',
                    'cross_attention_dim'):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.torch_dtype

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        blocks = getattr(self.transformer, 'transformer_blocks', None)
        if blocks is None or len(blocks) == 0:
            return nn.Module
        return type(blocks[0])

    def set_frozen_modules_to_eval_mode(self) -> None:
        if not self.trainable:
            self.eval()
            return
        for module in self._iter_frozen_submodules():
            module.eval()

    def freeze_configured_submodules(self) -> None:
        for module in self._iter_frozen_submodules():
            module.requires_grad_(False)
            module.eval()

    def _iter_frozen_submodules(self):
        aliases = {
            'backbone_interface.extractor.text_encoder': 'text_encoder',
            'backbone_interface.extractor.vae': 'vae',
            'backbone_interface.extractor.transformer': 'transformer',
            'extractor.text_encoder': 'text_encoder',
            'extractor.vae': 'vae',
            'extractor.transformer': 'transformer',
        }
        for name in self.frozen_submodules:
            module_name = aliases.get(name, name)
            module = self
            for part in module_name.split('.'):
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None:
                raise ValueError(
                    f"Cannot freeze unknown Cosmos submodule '{name}'.")
            yield module

    @staticmethod
    def _enable_module_gradient_checkpointing(module: nn.Module) -> None:
        fn = getattr(module, 'enable_gradient_checkpointing', None)
        if callable(fn):
            supports = getattr(module, '_supports_gradient_checkpointing',
                               True)
            if supports is False:
                return
            try:
                fn()
            except ValueError as exc:
                if 'does not support gradient checkpointing' in str(exc):
                    return
                raise
        elif hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

    def enable_gradient_checkpointing(self) -> None:
        # AutoencoderKLWan advertises enable_gradient_checkpointing through
        # diffusers.ModelMixin but explicitly does not support it. The VAE is
        # frozen/no-grad for DiT4DiT, so checkpoint only supported modules.
        for module in (self.transformer, self.text_encoder, self.vae):
            self._enable_module_gradient_checkpointing(module)

    def get_fsdp_wrapping_policy(self):
        from functools import partial

        from torch.distributed.fsdp.wrap import (_module_wrap_policy,
                                                 _or_policy,
                                                 size_based_auto_wrap_policy)

        module_classes = set()
        if self.transformer_layer_cls is not nn.Module:
            module_classes.add(self.transformer_layer_cls)

        optional_fsdp_classes = {
            'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl': (
                'Qwen2_5_VLDecoderLayer',
                'Qwen2_5_VLVisionBlock',
            ),
            'diffusers.models.autoencoders.autoencoder_kl_wan': (
                'WanResidualBlock',
                'WanAttentionBlock',
                'WanResidualDownBlock',
                'WanResidualUpBlock',
                'WanMidBlock',
            ),
        }
        for module_path, class_names in optional_fsdp_classes.items():
            try:
                module = __import__(module_path, fromlist=list(class_names))
            except Exception:
                continue
            for class_name in class_names:
                cls = getattr(module, class_name, None)
                if cls is not None:
                    module_classes.add(cls)

        policies = []
        if module_classes:
            policies.append(
                partial(
                    _module_wrap_policy,
                    module_classes=module_classes,
                ))
        if self.fsdp_min_num_params > 0:
            policies.append(
                partial(
                    size_based_auto_wrap_policy,
                    min_num_params=self.fsdp_min_num_params,
                ))
        if not policies:
            return None
        if len(policies) == 1:
            return policies[0]
        return partial(
            _or_policy,
            policies=policies,
        )

    def _register_hidden_hook(self) -> None:
        blocks = getattr(self.transformer, 'transformer_blocks', None)
        if not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
            raise ValueError(
                'Cosmos transformer does not expose `transformer_blocks`. '
                'cannot capture DiT4DiT conditioning features.')
        if self.extract_layer < 0 or self.extract_layer >= len(blocks):
            raise ValueError(
                f'extract_layer={self.extract_layer} out of bounds for '
                f'{len(blocks)} Cosmos transformer blocks.')

        def hook_fn(module, inp, out):
            if not self._capture_hidden_enabled:
                return
            if torch.is_tensor(out):
                hidden = out
            elif isinstance(out,
                            (tuple, list)) and out and torch.is_tensor(out[0]):
                hidden = out[0]
            else:
                return
            if not self.trainable or self.detach_hidden_states:
                hidden = hidden.detach()
            self._cached_hidden.append(hidden)

        self._hook_handle = blocks[self.extract_layer].register_forward_hook(
            hook_fn)

    def __del__(self):
        handle = getattr(self, '_hook_handle', None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass

    @staticmethod
    def _to_tensor(value, device=None, dtype=None) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    @staticmethod
    def _image_to_chw_float(image: Any) -> torch.Tensor:
        if torch.is_tensor(image):
            tensor = image.detach()
            if tensor.ndim != 3:
                raise ValueError(
                    f'Expected 3D image tensor, got {tuple(tensor.shape)}.')
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor[:3]
            elif tensor.shape[-1] in (1, 3, 4):
                tensor = tensor[..., :3].permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f'Cannot infer channel dimension for image '
                                 f'{tuple(tensor.shape)}.')
            tensor = tensor.float()
            return tensor / 255.0 if tensor.max() > 1.0 else tensor

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f'Expected HWC image array, got {arr.shape}.')
        arr = arr[..., :3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        return tensor / 255.0 if tensor.max() > 1.0 else tensor

    def _normalize_videos(
        self,
        images: Union[torch.Tensor, Sequence[Any]],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return condition video B,C,T,H,W and optional future B,T,C,H,W."""
        if torch.is_tensor(images):
            videos = images
            if videos.ndim == 4:
                # B,C,H,W -> B,C,1,H,W
                videos = videos.unsqueeze(2)
            elif videos.ndim != 5:
                raise ValueError(
                    '`images` must be 4D/5D tensor or a sequence of images. '
                    f'got shape {tuple(videos.shape)}.')

            if videos.shape[1] == 3:
                bcthw = videos.float()
            elif videos.shape[2] == 3:
                bcthw = videos.permute(0, 2, 1, 3, 4).contiguous().float()
            else:
                raise ValueError(
                    'Could not infer channel dimension for video tensor '
                    f'{tuple(videos.shape)}.')
        else:
            samples = []
            for sample in images:
                frames = sample if isinstance(sample,
                                              (list, tuple)) else [sample]
                samples.append(
                    torch.stack(
                        [self._image_to_chw_float(frame) for frame in frames],
                        dim=0))
            # B,T,C,H,W -> B,C,T,H,W
            bcthw = torch.stack(
                samples, dim=0).permute(0, 2, 1, 3, 4).contiguous()

        if bcthw.max() > 1.0:
            bcthw = bcthw / 255.0

        future = None
        if self.split_future_frames and bcthw.shape[2] > 1:
            future = bcthw[:, :, 1:].permute(0, 2, 1, 3, 4).contiguous()
            bcthw = bcthw[:, :, :1]

        return bcthw, future

    def build_cosmos_inputs(
        self,
        images: Union[torch.Tensor, Sequence[Any]],
        instructions: Sequence[str],
        **kwargs,
    ) -> dict:
        """Build the same input dict used by DiT4DiT's Cosmos interface."""
        if isinstance(instructions, str):
            instructions = [instructions]

        if torch.is_tensor(images):
            batch_size = images.shape[0]
            if batch_size != len(instructions):
                raise ValueError(
                    '`images` and `instructions` must have the same batch '
                    f'size, got {batch_size} and {len(instructions)}.')
            condition_video, future_videos = self._normalize_videos(images)
            videos = condition_video.permute(0, 2, 1, 3, 4).contiguous()
        else:
            if len(images) != len(instructions):
                raise ValueError(
                    '`images` and `instructions` must have the same batch '
                    f'size, got {len(images)} and {len(instructions)}.')

            all_cond = []
            all_future = []
            for sample in images:
                frames = sample if isinstance(sample,
                                              (list, tuple)) else [sample]
                if len(frames) == 0:
                    raise ValueError(
                        'Each sample must provide at least one image frame.')
                chw_frames = [
                    self._image_to_chw_float(frame) for frame in frames
                ]
                all_cond.append(chw_frames[0])
                if len(chw_frames) > 1:
                    all_future.append(torch.stack(chw_frames[1:], dim=0))

            videos = torch.stack(all_cond, dim=0).unsqueeze(1)
            if all_future and len(all_future) != len(all_cond):
                raise ValueError(
                    'Future frames must be present for every sample or for '
                    'no samples in a batch.')
            future_videos = (
                torch.stack(all_future, dim=0) if all_future else None)

        height = int(videos.shape[-2])
        width = int(videos.shape[-1])
        out = {
            'prompts': list(instructions),
            'videos': videos,
            'height': height,
            'width': width,
            'future_videos': future_videos,
        }
        out.update(kwargs)
        return out

    def _resolve_prompts(
        self,
        batch_size: int,
        prompts: Optional[Union[str, Sequence[str]]] = None,
        task_description: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> list[str]:
        prompts = prompts if prompts is not None else task_description
        prompts = prompts if prompts is not None else kwargs.get('prompt')
        prompts = prompts if prompts is not None else kwargs.get('instruction')
        prompts = prompts if prompts is not None else kwargs.get(
            'instructions')
        prompts = prompts if prompts is not None else kwargs.get('lang')

        if prompts is None:
            raise ValueError(
                'Cosmos25Backbone requires raw text prompts via `prompts`, '
                '`task_description`, `prompt`, `instruction`, '
                '`instructions`, or `lang`. Token IDs from other FluxVLA '
                'tokenizers are not valid Cosmos prompts.')

        if isinstance(prompts, str):
            prompts = [prompts] * batch_size
        else:
            prompts = list(prompts)

        if len(prompts) != batch_size:
            raise ValueError(
                f'Prompt batch size {len(prompts)} does not match video '
                f'batch size {batch_size}.')
        return [str(prompt) for prompt in prompts]

    def _get_prompt_embeds(
        self,
        prompts: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        def to_input_ids(conversations) -> torch.Tensor:
            ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=self.max_sequence_length,
                truncation=True,
                padding='max_length',
            )
            if isinstance(ids, str):
                ids = self.tokenizer(
                    ids,
                    add_special_tokens=False,
                    max_length=self.max_sequence_length,
                    truncation=True,
                    padding='max_length',
                )
            if isinstance(ids, dict):
                ids = ids['input_ids']
            elif hasattr(ids, 'input_ids'):
                ids = ids.input_ids

            ids = torch.as_tensor(ids, dtype=torch.long)
            if ids.ndim == 2 and ids.shape[0] == 1:
                ids = ids.squeeze(0)
            if ids.ndim != 1:
                raise ValueError(
                    'Cosmos tokenizer must return one input-id sequence per '
                    f'prompt, got shape {tuple(ids.shape)}.')
            return ids

        input_ids = []
        for prompt in prompts:
            conversations = [
                {
                    'role':
                    'system',
                    'content': [{
                        'type':
                        'text',
                        'text':
                        'You are a helpful assistant who will provide prompts to an image generator.',  # noqa: E501
                    }],
                },
                {
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': prompt
                    }]
                },
            ]
            input_ids.append(to_input_ids(conversations))

        input_ids = torch.stack(input_ids, dim=0).to(device)
        outputs = self.text_encoder(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        normalized = []
        for layer_idx in range(1, len(hidden_states)):
            hs = hidden_states[layer_idx]
            hs = (hs - hs.mean(dim=-1, keepdim=True)) / (
                hs.std(dim=-1, keepdim=True) + 1e-8)
            normalized.append(hs)
        return torch.cat(normalized, dim=-1).to(device=device, dtype=dtype)

    def _encode_condition_latents(
        self,
        video_bcthw: torch.Tensor,
        num_frames_out: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        vae_dtype = getattr(self.vae, 'dtype', self.dtype)
        video = video_bcthw.to(device=self.device, dtype=vae_dtype)
        if video.min() >= 0.0:
            video = video * 2.0 - 1.0
        if video.shape[2] < num_frames_out:
            pad = video.new_zeros(video.shape[0], video.shape[1],
                                  num_frames_out - video.shape[2],
                                  video.shape[3], video.shape[4])
            video = torch.cat([video, pad], dim=2)

        encoded = self.vae.encode(video)
        if hasattr(encoded, 'latent_dist'):
            latents = encoded.latent_dist.sample()
        else:
            latents = encoded
        return latents.to(dtype=dtype)

    def _latent_mean_std(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._to_tensor(
            self.latents_mean, device=latents.device, dtype=latents.dtype)
        std = self._to_tensor(
            self.latents_std, device=latents.device, dtype=latents.dtype)
        if (mean.ndim == 5 and latents.ndim == 5
                and mean.shape[2] >= latents.shape[2]):
            mean = mean[:, :, :latents.shape[2]]
        if (std.ndim == 5 and latents.ndim == 5
                and std.shape[2] >= latents.shape[2]):
            std = std[:, :, :latents.shape[2]]
        return mean, std

    def _encode_video_to_latents_norm(
        self,
        video_bcthw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pixel video [B,3,T,H,W] into normalized VAE latents."""
        if video_bcthw.ndim != 5 or video_bcthw.shape[1] != 3:
            raise ValueError('`video_bcthw` must be [B,3,T,H,W], got '
                             f'{tuple(video_bcthw.shape)}.')
        if self.latents_mean is None or self.latents_std is None:
            raise ValueError(
                'Cosmos VAE must expose `latents_mean` and `latents_std`.')

        vae_dtype = getattr(self.vae, 'dtype', self.dtype)
        video = video_bcthw.to(device=self.device, dtype=vae_dtype)
        if video.min() >= 0.0:
            video = video * 2.0 - 1.0

        encoded = self.vae.encode(video)
        if hasattr(encoded, 'latent_dist'):
            latents = encoded.latent_dist.mean
        else:
            latents = encoded
        latents = latents.to(dtype=torch.float32)
        mean, std = self._latent_mean_std(latents)
        return (latents - mean) / std

    def _prepare_latents(
        self,
        video_bcthw: torch.Tensor,
        num_frames_out: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.latents_mean is None or self.latents_std is None:
            raise ValueError(
                'Cosmos VAE must expose `latents_mean` and `latents_std`.')

        b, _, _, height, width = video_bcthw.shape
        latent_channels = int(self.transformer.config.in_channels) - 1
        t_lat = (num_frames_out - 1) // max(1,
                                            self.vae_scale_factor_temporal) + 1
        h_lat = height // max(1, self.vae_scale_factor_spatial)
        w_lat = width // max(1, self.vae_scale_factor_spatial)
        shape = (b, latent_channels, t_lat, h_lat, w_lat)

        generator = None
        if self.fixed_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.fixed_seed))
        latents = torch.randn(
            shape, generator=generator, device=self.device, dtype=dtype)

        cond_latents = self._encode_condition_latents(video_bcthw,
                                                      num_frames_out, dtype)
        mean, std = self._latent_mean_std(cond_latents)
        cond_latents = (cond_latents - mean) / std

        if cond_latents.shape != latents.shape:
            adjusted = latents.new_zeros(shape)
            t_copy = min(adjusted.shape[2], cond_latents.shape[2])
            h_copy = min(adjusted.shape[3], cond_latents.shape[3])
            w_copy = min(adjusted.shape[4], cond_latents.shape[4])
            adjusted[:, :, :t_copy, :h_copy, :
                     w_copy] = cond_latents[:, :, :t_copy, :h_copy, :w_copy]
            cond_latents = adjusted

        cond_latent_frames = min(
            t_lat,
            (video_bcthw.shape[2] - 1) //
            max(1, self.vae_scale_factor_temporal) + 1,
        )
        cond_indicator = latents.new_zeros(b, 1, t_lat, 1, 1)
        cond_indicator[:, :, :cond_latent_frames] = 1.0
        cond_mask = cond_indicator.expand(b, 1, t_lat, h_lat, w_lat)
        return latents, cond_latents, cond_mask, cond_indicator

    @staticmethod
    def _future_video_to_btchw(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError('`future_videos` must be 5D, got '
                             f'{tuple(video.shape)}.')
        if video.shape[2] == 3:
            return video
        if video.shape[1] == 3:
            return video.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError('`future_videos` must be [B, T, 3, H, W] or '
                         f'[B, 3, T, H, W], got {tuple(video.shape)}.')

    @staticmethod
    def _is_flow_matching_loss(loss_type: Optional[str]) -> bool:
        return loss_type in {
            'flow_matching',
            'latent_flow_matching',
            'rectified_flow',
            'rf',
        }

    def _sample_flow_time(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.flow_matching_time_distribution == 'logit_normal':
            t = torch.sigmoid(
                torch.randn((batch_size, ), device=device,
                            dtype=torch.float32))
        else:
            t = torch.rand((batch_size, ), device=device, dtype=torch.float32)

        high_sigma_ratio = self.flow_matching_high_sigma_ratio
        high_sigma_min = self.flow_matching_high_sigma_min
        if high_sigma_ratio is not None and float(high_sigma_ratio) > 0:
            high_sigma_min = 0.98 if high_sigma_min is None else float(
                high_sigma_min)
            high_mask = torch.rand(
                (batch_size, ), device=device) < float(high_sigma_ratio)
            high_t = (
                torch.rand(
                    (batch_size, ), device=device, dtype=torch.float32) *
                (1.0 - high_sigma_min) + high_sigma_min)
            t = torch.where(high_mask, high_t, t)
        return t

    def _future_video_flow_matching_loss(
        self,
        condition_video: torch.Tensor,
        future_videos: torch.Tensor,
        latents: torch.Tensor,
        cond_latents: torch.Tensor,
        cond_mask_t: torch.Tensor,
        cond_indicator: torch.Tensor,
        cond_timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        padding_mask: torch.Tensor,
        transformer_dtype: torch.dtype,
    ) -> torch.Tensor:
        gt = self._future_video_to_btchw(future_videos)
        gt_bcthw = gt.permute(0, 2, 1, 3, 4).contiguous()
        if gt_bcthw.min() >= 0.0:
            gt_bcthw = gt_bcthw * 2.0 - 1.0

        cond = condition_video.to(device=gt_bcthw.device, dtype=gt_bcthw.dtype)
        if cond.min() >= 0.0:
            cond = cond * 2.0 - 1.0
        full_bcthw = torch.cat([cond, gt_bcthw], dim=2)

        temporal_factor = max(1, int(self.vae_scale_factor_temporal))
        min_full_frames = 1 + temporal_factor
        if full_bcthw.shape[2] < min_full_frames:
            pad_len = min_full_frames - int(full_bcthw.shape[2])
            last = full_bcthw[:, :, -1:].repeat(1, 1, pad_len, 1, 1)
            full_bcthw = torch.cat([full_bcthw, last], dim=2)

        with torch.no_grad():
            gt_latents = self._encode_video_to_latents_norm(full_bcthw)

        cond_count = int(cond_indicator[0, 0, :, 0, 0].sum().item())
        pred_latents_future = latents[:, :, cond_count:]
        gt_latents_future = gt_latents[:, :, cond_count:cond_count +
                                       pred_latents_future.shape[2]]
        no_future_latents = (
            pred_latents_future.numel() == 0 or gt_latents_future.numel() == 0)
        if no_future_latents:
            return torch.tensor(
                0.0, device=latents.device, dtype=latents.dtype)

        time_steps = min(pred_latents_future.shape[2],
                         gt_latents_future.shape[2])
        x0_future = gt_latents_future[:, :, :time_steps].to(
            device=latents.device, dtype=torch.float32)
        batch_size = latents.shape[0]
        t = self._sample_flow_time(batch_size, latents.device).view(
            batch_size, 1, 1, 1, 1)
        z_future = torch.randn_like(x0_future)
        xt_future = (1.0 - t) * x0_future + t * z_future

        xt_full = torch.randn_like(latents.float())
        xt_full[:, :, cond_count:cond_count + time_steps] = xt_future

        t_b1t11 = latents.new_zeros(cond_indicator.shape, dtype=torch.float32)
        t_b1t11[:, :, cond_count:cond_count + time_steps] = t
        t_b1t11 = t_b1t11.to(dtype=transformer_dtype)

        in_latents = (
            cond_mask_t * cond_latents.to(transformer_dtype) +
            (1.0 - cond_mask_t) * xt_full.to(transformer_dtype))
        in_timestep = (
            cond_indicator.to(transformer_dtype) *
            cond_timestep.to(transformer_dtype) +
            (1.0 - cond_indicator.to(transformer_dtype)) * t_b1t11)

        v_pred = self.transformer(
            hidden_states=in_latents,
            condition_mask=cond_mask_t,
            timestep=in_timestep,
            encoder_hidden_states=prompt_embeds,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]

        v_target = (z_future - x0_future).to(
            device=v_pred.device, dtype=v_pred.dtype)
        v_pred_future = v_pred[:, :, cond_count:cond_count + time_steps]
        return F.mse_loss(v_pred_future.float(), v_target.float())

    @staticmethod
    def _hidden_to_bsd(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            return hidden
        if hidden.ndim == 5:
            b, c, t, h, w = hidden.shape
            return hidden.permute(0, 2, 3, 4,
                                  1).contiguous().view(b, t * h * w, c)
        raise ValueError(
            f'Unsupported Cosmos hidden shape {tuple(hidden.shape)}. '
            'expected [B,S,D] or [B,C,T,H,W].')

    def _infer_num_frames_out(
        self,
        condition_video: torch.Tensor,
        future_video: Optional[torch.Tensor],
        num_frames_out: Optional[int],
    ) -> int:
        if num_frames_out is not None:
            return int(num_frames_out)
        if future_video is not None:
            return int(condition_video.shape[2] + future_video.shape[1])
        return int(condition_video.shape[2])

    def forward(
        self,
        images: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
        videos: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
        prompts: Optional[Union[str, Sequence[str]]] = None,
        task_description: Optional[Union[str, Sequence[str]]] = None,
        future_videos: Optional[torch.Tensor] = None,
        num_frames_out: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        conditional_frame_timestep: Optional[float] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs,
    ) -> Cosmos25BackboneOutput:
        _ = output_hidden_states, return_dict
        source_video = videos if videos is not None else images
        if source_video is None:
            raise ValueError('Cosmos25Backbone.forward requires `images` or '
                             '`videos`.')

        condition_video, inferred_future = self._normalize_videos(source_video)
        if future_videos is None:
            future_videos = inferred_future

        batch_size = condition_video.shape[0]
        prompt_list = self._resolve_prompts(
            batch_size,
            prompts=prompts,
            task_description=task_description,
            **kwargs,
        )
        condition_video = condition_video.to(self.device)
        height = int(condition_video.shape[-2])
        width = int(condition_video.shape[-1])
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f'Cosmos requires height/width divisible by 16, got '
                f'{height}x{width}.')

        if self.training:
            self.set_frozen_modules_to_eval_mode()

        transformer_dtype = getattr(self.transformer, 'dtype', self.dtype)
        prompt_embeds = self._get_prompt_embeds(
            prompt_list, device=self.device, dtype=transformer_dtype)
        num_frames_out = self._infer_num_frames_out(condition_video,
                                                    future_videos,
                                                    num_frames_out)
        compute_future_loss = (
            future_videos is not None
            and self._is_flow_matching_loss(self.future_loss_type))
        if future_videos is not None and self.future_loss_type is not None \
                and not compute_future_loss:
            raise NotImplementedError(
                'Cosmos25Backbone currently supports future video loss only '
                f'for flow-matching types, got {self.future_loss_type!r}.')
        if compute_future_loss:
            num_frames_out = max(
                int(num_frames_out),
                1 + max(1, self.vae_scale_factor_temporal))
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if conditional_frame_timestep is None:
            conditional_frame_timestep = self.conditional_frame_timestep
        latents, cond_latents, cond_mask, cond_indicator = (
            self._prepare_latents(condition_video, num_frames_out,
                                  torch.float32))

        self.scheduler.set_timesteps(
            max(1, int(num_inference_steps)), device=self.device)
        timesteps = self.scheduler.timesteps
        cond_mask_t = cond_mask.to(transformer_dtype)
        cond_timestep = torch.ones_like(cond_indicator) * float(
            conditional_frame_timestep)
        padding_mask = latents.new_zeros(
            1, 1, height, width, dtype=transformer_dtype)

        self._cached_hidden.clear()
        self._capture_hidden_enabled = True
        hidden_first = None

        for idx, timestep in enumerate(timesteps):
            sigma = torch.as_tensor(
                self.scheduler.sigmas[idx].item(),
                device=self.device,
                dtype=transformer_dtype,
            ).view(1)
            in_latents = (
                cond_mask_t * cond_latents.to(transformer_dtype) +
                (1.0 - cond_mask_t) * latents.to(transformer_dtype))
            in_timestep = cond_indicator.to(
                transformer_dtype) * cond_timestep.to(transformer_dtype) + (
                    1.0 - cond_indicator.to(transformer_dtype)) * sigma

            model_out = self.transformer(
                hidden_states=in_latents,
                condition_mask=cond_mask_t,
                timestep=in_timestep,
                encoder_hidden_states=prompt_embeds,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

            if idx == 0 and self._cached_hidden:
                hidden_first = self._cached_hidden[-1]
                self._capture_hidden_enabled = False
                break

            # Keep a minimal denoising path for callers that request more than
            # one step before the selected layer emits hidden states.
            latents = self.scheduler.step(
                model_out, timestep, latents, return_dict=False)[0]

        if hidden_first is None:
            if not self._cached_hidden:
                raise RuntimeError(
                    'No Cosmos transformer hidden state was captured. Check '
                    '`extract_layer` and the installed Cosmos transformer.')
            hidden_first = self._cached_hidden[-1]

        future_video_loss = None
        if compute_future_loss:
            future_video_loss = self._future_video_flow_matching_loss(
                condition_video=condition_video,
                future_videos=future_videos,
                latents=latents,
                cond_latents=cond_latents,
                cond_mask_t=cond_mask_t,
                cond_indicator=cond_indicator,
                cond_timestep=cond_timestep,
                prompt_embeds=prompt_embeds,
                padding_mask=padding_mask,
                transformer_dtype=transformer_dtype,
            )

        hidden = self._hidden_to_bsd(hidden_first)
        return Cosmos25BackboneOutput(
            hidden_states=[hidden],
            future_video_loss=future_video_loss,
            pred_future_video=None,
        )


__all__ = ['Cosmos25Backbone', 'Cosmos25BackboneOutput']
