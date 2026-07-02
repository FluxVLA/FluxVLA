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
from typing import List, Optional

import torch

from ..third_party_models.cosmos3.data.vfm.sequence_packing import (
    GenerationDataClean, PackedSequence, SequencePlan, pack_input_sequence)
from ..third_party_models.cosmos3.model.vfm.diffusion.rectified_flow import \
    TrainTimeSampler
from .cosmos3_flowmatching_utils import (_get_vision_data_resolution,
                                         _move_packed_sequence_to_device)


class Cosmos3ScheduleMixin:
    """Rectified-flow schedule and noising helpers.

    The original Cosmos3 code keeps image/video/action schedule semantics close
    to the VFM training loop.  FluxVLA keeps that behavior at the VLA layer so
    configs can expose the training and inference flow blocks directly.
    """

    def _sample_train_time(self, batch_size: int,
                           distribution: str) -> torch.Tensor:
        return TrainTimeSampler(distribution)(
            batch_size,
            device=self.device,
            dtype=torch.float32,
        )

    def _resolve_vision_resolutions(
        self,
        images: Optional[torch.Tensor],
        vision_tokens: Optional[List[torch.Tensor]],
        batch_size: int,
    ) -> Optional[List[str]]:
        if images is not None:
            if not isinstance(images, torch.Tensor):
                raise TypeError(
                    'Cosmos3FlowMatching expects images to be a torch.Tensor.')
            if images.dim() == 5:
                return [
                    _get_vision_data_resolution(
                        (int(images.shape[-2]), int(images.shape[-1])))
                    for _ in range(images.shape[0])
                ]
            if images.dim() == 4:
                return [
                    _get_vision_data_resolution(
                        (int(images.shape[-2]), int(images.shape[-1])))
                    for _ in range(batch_size)
                ]
            raise ValueError(
                f'images must have shape [B,C,T,H,W] or [C,T,H,W], '
                f'got {images.shape}')

        if vision_tokens is not None:
            spatial = self.vision_vae.spatial_compression_factor
            resolutions = []
            for latent in vision_tokens:
                resolutions.append(
                    _get_vision_data_resolution(
                        (int(latent.shape[-2]) * spatial,
                         int(latent.shape[-1]) * spatial)))
            return resolutions
        return None

    def _resolve_flow_shifts(
        self,
        shift_config,
        *,
        batch_size: int,
        for_action: bool,
        vision_resolutions: Optional[List[str]] = None,
        num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if isinstance(shift_config, (int, float)):
            return torch.full(
                (batch_size, ),
                float(shift_config),
                device=self.device,
                dtype=torch.float32,
            )
        if not isinstance(shift_config, dict):
            raise TypeError(f'Flow shift must be a number or dict, '
                            f'got {type(shift_config).__name__}.')
        if for_action:
            raise ValueError(
                'shift_action must be an int/float for independent action '
                'schedules. Dict-valued shift is vision-only, matching the '
                'original Cosmos3 implementation.')

        shift_dict = dict(shift_config)
        if 'dynamic_shift_base_num_tokens_video' in shift_dict:
            if num_tokens is None:
                raise ValueError(
                    'Dynamic shift requires num_tokens per vision sample.')
            base_num_tokens = float(
                shift_dict['dynamic_shift_base_num_tokens_video'])
            return torch.sqrt(
                torch.tensor(
                    num_tokens, device=self.device, dtype=torch.float32) /
                base_num_tokens)

        resolutions = vision_resolutions
        if resolutions is None:
            resolution = self.rectified_flow_training_config.get('resolution')
            if resolution is None:
                raise ValueError(
                    'Dict-valued rectified-flow `shift` requires pixel '
                    'images, vision latents, or `resolution` in '
                    'rectified_flow_training_config.')
            if isinstance(resolution, (str, int)):
                resolutions = [str(resolution)] * batch_size
            else:
                resolutions = [str(item) for item in resolution]
        if len(resolutions) != batch_size:
            raise ValueError(f'Expected {batch_size} vision resolutions, got '
                             f'{len(resolutions)}.')

        shifts = []
        for resolution in resolutions:
            key = str(resolution)
            if key not in shift_dict:
                raise ValueError(
                    f'Resolution {key!r} not found in rectified-flow shift '
                    f'dict. Available keys: {sorted(shift_dict)}.')
            shifts.append(float(shift_dict[key]))
        return torch.tensor(shifts, device=self.device, dtype=torch.float32)

    def _apply_high_sigma_strategy(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.rectified_flow_training_config
        mask = torch.rand_like(timesteps) < float(cfg['high_sigma_ratio'])
        high = torch.rand_like(timesteps) * (
            float(cfg['high_sigma_timesteps_max']) -
            float(cfg['high_sigma_timesteps_min'])) + float(
                cfg['high_sigma_timesteps_min'])
        return torch.where(mask, high, timesteps)

    def _sample_flow_schedule(
        self,
        *,
        batch_size: int,
        modality: str,
        vision_resolutions: Optional[List[str]] = None,
        num_tokens: Optional[List[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.rectified_flow_training_config
        if modality == 'action':
            distribution = cfg['train_time_action_distribution']
            shift_config = cfg['shift_action']
            if shift_config is None:
                shift_config = cfg['shift']
            use_high_sigma = bool(cfg['use_high_sigma_strategy_action'])
            shifts = self._resolve_flow_shifts(
                shift_config,
                batch_size=batch_size,
                for_action=True,
            )
        elif modality == 'vision':
            distribution = cfg['train_time_video_distribution']
            shifts = self._resolve_flow_shifts(
                cfg['shift'],
                batch_size=batch_size,
                for_action=False,
                vision_resolutions=vision_resolutions,
                num_tokens=num_tokens,
            )
            use_high_sigma = bool(cfg['use_high_sigma_strategy'])
        else:
            raise ValueError(f'Unsupported flow schedule modality: {modality}')

        t_raw = self._sample_train_time(batch_size, distribution)
        t = 1.0 - t_raw
        timesteps = shifts * t / (1.0 + (shifts - 1.0) * t)
        timesteps = timesteps * float(self.num_train_timesteps)
        if use_high_sigma:
            timesteps = self._apply_high_sigma_strategy(timesteps)
        sigmas = timesteps / float(self.num_train_timesteps)
        return timesteps.unsqueeze(1), sigmas.clamp(0.0, 1.0).unsqueeze(1)

    def _sample_noisy_inputs(
        self,
        vision_tokens: Optional[List[torch.Tensor]],
        raw_state_action: Optional[List[torch.Tensor]],
        sequence_plans: List[SequencePlan],
        raw_action_dim: Optional[List[torch.Tensor]],
        vision_resolutions: Optional[List[str]],
    ) -> tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]],
               Optional[List[torch.Tensor]], Optional[List[torch.Tensor]],
               torch.Tensor, Optional[torch.Tensor]]:
        device = self.device
        batch_size = len(sequence_plans)
        num_tokens = None
        if vision_tokens is not None:
            num_tokens = [
                int(latent.shape[-3] * latent.shape[-2] * latent.shape[-1])
                for latent in vision_tokens
            ]
        timesteps_vision, sigmas_vision = self._sample_flow_schedule(
            batch_size=batch_size,
            modality='vision',
            vision_resolutions=vision_resolutions,
            num_tokens=num_tokens,
        )
        if (raw_state_action is not None and self.
                rectified_flow_training_config['independent_action_schedule']):
            timesteps_action, sigmas_action = self._sample_flow_schedule(
                batch_size=batch_size,
                modality='action',
            )
        else:
            timesteps_action = None
            sigmas_action = sigmas_vision

        noised_vision = None
        target_vision = None
        if vision_tokens is not None:
            noised_vision = []
            target_vision = []
            for latent, plan, sigma in zip(vision_tokens, sequence_plans,
                                           sigmas_vision):
                latent = latent.to(device=device)
                eps = torch.randn_like(latent)
                mask = torch.ones((1, 1, latent.shape[2], 1, 1),
                                  device=device,
                                  dtype=latent.dtype)
                if plan.condition_frame_indexes_vision:
                    clean_idx = torch.tensor(
                        plan.condition_frame_indexes_vision,
                        device=device,
                        dtype=torch.long)
                    clean_idx = clean_idx[(clean_idx >= 0)
                                          & (clean_idx < latent.shape[2])]
                    mask[:, :, clean_idx] = 0
                sigma = sigma.to(dtype=latent.dtype)
                xt = (1.0 - sigma) * latent + sigma * eps
                vt = eps - latent
                noised_vision.append(torch.where(mask.bool(), xt, latent))
                target_vision.append(vt * mask)

        noised_action = None
        target_action = None
        if raw_state_action is not None:
            noised_action = []
            target_action = []
            for index, (action, plan, sigma) in enumerate(
                    zip(raw_state_action, sequence_plans, sigmas_action)):
                action = action.to(device=device)
                eps = torch.randn_like(action)
                mask = torch.ones((action.shape[0], 1),
                                  device=device,
                                  dtype=action.dtype)
                if plan.condition_frame_indexes_action:
                    clean_idx = torch.tensor(
                        plan.condition_frame_indexes_action,
                        device=device,
                        dtype=torch.long)
                    clean_idx = clean_idx[(clean_idx >= 0)
                                          & (clean_idx < action.shape[0])]
                    mask[clean_idx] = 0

                valid_dim = int(raw_action_dim[index].item(
                )) if raw_action_dim is not None else self.max_action_dim
                dim_mask = torch.arange(
                    action.shape[-1], device=device) < valid_dim
                dim_mask = dim_mask.to(dtype=action.dtype).view(1, -1)
                sigma = sigma.to(dtype=action.dtype)
                xt = (1.0 - sigma) * action + sigma * eps
                vt = (eps - action) * dim_mask
                noised_action.append(
                    torch.where(mask.bool(), xt, action) * dim_mask)
                target_action.append(vt * mask)

        return (noised_vision, target_vision, noised_action, target_action,
                timesteps_vision, timesteps_action)

    def _override_action_timesteps(
        self,
        packed_seq: PackedSequence,
        timesteps_action: Optional[torch.Tensor],
    ) -> None:
        if timesteps_action is None or packed_seq.action is None:
            return

        # pack_input_sequence only accepts one timestep tensor, so
        # action tokens initially receive the vision timestep. When action
        # uses an independent schedule, replace those per-noisy-token
        # timesteps with action timesteps.
        sample_timesteps = timesteps_action.to(
            device=self.device, dtype=torch.float32).view(-1)
        noisy_frame_indexes = packed_seq.action.noisy_frame_indexes
        if not any(indexes.numel() > 0 for indexes in noisy_frame_indexes):
            return

        if sample_timesteps.numel() != len(noisy_frame_indexes):
            raise ValueError(
                'Expected one action timestep per action sample, got '
                f'{sample_timesteps.numel()} timesteps for '
                f'{len(noisy_frame_indexes)} action samples.')
        expanded = [
            sample_timesteps[index:index + 1].expand(indexes.numel())
            for index, indexes in enumerate(noisy_frame_indexes)
            if indexes.numel() > 0
        ]
        packed_seq.action.timesteps = (
            torch.cat(expanded, dim=0)
            if expanded else sample_timesteps.new_empty((0, )))


class Cosmos3LossMixin:
    """Flow-matching losses for action and Wan latent predictions."""

    def _compute_action_loss(
        self,
        preds_action: List[torch.Tensor],
        target_action: Optional[List[torch.Tensor]],
        raw_action_dim: Optional[List[torch.Tensor]],
        condition_mask: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if target_action is None:
            if preds_action:
                return 0.0 * sum(pred.sum() for pred in preds_action)
            return next(self.parameters()).sum() * 0.0
        if not preds_action:
            return next(self.parameters()).sum() * 0.0

        normalize_by_active = bool(
            self.rectified_flow_training_config['normalize_loss_by_active'])
        losses = []
        for index, (pred,
                    target) in enumerate(zip(preds_action, target_action)):
            valid_dim = int(raw_action_dim[index].item()
                            ) if raw_action_dim is not None else pred.shape[-1]
            sqerr = (pred[:, :valid_dim].float() -
                     target[:, :valid_dim].float())**2
            if condition_mask is None:
                losses.append(sqerr.mean())
                continue
            noisy_mask = 1.0 - condition_mask[index].to(
                device=sqerr.device, dtype=sqerr.dtype)
            if normalize_by_active:
                active_count = (noisy_mask.sum() *
                                (sqerr.numel() // noisy_mask.numel())).clamp(
                                    min=1)
                losses.append((sqerr * noisy_mask).sum() / active_count)
            else:
                losses.append((sqerr * noisy_mask).mean())
        return torch.stack(losses).mean() if losses else next(
            self.parameters()).sum() * 0.0

    def _compute_vision_loss(
        self,
        preds_vision: List[torch.Tensor],
        target_vision: Optional[List[torch.Tensor]],
        condition_mask: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if target_vision is None:
            if preds_vision:
                return 0.0 * sum(pred.sum() for pred in preds_vision)
            return next(self.parameters()).sum() * 0.0
        if not preds_vision:
            return next(self.parameters()).sum() * 0.0
        normalize_by_active = bool(
            self.rectified_flow_training_config['normalize_loss_by_active'])
        losses = []
        for index, (pred,
                    target) in enumerate(zip(preds_vision, target_vision)):
            sqerr = (pred.float() - target.float())**2
            if condition_mask is None:
                losses.append(sqerr.mean())
                continue
            noisy_mask = 1.0 - condition_mask[index].to(
                device=sqerr.device, dtype=sqerr.dtype)
            noisy_mask = noisy_mask.view(1, 1, noisy_mask.shape[0], 1, 1)
            if normalize_by_active:
                active_count = (noisy_mask.sum() *
                                (sqerr.numel() // noisy_mask.numel())).clamp(
                                    min=1)
                losses.append((sqerr * noisy_mask).sum() / active_count)
            else:
                losses.append((sqerr * noisy_mask).mean())
        return torch.stack(losses).mean() if losses else next(
            self.parameters()).sum() * 0.0


class Cosmos3SequenceMixin:

    def _pack_generation_data(
        self,
        *,
        sequence_plans: List[SequencePlan],
        text_token_ids: List[List[int]],
        vision_tokens: Optional[List[torch.Tensor]],
        action_tokens: Optional[List[torch.Tensor]],
        embodiment_ids: Optional[List[torch.Tensor]],
        raw_action_dim: Optional[List[torch.Tensor]],
        timesteps: torch.Tensor,
        fps_vision: Optional[torch.Tensor],
        fps_action: Optional[torch.Tensor],
    ) -> PackedSequence:
        gen_data_clean = GenerationDataClean(
            batch_size=len(sequence_plans),
            is_image_batch=False,
            x0_tokens_vision=vision_tokens,
            fps_vision=fps_vision,
            x0_tokens_action=action_tokens,
            fps_action=fps_action,
            action_domain_id=embodiment_ids,
            raw_action_dim=raw_action_dim,
        )
        packed_seq = pack_input_sequence(
            sequence_plans=sequence_plans,
            input_text_indexes=text_token_ids,
            gen_data_clean=gen_data_clean,
            input_timesteps=timesteps.detach().cpu(),
            special_tokens=self.special_tokens,
            latent_patch_size=self.latent_patch_size,
            include_end_of_generation_token=False,
            position_embedding_type=self.position_embedding_type,
            unified_3d_mrope_reset_spatial_ids=self.
            unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.
            unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.enable_fps_modulation,
            base_fps=self.base_fps,
            temporal_compression_factor=(
                self.vision_vae.temporal_compression_factor),
            action_dim=self.max_action_dim,
        )
        return _move_packed_sequence_to_device(packed_seq, self.device)
