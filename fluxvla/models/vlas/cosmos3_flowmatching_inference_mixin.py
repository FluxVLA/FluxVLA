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
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..third_party_models.cosmos3.data.vfm.sequence_packing import SequencePlan
from .cosmos3_flowmatching_utils import (_as_text_ids,
                                         _expand_sampler_timestep,
                                         _sample_arch_invariant_noise)


@dataclass
class _FlowRequest:
    batch_size: int
    device: torch.device
    sampling_config: Dict
    guidance: float
    text_ids: List[List[int]]
    negative_text_ids: List[List[int]]
    sequence_plan: SequencePlan
    conditioning_fps: Optional[torch.Tensor]
    action_fps: Optional[torch.Tensor]
    vision_latents: Optional[torch.Tensor]
    action_tokens: Optional[torch.Tensor]
    embodiment_id_value: Optional[int]
    raw_action_dim_value: Optional[int]


class Cosmos3InferenceMixin:

    def _sampling_config(self) -> Dict:
        sampling_config = dict(self.rectified_flow_inference_config)
        sampling_config['scheduler_type'] = str(
            sampling_config['scheduler_type']).lower()
        if sampling_config['scheduler_type'] != 'unipc':
            raise ValueError(
                'Cosmos3FlowMatching supports scheduler_type="unipc" only, '
                f'got {sampling_config["scheduler_type"]!r}.')
        return sampling_config

    def _build_unipc_scheduler(self, sampling_config: Dict,
                               device: torch.device):
        from ..third_party_models.cosmos3.model.vfm.diffusion.samplers import \
            fm_solvers_unipc  # noqa: E501

        use_dynamic_shifting = bool(sampling_config['use_dynamic_shifting'])
        scheduler = fm_solvers_unipc.FlowUniPCMultistepScheduler(
            num_train_timesteps=sampling_config['num_train_timesteps'],
            shift=sampling_config['shift'],
            use_dynamic_shifting=use_dynamic_shifting,
            final_sigmas_type='zero',
        )
        dynamic_shift_mu = sampling_config['dynamic_shift_mu']
        if use_dynamic_shifting and dynamic_shift_mu is None:
            dynamic_shift_mu = math.log(float(sampling_config['shift']))
        scheduler.set_timesteps(
            max(sampling_config['num_steps'], 1),
            device=device,
            shift=sampling_config['shift'],
            mu=dynamic_shift_mu,
            use_kerras_sigma=sampling_config['use_karras_sigmas'],
            sigma_min=sampling_config['sigma_min'],
            sigma_max=sampling_config['sigma_max'],
            rho=sampling_config['rho'],
        )
        return scheduler

    @staticmethod
    def _text_batch_size(text_token_ids) -> int:
        if text_token_ids is None:
            raise ValueError(
                'Cosmos3 flow inference requires `text_token_ids`.')
        if isinstance(text_token_ids, torch.Tensor):
            if text_token_ids.dim() == 1:
                return 1
            if text_token_ids.dim() != 2:
                raise ValueError(
                    'Cosmos3 text_token_ids must have shape [L] or [B,L], '
                    f'got {tuple(text_token_ids.shape)}.')
            batch_size = int(text_token_ids.shape[0])
        else:
            batch_size = len(text_token_ids)
        if batch_size < 1:
            raise ValueError('Cosmos3 text_token_ids must not be empty.')
        return batch_size

    @staticmethod
    def _has_generated_indexes(condition_indexes: List[int],
                               length: int) -> bool:
        conditioned = {
            int(index)
            for index in condition_indexes if 0 <= int(index) < length
        }
        return len(conditioned) < length

    def _plan_samples_vision(self, plan: SequencePlan,
                             num_frames: int) -> bool:
        return plan.has_vision and self._has_generated_indexes(
            plan.condition_frame_indexes_vision, num_frames)

    def _plan_samples_action(self, plan: SequencePlan,
                             action_length: int) -> bool:
        return plan.has_action and self._has_generated_indexes(
            plan.condition_frame_indexes_action, action_length)

    def _prepare_action_tokens(
        self,
        actions: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
        raw_action_dim_value: int,
    ) -> torch.Tensor:
        if actions.dim() != 3:
            raise ValueError('Cosmos3 actions must have shape [B,T,D], '
                             f'got {tuple(actions.shape)}.')
        if actions.shape[-1] > raw_action_dim_value:
            raise ValueError(f'Cosmos3 expected action dim <= '
                             f'{raw_action_dim_value}, got '
                             f'{actions.shape[-1]}.')
        actions = actions.to(device=device, dtype=dtype).clone()
        if actions.shape[-1] < self.max_action_dim:
            padding = actions.new_zeros(
                actions.shape[0],
                actions.shape[1],
                self.max_action_dim - actions.shape[-1],
            )
            actions = torch.cat([actions, padding], dim=-1)
        return actions

    @staticmethod
    def _stack_vision_latents(
            vision_tokens: Optional[List[torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        if vision_tokens is None:
            return None
        return torch.cat(vision_tokens, dim=0)

    @staticmethod
    def _vision_list_from_latents(
            latents: Optional[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        if latents is None:
            return None
        return [latents[i:i + 1] for i in range(latents.shape[0])]

    @staticmethod
    def _action_list_from_tokens(
            action_tokens: Optional[torch.Tensor]
    ) -> Optional[List[torch.Tensor]]:
        if action_tokens is None:
            return None
        return [action_tokens[i] for i in range(action_tokens.shape[0])]

    def _copy_conditioned_vision(
        self,
        target: torch.Tensor,
        source: Optional[torch.Tensor],
        sequence_plan: SequencePlan,
    ) -> torch.Tensor:
        if source is None:
            if sequence_plan.condition_frame_indexes_vision:
                raise ValueError('Vision condition indexes require `images`.')
            return target

        source = source.to(device=target.device, dtype=target.dtype)
        if source.shape[0] != target.shape[0]:
            raise ValueError('conditioning vision batch size does not match '
                             f'target latents: {source.shape[0]} vs '
                             f'{target.shape[0]}.')
        if source.shape[1] != target.shape[1]:
            raise ValueError(
                'conditioning vision channel count does not match '
                f'target latents: {source.shape[1]} vs '
                f'{target.shape[1]}.')
        if (source.shape[-2], source.shape[-1]) != (target.shape[-2],
                                                    target.shape[-1]):
            raise ValueError(
                'conditioning vision latent spatial shape does not '
                f'match target latents: {tuple(source.shape[-2:])} '
                f'vs {tuple(target.shape[-2:])}.')

        for sample_idx in range(target.shape[0]):
            for frame_idx in sequence_plan.condition_frame_indexes_vision:
                if frame_idx < 0 or frame_idx >= target.shape[2]:
                    continue
                if frame_idx >= source.shape[2]:
                    raise ValueError(
                        'conditioning vision latents do not contain frame '
                        f'{frame_idx}; source has {source.shape[2]} frames.')
                target[sample_idx:sample_idx + 1, :, frame_idx:frame_idx +
                       1] = source[sample_idx:sample_idx + 1, :,
                                   frame_idx:frame_idx + 1]
        return target

    @staticmethod
    def _copy_conditioned_actions(
        target: torch.Tensor,
        source: Optional[torch.Tensor],
        sequence_plan: SequencePlan,
    ) -> torch.Tensor:
        if source is None:
            if sequence_plan.condition_frame_indexes_action:
                raise ValueError('Action condition indexes require `actions`.')
            return target

        source = source.to(device=target.device, dtype=target.dtype)
        if source.shape[0] != target.shape[0]:
            raise ValueError('conditioning action batch size does not match '
                             f'target tokens: {source.shape[0]} vs '
                             f'{target.shape[0]}.')
        if source.shape[-1] != target.shape[-1]:
            raise ValueError('conditioning action dim does not match target '
                             f'tokens: {source.shape[-1]} vs '
                             f'{target.shape[-1]}.')
        for sample_idx in range(target.shape[0]):
            for frame_idx in sequence_plan.condition_frame_indexes_action:
                if frame_idx < 0 or frame_idx >= target.shape[1]:
                    continue
                if frame_idx >= source.shape[1]:
                    raise ValueError(
                        'conditioning actions do not contain step '
                        f'{frame_idx}; source has {source.shape[1]} steps.')
                target[sample_idx, frame_idx] = source[sample_idx, frame_idx]
        return target

    def _prepare_flow_vision_latents(
        self,
        *,
        conditioning_vision_tokens: Optional[List[torch.Tensor]],
        sequence_plan: SequencePlan,
        initial_latents: Optional[torch.Tensor],
        batch_size: int,
        num_frames: int,
        latent_height: Optional[int],
        latent_width: Optional[int],
        seed: Optional[int | List[int]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        condition_latents = self._stack_vision_latents(
            conditioning_vision_tokens)
        if condition_latents is not None and condition_latents.dim() != 5:
            raise ValueError(
                'conditioning vision latents must have shape [B,C,T,H,W], '
                f'got {tuple(condition_latents.shape)}.')
        if not self._plan_samples_vision(sequence_plan, num_frames):
            if condition_latents is None:
                raise ValueError('Vision condition indexes require `images`.')
            return condition_latents

        if (latent_height is None) != (latent_width is None):
            raise ValueError(
                '`latent_height` and `latent_width` must be provided '
                'together.')
        explicit_latent_size = latent_height is not None
        dtype = self._first_parameter(self.vision_in_proj).dtype
        if initial_latents is None:
            if condition_latents is not None and latent_height is None:
                latent_height = int(condition_latents.shape[-2])
                latent_width = int(condition_latents.shape[-1])
            elif latent_height is None:
                latent_height = 16
                latent_width = 16
            vision_latents = _sample_arch_invariant_noise(
                (self.latent_channel, num_frames, latent_height, latent_width),
                batch_size,
                dtype=dtype,
                device=device,
                seed=seed,
                label='vision',
            )
        else:
            vision_latents = initial_latents.to(device=device, dtype=dtype)
            if vision_latents.dim() != 5:
                raise ValueError(
                    'initial_latents must have shape [B,C,T,H,W], got '
                    f'{tuple(vision_latents.shape)}.')
            if vision_latents.shape[0] != batch_size:
                raise ValueError(
                    f'Expected initial_latents batch size {batch_size}, '
                    f'got {vision_latents.shape[0]}.')
            if (explicit_latent_size
                    and (int(latent_height), int(latent_width)) != tuple(
                        int(dim) for dim in vision_latents.shape[-2:])):
                raise ValueError(
                    'initial_latents spatial shape does not match requested '
                    'latent size: '
                    f'{tuple(vision_latents.shape[-2:])} vs '
                    f'{(latent_height, latent_width)}.')
        return self._copy_conditioned_vision(vision_latents, condition_latents,
                                             sequence_plan)

    def _prepare_flow_action_tokens(
        self,
        *,
        action_tokens_input: Optional[torch.Tensor],
        sequence_plan: SequencePlan,
        batch_size: int,
        action_horizon: int,
        raw_action_dim_value: int,
        seed: Optional[int | List[int]],
        device: torch.device,
    ) -> torch.Tensor:
        denoise_action = self._plan_samples_action(sequence_plan,
                                                   action_horizon)
        action_dtype = self._first_parameter(self.action_in_proj).dtype
        conditioned_actions = None
        if action_tokens_input is not None:
            conditioned_actions = self._prepare_action_tokens(
                action_tokens_input,
                dtype=action_dtype,
                device=device,
                raw_action_dim_value=raw_action_dim_value,
            )

        if not denoise_action:
            if conditioned_actions is None:
                raise ValueError('Action condition indexes require `actions`.')
            return conditioned_actions

        action_tokens = _sample_arch_invariant_noise(
            (action_horizon, self.max_action_dim),
            batch_size,
            dtype=action_dtype,
            device=device,
            seed=seed,
            label='action',
        )
        action_tokens = action_tokens.to(device=device, dtype=action_dtype)
        action_tokens[:, :, raw_action_dim_value:] = 0
        action_tokens = self._copy_conditioned_actions(action_tokens,
                                                       conditioned_actions,
                                                       sequence_plan)

        return action_tokens

    def _prepare_flow_request(
        self,
        *,
        images: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor | List[List[int]]] = None,
        negative_text_token_ids: Optional[torch.Tensor
                                          | List[List[int]]] = None,
        actions: Optional[torch.Tensor] = None,
        embodiment_id: Optional[int | torch.Tensor] = None,
        raw_action_dim: Optional[int] = None,
        sequence_plan: SequencePlan,
        num_frames: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        action_horizon: Optional[int] = None,
        initial_latents: Optional[torch.Tensor] = None,
        seed: Optional[int | List[int]] = None,
        guidance: float = 1.0,
        conditioning_fps: Optional[float | torch.Tensor] = None,
        action_fps: Optional[float | torch.Tensor] = None,
    ) -> _FlowRequest:
        device = self.device
        if sequence_plan is None:
            raise ValueError(
                'Cosmos3 flow inference requires `sequence_plan`.')
        batch_size = self._text_batch_size(text_token_ids)
        conditioning_vision_tokens = self._tokenize_vision(images)
        action_tokens_input = actions
        pad_token_id = self._text_pad_token_id()
        text_ids = _as_text_ids(
            text_token_ids,
            batch_size,
            pad_token_id=pad_token_id,
        )
        negative_text_ids = _as_text_ids(
            negative_text_token_ids,
            batch_size,
            pad_token_id=pad_token_id,
        )
        has_generation = False
        if sequence_plan.has_vision:
            if num_frames is None:
                raise ValueError(
                    'Cosmos3 vision SequencePlan requires explicit '
                    '`num_frames`.')
            has_generation = self._plan_samples_vision(sequence_plan,
                                                       num_frames)

        if sequence_plan.has_action:
            if action_horizon is None:
                raise ValueError(
                    'Cosmos3 action SequencePlan requires explicit '
                    '`action_horizon`.')
            has_generation = (
                has_generation
                or self._plan_samples_action(sequence_plan, action_horizon))
        if not has_generation:
            raise ValueError(
                'Cosmos3 flow inference SequencePlan has no noisy '
                'vision/action tokens to generate.')

        vision_latents = None
        if sequence_plan.has_vision:
            vision_latents = self._prepare_flow_vision_latents(
                conditioning_vision_tokens=conditioning_vision_tokens,
                sequence_plan=sequence_plan,
                initial_latents=initial_latents,
                batch_size=batch_size,
                num_frames=num_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                seed=seed,
                device=device,
            )

        action_tokens = None
        embodiment_id_value = None
        raw_action_dim_value = None
        if sequence_plan.has_action:
            if embodiment_id is None:
                raise ValueError(
                    'Cosmos3 action generation requires `embodiment_id`.')
            embodiment_id_value = int(embodiment_id)
            raw_action_dim_value = (
                int(self.ori_action_dim)
                if raw_action_dim is None else int(raw_action_dim))
            if (raw_action_dim_value < 1
                    or raw_action_dim_value > self.max_action_dim):
                raise ValueError(
                    'raw_action_dim must be in '
                    f'[1, {self.max_action_dim}], got {raw_action_dim_value}.')
            action_tokens = self._prepare_flow_action_tokens(
                action_tokens_input=action_tokens_input,
                sequence_plan=sequence_plan,
                batch_size=batch_size,
                action_horizon=action_horizon,
                raw_action_dim_value=raw_action_dim_value,
                seed=seed,
                device=device,
            )

        conditioning_fps_tensor = None
        action_fps_tensor = None
        if self.enable_fps_modulation:
            fps_dtype = self._first_parameter(self.vision_in_proj).dtype
            conditioning_fps_value = (
                self.base_fps
                if conditioning_fps is None else float(conditioning_fps))
            action_fps_value = (
                conditioning_fps_value
                if action_fps is None else float(action_fps))
            conditioning_fps_tensor = torch.full(
                (batch_size, ),
                conditioning_fps_value,
                device=device,
                dtype=fps_dtype,
            )
            action_fps_tensor = torch.full(
                (batch_size, ),
                action_fps_value,
                device=device,
                dtype=fps_dtype,
            )

        return _FlowRequest(
            batch_size=batch_size,
            device=device,
            sampling_config=self._sampling_config(),
            guidance=guidance,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            sequence_plan=sequence_plan,
            conditioning_fps=conditioning_fps_tensor,
            action_fps=action_fps_tensor,
            vision_latents=vision_latents,
            action_tokens=action_tokens,
            embodiment_id_value=embodiment_id_value,
            raw_action_dim_value=raw_action_dim_value,
        )

    def _restore_conditioned_vision(
        self,
        flow: _FlowRequest,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self._copy_conditioned_vision(latent_tokens,
                                             flow.vision_latents,
                                             flow.sequence_plan)

    def _restore_conditioned_actions(
        self,
        flow: _FlowRequest,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        assert flow.raw_action_dim_value is not None
        action_tokens = self._copy_conditioned_actions(action_tokens,
                                                       flow.action_tokens,
                                                       flow.sequence_plan)
        action_tokens[:, :, flow.raw_action_dim_value:] = 0
        return action_tokens

    def _flow_samples_vision(self, flow: _FlowRequest) -> bool:
        if flow.vision_latents is None:
            return False
        return self._plan_samples_vision(
            flow.sequence_plan,
            int(flow.vision_latents.shape[2]),
        )

    def _flow_samples_action(self, flow: _FlowRequest) -> bool:
        if flow.action_tokens is None:
            return False
        return self._plan_samples_action(
            flow.sequence_plan,
            int(flow.action_tokens.shape[1]),
        )

    def _predict_flow_velocity(
        self,
        flow: _FlowRequest,
        vision_latents: Optional[torch.Tensor],
        action_tokens: Optional[torch.Tensor],
        token_ids: List[List[int]],
        timestep,
    ) -> Dict[str, torch.Tensor]:
        timestep_value = _expand_sampler_timestep(
            timestep, batch_size=flow.batch_size, device=flow.device)
        needs_vision_velocity = self._flow_samples_vision(flow)
        needs_action_velocity = self._flow_samples_action(flow)

        if needs_vision_velocity:
            vision_latents = self._restore_conditioned_vision(
                flow, vision_latents.clone())
        vision_tokens = self._vision_list_from_latents(vision_latents)

        if needs_action_velocity:
            action_tokens = self._restore_conditioned_actions(
                flow, action_tokens.clone())
        action_tokens_for_pack = self._action_list_from_tokens(action_tokens)
        action_domain_ids = None
        raw_action_dims = None
        if flow.embodiment_id_value is not None:
            assert flow.raw_action_dim_value is not None
            action_domain_id = torch.tensor(
                [flow.embodiment_id_value],
                dtype=torch.long,
                device=flow.device,
            )
            raw_action_dim = torch.tensor(
                [flow.raw_action_dim_value],
                dtype=torch.long,
                device=flow.device,
            )
            action_domain_ids = [action_domain_id] * flow.batch_size
            raw_action_dims = [raw_action_dim] * flow.batch_size

        packed_seq = self._pack_generation_data(
            sequence_plans=[flow.sequence_plan] * flow.batch_size,
            text_token_ids=token_ids,
            vision_tokens=vision_tokens,
            action_tokens=action_tokens_for_pack,
            embodiment_ids=action_domain_ids,
            raw_action_dim=raw_action_dims,
            timesteps=timestep_value,
            fps_vision=flow.conditioning_fps,
            fps_action=flow.action_fps,
        )
        packed_sequence, target_dtype = self._encode_text(packed_seq)
        original_latent_shapes = self._encode_vision(packed_seq,
                                                     packed_sequence,
                                                     target_dtype)
        self._encode_action(packed_seq, packed_sequence, target_dtype)
        last_hidden_state = self._run_backbone(packed_seq, packed_sequence)

        velocities: Dict[str, torch.Tensor] = {}
        if needs_vision_velocity:
            decoded_vision = self._decode_vision(
                packed_seq,
                last_hidden_state,
                original_latent_shapes,
            )
            velocities['vision'] = torch.cat(
                decoded_vision, dim=0).to(dtype=vision_latents.dtype)
        if needs_action_velocity:
            assert flow.raw_action_dim_value is not None
            decoded_actions = self._decode_action(packed_seq,
                                                  last_hidden_state)
            velocity_action = torch.stack(
                decoded_actions, dim=0).to(dtype=action_tokens.dtype)
            velocity_action[:, :, flow.raw_action_dim_value:] = 0
            velocities['action'] = velocity_action
        return velocities

    def _guided_flow_velocity(
        self,
        flow: _FlowRequest,
        vision_latents: Optional[torch.Tensor],
        action_tokens: Optional[torch.Tensor],
        timestep,
    ) -> Dict[str, torch.Tensor]:
        cond_velocity = self._predict_flow_velocity(
            flow,
            vision_latents,
            action_tokens,
            flow.text_ids,
            timestep,
        )
        if flow.guidance == 1.0:
            return cond_velocity
        uncond_velocity = self._predict_flow_velocity(
            flow,
            vision_latents,
            action_tokens,
            flow.negative_text_ids,
            timestep,
        )
        return {
            key: uncond_velocity[key] + flow.guidance *
            (cond_velocity[key] - uncond_velocity[key])
            for key in cond_velocity
        }

    def _target_order(self, flow: _FlowRequest) -> List[str]:
        targets = []
        if self._flow_samples_vision(flow):
            targets.append('vision')
        if self._flow_samples_action(flow):
            targets.append('action')
        return targets

    def _run_unipc_flow(
        self,
        flow: _FlowRequest,
        vision_latents: Optional[torch.Tensor],
        action_tokens: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        target_order = self._target_order(flow)
        schedulers = {
            target: self._build_unipc_scheduler(flow.sampling_config,
                                                flow.device)
            for target in target_order
        }
        for timestep in schedulers[target_order[0]].timesteps:
            velocities = self._guided_flow_velocity(flow, vision_latents,
                                                    action_tokens, timestep)
            if 'vision' in target_order:
                vision_latents = schedulers['vision'].step(
                    model_output=velocities['vision'],
                    timestep=timestep,
                    sample=vision_latents,
                    return_dict=False,
                )[0]
                vision_latents = self._restore_conditioned_vision(
                    flow, vision_latents)
            if 'action' in target_order:
                action_tokens = schedulers['action'].step(
                    model_output=velocities['action'],
                    timestep=timestep,
                    sample=action_tokens,
                    return_dict=False,
                )[0]
                action_tokens = self._restore_conditioned_actions(
                    flow, action_tokens)
        return vision_latents, action_tokens

    def _format_flow_outputs(
        self,
        flow: _FlowRequest,
        vision_latents: Optional[torch.Tensor],
        action_tokens: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        outputs: Dict[str, torch.Tensor | List[torch.Tensor]] = {}
        if self._flow_samples_vision(flow):
            outputs['vision_latents'] = vision_latents
        if self._flow_samples_action(flow):
            outputs['actions'] = action_tokens
        return outputs

    @torch.no_grad()
    def _generate_flow(
        self,
        flow: _FlowRequest,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        vision_latents = (
            flow.vision_latents.clone()
            if flow.vision_latents is not None else None)
        action_tokens = (
            flow.action_tokens.clone()
            if flow.action_tokens is not None else None)
        if self._flow_samples_vision(flow):
            vision_latents = self._restore_conditioned_vision(
                flow, vision_latents)
        if self._flow_samples_action(flow):
            action_tokens = self._restore_conditioned_actions(
                flow, action_tokens)

        vision_latents, action_tokens = self._run_unipc_flow(
            flow, vision_latents, action_tokens)
        return self._format_flow_outputs(flow, vision_latents, action_tokens)

    @torch.no_grad()
    def generate_vision_latents(
        self,
        *,
        images: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor | List[List[int]]] = None,
        negative_text_token_ids: Optional[torch.Tensor
                                          | List[List[int]]] = None,
        actions: Optional[torch.Tensor] = None,
        embodiment_id: Optional[int] = None,
        raw_action_dim: Optional[int] = None,
        sequence_plan: SequencePlan,
        num_frames: int,
        action_horizon: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        initial_latents: Optional[torch.Tensor] = None,
        seed: Optional[int | List[int]] = 7,
        guidance: float = 1.0,
        conditioning_fps: Optional[float] = None,
        action_fps: Optional[float] = None,
    ) -> torch.Tensor:
        flow = self._prepare_flow_request(
            images=images,
            text_token_ids=text_token_ids,
            negative_text_token_ids=negative_text_token_ids,
            actions=actions,
            embodiment_id=embodiment_id,
            raw_action_dim=raw_action_dim,
            sequence_plan=sequence_plan,
            num_frames=num_frames,
            action_horizon=action_horizon,
            latent_height=latent_height,
            latent_width=latent_width,
            initial_latents=initial_latents,
            seed=seed,
            guidance=guidance,
            conditioning_fps=conditioning_fps,
            action_fps=action_fps,
        )
        return self._generate_flow(flow)['vision_latents']

    @torch.no_grad()
    def generate_inverse_dynamics(
        self,
        *,
        images: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor | List[List[int]]] = None,
        negative_text_token_ids: Optional[torch.Tensor
                                          | List[List[int]]] = None,
        embodiment_id: Optional[int] = None,
        raw_action_dim: Optional[int] = None,
        sequence_plan: SequencePlan,
        num_frames: int,
        action_horizon: int,
        seed: Optional[int | List[int]] = None,
        guidance: float = 1.0,
        conditioning_fps: Optional[float] = None,
        action_fps: Optional[float] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        action_dim = int(
            self.ori_action_dim if raw_action_dim is None else raw_action_dim)
        flow = self._prepare_flow_request(
            images=images,
            text_token_ids=text_token_ids,
            negative_text_token_ids=negative_text_token_ids,
            actions=None,
            embodiment_id=embodiment_id,
            raw_action_dim=action_dim,
            sequence_plan=sequence_plan,
            num_frames=num_frames,
            action_horizon=action_horizon,
            seed=seed,
            guidance=guidance,
            conditioning_fps=conditioning_fps,
            action_fps=action_fps,
        )
        result = self._generate_flow(flow)
        result['actions'] = result['actions'][..., :action_dim].clone()
        return result

    @torch.no_grad()
    def generate_joint(
        self,
        *,
        images: torch.Tensor,
        text_token_ids: Optional[torch.Tensor | List[List[int]]] = None,
        embodiment_id: Optional[int] = None,
        raw_action_dim: Optional[int] = None,
        sequence_plan: SequencePlan,
        num_frames: int,
        action_horizon: int,
        seed: Optional[int | List[int]] = None,
        guidance: float = 1.0,
        conditioning_fps: Optional[float] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        action_dim = int(
            self.ori_action_dim if raw_action_dim is None else raw_action_dim)
        flow = self._prepare_flow_request(
            images=images,
            text_token_ids=text_token_ids,
            embodiment_id=embodiment_id,
            raw_action_dim=action_dim,
            sequence_plan=sequence_plan,
            num_frames=num_frames,
            action_horizon=action_horizon,
            seed=seed,
            guidance=guidance,
            conditioning_fps=conditioning_fps,
            action_fps=conditioning_fps,
        )
        result = self._generate_flow(flow)
        result['actions'] = result['actions'][..., :action_dim].clone()
        return result

    def _prepare_reasoner_input_ids(self, input_ids, pad_token_id: int):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            return input_ids, None

        if isinstance(input_ids, (list, tuple)):
            if len(input_ids) == 0:
                raise ValueError('input_ids must not be empty.')
            if isinstance(input_ids[0], int):
                return torch.tensor(
                    [input_ids],
                    device=self.device,
                    dtype=torch.long,
                ), None

            rows = [list(row) for row in input_ids]
            max_len = max(len(row) for row in rows)
            if max_len == 0:
                raise ValueError('input_ids rows must not be empty.')
            tensor = torch.full(
                (len(rows), max_len),
                int(pad_token_id),
                device=self.device,
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(tensor)
            for index, row in enumerate(rows):
                row_tensor = torch.tensor(
                    row,
                    device=self.device,
                    dtype=torch.long,
                )
                tensor[index, :row_tensor.numel()] = row_tensor
                attention_mask[index, :row_tensor.numel()] = 1
            return tensor, attention_mask

        raise TypeError('input_ids must be a tensor or a list of token ids.')

    @torch.no_grad()
    def generate_reasoner_text(
        self,
        input_ids=None,
        *,
        text_token_ids=None,
        max_new_tokens: int = 128,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int | List[int]] = None,
        pad_token_id: Optional[int] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        if input_ids is None:
            input_ids = text_token_ids
        if input_ids is None:
            raise ValueError('generate_reasoner_text requires `input_ids` or '
                             '`text_token_ids`.')

        if eos_token_id is None:
            eos_token_id = self.special_tokens.get('eos_token_id')
        if pad_token_id is None:
            pad_token_id = self.special_tokens.get(
                'pad_token_id',
                eos_token_id if isinstance(eos_token_id, int) else 0,
            )

        input_ids, inferred_attention_mask = self._prepare_reasoner_input_ids(
            input_ids, int(pad_token_id))
        if attention_mask is None:
            attention_mask = inferred_attention_mask
        elif not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.as_tensor(
                attention_mask,
                device=self.device,
                dtype=torch.long,
            )
        else:
            attention_mask = attention_mask.to(device=self.device)
        if isinstance(attention_mask,
                      torch.Tensor) and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        for key, value in list(generation_kwargs.items()):
            if isinstance(value, torch.Tensor):
                generation_kwargs[key] = value.to(device=self.device)

        if (pixel_values is not None
                or image_grid_thw is not None) and not bool(
                    getattr(self.vlm_backbone, 'include_visual', False)):
            raise ValueError(
                'Cosmos3 reasoner image inputs require '
                '`vlm_backbone.include_visual=True`; the default VLA config '
                'keeps the Qwen3VL visual tower unloaded.')

        if pixel_values is not None:
            pixel_values = pixel_values.to(device=self.device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=self.device)

        return_only_new_tokens = bool(
            generation_kwargs.pop('return_only_new_tokens', False))
        output_ids = self.vlm_backbone.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **generation_kwargs,
        )
        if return_only_new_tokens:
            return output_ids[:, input_ids.shape[1]:]
        return output_ids

    def _policy_condition_actions_from_states(self, states) -> torch.Tensor:
        if states.ndim != 2 or states.shape[-1] != self.max_action_dim:
            raise ValueError(
                'Cosmos3 predict_action expects states with shape '
                f'[B, {self.max_action_dim}], got {tuple(states.shape)}.')
        action_dtype = self._first_parameter(self.action_in_proj).dtype
        states = states.to(device=self.device, dtype=action_dtype)
        return states.unsqueeze(1)

    def _policy_num_vision_latents(self, action_horizon: int) -> int:
        temporal_compression = int(self.vision_vae.temporal_compression_factor)
        if action_horizon % temporal_compression != 0:
            raise ValueError(
                'Cosmos3 policy inference expects action_horizon to match '
                'the Wan VAE 4n+1 video convention; got '
                f'action_horizon={action_horizon}, '
                f'temporal_compression_factor={temporal_compression}.')
        return action_horizon // temporal_compression + 1

    @staticmethod
    def _policy_sequence_plan(prepend_state_to_action: bool) -> SequencePlan:
        return SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=[0],
            has_action=True,
            condition_frame_indexes_action=([0] if prepend_state_to_action else
                                            []),
            action_start_frame_offset=(0 if prepend_state_to_action else 1),
        )

    @torch.no_grad()
    def _predict_action_joint(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        sequence_plan: SequencePlan,
        states: Optional[torch.Tensor] = None,
        embodiment_id: Optional[int | torch.Tensor] = None,
        raw_action_dim: Optional[int | torch.Tensor] = None,
        conditioning_fps: Optional[float | torch.Tensor] = None,
        prepend_state_to_action: bool | torch.Tensor = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        del kwargs
        future_horizon = int(self.action_horizon)
        if future_horizon < 1:
            raise ValueError(
                'Cosmos3 predict_action requires action_horizon > 0.')

        actions = None
        if prepend_state_to_action:
            if states is None:
                raise ValueError(
                    'Cosmos3 predict_action requires `states` when '
                    'prepend_state_to_action=True.')
            actions = self._policy_condition_actions_from_states(states)

        flow = self._prepare_flow_request(
            images=images,
            text_token_ids=lang_tokens,
            actions=actions,
            embodiment_id=embodiment_id,
            raw_action_dim=raw_action_dim,
            sequence_plan=sequence_plan,
            num_frames=self._policy_num_vision_latents(future_horizon),
            action_horizon=future_horizon + int(prepend_state_to_action),
            conditioning_fps=conditioning_fps,
            action_fps=conditioning_fps,
        )
        return self._generate_flow(flow)

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        raw_action_dim: Optional[int | torch.Tensor] = None,
        conditioning_fps: Optional[torch.Tensor] = None,
        prepend_state_to_action: bool | torch.Tensor = False,
        decode_video: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | List[torch.Tensor]]:
        sequence_plan = self._policy_sequence_plan(prepend_state_to_action)
        result = self._predict_action_joint(
            images=images,
            lang_tokens=lang_tokens,
            states=states,
            embodiment_id=embodiment_ids,
            raw_action_dim=raw_action_dim,
            conditioning_fps=conditioning_fps,
            prepend_state_to_action=prepend_state_to_action,
            sequence_plan=sequence_plan,
            **kwargs,
        )
        action_dim = int(
            self.ori_action_dim if raw_action_dim is None else raw_action_dim)
        actions = result['actions'][..., :action_dim].clone()
        if prepend_state_to_action:
            actions = actions[:, 1:]
        if not decode_video:
            return actions
        return actions, self.decode_vision_latents(result['vision_latents'])
