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
"""VPP2 policy with episode history supplied by the evaluation client."""

from copy import deepcopy
from pathlib import Path

import torch

from fluxvla.engines import (VLAS, build_head_from_cfg,
                             build_vlm_backbone_from_cfg)
from fluxvla.engines.utils.name_map import str_to_dtype
from .base_vla import BaseVLA


def replan_seeds(replan_index, seed_mode='vpp2', seed=None, base=1):
    """Derive the action and video noise seeds for one replan.

    Args:
        replan_index (int | torch.Tensor): Zero-based replan counter within
            the episode, supplied by the history-enabled policy client.
        seed_mode (str): ``'vpp2'`` offsets ``base`` as in the reference
            evaluation; ``'policy'`` offsets the request's policy ``seed``.
        seed (int | None): Policy seed, required for ``seed_mode='policy'``.
        base (int): Base seed for ``seed_mode='vpp2'``.

    Returns:
        tuple[int, int]: ``(action_seed, video_seed)``, where
        ``video_seed = action_seed + 1000003``.
    """
    if replan_index is None:
        raise ValueError(
            'replan_index is required: use FluxVLAPolicyClient with history '
            'enabled, which supplies camera history and replan_index')
    value = torch.as_tensor(replan_index)
    if value.numel(
    ) != 1 or value.dtype == torch.bool or value.is_floating_point(
    ) or value.item() < 0:
        raise ValueError('replan_index must be one nonnegative integer')
    if seed_mode not in ('vpp2', 'policy'):
        raise ValueError('seed_mode must be vpp2 or policy')
    if seed_mode == 'policy' and seed is None:
        raise ValueError('seed_mode=policy requires a policy seed')
    action = int(base if seed_mode == 'vpp2' else seed) + int(value.item())
    return action, action + 1000003


@VLAS.register_module()
class VPP2VLA(BaseVLA):
    """VPP2 (Video Prediction Policy 2) inference policy.

    Implemented from the VPP2 reference code:
    https://github.com/roboterax/video-prediction-policy-2
    (revision 3365872910244f14df83003062f7c563f14c6d9e), with released
    weights at https://huggingface.co/Haodong082399/VPP2. See
    ``docs/vpp2.md`` for checkpoint conversion and evaluation.

    ``VPP2Wan21Backbone`` (vlm_backbone) encodes the condition video (VAE,
    CLIP) and prompt (T5); ``VPP2Head`` (vla_head) samples actions. The
    evaluation client owns episode history and supplies ``replan_index``,
    from which the action and video noise seeds are derived.

    Args:
        vlm_backbone (dict | torch.nn.Module): ``VPP2Wan21Backbone`` config
            or module.
        vla_head (dict): ``VPP2Head`` config. Built on the meta device when
            a checkpoint is given and materialized by :meth:`load_state_dict`.
        pretrained_name_or_path (str | None): Converted native checkpoint
            directory. Official VPP2 bundles must be converted with
            ``tools.vpp2.convert_vpp2_checkpoint`` first.
        action_horizon (int): Actions per chunk. Defaults to 32.
        num_inference_steps (int): Euler steps for the action flow.
            Defaults to 10.
        sigma_shift (float): Shift of the action inference schedule.
            Defaults to 1.0.
        num_video_frames (int): Target video length in pixel frames; must be
            ``4k + 1``. Defaults to 17.
        seed_mode (str): Seed derivation, see :func:`replan_seeds`.
            Defaults to ``'vpp2'``.
        replan_seed_base (int): Base seed for ``seed_mode='vpp2'``.
            Defaults to 1.
        device (str | torch.device): Device for the head and encoders.
        torch_dtype (torch.dtype | str): Head and text-encoder dtype.
            Defaults to ``torch.bfloat16``.
    """

    def __init__(self,
                 vlm_backbone,
                 vla_head,
                 pretrained_name_or_path=None,
                 action_horizon=32,
                 num_inference_steps=10,
                 sigma_shift=1.,
                 num_video_frames=17,
                 seed_mode='vpp2',
                 replan_seed_base=1,
                 device='cpu',
                 torch_dtype=torch.bfloat16):
        super().__init__(
            pretrained_name_or_path=(str(pretrained_name_or_path)
                                     if pretrained_name_or_path else None),
            strict_mapping=True,
            pretrained_skip_prefixes=['vlm_backbone.'])
        torch_dtype = str_to_dtype(torch_dtype) if isinstance(
            torch_dtype, str) else torch_dtype
        self._load_device = device
        self._load_dtype = torch_dtype
        path = Path(
            pretrained_name_or_path) if pretrained_name_or_path else None
        if path is not None and (path / 'action.pt').is_file():
            raise ValueError('Convert the official bundle with '
                             'tools.vpp2.convert_vpp2_checkpoint first; '
                             'VPP2VLA loads native FluxVLA checkpoints.')
        # The native config owns the architecture and inference settings.
        # Checkpoint metadata must never silently override it.
        head = deepcopy(vla_head)
        head_device = 'meta' if path is not None else device
        head.update(device=head_device, torch_dtype=torch_dtype)
        with torch.device(head_device):
            self.vla_head = build_head_from_cfg(head)
        if isinstance(vlm_backbone, torch.nn.Module):
            self.vlm_backbone = vlm_backbone
        else:
            backbone = deepcopy(vlm_backbone)
            backbone.update(device=device, torch_dtype=torch_dtype)
            self.vlm_backbone = build_vlm_backbone_from_cfg(backbone)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = float(sigma_shift)
        self.replan_seed_base = int(replan_seed_base)
        self.num_video_frames = int(num_video_frames)
        if seed_mode not in ('vpp2', 'policy'):
            raise ValueError('seed_mode must be vpp2 or policy')
        self.seed_mode = seed_mode
        self.all_module_keys = ['vlm_backbone', 'vla_head']

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load a converted checkpoint, materializing a meta-built head.

        Accepts either a full state dict or one containing only
        ``vla_head.*`` keys. A meta-built head is always loaded strictly.
        """
        # Serving calls load_state_dict directly, rather than from_pretrained.
        # Converted bundles contain the complete head; the three frozen
        # encoders were already loaded strictly from their separate files.
        materialize = any(p.is_meta for p in self.vla_head.parameters())
        if state_dict and all(k.startswith('vla_head.') for k in state_dict):
            result = self.vla_head.load_state_dict(
                {
                    k.removeprefix('vla_head.'): v
                    for k, v in state_dict.items()
                },
                strict=strict or materialize,
                assign=assign or materialize)
        else:
            result = super().load_state_dict(
                state_dict, strict=strict, assign=assign or materialize)
        if materialize:
            self.vla_head.to(device=self._load_device, dtype=self._load_dtype)
        return result

    def _encode(self, images, lang_tokens, lang_masks, context, context_mask):
        if context is None:
            context, context_mask = self.vlm_backbone.encode_prompt_cached(
                lang_tokens, lang_masks)
        dtype = next(self.vla_head.parameters()).dtype
        context = context.to(device=self.device, dtype=dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        return (self.vlm_backbone.encode_video_latents(images),
                self.vlm_backbone.encode_clip(images), context, context_mask)

    def forward(self, *args, **kwargs):
        """Alias of :meth:`predict_action`; VPP2 is inference only."""
        return self.predict_action(*args, **kwargs)

    @torch.no_grad()
    def predict_action(self,
                       images,
                       states,
                       lang_tokens=None,
                       lang_masks=None,
                       context=None,
                       context_mask=None,
                       seed=None,
                       replan_index=None,
                       **kwargs):
        """Predict one normalized action chunk from the camera history.

        Args:
            images (torch.Tensor): Condition video ``[1, 3, T, H, W]`` in
                ``[-1, 1]``, with ``T = 4k + 1`` history frames.
            states (torch.Tensor): Normalized EE16 state ``[1, 16]``.
            lang_tokens (torch.Tensor | None): UMT5 token ids ``[1, L]``.
            lang_masks (torch.Tensor | None): Token mask ``[1, L]``.
            context (torch.Tensor | None): Precomputed T5 context
                ``[1, L, 4096]``; replaces ``lang_tokens`` when given.
            context_mask (torch.Tensor | None): Mask for ``context``.
            seed (int | None): Policy seed for ``seed_mode='policy'``.
            replan_index (int | None): Replan counter from the client.
            **kwargs: Other batch keys from the serving pipeline, ignored.

        Returns:
            torch.Tensor: Normalized actions ``[1, action_horizon,
            action_dim]`` in float32.
        """
        action_seed, video_seed = replan_seeds(replan_index, self.seed_mode,
                                               seed, self.replan_seed_base)
        condition, clip, context, mask = self._encode(images, lang_tokens,
                                                      lang_masks, context,
                                                      context_mask)
        return self.vla_head(
            forward_mode='predict_action',
            condition_latents=condition,
            clip_fea=clip,
            context=context,
            context_mask=mask,
            states=states,
            action_seed=action_seed,
            video_seed=video_seed,
            action_horizon=self.action_horizon,
            num_video_frames=self.num_video_frames,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift)

    @property
    def config(self):
        """Minimal ``PretrainedConfig`` expected by shared utilities."""
        from transformers import PretrainedConfig
        return PretrainedConfig(is_encoder_decoder=False)
