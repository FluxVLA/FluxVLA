"""VPP2 policy with episode history supplied by the evaluation client."""

from copy import deepcopy
from pathlib import Path

import torch

from fluxvla.engines import (VLAS, build_head_from_cfg,
                             build_vlm_backbone_from_cfg)
from fluxvla.engines.utils.name_map import str_to_dtype
from .base_vla import BaseVLA


def replan_seeds(replan_index, seed_mode='vpp2', seed=None, base=1):
    if replan_index is None:
        raise ValueError(
            'replan_index is required; history belongs to the client')
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

    def _encode(self, condition_video, lang_tokens, lang_masks, context,
                context_mask):
        if context is None:
            context, context_mask = self.vlm_backbone.encode_prompt_cached(
                lang_tokens, lang_masks)
        dtype = next(self.vla_head.parameters()).dtype
        context = context.to(device=self.device, dtype=dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        return (self.vlm_backbone.encode_video_latents(condition_video),
                self.vlm_backbone.encode_clip(condition_video), context,
                context_mask)

    def forward(self, *args, **kwargs):
        return self.predict_action(*args, **kwargs)

    @torch.no_grad()
    def predict_action(self,
                       condition_video,
                       states,
                       lang_tokens=None,
                       lang_masks=None,
                       context=None,
                       context_mask=None,
                       seed=None,
                       replan_index=None,
                       **kwargs):
        action_seed, video_seed = replan_seeds(replan_index, self.seed_mode,
                                               seed, self.replan_seed_base)
        condition, clip, context, mask = self._encode(condition_video,
                                                      lang_tokens, lang_masks,
                                                      context, context_mask)
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
        from transformers import PretrainedConfig
        return PretrainedConfig(is_encoder_decoder=False)
