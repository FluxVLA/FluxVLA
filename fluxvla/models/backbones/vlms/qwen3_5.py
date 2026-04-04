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
"""Qwen3.5 backbone for VLM SFT and GR00T VLA. Qwen3.5 natively supports
multimodal (no VL suffix). Compatible with Qwen3.5-0.8B etc."""

from functools import partial
from typing import Callable, Dict, Optional, Type, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import \
        Qwen3_5DecoderLayer
    _HAS_QWEN3_5 = True
except ImportError:
    Qwen3_5DecoderLayer = None
    _HAS_QWEN3_5 = False

from fluxvla.engines import VLM_BACKBONES
from .hf_vlm import VLMBackbone

if _HAS_QWEN3_5:

    @VLM_BACKBONES.register_module()
    class Qwen3_5(VLMBackbone):
        """
        HuggingFace-compatible wrapper for Qwen3.5 (e.g. Qwen3.5-0.8B).
        Qwen3.5 natively supports multimodal. Supports VLM SFT
        (input_ids/labels) and VLA (images + lang_tokens).
        """

        def __init__(self,
                     vlm_backbone_id: str,
                     vlm_config: Dict = None,
                     vlm_path: Optional[str] = None,
                     vla_pretrained_path: Optional[str] = None,
                     use_projection: bool = False,
                     projection_output_dim: Optional[int] = None,
                     projection_type: str = 'linear',
                     projection_mlp_hidden_dim: Optional[int] = None,
                     attn_implementation: str = 'flash_attention_2',
                     torch_dtype: Optional[Union[torch.dtype, str]] = None,
                     **kwargs) -> None:
            self._attn_implementation = attn_implementation
            load_dtype = (
                torch_dtype if torch_dtype is not None else torch.bfloat16)
            if isinstance(load_dtype, str):
                load_dtype = getattr(torch, load_dtype, torch.bfloat16)
            super().__init__(
                vlm_backbone_id,
                vlm_config,
                vlm_path=vlm_path,
                vla_pretrained_path=vla_pretrained_path,
                attn_implementation=attn_implementation,
                torch_dtype=load_dtype,
                **kwargs)
            if hasattr(self.vlm.config, 'attn_implementation'):
                self.vlm.config.attn_implementation = attn_implementation

            self._use_projection = use_projection
            self._embed_dim_override: Optional[int] = None
            if use_projection:
                if projection_output_dim is None:
                    raise ValueError('projection_output_dim is required when '
                                     'use_projection=True')
                in_dim = self.vlm.model.language_model.config.hidden_size
                if projection_type == 'linear':
                    self.projection = nn.Linear(in_dim, projection_output_dim)
                elif projection_type == 'mlp':
                    hid = projection_mlp_hidden_dim or max(
                        in_dim, projection_output_dim)
                    self.projection = nn.Sequential(
                        nn.Linear(in_dim, hid),
                        nn.GELU(),
                        nn.Linear(hid, projection_output_dim),
                    )
                else:
                    raise ValueError(
                        "projection_type must be 'linear' or 'mlp', "
                        'got {}'.format(projection_type))
                self._embed_dim_override = projection_output_dim
            else:
                self.projection = None

        @property
        def embed_dim(self) -> int:
            if self._embed_dim_override is not None:
                return self._embed_dim_override
            return self.vlm.model.language_model.config.hidden_size

        @property
        def transformer_layer_cls(self) -> Type[nn.Module]:
            return Qwen3_5DecoderLayer

        def _compute_position_ids_for_vla(
            self,
            batch_size: int,
            seq_len: int,
            L_img: int,
            num_images: int,
            image_grid_thw_flat: torch.LongTensor,
            device: torch.device,
        ) -> torch.LongTensor:
            """Compute 3D position_ids for [img|text] in Qwen3.5 M-RoPE.
            Image tokens get (temporal, height, width) from
            get_vision_position_ids;
            text tokens get 1D arange. Returns (3, batch_size, seq_len).
            """
            model = self.vlm.model
            vc = getattr(model, 'visual', None)
            spatial_merge_size = (
                getattr(vc.config, 'spatial_merge_size', 2)
                if vc is not None else 2)
            position_ids = torch.zeros(
                3, batch_size, seq_len, dtype=torch.long, device=device)
            for b in range(batch_size):
                start_pos = 0
                llm_pos_list = []
                # Image tokens: one chunk per image in this batch
                for m in range(num_images):
                    idx = b * num_images + m
                    if idx >= image_grid_thw_flat.shape[0]:
                        break
                    grid_thw = image_grid_thw_flat[idx]
                    n_tokens = int(grid_thw.prod().item() //
                                   (spatial_merge_size**2))
                    if n_tokens <= 0:
                        continue
                    vision_pos = model.get_vision_position_ids(
                        start_pos,
                        grid_thw,
                        temp_merge_size=1,
                        spatial_merge_size=spatial_merge_size,
                        device=device,
                    )
                    llm_pos_list.append(vision_pos)
                    start_pos += n_tokens
                # Text tokens: 1D arange (same in all 3 dims for text)
                L_text = seq_len - L_img
                if L_text > 0:
                    text_pos = torch.arange(
                        L_text, device=device, dtype=torch.long).view(
                            1, -1).expand(3, -1) + start_pos
                    llm_pos_list.append(text_pos)
                if llm_pos_list:
                    cat_pos = torch.cat(llm_pos_list, dim=1)
                    fill_len = min(cat_pos.shape[1], seq_len)
                    position_ids[:, b, :fill_len] = cat_pos[:, :fill_len]
            return position_ids

        def _compute_grid_thw(
            self,
            pixel_values: torch.Tensor,
            batch_size: int,
            image_grid_thw: Optional[torch.LongTensor],
        ) -> torch.LongTensor:
            """Compute grid_thw in Qwen3.5 convention from pixel_values."""
            ndim = pixel_values.dim()
            if ndim == 5:
                num_imgs = pixel_values.shape[0] * pixel_values.shape[1]
                H, W = pixel_values.shape[3], pixel_values.shape[4]
            elif ndim == 4:
                num_imgs = pixel_values.shape[0]
                H, W = pixel_values.shape[2], pixel_values.shape[3]
            elif ndim == 3:
                num_imgs = pixel_values.shape[0]
                H, W = pixel_values.shape[1], pixel_values.shape[2]
            else:
                raise ValueError(
                    f'qwen3_5: pixel_values must be 3D/4D/5D, got ndim={ndim}')
            visual = getattr(self.vlm.model, 'visual', None)
            patch_size = getattr(visual.config, 'patch_size',
                                 16) if visual else 16
            merge_size = getattr(visual.config, 'spatial_merge_size',
                                 2) if visual else 2
            grid_h = max(2, H // patch_size)
            grid_w = max(2, W // patch_size)
            if grid_h % merge_size != 0 or grid_w % merge_size != 0:
                grid_h = max(merge_size, (grid_h + merge_size - 1) //
                             merge_size * merge_size)
                grid_w = max(merge_size, (grid_w + merge_size - 1) //
                             merge_size * merge_size)
            device = pixel_values.device
            grid_thw = torch.zeros(
                num_imgs, 3, dtype=torch.long, device=device)
            grid_thw[:, 0] = 1
            grid_thw[:, 1] = grid_h
            grid_thw[:, 2] = grid_w
            return grid_thw

        def forward(self, *args, **kwargs):
            if 'input_ids' in kwargs or 'labels' in kwargs:
                return self.vlm(**kwargs)
            return self._forward_vla(*args, **kwargs)

        def _forward_vla(self,
                         images: torch.Tensor,
                         lang_tokens: torch.Tensor,
                         img_masks: torch.Tensor,
                         lang_masks: Optional[torch.Tensor] = None,
                         image_grid_thw: Optional[torch.LongTensor] = None,
                         *args,
                         **kwargs):
            device = next(self.vlm.parameters()).device
            images = images.to(device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            img_masks = img_masks.to(device)
            if lang_masks is not None:
                lang_masks = lang_masks.to(device)
            lang_tokens = lang_tokens.to(device)

            batch_size = images.shape[0]
            pixel_values = images
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.view(-1, pixel_values.shape[2],
                                                 pixel_values.shape[3],
                                                 pixel_values.shape[4])
            elif pixel_values.dim() == 3:
                # Raw spatial (B, H, W) -> (B, 3, H, W); or flattened (B, N, D)
                # from processor needs image_grid_thw (handled below)
                vc = getattr(self.vlm.model, 'visual', None)
                if vc is not None:
                    patch_feat_dim = (
                        getattr(vc.config, 'in_channels', 3) *
                        getattr(vc.config, 'temporal_patch_size', 2) *
                        getattr(vc.config, 'patch_size', 16)**2)
                    if pixel_values.shape[-1] == patch_feat_dim:
                        pixel_values = pixel_values.reshape(
                            -1, pixel_values.shape[-1])
                    else:
                        pixel_values = pixel_values.unsqueeze(1).expand(
                            -1, 3, pixel_values.shape[1],
                            pixel_values.shape[2])
                else:
                    pixel_values = pixel_values.unsqueeze(1).expand(
                        -1, 3, pixel_values.shape[1], pixel_values.shape[2])
            pixel_values = pixel_values.contiguous()

            if pixel_values.dim() in (4, 5):
                image_grid_thw_flat = self._compute_grid_thw(
                    pixel_values, batch_size, image_grid_thw)
            else:
                if image_grid_thw is None:
                    raise ValueError('qwen3_5: image_grid_thw required when '
                                     'pixel_values are flattened.')
                image_grid_thw_flat = image_grid_thw.view(-1, 3)

            image_outputs = self.vlm.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw_flat,
                return_dict=True,
            )
            # pooler_output: list of (L_i, H) per image or single tensor
            image_embeds = image_outputs.pooler_output
            if isinstance(image_embeds, torch.Tensor):
                img_emb = image_embeds
            else:
                img_emb = torch.cat(image_embeds, dim=0)
            hidden_size = img_emb.shape[-1]
            num_images = image_grid_thw_flat.shape[0] // batch_size
            if num_images <= 0:
                num_images = 1
            if isinstance(image_embeds, torch.Tensor):
                img_emb = img_emb.reshape(batch_size, -1, hidden_size)
            elif isinstance(image_embeds, (list, tuple)):
                if all(e.shape[0] == image_embeds[0].shape[0]
                       for e in image_embeds):
                    img_emb = img_emb.reshape(batch_size, -1, hidden_size)
                else:
                    per_batch = [
                        sum(emb.shape[0]
                            for emb in image_embeds[i * num_images:(i + 1) *
                                                    num_images])
                        for i in range(batch_size)
                    ]
                    chunks = torch.split(img_emb, per_batch)
                    max_len = max(c.shape[0] for c in chunks)
                    img_emb = torch.stack([
                        torch.nn.functional.pad(
                            c, (0, 0, 0, max_len - c.shape[0])) for c in chunks
                    ])
            else:
                img_emb = img_emb.reshape(batch_size, -1, hidden_size)

            text_embeds = self.vlm.get_input_embeddings()(lang_tokens)
            inputs_embeds = torch.cat([img_emb, text_embeds], dim=1)
            L_img = img_emb.shape[1]
            img_part = torch.cat([
                img_masks[:, i:i + 1].repeat(1, img_emb.shape[1] // num_images)
                for i in range(img_masks.shape[1])
            ],
                                 dim=1)
            attention_mask_2d = torch.cat([img_part, lang_masks], dim=1)
            seq_len = inputs_embeds.shape[1]
            if attention_mask_2d.shape[1] != seq_len:
                attention_mask_2d = attention_mask_2d[:, :seq_len].contiguous()

            # Build position_ids for [img|text] layout. Qwen3.5 uses M-RoPE and
            # requires correct 3D positions for image tokens. When we pass
            # pre-fused inputs_embeds with pixel_values=None, model() cannot
            # compute position_ids (needs input_ids + mm_token_type_ids +
            # image_grid_thw).
            # We compute position_ids explicitly and call language_model.
            position_ids = self._compute_position_ids_for_vla(
                batch_size=batch_size,
                seq_len=seq_len,
                L_img=L_img,
                num_images=num_images,
                image_grid_thw_flat=image_grid_thw_flat,
                device=inputs_embeds.device,
            )

            # SDPA needs 4D causal+padding mask; Flash uses 2D.
            # Keep behavior aligned with Qwen3VL.
            attn_impl = getattr(self, '_attn_implementation', 'sdpa') or 'sdpa'
            if attn_impl == 'flash_attention_2':
                mask_for_lm = attention_mask_2d
            else:
                causal = torch.tril(
                    torch.ones(
                        seq_len,
                        seq_len,
                        device=inputs_embeds.device,
                        dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
                padding_ok = attention_mask_2d.unsqueeze(1).unsqueeze(3)
                combined = causal & padding_ok
                zero = torch.zeros((),
                                   device=inputs_embeds.device,
                                   dtype=inputs_embeds.dtype)
                inf = torch.full((),
                                 float('-inf'),
                                 device=inputs_embeds.device,
                                 dtype=inputs_embeds.dtype)
                mask_for_lm = torch.where(combined, zero, inf)

            outputs = self.vlm.model.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=mask_for_lm,
                position_ids=position_ids,
                use_cache=False,
            )
            last_hidden_state = outputs.last_hidden_state
            if self.projection is not None:
                last_hidden_state = self.projection(last_hidden_state)
            return last_hidden_state, attention_mask_2d, attention_mask_2d

        def enable_gradient_checkpointing(self) -> None:
            if hasattr(self.vlm, 'gradient_checkpointing_enable'):
                self.vlm.gradient_checkpointing_enable()

        def get_fsdp_wrapping_policy(self) -> Callable:
            transformer_block_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen3_5DecoderLayer})
            return transformer_block_policy

else:
    Qwen3_5 = None  # type: ignore
