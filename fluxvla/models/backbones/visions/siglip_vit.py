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

from typing import Callable, Dict, Tuple

import torch
from timm.models.vision_transformer import Block, VisionTransformer

from fluxvla.engines import VISION_BACKBONES
from fluxvla.engines.utils.fsdp_wrapping import (
    build_combined_wrap_policy, build_module_wrap_policy,
    build_transformer_layer_wrap_policy)
from .base_vision import VisionBackbone
from .configs import VISION_BACKBONE_CONFIGS


@VISION_BACKBONES.register_module()
class SigLIPViTBackbone(VisionBackbone):
    """
    A vision backbone that uses the SigLIP ViT model for feature extraction.

    Args:
        vision_backbone_id (str): Identifier string for the backbone.
        pretrained_cfg (Dict): Configuration for loading the pretrained model.
    """

    def __init__(self,
                 vision_backbone_id: str,
                 vision_config: Dict = None,
                 pretrained_cfg: Dict = None,
                 openpi_stem_fp32: bool = False) -> None:
        super().__init__(vision_backbone_id)
        self.openpi_stem_fp32 = bool(openpi_stem_fp32)
        vision_cls = VISION_BACKBONE_CONFIGS[vision_backbone_id]['model_cls']
        if pretrained_cfg is None:
            assert vision_config is not None, 'vision_cfg must be provided if pretrained_cfg is specified'  # noqa: E501
            vision_cfg = VISION_BACKBONE_CONFIGS[vision_backbone_id]['config']
            vision_config = vision_cfg(**vision_config)
            self.vision = vision_cls(vision_config)
        else:
            pretrained_cfg = dict(pretrained_cfg)
            pretrained_cfg.setdefault('trust_remote_code', True)
            self.vision = vision_cls.from_pretrained(**pretrained_cfg)

    def _forward_openpi_stem(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run patch extraction and positional embedding in FP32."""
        vision_model = self.vision.vision_model
        with torch.autocast(
                device_type=pixel_values.device.type, enabled=False):
            hidden_states = vision_model.embeddings(pixel_values.float())

        if torch.is_autocast_enabled(pixel_values.device.type):
            encoder_dtype = torch.get_autocast_dtype(pixel_values.device.type)
        else:
            encoder_dtype = next(vision_model.encoder.parameters()).dtype
        hidden_states = hidden_states.to(encoder_dtype)
        encoder_outputs = vision_model.encoder(inputs_embeds=hidden_states)
        return vision_model.post_layernorm(encoder_outputs.last_hidden_state)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Define FSDP wrapping policy that includes full ViTs and
            transformer blocks.

        Returns:
            Callable: A composite policy for FSDP module wrapping.
        """
        vit_wrap_policy = build_module_wrap_policy({VisionTransformer})
        transformer_block_policy = build_transformer_layer_wrap_policy({Block})
        return build_combined_wrap_policy(
            [vit_wrap_policy, transformer_block_policy])

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        """
        Get the default input resolution for the DINO backbone.

        Returns:
            Tuple[int, int, int]: Image shape as (C, H, W).
        """
        return self.dino_data_cfg['input_size']

    @property
    def embed_dim(self) -> int:
        """
        Get the combined embedding dimension from both backbones.

        Returns:
            int: Sum of DINO and SigLIP embedding dimensions.
        """
        return self.vision.embed_dim

    @property
    def num_patches(self) -> int:
        """
        Get the number of spatial patches output by the backbones.

        Returns:
            int: Number of patches (same for both models).
        """
        return self.vision.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """
        Preferred half-precision dtype for mixed-precision training.

        Returns:
            torch.dtype: Typically bfloat16.
        """
        return torch.bfloat16

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a forward pass using both DINO and SigLIP backbones.

        Args:
            pixel_values (Dict[str, torch.Tensor]): Dictionary containing:
                - "dino": tensor input for DINOv2 model
                - "siglip": tensor input for SigLIP model

        Returns:
            torch.Tensor: Concatenated patch features from both models.
        """
        token_embeddings = list()
        pixel_values = pixel_values.unflatten(1, (-1, 3))
        for i in range(pixel_values.shape[1]):
            view = pixel_values[:, i, :, :]
            if self.openpi_stem_fp32:
                token_embeddings.append(self._forward_openpi_stem(view))
            else:
                token_embeddings.append(self.vision(view).last_hidden_state)

        token_embeddings = torch.cat(token_embeddings, dim=1)
        return token_embeddings
