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

import torch
import torch.nn as nn
from transformers import SiglipVisionModel


class SigLIPViTBackbone(nn.Module):
    """Apply a constructed SigLIP model to channel-packed RGB views.

    ``vision`` is registered directly as ``.vision`` so existing checkpoint
    keys are unchanged. Model construction and pretrained loading belong to
    the caller. Extra keyword arguments are forwarded for cooperative module
    initialization by training adapters.
    """

    def __init__(self,
                 vision: SiglipVisionModel,
                 openpi_stem_fp32: bool = False,
                 preserve_fp32_residuals: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.openpi_stem_fp32 = bool(openpi_stem_fp32)
        self.preserve_fp32_residuals = bool(preserve_fp32_residuals)
        self.vision = vision

    def _forward_openpi_stem(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run patch extraction and positional embedding in FP32."""
        vision_model = self.vision.vision_model
        with torch.autocast(
                device_type=pixel_values.device.type, enabled=False):
            hidden_states = vision_model.embeddings(pixel_values.float())

        if (not self.preserve_fp32_residuals
                and torch.is_autocast_enabled(pixel_values.device.type)):
            encoder_dtype = torch.get_autocast_dtype(pixel_values.device.type)
        else:
            encoder_dtype = next(vision_model.encoder.parameters()).dtype
        hidden_states = hidden_states.to(encoder_dtype)
        encoder_outputs = vision_model.encoder(inputs_embeds=hidden_states)
        return vision_model.post_layernorm(encoder_outputs.last_hidden_state)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Map ``[B, 3 * views, H, W]`` to view-major patch embeddings."""
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
