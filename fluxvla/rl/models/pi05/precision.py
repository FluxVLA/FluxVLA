"""OpenPI-compatible RoboTwin compute with FP32 master parameters."""

import torch
import torch.nn.functional as F

from fluxvla.models.backbones.llms.condition_gemma import GemmaRMSNorm
from fluxvla.models.backbones.visions.siglip_vit import SigLIPViTBackbone


class OpenPILayerNorm(torch.nn.LayerNorm):

    def forward(self, x):
        # CUDA autocast promotes LayerNorm to FP32, whereas native OpenPI
        # executes SigLIP norms in BF16. Keep master weights FP32 and cast only
        # this operation's operands, preserving the reference residual stream.
        dtype = torch.get_autocast_dtype(
            x.device.type) if torch.is_autocast_enabled(
                x.device.type) else x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.layer_norm(
                x.to(dtype), self.normalized_shape, self.weight.to(dtype),
                self.bias.to(dtype), self.eps)


class OpenPISiglipBackbone(SigLIPViTBackbone):

    def _forward_openpi_stem(self, pixel_values):
        # OpenPI transposes HWC observations without materializing NCHW. Match
        # its convolution memory layout as well as dtype: tiny FP32 stem
        # rounding differences can cross BF16 quantization boundaries.
        return super()._forward_openpi_stem(
            pixel_values.contiguous(memory_format=torch.channels_last))


def align_openpi_vision(module):
    module.__class__ = OpenPISiglipBackbone
    for child in module.modules():
        if isinstance(child, torch.nn.LayerNorm):
            child.__class__ = OpenPILayerNorm


class FP32AdaptiveNorm(GemmaRMSNorm):

    def forward(self, x, cond=None):
        # The reference keeps adaRMS dense weights and timestep conditioning
        # FP32. Generic BF16 autocast would silently lower this projection.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x, cond)


def keep_adaptive_norm_fp32(module):
    for child in module.modules():
        if isinstance(child, GemmaRMSNorm) and child.dense is not None:
            # Same module/parameters, a compute-only specialization; names and
            # optimizer/FSDP parameter references are preserved exactly.
            child.__class__ = FP32AdaptiveNorm


def align_openpi_rope(module):
    # Native OpenPI converts the whole Gemma module to BF16 before selectively
    # restoring FP32 norm parameters. That also rounds its nonpersistent RoPE
    # frequency buffers. Retain those values in FP32 without rounding masters.
    with torch.no_grad():
        rotary = module.rotary_emb
        for name in ('inv_freq', 'original_inv_freq'):
            buffer = getattr(rotary, name, None)
            if buffer is not None:
                buffer.copy_(buffer.bfloat16().to(buffer.dtype))
