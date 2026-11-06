# Origin: Modified from
# Upstream-URL: https://github.com/ZibinDong/openpi_pytorch/blob/main/pi0/utils.py  # noqa: E501
# Upstream-Ref: main
# SPDX-License-Identifier: Apache-2.0
# Notes: Attribution normalized and reused internal equivalents.

import numpy as np
import torch
import torch.nn.functional as F
from fluxvla_export.components.tensor_utils import (  # noqa: F401
    apply_rope, apply_rotary_pos_emb, create_sinusoidal_pos_embedding,
    eager_attention_forward, gated_residual, make_att_2d_masks, repeat_kv,
    rotate_half, sdpa_attention_forward, sdpa_math_fp32_attention_forward)


def find_next_divisible_by_8_numpy(n: np.ndarray) -> np.ndarray:
    """
    Finds the smallest integers greater than each element in a
    NumPy array 'n'
    that are divisible by 8. Assumes non-negative integers.

    Args:
        n: A NumPy array of integers.

    Returns:
        A NumPy array containing the smallest integers
            greater than each input element that are divisible by 8.
    """
    remainder = n % 8
    # Calculate the amount to add: 0 if already divisible,
    # otherwise 8 - remainder
    # np.where is efficient for conditional operations on arrays
    amount_to_add = np.where(remainder == 0, 8, 8 - remainder)
    return n + amount_to_add


def sample_beta(alpha,
                beta,
                bsize,
                device,
                sampler: str = 'legacy_power_ratio') -> torch.Tensor:
    """Sample flow times from OpenPI's Beta or the legacy FluxVLA rule."""
    if alpha <= 0 or beta <= 0:
        raise ValueError(
            f'alpha and beta must be positive, got {alpha=} and {beta=}')
    if sampler == 'beta':
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
        return torch.distributions.Beta(alpha_t, beta_t).sample((int(bsize), ))
    if sampler == 'legacy_power_ratio':
        gamma1 = torch.rand((bsize, ), device=device).pow(1 / alpha)
        gamma2 = torch.rand((bsize, ), device=device).pow(1 / beta)
        return gamma1 / (gamma1 + gamma2)
    raise ValueError(f'Unsupported beta sampler {sampler!r}. Expected '
                     "'beta' or 'legacy_power_ratio'.")


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f'(b, c, h, w) expected, but {img.shape}')

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img,
        size=(resized_height, resized_width),
        mode='bilinear',
        align_corners=False)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(
        resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img
