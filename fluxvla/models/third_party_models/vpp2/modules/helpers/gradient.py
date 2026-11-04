# Copyright (c) 2026 The FastWAM Authors.
# SPDX-License-Identifier: MIT
#
# Origin: Source
# Upstream-URL: https://github.com/roboterax/video-prediction-policy-2/blob/3365872910244f14df83003062f7c563f14c6d9e/src/vpp2/models/wan21_14b/helpers/gradient.py
# Upstream-Ref: 3365872910244f14df83003062f7c563f14c6d9e
# Notes: No functional change. License text: ../../LICENSE; attribution:
# ../../ATTRIBUTION.txt.
import torch


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)

    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
