# Copyright 2026 Limx Dynamics
"""Aloha training and local/remote serving with selective FP32 computation.

Keep BF16 autocast except for the first six expert blocks, Gemma math
attention, AdaRMS and residuals. Flow projections and the vision stem
retain the base recipe's FP32 computation. Keep FP32 master parameters.
The original Aloha recipe, action dimensions and data transforms are unchanged.
For remote serving, use this config on the GPU server with the existing
pi05_paligemma_aloha_remote_inference.py client config on the robot.
"""

_base_ = ['./pi05_paligemma_aloha_full_finetune.py']

model = dict(
    enable_mixed_precision_training=True,
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True),
    llm_backbone=dict(attention_math_fp32=True),
    llm_expert=dict(
        adarms_fp32=True,
        attention_math_fp32=True,
        fp32_layers=(0, 1, 2, 3, 4, 5)))
inference_model = model.copy()
# The base training runner already keeps FP32 parameters under BF16 autocast.
inference = dict(
    enable_mixed_precision=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True)
