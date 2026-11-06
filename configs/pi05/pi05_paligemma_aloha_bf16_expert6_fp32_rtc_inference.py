# Copyright 2026 Limx Dynamics
"""Aloha RTC with selective FP32 through the standard PI05 model.

The separate Triton RTC model uses fixed BF16 buffers and does not consume
these precision switches. Keep that existing kernel recipe unchanged.
"""

_base_ = ['./pi05_paligemma_aloha_bf16_expert6_fp32_full_finetune.py']

inference = dict(
    type='AlohaRTCInferenceRunner',
    async_execution=True,
    execute_horizon=0,
    rtc_config=dict(enabled=True, method='prefix', prefix_len=5),
    publish_rate=150)
