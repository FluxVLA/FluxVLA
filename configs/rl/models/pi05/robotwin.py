"""Same Flux PI0.5 parameter layout; reference OpenPI FP32 flow projections."""
_base_ = '../../../pi05/pi05_paligemma_libero_10_full_finetune.py'

model = dict(
    n_action_steps=50,
    num_steps=5,
    attention_implementation='eager',
    openpi_fp32_flow=True,
    vision_backbone=dict(openpi_stem_fp32=True),
)
