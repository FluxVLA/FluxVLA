"""Deterministic eager-attention/VLA-only parity configuration.

The native LingBot YAML enables flex attention and auxiliary alignment heads by
default.  This explicit overlay matches the official eager reference contract.
"""

_base_ = ["./lingbot_vla_v2_robotwin_dense_finetune.py"]

model = dict(
    attention_implementation="eager",
    vit_attn_implementation="eager",
    moe_implementation="eager",
    align_params={},
)

runner = dict(max_steps=1)
