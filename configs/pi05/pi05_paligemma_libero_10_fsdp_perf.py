"""Throughput-oriented FSDP recipe for PI0.5 LIBERO.

This keeps the regular full-finetune config as the source of model and data
settings while selecting coarse execution-block wrapping and BF16 FSDP
parameters.  Set batch size and accumulation at launch to match the target
global batch (for GB32 on DP4: MB=8, GA=1).
"""

_base_ = ['./pi05_paligemma_libero_10_full_finetune.py']

runner = dict(
    sharding_strategy='full-shard',
    fsdp_wrap_policy='execution-block',
    pre_fsdp_param_dtype='bf16',
    reduce_in_full_precision=False,
    keep_params_fp32=False,
    deterministic_algorithms=False,
)
