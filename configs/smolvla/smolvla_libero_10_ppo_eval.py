"""Native LIBERO evaluation of an exported SmolVLA PPO policy.

Pass the value-head-free policy with --ckpt-path and the original SFT
statistics with eval.norm_stats_path. Keep the checkpoint-local tokenizer
or override eval.dataset.transforms.2.tokenizer.model_path.
"""

_base_ = './smolvla_libero_10_finetune.py'

# PPO saves FP32 master weights. Loading them into BF16 parameters would
# round off small PPO updates before autocast ever runs.
model = dict(
    pretrained_name_or_path=None,
    name_mapping=None,
    strict_mapping=True,
    vlm_backbone=dict(torch_dtype='float32'),
    llm_expert=dict(torch_dtype='float32'))

eval = dict(
    model_build_dtype='float32',
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16')
