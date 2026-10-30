"""PI0.5 native MCore Transformer TP recipe.

Unlike the compatibility ``gemma_mlp`` recipe, this provider converts every
Gemma attention and gated-MLP projection using an explicit TP contract while
preserving FluxVLA's PI0.5 joint-attention and flow-loss implementation.
"""

_base_ = ["./pi05_paligemma_libero_10_full_finetune.py"]

runner = dict(
    type="MegatronNativeTrainRunner",
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    sequence_parallel=False,
    tensor_parallel_strategy="gemma_transformer",
    distributed_optimizer=False,
)
