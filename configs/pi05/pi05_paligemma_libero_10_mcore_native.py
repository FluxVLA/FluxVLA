"""Native MCore PI0.5 provider recipe, first phase (DP/TP/SP)."""

_base_ = ["./pi05_paligemma_libero_10_full_finetune.py"]

runner = dict(
    type="MegatronNativeTrainRunner",
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    sequence_parallel=False,
    distributed_optimizer=True,
)
