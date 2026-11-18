"""PI0.5 Libero recipe using the optional Megatron-Core compatibility runner."""

_base_ = ['./pi05_paligemma_libero_10_full_finetune.py']

runner = dict(
    type='MegatronNativeTrainRunner',
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)
