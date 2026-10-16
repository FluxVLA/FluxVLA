"""SmolVLA RL model defaults, registered as Hydra models/smolvla."""

model = dict(
    model_type='fluxvla_smolvla',
    model_path='???',
    precision='fp32',
    load_to_device=True,
    is_lora=False,
    lora_rank=32,
    num_action_chunks=10,
    action_dim=7,
    num_steps=10,
    use_proprio=True,
    add_value_head=True,
    joint_logprob=False,
    fluxvla=dict(
        config_path=('${oc.env:FLUXVLA_ROOT}/configs/smolvla/'
                     'smolvla_libero_10_finetune.py'),
        tokenizer_path='???',
        norm_stats_path='???',
        observation_adapter='libero',
        action_horizon=50,
        compute_dtype='bf16',
        train_expert_only=True,
        noise_method='flow_sde',
        noise_level=0.5,
        rollout_micro_batch_size='${actor.micro_batch_size}',
    ),
)
