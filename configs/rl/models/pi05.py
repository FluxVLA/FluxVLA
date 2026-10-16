"""PI0.5 RL model defaults, registered as Hydra models/pi05."""

model = dict(
    model_type='fluxvla_pi05',
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
        config_path=('${oc.env:FLUXVLA_ROOT}/configs/pi05/'
                     'pi05_paligemma_libero_10_full_finetune.py'),
        tokenizer_path=None,
        norm_stats_path=None,
        compute_dtype='bf16',
        train_expert_only=True,
        noise_method='flow_sde',
        noise_level=0.5,
    ),
)
