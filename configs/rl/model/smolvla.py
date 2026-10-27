"""Python RL defaults registered with Hydra by RLinfTrainRunner."""

model = {
    'model_type': 'fluxvla_smolvla',
    'model_path': '???',
    'precision': 'fp32',
    'load_to_device': True,
    'is_lora': False,
    'lora_rank': 32,
    'num_action_chunks': '${simulator.action_chunk}',
    'action_dim': '${simulator.action_dim}',
    'num_steps': '${simulator.num_steps}',
    'use_proprio': True,
    'add_value_head': True,
    'joint_logprob': False,
    'experiment_tag': 'fluxvla_smolvla',
    'fsdp_wrap_policy': {
        'transformer_layer_cls_to_wrap': ['SmolVLMEncoderLayer'],
        'module_classes_to_wrap': ['LinearProjector', 'ValueHead']
    },
    'fluxvla': {
        'config_path':
        ('${oc.env:FLUXVLA_ROOT}/configs/smolvla/smolvla_libero_1'
         '0_finetune.py'),
        'tokenizer_path':
        '${simulator.tokenizer_path}',
        'norm_stats_path':
        '${simulator.norm_stats_path}',
        'compute_dtype':
        'bf16',
        'train_expert_only':
        True,
        'noise_method':
        'flow_sde',
        'noise_level':
        '${simulator.noise_level}',
        'observation_adapter':
        '${simulator.adapter}',
        'action_horizon':
        50,
        'rollout_micro_batch_size':
        '${actor.micro_batch_size}',
        'model_overrides':
        '${simulator.model_overrides}'
    }
}
