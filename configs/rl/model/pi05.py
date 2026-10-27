"""Python RL defaults registered with Hydra by RLinfTrainRunner."""

model = {
    'model_type': 'fluxvla_pi05',
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
    'experiment_tag': 'fluxvla_pi05',
    'fsdp_wrap_policy': {
        'transformer_layer_cls_to_wrap':
        ['GemmaDecoderLayer', 'SiglipEncoderLayer'],
        'module_classes_to_wrap': ['LinearProjector', 'ValueHead']
    },
    'fluxvla': {
        'config_path':
        ('${oc.env:FLUXVLA_ROOT}/configs/pi05/pi05_paligemma_libe'
         'ro_10_full_finetune.py'),
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
        '${simulator.action_horizon}',
        'rollout_micro_batch_size':
        '${simulator.rollout_micro_batch_size}',
        'model_overrides':
        '${simulator.model_overrides}'
    }
}
