"""Python RL defaults registered with Hydra by RLinfTrainRunner."""

defaults = [{
    '/env@env.train': 'libero_10'
}, {
    '/env@env.eval': 'libero_10'
}, '_self_']

simulator = {
    'name': 'libero10',
    'placement': {
        'actor,env,rollout': 'all'
    },
    'action_chunk': 10,
    'action_dim': 7,
    'action_horizon': 10,
    'num_steps': 10,
    'noise_level': 0.5,
    'tokenizer_path': None,
    'norm_stats_path': None,
    'rollout_micro_batch_size': None,
    'adapter': {
        'type': 'LiberoRLAdapter'
    },
    'model_overrides': {}
}

runner = {
    'logger': {
        'experiment_name': 'libero_10_ppo_${actor.model.experiment_tag}'
    }
}

env = {
    'train': {
        'rollout_epoch': 1,
        'total_num_envs': 16,
        'max_episode_steps': 480,
        'max_steps_per_rollout_epoch': 480,
        'reward_coef': '${algorithm.reward_coef}',
        'video_cfg': {
            'save_video': False
        }
    },
    'eval': {
        'rollout_epoch': 1,
        'total_num_envs': 16,
        'max_episode_steps': 480,
        'max_steps_per_rollout_epoch': 480,
        'auto_reset': True,
        'ignore_terminations': True,
        'is_eval': True,
        'group_size': 1,
        'video_cfg': {
            'save_video': False,
            'video_base_dir': '${runner.logger.log_path}/video/eval'
        }
    }
}
