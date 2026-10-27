"""Python RL defaults registered with Hydra by RLinfTrainRunner."""

defaults = [{
    '/env@env.train': 'robotwin_adjust_bottle'
}, {
    '/env@env.eval': 'robotwin_adjust_bottle'
}, '_self_']

simulator = {
    'name':
    'robotwin_adjust_bottle',
    'placement': {
        'actor': '0-1',
        'env,rollout': '2-3'
    },
    'action_chunk':
    50,
    'action_dim':
    14,
    'action_horizon':
    50,
    'num_steps':
    5,
    'noise_level':
    0.3,
    'tokenizer_path':
    '${actor.model.model_path}/paligemma_tokenizer.model',
    'norm_stats_path':
    ('${actor.model.model_path}/physical-intelligence/robotwi'
     'n/norm_stats.json'),
    'rollout_micro_batch_size':
    '${actor.micro_batch_size}',
    'adapter': {
        'type': 'RoboTwinRLAdapter',
        'image_size': 224,
        'max_token_len': 200
    },
    'model_overrides': {
        'n_action_steps': 50,
        'num_steps': 5,
        'attention_implementation': 'eager',
        'openpi_fp32_flow': True,
        'vision_backbone': {
            'openpi_stem_fp32': True
        }
    }
}

experiment = {
    'root': ('${oc.env:FLUX_ROBOTWIN_ROOT,${oc.env:FLUX_RL_ROOT,${oc.'
             'env:FLUXVLA_ROOT}/work_dirs/rlinf}}'),
    'split':
    'validation'
}

runner = {
    'max_epochs': 200,
    'max_steps': 200,
    'val_check_interval': 10,
    'save_interval': 10,
    'logger': {
        'log_path': '${experiment.root}/results',
        'experiment_name': 'adjust_bottle_pi05_seed1234',
        'project_name': 'fluxvla-rl-robotwin',
        'logger_backends': ['wandb', 'tensorboard']
    }
}

algorithm = {'update_epoch': 5}

actor = {
    'micro_batch_size': 2,
    'global_batch_size': 64,
    'seed': 1234,
    'enable_offload': False,
    'model': {
        'model_path':
        '${experiment.root}/weights/RLinf-Pi05-RoboTwin-SFT-adjust_bottle'
    },
    'optim': {
        'lr': 5e-06,
        'value_lr': 0.0001
    }
}

rollout = {'enable_offload': False}

env = {
    'enable_offload': False,
    'train': {
        'rollout_epoch':
        1,
        'total_num_envs':
        32,
        'seed':
        1234,
        'auto_reset':
        False,
        'ignore_terminations':
        True,
        'max_episode_steps':
        200,
        'max_steps_per_rollout_epoch':
        200,
        'center_crop':
        False,
        'assets_path':
        '${experiment.root}',
        'seeds_path':
        '${experiment.root}/protocol/train_seeds.json',
        'episode_log_dir':
        ('${runner.logger.log_path}/${runner.logger.experiment_na'
         'me}/episodes/train'),
        'video_cfg': {
            'save_video': False
        },
        'task_config': {
            'embodiment': ['aloha-agilex'],
            'planner_backend': 'mplib',
            'step_lim': 200,
            'save_path': '${experiment.root}/simulator_data',
            'eval_video_log': False,
            'camera': {
                'collect_wrist_camera': True
            },
            'domain_randomization': {
                'random_background': False,
                'cluttered_table': False,
                'clean_background_rate': 1,
                'random_head_camera_dis': 0,
                'random_table_height': 0,
                'random_light': False,
                'crazy_random_light_rate': 0
            }
        }
    },
    'eval': {
        'rollout_epoch':
        4,
        'total_num_envs':
        8,
        'seed':
        1234,
        'auto_reset':
        True,
        'ignore_terminations':
        True,
        'is_eval':
        True,
        'use_fixed_reset_state_ids':
        False,
        'max_episode_steps':
        200,
        'max_steps_per_rollout_epoch':
        200,
        'center_crop':
        False,
        'assets_path':
        '${experiment.root}',
        'seeds_path':
        '${experiment.root}/protocol/${experiment.split}_seeds.json',
        'episode_log_dir':
        ('${runner.logger.log_path}/${runner.logger.experiment_na'
         'me}/episodes/${experiment.split}'),
        'video_cfg': {
            'save_video': False
        },
        'task_config':
        '${env.train.task_config}'
    }
}
