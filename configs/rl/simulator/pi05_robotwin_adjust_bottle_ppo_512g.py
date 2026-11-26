"""PI0.5 adjust_bottle PPO with fewer concurrent training scenes.

The 512 GiB host exceeded Ray's memory threshold with 256 training scenes.
64 environments x 16 rounds retain the reference 1024 rollout episodes per
iteration. Keep all 128 fixed-seed evaluation environments in one round.
"""

_base_ = './pi05_robotwin_adjust_bottle_ppo.py'

runner = dict(
    logger=dict(experiment_name='pi05_adjust_bottle_ppo_512g_seed1234'))

env = dict(
    train=dict(
        total_num_envs=64, rollout_epoch=16, video_cfg=dict(save_video=False)))
