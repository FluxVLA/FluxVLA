"""Two-GPU, one-update validation of the complete SmolVLA PPO pipeline."""

_base_ = './smolvla_libero_10_ppo.py'

simulator = dict(
    placement=dict(_delete_=True, actor='0-1', env='0-1', rollout='0-1'))

runner = dict(
    max_epochs=1,
    max_steps=1,
    val_check_interval=1,
    save_interval=1,
    logger=dict(experiment_name='smolvla_libero10_ppo_smoke'))

algorithm = dict(update_epoch=1)

actor = dict(micro_batch_size=2, global_batch_size=8)

env = dict(
    train=dict(
        total_num_envs=4,
        rollout_epoch=1,
        max_episode_steps=20,
        max_steps_per_rollout_epoch=20),
    eval=dict(
        total_num_envs=4,
        rollout_epoch=1,
        max_episode_steps=20,
        max_steps_per_rollout_epoch=20))
