"""FluxVLA SmolVLA PPO with the published LIBERO-10 SFT artifacts.

PPO hyperparameters inherit FluxVLA's RL bridge defaults, not an upstream
RLinf SmolVLA recipe. Normalization and tokenizer match the SFT checkpoint.
"""

_base_ = './libero10.py'

actor = dict(
    model=dict(
        model_path=(
            '${oc.env:SMOLVLA_LIBERO10_ROOT,'
            '/mnt/data/cpfs/users/danny/ckpt/smolvla/libero_10}'
            '/checkpoints/step-057096-epoch-36-loss=0.2340.safetensors')))

simulator = dict(
    tokenizer_path=(
        '${oc.env:SMOLVLA_LIBERO10_ROOT,'
        '/mnt/data/cpfs/users/danny/ckpt/smolvla/libero_10}/tokenizer'),
    norm_stats_path=(
        '${oc.env:SMOLVLA_LIBERO10_ROOT,'
        '/mnt/data/cpfs/users/danny/ckpt/smolvla/libero_10}'
        '/dataset_statistics.json'))

runner = dict(
    val_check_interval=10,
    save_interval=10,
    logger=dict(
        experiment_name='smolvla_libero10_ppo',
        swanlab_mode='cloud',
        logger_backends=['swanlab', 'tensorboard']))

env = dict(
    eval=dict(
        reward_coef='${algorithm.reward_coef}',
        use_ordered_reset_state_ids=True))
