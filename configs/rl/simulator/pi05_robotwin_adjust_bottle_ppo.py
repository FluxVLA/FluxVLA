"""Official adjust_bottle PPO hyperparameters for FluxVLA PI0.5 + RLinf.

Select with model=pi05 simulator=pi05_robotwin_adjust_bottle_ppo.
Reference: RLinf commit 88988ad5999619953b18d10b15bea1d3a6e1cfce,
examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml.
"""

_base_ = './robotwin.py'

simulator = dict(
    placement=dict(_delete_=True, actor='0-7', env='0-7', rollout='0-7'),
    action_chunk=50,
    action_dim=14,
    action_horizon=50,
    num_steps=5,
    noise_level=0.3,
    rollout_micro_batch_size=32)

runner = dict(
    max_epochs=1000,
    max_steps=-1,
    val_check_interval=10,
    save_interval=10,
    resume_dir=None,
    ckpt_path=None,
    logger=dict(
        log_path='${oc.env:FLUXVLA_ROOT}/work_dirs/rlinf',
        project_name='fluxvla-rl-robotwin',
        experiment_name='pi05_adjust_bottle_ppo_official_seed1234',
        swanlab_mode='cloud',
        logger_backends=['swanlab', 'tensorboard']))

algorithm = dict(
    normalize_advantages=True,
    adv_type='gae',
    loss_type='actor_critic',
    group_size=1,
    reward_coef=1.0,
    reward_type='chunk_level',
    logprob_type='chunk_level',
    entropy_type='token_level',
    loss_agg_func='token-mean',
    update_epoch=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio_high=0.2,
    clip_ratio_low=0.2,
    clip_ratio_c=3.0,
    value_clip=0.2,
    huber_delta=10.0,
    kl_beta=0.0,
    entropy_bonus=0.0,
    filter_rewards=False)

actor = dict(
    micro_batch_size=32,
    global_batch_size=2048,
    seed=1234,
    enable_offload=True,
    model=dict(
        model_path=('${oc.env:FLUXVLA_ROOT}/checkpoints/'
                    'RLinf-Pi05-RoboTwin-SFT-adjust_bottle'),
        num_action_chunks=50,
        action_dim=14,
        num_steps=5,
        add_value_head=True,
        is_lora=False,
        precision='fp32',
        fluxvla=dict(
            compute_dtype='bf16',
            train_expert_only=True,
            noise_method='flow_sde',
            noise_level=0.3,
            rollout_micro_batch_size=32)),
    optim=dict(
        lr=5e-6,
        value_lr=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.01,
        clip_grad=1.0,
        critic_warmup_steps=0,
        total_training_steps=1000,
        lr_scheduler='constant',
        min_lr_rate=1.0),
    # Keep the FluxVLA FSDP2 / FP32 master-weight implementation. Synchronize
    # every microbatch while accumulating optimizer gradients to batch 2048.
    fsdp_config=dict(
        strategy='fsdp2',
        sharding_strategy='full_shard',
        gradient_checkpointing=False,
        enable_gradient_accumulation=False))

rollout = dict(
    enable_offload=True,
    pipeline_stage_num=1,
    recompute_logprobs=False)

env = dict(
    enable_offload=True,
    train=dict(
        enable_offload=True,
        total_num_envs=256,
        rollout_epoch=4,
        seed=0,
        auto_reset=False,
        ignore_terminations=False,
        max_episode_steps=200,
        max_steps_per_rollout_epoch=200,
        assets_path='/mnt/data/cpfs/users/danny/env/src/RoboTwin',
        center_crop=False,
        use_custom_reward=True,
        use_rel_reward=True,
        video_cfg=dict(
            save_video=True,
            video_base_dir='${runner.logger.log_path}/video/train'),
        task_config=dict(
            task_name='adjust_bottle',
            step_lim=200,
            clear_cache_on_reset=False,
            planner_backend='mplib',
            embodiment=['aloha-agilex'],
            camera=dict(
                collect_head_camera=True, collect_wrist_camera=True),
            data_type=dict(
                rgb=True,
                third_view=False,
                depth=False,
                pointcloud=False,
                observer=False,
                endpose=False,
                qpos=True,
                mesh_segmentation=False,
                actor_segmentation=False),
            domain_randomization=dict(
                random_background=False,
                cluttered_table=False,
                clean_background_rate=1,
                random_head_camera_dis=0,
                random_table_height=0,
                random_light=False,
                crazy_random_light_rate=0))),
    eval=dict(
        enable_offload=True,
        total_num_envs=128,
        rollout_epoch=1,
        seed=0,
        auto_reset=True,
        ignore_terminations=True,
        use_fixed_reset_state_ids=True,
        max_episode_steps=200,
        max_steps_per_rollout_epoch=200,
        assets_path='${env.train.assets_path}',
        center_crop=False,
        video_cfg=dict(
            save_video=True,
            video_base_dir='${runner.logger.log_path}/video/eval')))
