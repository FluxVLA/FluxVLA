"""Python RL defaults registered with Hydra by RLinfTrainRunner."""
defaults = [
    'hybrid_engines/fsdp@actor.fsdp_config',
    'weight_syncer/bucket_syncer@weight_syncer', '_self_',
    dict(model='pi05'),
    dict(simulator='libero10'),
    dict(**{'override hydra/job_logging': 'stdout'})
]
hydra = dict(
    searchpath=['file://${oc.env:RLINF_CONFIG_DIR}'],
    run=dict(dir='.'),
    job=dict(chdir=False),
    output_subdir=None)
cluster = dict(num_nodes=1, component_placement='${simulator.placement}')
runner = dict(
    task_type='embodied',
    logger=dict(
        log_path='${oc.env:FLUXVLA_ROOT}/work_dirs/rlinf',
        project_name='fluxvla_rl',
        experiment_name='libero_10_ppo_fluxvla_pi05',
        logger_backends=['tensorboard']),
    max_epochs=1000,
    max_steps=1000,
    weight_sync_interval=1,
    only_eval=False,
    val_check_interval=-1,
    save_interval=50,
    resume_dir=None,
    ckpt_path=None)
algorithm = dict(
    normalize_advantages=True,
    kl_penalty='kl',
    group_size=1,
    reward_coef=1.0,
    reward_type='chunk_level',
    logprob_type='chunk_level',
    entropy_type='token_level',
    update_epoch=4,
    adv_type='gae',
    loss_type='actor_critic',
    loss_agg_func='token-mean',
    kl_beta=0.0,
    entropy_bonus=0.0,
    clip_ratio_high=0.2,
    clip_ratio_low=0.2,
    clip_ratio_c=3.0,
    value_clip=0.2,
    huber_delta=10.0,
    gamma=0.99,
    gae_lambda=0.95,
    filter_rewards=False,
    rewards_lower_bound=0.1,
    rewards_upper_bound=0.9)
env = dict(
    group_name='EnvGroup',
    enable_offload=False,
    train=dict(
        rollout_epoch=1,
        total_num_envs=16,
        max_episode_steps=480,
        max_steps_per_rollout_epoch=480,
        reward_coef='${algorithm.reward_coef}',
        video_cfg=dict(save_video=False)),
    eval=dict(
        rollout_epoch=1,
        total_num_envs=16,
        max_episode_steps=480,
        max_steps_per_rollout_epoch=480,
        auto_reset=True,
        ignore_terminations=True,
        is_eval=True,
        group_size=1,
        video_cfg=dict(
            save_video=False,
            video_base_dir='${runner.logger.log_path}/video/eval')))
rollout = dict(
    group_name='RolloutGroup',
    generation_backend='huggingface',
    recompute_logprobs=False,
    enable_offload=True,
    pipeline_stage_num=1,
    enable_torch_compile=False,
    enable_cuda_graph=False,
    model=dict(model_path='${actor.model.model_path}', precision='fp32'))
actor = dict(
    group_name='ActorGroup',
    training_backend='fsdp',
    micro_batch_size=2,
    global_batch_size=32,
    seed=42,
    enable_offload=True,
    enable_sft_co_train=False,
    optim=dict(
        lr=1e-06,
        value_lr=0.0001,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-08,
        weight_decay=0.01,
        clip_grad=1.0,
        critic_warmup_steps=0,
        total_training_steps='${runner.max_steps}',
        min_lr_rate=1.0,
        lr_scheduler='constant'),
    fsdp_config=dict(
        strategy='fsdp2',
        sharding_strategy='full_shard',
        use_orig_params=True,
        gradient_checkpointing=False,
        wrap_policy=dict(
            transformer_layer_cls_to_wrap=(
                '${actor.model.fsdp_wrap_policy.transform'
                'er_layer_cls_to_wrap}'),
            module_classes_to_wrap=('${actor.model.fsdp_wrap_policy.module_cl'
                                    'asses_to_wrap}')),
        mixed_precision=dict(
            param_dtype='fp32',
            reduce_dtype='fp32',
            buffer_dtype='fp32',
            cast_forward_inputs=False),
        amp_autocast=dict(enabled=False)))
reward = dict(use_reward_model=False)
critic = dict(use_critic_model=False)
