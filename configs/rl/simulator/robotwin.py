"""Shared RoboTwin settings; select a task with env.train.task_config."""
defaults = [
    dict(**{'/env@env.train': 'robotwin_adjust_bottle'}),
    dict(**{'/env@env.eval': 'robotwin_adjust_bottle'}), '_self_'
]
simulator = dict(
    name='robotwin',
    placement=dict(actor='0-1', **{'env,rollout': '2-3'}),
    action_chunk=50,
    action_dim=14,
    action_horizon=50,
    num_steps=5,
    noise_level=0.3,
    tokenizer_path=('${actor.model.model_path}/paligemma_toke'
                    'nizer.model'),
    norm_stats_path=('${actor.model.model_path}/physical-intel'
                     'ligence/robotwin/norm_stats.json'),
    rollout_micro_batch_size='${actor.micro_batch_size}',
    adapter=dict(type='RoboTwinRLAdapter', image_size=224, max_token_len=200),
    model_overrides=dict(
        n_action_steps=50,
        num_steps=5,
        attention_implementation='eager',
        openpi_fp32_flow=True,
        vision_backbone=dict(openpi_stem_fp32=True)))
experiment = dict(
    root=('${oc.env:FLUX_ROBOTWIN_ROOT,${oc.env:FLU'
          'X_RL_ROOT,${oc.env:FLUXVLA_ROOT}/work_di'
          'rs/rlinf}}'),
    split='validation')
runner = dict(
    max_epochs=200,
    max_steps=200,
    val_check_interval=10,
    save_interval=10,
    logger=dict(
        log_path='${experiment.root}/results',
        experiment_name=('${env.train.task_config.task_name}_'
                         '${actor.model.experiment_tag}_seed${actor.seed}'),
        project_name='fluxvla-rl-robotwin',
        logger_backends=['wandb', 'tensorboard']))
algorithm = dict(update_epoch=5)
actor = dict(
    micro_batch_size=2,
    global_batch_size=64,
    seed=1234,
    enable_offload=False,
    # Torch 2.8 FSDP2 deferred sync corrupts later PI05 prefix forwards.
    # Sync each microbatch; RLinf still accumulates gradients until the
    # configured global batch is complete.
    fsdp_config=dict(enable_gradient_accumulation=False),
    optim=dict(lr=5e-06, value_lr=0.0001))
rollout = dict(enable_offload=False)
env = dict(
    enable_offload=False,
    train=dict(
        rollout_epoch=1,
        total_num_envs=32,
        seed=1234,
        auto_reset=False,
        ignore_terminations=True,
        max_episode_steps='${env.train.task_config.step_lim}',
        max_steps_per_rollout_epoch='${env.train.task_config.step_lim}',
        center_crop=False,
        assets_path='${experiment.root}',
        seeds_path='${experiment.root}/protocol/train_seeds.json',
        episode_log_dir=('${runner.logger.log_path}/${runner.logge'
                         'r.experiment_name}/episodes/train'),
        video_cfg=dict(save_video=False),
        task_config=dict(
            embodiment=['aloha-agilex'],
            planner_backend='mplib',
            step_lim=200,
            save_path='${experiment.root}/simulator_data',
            eval_video_log=False,
            camera=dict(collect_wrist_camera=True),
            domain_randomization=dict(
                random_background=False,
                cluttered_table=False,
                clean_background_rate=1,
                random_head_camera_dis=0,
                random_table_height=0,
                random_light=False,
                crazy_random_light_rate=0))),
    eval=dict(
        rollout_epoch=4,
        total_num_envs=8,
        seed=1234,
        auto_reset=True,
        ignore_terminations=True,
        is_eval=True,
        use_fixed_reset_state_ids=False,
        max_episode_steps='${env.train.task_config.step_lim}',
        max_steps_per_rollout_epoch='${env.train.task_config.step_lim}',
        center_crop=False,
        assets_path='${experiment.root}',
        seeds_path=('${experiment.root}/protocol/${experiment'
                    '.split}_seeds.json'),
        episode_log_dir=('${runner.logger.log_path}/${runner.logge'
                         'r.experiment_name}/episodes/${experiment'
                         '.split}'),
        video_cfg=dict(save_video=False),
        task_config='${env.train.task_config}'))
