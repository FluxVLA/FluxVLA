# Copyright 2026 Limx Dynamics
"""GPT planner + subtask-trained PI05, 10 RoboDojo stack_blocks episodes.

Start Themis (it waits for the model), then the VLA server from launch.json.
Alternatively, use the Themis Eval + Video launch entry after the VLA server.
The ordinary Themis client sends observations; FluxVLA runs HierarchicalPolicy
and PI05 in the same debugged model process. Candidate instructions come
verbatim from training metadata, filtered by episode goal.
Training frame/episode ids never determine the online subtask. Only
stack_blocks is configured; the agent judges completion from camera evidence.
Native simulator success is recorded separately, never supplied to the agent.
"""

_base_ = ['../pi05/pi05_paligemma_robodojo_full_finetune.py']

from os import environ as _environ  # noqa: E402

_run_dir = _environ.get(
    'PI05_ROBODOJO_CHECKPOINT_DIR',
    '/mnt/data/oss/users/liyinhao/projects/FluxVLA/work_dirs/'
    'pi05_robodojo_subtask_aee0551af4e_8node_bs128_'
    '20261110_dlc1c8el1nm8e4rz')
_checkpoint = _environ.get(
    'PI05_ROBODOJO_CHECKPOINT',
    f'{_run_dir}/checkpoints/step-060000-epoch-04-loss=0.0010.safetensors')
_dataset = _base_.eval.dataset.copy()
for _transform in _dataset['transforms']:
    if _transform['type'] == 'ProcessPrompts':
        _transform['tokenizer']['model_path'] = f'{_run_dir}/tokenizer'

# This is a FluxVLA-format fine-tuned checkpoint, not an OpenPI base export.
model = dict(
    pretrained_name_or_path=_checkpoint, name_mapping=dict(_delete_=True))
inference_model = model
train_dataloader = None
runner = None

eval = dict(
    dataset=_dataset,
    num_trials_per_task=10,
    num_trials_per_task_overrides=dict(_delete_=True),
    output_dir='work_dirs/robodojo_pi05_gpt6',
    feishu_report=False,
    policy=dict(
        type='HierarchicalPolicy',
        stable_observations=1,
        max_rejections=1,
        replan_every_chunks=1,
        max_episode_chunks=70,
        # Two extra planner choices per snapshot, twelve per whole episode.
        # No action/step is fabricated while reconsidering blocked.
        max_control_replans=2,
        max_episode_control_replans=12,
        completion_mode='agent',
        planner_image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        completion_guidance='Judge from the CURRENT camera views whether '
        'all three blocks form one upright stack on the table, the top block '
        'is released rather than held, and both arms have withdrawn to their '
        'visible resting region (compare INITIAL views). If these are clear, '
        'choose request_finish yourself. Do not infer completion from label '
        'wording or elapsed steps. If still holding/releasing/withdrawing, '
        'continue an appropriate original training instruction. Do not '
        'invent a home skill or regrasp an already completed stack.',
        subtask_source=dict(
            data_root=_environ.get('ROBODOJO_SUBTASK_DATA_ROOT')
            or '/mnt/data/oss/datasets/RoboDojo_lerobot_v3',
            subtask_prefix='dense',
            max_candidates=1024),
        planner=dict(
            type='OpenAISubtaskPlanner',
            # Agent judges completion directly; no simulator verifier.
            completion_mode='agent',
            allow_request_finish=True,
            model=_environ.get('OPENAI_PLANNER_MODEL') or 'gpt-5.6-luna',
            base_url=_environ.get('OPENAI_BASE_URL')
            or 'https://llm-api.limxint.cn/v1',
            reasoning_effort='low',
            max_output_tokens=4096,
            request_timeout=120.0,
            max_retries=1,
            image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist']),
        monitor=None,
        goal_conditions=None))

themis = dict(
    transport=dict(endpoint='tcp://127.0.0.1:15565'),
    evaluation=dict(
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='stack_blocks',
            device_id=0,
            completion_diagnostics=False),
        episodes_per_task=10,
        episodes_per_task_overrides=dict(_delete_=True),
        max_episode_steps=550,
        execute_horizon=8,
        policy_client=dict(
            type='FluxVLAPolicyClient',
            send_policy_context=True,
            send_environment_feedback=False)),
    server=dict(
        mode='stateful',
        ckpt_path=_checkpoint,
        norm_stats_path=f'{_run_dir}/dataset_statistics.json'))

del _environ, _run_dir, _checkpoint, _dataset, _transform
