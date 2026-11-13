# Copyright 2026 Limx Dynamics
"""GPT planner + subtask-trained PI05, full RoboDojo benchmark evaluation.

Inherit the standard task=all, 50 episodes per task, 25 for generalization
base/random variants, and 2000-step limit from the PI05 benchmark config.
Small-batch debugging is selected with --cfg-options in launch.json on BOTH
the VLA Server and Themis Eval + Video entries, not in this shared config.
The ordinary Themis client sends observations; FluxVLA runs HierarchicalPolicy
and PI05 in the same debugged model process. Candidate instructions come
verbatim from training metadata, filtered by episode goal.
Training frame/episode ids never determine the online subtask. Tasks run
serially; the agent judges each goal's completion from camera evidence.
Native simulator success is recorded separately, never supplied to the agent.
Full scheduling does not imply full annotation coverage: a goal absent from
training metadata still fails explicitly, without invented fallback labels.
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
    output_dir='work_dirs/robodojo_pi05_gpt6',
    feishu_report=False,
    policy=dict(
        type='HierarchicalPolicy',
        stable_observations=1,
        max_rejections=1,
        replan_every_chunks=1,
        # Cover the full 2000-step limit at eight executed actions per chunk.
        max_episode_chunks=250,
        # Two extra planner choices per snapshot, twelve per whole episode.
        # No action/step is fabricated while reconsidering blocked.
        max_control_replans=2,
        max_episode_control_replans=12,
        completion_mode='agent',
        planner_image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        # Visual criteria for this agent, NOT simulator predicate checks.
        completion_guidance='Declare the episode goal complete only when '
        'ALL conditions are clearly visible in CURRENT views: (1) all '
        'required objects are in the intended final arrangement; (2) both '
        'grippers are open and empty, with the objects released; (3) BOTH '
        'arms have returned to their INITIAL home poses. Check the LEFT '
        'arm and RIGHT arm separately: compare each gripper position AND '
        'orientation in CURRENT cam_high against INITIAL cam_high, using '
        'the corresponding wrist views as supporting evidence. Merely '
        'withdrawing from the objects or moving to the side is NOT home. '
        'If either arm is still extended, raised, or differently oriented, '
        'or its home pose is occluded or uncertain, do not request_finish. '
        'INITIAL views are home references, not current task-completion '
        'evidence. Do not invent metric distances or angles from images, '
        'or infer completion from label wording or elapsed steps. Before '
        'request_finish, describe the object outcome, left-arm home '
        'comparison and right-arm home comparison separately in evidence. '
        'If release or return-home motion is still needed, continue an '
        'appropriate original training instruction only when supported '
        'by the current scene. Do not invent a home skill or regrasp or '
        'disturb completed work. If no candidate can safely finish the '
        'remaining motion, choose blocked rather than claiming success.',
        subtask_source=dict(
            data_root=_environ.get('ROBODOJO_SUBTASK_DATA_ROOT')
            or '/mnt/data/oss/datasets/RoboDojo_lerobot_v3',
            subtask_prefix='dense',
            # Full training tasks contain up to 1743 distinct instructions.
            max_candidates=2048),
        planner=dict(
            type='OpenAISubtaskPlanner',
            # Agent judges completion directly; no simulator verifier.
            completion_mode='agent',
            allow_request_finish=True,
            model=_environ.get('OPENAI_PLANNER_MODEL') or 'gpt-5.6-sol',
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
            device_id=0,
            completion_diagnostics=False),
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
