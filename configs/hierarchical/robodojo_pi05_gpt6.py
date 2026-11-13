# Copyright 2026 Limx Dynamics
"""GPT planner + subtask-trained PI05, full RoboDojo benchmark evaluation.

Inherit the standard task=all, 50 episodes per task, 25 for generalization
base/random variants, and 2000-step limit from the PI05 benchmark config.
Small-batch debugging is selected with --cfg-options in launch.json on BOTH
the VLA Server and Themis Eval + Video entries, not in this shared config.
Output paths are launch-time settings: the VLA server's --cfg-options
themis.server.evaluation_reporting.result_output_dir and Themis --video-dir.
The ordinary Themis client sends observations; FluxVLA runs HierarchicalPolicy
and PI05 in the same debugged model process. Candidate instructions come
verbatim from training metadata, filtered by episode goal.
Training frame/episode ids never determine the online subtask. Tasks run
using the unchanged Themis main action-only protocol. Each request selects
a subtask independently from current cameras; no planner episode history or
stop decisions are used. Themis owns termination and scoring, without sending
success predicates to the planner.
Full scheduling does not imply full annotation coverage: a goal absent from
training metadata still fails explicitly, without invented fallback labels.
"""

_base_ = ['../pi05/pi05_paligemma_robodojo_full_finetune.py']

from os import environ as _environ  # noqa: E402

# Input checkpoint artifacts, not the output directory of this evaluation.
_checkpoint_dir = _environ.get(
    'PI05_ROBODOJO_CHECKPOINT_DIR',
    '/mnt/data/oss/users/liyinhao/projects/FluxVLA/work_dirs/'
    'pi05_robodojo_subtask_aee0551af4e_8node_bs128_'
    '20261110_dlc1c8el1nm8e4rz')
_checkpoint = _environ.get(
    'PI05_ROBODOJO_CHECKPOINT', f'{_checkpoint_dir}/checkpoints/'
    'step-060000-epoch-04-loss=0.0010.safetensors')
_dataset = _base_.eval.dataset.copy()
for _transform in _dataset['transforms']:
    if _transform['type'] == 'ProcessPrompts':
        _transform['tokenizer']['model_path'] = f'{_checkpoint_dir}/tokenizer'

# This is a FluxVLA-format fine-tuned checkpoint, not an OpenPI base export.
model = dict(
    pretrained_name_or_path=_checkpoint, name_mapping=dict(_delete_=True))
inference_model = model
train_dataloader = None
runner = None

eval = dict(dataset=_dataset, feishu_report=False)

policy = dict(
    type='HierarchicalPolicy',
    stable_observations=1,
    max_rejections=1,
    replan_every_chunks=1,
    # Relevant only if reused by an episode-mode runner; serving resets
    # per request and leaves the whole-episode limit to Themis.
    max_episode_chunks=250,
    completion_mode='disabled',
    planner_image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    subtask_source=dict(
        data_root=_environ.get('ROBODOJO_SUBTASK_DATA_ROOT')
        or '/mnt/data/oss/datasets/RoboDojo_lerobot_v3',
        subtask_prefix='dense',
        # Full training tasks contain up to 1743 distinct instructions.
        max_candidates=2048),
    planner=dict(
        type='OpenAISubtaskPlanner',
        # Candidate-only selection; stop prediction remains available in
        # agent-mode configs, but is not part of this benchmark protocol.
        completion_mode='disabled',
        allow_request_finish=False,
        model=_environ.get('OPENAI_PLANNER_MODEL') or 'gpt-5.6-sol',
        base_url=_environ.get('OPENAI_BASE_URL')
        or 'https://llm-api.limxint.cn/v1',
        reasoning_effort='low',
        max_output_tokens=4096,
        request_timeout=120.0,
        max_retries=1,
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist']),
    monitor=None,
    goal_conditions=None)

themis = dict(
    transport=dict(endpoint='tcp://127.0.0.1:15565'),
    evaluation=dict(
        environment=dict(type='RoboDojoEnvironment', device_id=0),
        execute_horizon=8,
        policy_client=dict(type='FluxVLAPolicyClient')),
    server=dict(
        mode='stateless',
        runner=dict(policy_execution='per_request'),
        ckpt_path=_checkpoint,
        norm_stats_path=f'{_checkpoint_dir}/dataset_statistics.json'))

del _environ, _checkpoint_dir, _checkpoint, _dataset, _transform
