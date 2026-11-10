# Copyright 2026 Limx Dynamics
"""GPT planner + fine-tuned PI05, one RoboCasa cup/drawer episode.

Use scripts/eval.py (not the ROS-only hierarchical inference entry point).
The simulator verifies grasp/placement/closure; only GPT selects subtasks.
This demonstrates the wiring, not certified subtask capability: the PI05
checkpoint was trained with whole-task instructions, not subtask labels.

OPENAI_API_KEY is inherited from the launch environment. OPENAI_BASE_URL,
OPENAI_API_KEY_ENV and OPENAI_PLANNER_MODEL can override the service.
PI05_ROBOCASA_CHECKPOINT_DIR overrides the training output directory.
Pass its checkpoint FILE to --ckpt-path.
"""

from os import environ as _environ

_base_ = ['../pi05/pi05_paligemma_robocasa_full_data_full_finetune.py']

_checkpoint_dir = _environ.get(
    'PI05_ROBOCASA_CHECKPOINT_DIR',
    '/mnt/data/cpfs/mnt/data/liyinhao/checkpoints/'
    'pi05_paligemma_robocasa_full_data_full_finetune_bs256')

# Use the tokenizer saved by this training run, not the base model directory.
_dataset = _base_.eval.dataset.copy()
for _transform in _dataset['transforms']:
    if _transform['type'] == 'ProcessPrompts':
        _transform['tokenizer']['model_path'] = f'{_checkpoint_dir}/tokenizer'

eval = dict(
    task_ids=[2],  # PnPCupToDrawerClose, initially open drawer.
    num_trials_per_task=1,
    max_episode_steps=720,
    eval_chunk_size=8,
    dataset=_dataset,
    norm_stats_path=f'{_checkpoint_dir}/dataset_statistics.json',
    output_dir='work_dirs/robocasa_pi05_gpt6',
    run_id_suffix='gpt-planner',
    save_video=True,
    feishu_report=False,
    observation_adapter=dict(type='RobocasaDrawerObservationAdapter'),
    policy=dict(
        type='HierarchicalPolicy',
        instruction_prefix='unlocked_waist: ',
        # Simulator facts are exact snapshots; do not repeatedly count the
        # same paused scene as multiple independent visual observations.
        stable_observations=1,
        max_rejections=1,
        planner=dict(
            type='OpenAISubtaskPlanner',
            model=_environ.get('OPENAI_PLANNER_MODEL') or 'gpt-6-astra',
            reasoning_effort='low',
            max_output_tokens=4096,
            # The simulator is paused during these calls (no physical robot).
            request_timeout=120.0,
            max_retries=1,
            image_keys=['video.ego_view_bg_crop_pad_res256_freq20'],
            image_detail='high'),
        monitor=dict(type='ObservationPredicateMonitor'),
        goal_conditions={
            'inside:cup:drawer': True,
            'closed:drawer': True,
        },
        skill_catalog=dict(
            entities=dict(cup='cup', drawer='drawer'),
            skills=[
                dict(
                    name='pick_up',
                    parameters=dict(object=['cup']),
                    instruction='pick up the {object}',
                    preconditions={
                        'inside:cup:drawer': False,
                        'closed:drawer': False,
                    },
                    completion={'grasped:{object}': True},
                    max_chunks=60),
                dict(
                    name='put_into',
                    parameters=dict(object=['cup'], target=['drawer']),
                    instruction='place the {object} into the {target}',
                    preconditions={
                        'grasped:{object}': True,
                        'closed:{target}': False,
                    },
                    completion={'inside:{object}:{target}': True},
                    max_chunks=60),
                dict(
                    name='close_drawer',
                    parameters=dict(drawer=['drawer']),
                    instruction='close the {drawer}',
                    preconditions={'inside:cup:{drawer}': True},
                    completion={'closed:{drawer}': True},
                    max_chunks=60),
            ])))

del _environ, _checkpoint_dir, _dataset, _transform
