# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SmolVLA full-data fine-tuning and evaluation on RoboCasa GR1.

The RoboCasa LeRobot V2.1 conversion contains one 256x256 ego camera and
29-D absolute GR1 joint-position states/actions in FluxVLA order.  The model
keeps SmolVLA's native 32-D state/action width and 50-step action chunk; the
three padded dimensions are masked from the action loss and stripped before
the simulator receives an action.

Usage (adjust world size and batch accumulation for available GPUs):
    torchrun --nproc-per-node=8 --nnodes=2 scripts/train.py \\
        --config configs/smolvla/smolvla_robocasa_full_data_finetune.py \\
        --work-dir work_dirs/smolvla_robocasa_full_data_finetune

Evaluation:
    MUJOCO_GL=egl torchrun --nproc-per-node=1 scripts/eval.py \\
        --config configs/smolvla/smolvla_robocasa_full_data_finetune.py \\
        --ckpt-path <checkpoint.safetensors>
"""

import os

_CKPT_ROOT = './checkpoints'
_SMOLVLA_CHECKPOINT = os.environ.get(
    'SMOLVLA_CHECKPOINT', f'{_CKPT_ROOT}/smolvla_base/model.safetensors')
_SMOLVLA_TOKENIZER = os.environ.get(
    'SMOLVLA_TOKENIZER', f'{_CKPT_ROOT}/SmolVLM2-500M-Video-Instruct')

# Prefer a local link for portability, while allowing the mounted full-data
# V2.1 dataset used by the existing RoboCasa configurations without overrides.
_LOCAL_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_SHARED_ROBOCASA_DATA_ROOT = (
    '/mnt/data/cpfs/mnt/data/yiming/fluxvla/datasets/robocasa_lerobot_V2.1')
_DEFAULT_ROBOCASA_DATA_ROOT = (
    _LOCAL_ROBOCASA_DATA_ROOT if os.path.isdir(_LOCAL_ROBOCASA_DATA_ROOT) else
    _SHARED_ROBOCASA_DATA_ROOT)
_ROBOCASA_DATA_ROOT = os.environ.get('ROBOCASA_DATA_ROOT',
                                     _DEFAULT_ROBOCASA_DATA_ROOT)

_STATISTIC_NAME = 'robocasa_gr1_24tasks_absolute'
_ACTION_DIM = 29
_MAX_ACTION_DIM = 32
_MAX_STATE_DIM = 32
_ACTION_HORIZON = 50  # SmolVLA base checkpoint's native action chunk length.
_IMAGE_SIZE = 512
_PER_DEVICE_BATCH_SIZE = 64
_MAX_STEPS = 100000
_SAVE_ITER_INTERVAL = 5000

_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'
_ROBOCASA_TASK_DIRS = [
    'PnPBottleToCabinetClose',
    'PnPCanToDrawerClose',
    'PnPCupToDrawerClose',
    'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose',
    'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA',
    'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA',
    'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
    'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA',
    'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
    'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA',
    'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA',
    'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA',
    'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA',
    'PosttrainPnPNovelFromTrayToTieredshelfSplitA',
]


def _robocasa_data_path(task_name):
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name):
    return f'{_ROBOCASA_TASK_PREFIX}/{task_name}{_ROBOCASA_ENV_SUFFIX}'


# This is the same SmolVLA architecture/checkpoint mapping as the existing
# LIBERO recipes.  RoboCasa only changes input/output dimensions and data.
model = dict(
    type='SmolVLAFlowMatching',
    vlm_backbone=dict(
        type='SmolVLMBackbone',
        vision_config=dict(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=_IMAGE_SIZE,
            patch_size=16,
            intermediate_size=3072,
            hidden_act='gelu_pytorch_tanh',
            layer_norm_eps=1e-6,
        ),
        text_config=dict(
            hidden_size=960,
            num_hidden_layers=32,
            num_attention_heads=15,
            num_key_value_heads=5,
            head_dim=64,
            intermediate_size=2560,
            vocab_size=49280,
            rms_norm_eps=1e-5,
            hidden_act='silu',
            max_position_embeddings=8192,
        ),
        scale_factor=4,
        num_vlm_layers=16,
        torch_dtype='bfloat16',
    ),
    llm_expert=dict(
        type='SmolVLMExpert',
        hidden_size=720,
        num_hidden_layers=16,
        num_attention_heads=15,
        num_key_value_heads=5,
        head_dim=64,
        intermediate_size=-1,
        vocab_size=49280,
        attention_bias=False,
        rms_norm_eps=1e-5,
        hidden_act='silu',
        max_position_embeddings=8192,
        attention_mode='cross_attn',
        vlm_kv_dim=320,
        self_attn_every_n_layers=2,
        torch_dtype='bfloat16',
    ),
    state_proj=dict(
        type='LinearProjector', in_dim=_MAX_STATE_DIM, out_dim=960),
    action_in_proj=dict(
        type='LinearProjector', in_dim=_MAX_ACTION_DIM, out_dim=720),
    action_out_proj=dict(
        type='LinearProjector', in_dim=720, out_dim=_MAX_ACTION_DIM),
    action_time_mlp_in=dict(type='LinearProjector', in_dim=1440, out_dim=720),
    action_time_mlp_out=dict(type='LinearProjector', in_dim=720, out_dim=720),
    # Fine-tune the pretrained VLM together with the action expert/projections
    # for the RoboCasa embodiment and task distribution.
    freeze_vlm_backbone=False,
    max_action_dim=_MAX_ACTION_DIM,
    ori_action_dim=_ACTION_DIM,
    chunk_size=_ACTION_HORIZON,
    num_steps=10,
    add_image_special_tokens=False,
    pretrained_name_or_path=_SMOLVLA_CHECKPOINT,
    name_mapping={
        'vlm_backbone.vlm': 'model.vlm_with_expert.vlm.model',
        'llm_expert.expert': 'model.vlm_with_expert.lm_expert',
        'state_proj.projector': 'model.state_proj',
        'action_in_proj.projector': 'model.action_in_proj',
        'action_out_proj.projector': 'model.action_out_proj',
        'action_time_mlp_in.projector': 'model.action_time_mlp_in',
        'action_time_mlp_out.projector': 'model.action_time_mlp_out',
    },
)

_TOKENIZER = dict(
    type='PretrainedTokenizer',
    model_path=_SMOLVLA_TOKENIZER,
)

_TRANSFORMS = [
    dict(
        type='ProcessParquetInputs',
        parquet_keys=[
            'observation.state',
            'timestamp',
            'actions',
            'info',
            'stats',
            'action_masks',
        ],
        video_keys=['observation.images.ego_view'],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        },
    ),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_MAX_ACTION_DIM,
        state_dim=_MAX_STATE_DIM,
        # SmolVLA injects state and noisy action through separate projectors,
        # so retain its native per-field mean/std normalization.
        state_key='proprio',
        action_key='action',
        norm_type='mean_std',
    ),
    dict(
        type='ParquetPrompter',
        use_conversation=False,
        add_new_line=True,
    ),
    dict(type='ProcessPrompts', tokenizer=_TOKENIZER),
    # The source is square 256x256.  This retains the existing SmolVLA
    # letterbox convention and produces its expected [-1, 1] input range.
    dict(
        type='ResizeImagesWithPad',
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
        pad_direction='top-left',
    ),
    dict(type='SimpleNormalizeImages'),
]

train_dataloader = dict(
    per_device_batch_size=_PER_DEVICE_BATCH_SIZE,
    per_device_num_workers=4,
    dataset=dict(
        # Keep the 24 tasks equally represented even if their converted
        # sequence counts differ.  Their V2.1 meta/stats files are aggregated
        # into dataset_statistics.json by the training runner.
        type='DistributedBalancedRepeatingDataset',
        seed=42,
        reshuffle_each_epoch=True,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[
                _robocasa_data_path(task_dir)
                for task_dir in _ROBOCASA_TASK_DIRS
            ],
            transforms=_TRANSFORMS,
            action_window_size=_ACTION_HORIZON,
            action_key='action',
            use_delta=False,
            statistic_name=_STATISTIC_NAME,
            window_start_idx=0,
            supervise_terminal_padding=True,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=_MAX_STEPS,
    save_iter_interval=_SAVE_ITER_INTERVAL,
    max_keep_ckpts=2,
    # Match the repository's existing SmolVLA recipes.  This configuration
    # intentionally leaves accumulation at one; change it with world size
    # rather than silently changing the optimizer batch.
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0),
    max_grad_norm=10.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'lang_tokens',
            'lang_masks',
            'actions',
            'action_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats'],
    ),
    sampler=None,
    tokenizer=_TOKENIZER,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.03,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='no-shard',
    change_key_name=False,
)

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='smolvla',
    task_list=[
        _robocasa_task_env(task_dir) for task_dir in _ROBOCASA_TASK_DIRS
    ],
    total_tasks=len(_ROBOCASA_TASK_DIRS),
    # Predict SmolVLA's native 50-step chunk but replan after 10 actions
    # (0.5 s at the dataset's 20 Hz control rate).
    eval_chunk_size=10,
    max_episode_steps=720,
    num_trials_per_task=50,
    seed=7,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    unnorm_key=_STATISTIC_NAME,
    action_order='fluxvla',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=_IMAGE_SIZE,
                center_crop_scale=None,
                normalize=False,
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=_MAX_STATE_DIM,
                state_key='proprio',
                action_key=None,
                norm_type='mean_std',
            ),
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                add_new_line=True,
            ),
            dict(type='ProcessPrompts', tokenizer=_TOKENIZER),
            # ProcessRobocasaEvalInputs already resized the square image to
            # 512.  Map its uint8 HWC value to SmolVLA's [-1, 1] convention.
            dict(
                type='NormalizeImages',
                means=[[127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5]],
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='mean_std',
        action_dim=_ACTION_DIM,
        clip_actions=False,
        stats_order='native',
    ),
)
