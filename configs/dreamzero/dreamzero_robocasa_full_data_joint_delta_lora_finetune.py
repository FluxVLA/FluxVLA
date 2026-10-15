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
"""DreamZero RoboCasa GR1, source: ab790c198fbce33503358efbbd4187ce9a89adf3.

From the repo root (use a new work_dir; do not checkout the old commit):
  ROBOCASA_DATA_ROOT=/path/to/robocasa_lerobot_V2.1 WANDB_MODE=disabled \
  bash scripts/train.sh \
  configs/dreamzero/dreamzero_robocasa_full_data_joint_delta_lora_finetune.py \
  work_dirs/dreamzero_robocasa_joint_delta \
  --cfg-options train_dataloader.per_device_batch_size=2 \
  runner.grad_accumulation_steps=1

The collator pads variable windows to the longest sample in each batch.
Video/action losses exclude padding and retain equal weight per sample.
Batch 2 with accumulation 1 gives effective batch 32 on 16 GPUs.

Evaluation explicitly places weights in BF16: the merged FP32 checkpoint
alone occupies about 85.4 GiB, and autocast does not convert model weights.

Data: 1..4 blocks, each 24 actions / 8 future RGB frames (stride 3) / one
raw state anchor. Arms/waist are block-relative; Fourier-hand commands remain
absolute. Image/state/prompt use the checkpoint's gr1_unified transform,
not AgiBot views.
LoRA uses scripts/train/agibot_training.sh; sampling uses lerobot_sharded.py.
For another dataset, recompute h24 robocasa-joint-delta statistics with
 tools/compute_pi05_norm_stats.py --action-horizon 24 --window-start-index 0
 --exclude-terminal-padding, and set DREAMZERO_STATS_PATH to its JSON output.

Remaining differences: FluxVLA DDP/mixture and PyAV, rather than source
DeepSpeed/sharded mixture/decord. Full training, 1200-episode evaluation and
large-checkpoint numerical parity have NOT been established.
"""

import json
import os
import pathlib

_CKPT_ROOT = os.environ.get('DREAMZERO_CKPT_ROOT', './checkpoints')
_TOKENIZER = _CKPT_ROOT + '/Wan2.1-I2V-14B-480P/google/umt5-xxl'

_MODEL_NUM_FRAMES = 33
_TRAIN_FRAME_WINDOW_SIZE = 33
_FRAME_SAMPLE_STRIDE = 3
_ACTION_HORIZON = 24
_NUM_BLOCKS = 4
seed = 42

_NUM_VIEWS = 1
_IMAGE_SIZE = 256
_FRAME_SEQUENCE_LENGTH = (_IMAGE_SIZE // 16)**2
_PROMPT_TEMPLATE = 'A single view video shows that a human {task}'
_NEGATIVE_PROMPT = (
    'Vibrant colors, overexposed, static, blurry details, text, subtitles, '
    'style, artwork, painting, image, still, grayscale, dull, worst quality, '
    'low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, '
    'bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant '
    'image, cluttered background, three legs, many people in the background, '
    'walking backwards.')

model = dict(
    type='DreamZeroVLA',
    num_views=_NUM_VIEWS,
    frame_window_size=_MODEL_NUM_FRAMES,
    pretrained_name_or_path=  # noqa: E251
    _CKPT_ROOT + '/DreamZero-AgiBot',
    use_cache=True,
    strict_mapping=True,
    # Unused CLIP scalar is [] upstream vs [1] locally;
    # strict for all other keys.
    pretrained_skip_prefixes=['vlm_backbone.image_encoder.model.log_scale'],
    # New-embodiment recipe: rank-4 adapters and trainable
    # state/action projectors.
    use_lora=True,
    lora_rank=4,
    lora_alpha=4,
    lora_dropout=0.0,
    init_lora_weights=True,
    lora_target_modules=(
        r'^vla_head\.model\.blocks\.\d+\.(?:self_attn|cross_attn)\.'
        r'(?:q|k|v|o)$|^vla_head\.model\.blocks\.\d+\.ffn\.(?:0|2)$'),
    modules_to_save=[
        'state_encoder',
        'action_encoder',
        'action_decoder',
    ],
    vlm_backbone=dict(
        type='Wan21Backbone',
        text_encoder_path=None,
        image_encoder_path=None,
        vae_path=None,
        tiled=False,
    ),
    vla_head=dict(
        type='DreamZeroHead',
        action_dim=29,
        max_action_dim=32,
        action_horizon=_ACTION_HORIZON,
        max_state_dim=64,
        num_frames=_MODEL_NUM_FRAMES,
        num_frame_per_block=2,
        num_action_per_block=_ACTION_HORIZON,
        num_state_per_block=1,
        frame_seqlen=_FRAME_SEQUENCE_LENGTH,
        hidden_size=1024,
        input_embedding_dim=1536,
        dit_dim=5120,
        dit_ffn_dim=13824,
        dit_num_heads=40,
        dit_num_layers=40,
        dit_freq_dim=256,
        dit_in_dim=36,
        dit_out_dim=16,
        max_num_embodiments=32,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_inference_steps=16,
        # Fixed seed 1140; 16 BF16 UniPC updates / 8 DiT evaluations.
        official_inference=True,
        use_gradient_checkpointing=True,
        cfg_scale=5.0,
        max_chunk_size=4,
    ),
    name_mapping={
        'vla_head.model': 'action_head.model',
        'vlm_backbone.text_encoder': 'action_head.text_encoder',
        'vlm_backbone.image_encoder': 'action_head.image_encoder',
        'vlm_backbone.vae': 'action_head.vae',
    },
)

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_joint_delta'
_ROBOCASA_DATA_ROOT = os.environ.get('ROBOCASA_DATA_ROOT',
                                     './datasets/robocasa_lerobot_V2.1')
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'

_ROBOCASA_JOINT_DELTA_MASK = ([True] * 7 + [False] * 6 + [True] * 7 +
                              [False] * 6 + [True] * 3)
_ROBOCASA_N15_JOINT_DELTA_MASK = ([True] * 7 + [True] * 7 + [False] * 6 +
                                  [False] * 6 + [True] * 3)

# h24 stats from dd2039f1: 24,000 episodes / 6,020,058 frames.
# Raw FluxVLA order; retain only min/max consumed by this experiment.
_ROBOCASA_DATASET_STATISTICS = {
    'robocasa_gr1_24tasks_joint_delta': {
        'proprio': {
            'count': 6020058
        },
        'action': {
            'min': [
                -1.1532576084136963, -0.7710707187652588, -1.3112866878509521,
                -1.5591628551483154, -1.6683788299560547, -1.3634177446365356,
                -1.1948705911636353, -1.5, -1.5, -1.5, -1.5, -3.0, 0.0,
                -1.482505440711975, -1.5787591934204102, -1.6719448566436768,
                -1.9195635318756104, -1.7429898977279663, -1.7818541526794434,
                -2.0731394290924072, -1.5, -1.5, -1.5, -1.5, -3.0, 3.0,
                -0.4199202060699463, -0.26482921838760376, -0.2804606854915619
            ],
            'max': [
                1.206160545349121, 1.0281989574432373, 1.18000328540802,
                1.896183729171753, 1.784941554069519, 1.3858094215393066,
                1.2729451656341553, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                1.6917004585266113, 1.1616510152816772, 1.5212913751602173,
                1.9471081495285034, 1.8994455337524414, 1.6273449659347534,
                2.135958671569824, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.4214596152305603, 0.28732696175575256, 0.3515920639038086
            ],
            'count':
            137857392
        }
    }
}

_STATS_PATH = os.environ.get('DREAMZERO_STATS_PATH')
if _STATS_PATH:
    _loaded_stats = json.loads(
        pathlib.Path(_STATS_PATH).read_text(encoding='utf-8'))
    _stats_metadata = _loaded_stats.get('metadata', {})
    if _stats_metadata.get('action_horizon', 24) != 24:
        raise ValueError('DreamZero requires h24 relative-action statistics')
    if _stats_metadata.get('profile',
                           'robocasa-joint-delta') != 'robocasa-joint-delta':
        raise ValueError(
            'Expected the robocasa-joint-delta statistics profile')
    _ROBOCASA_DATASET_STATISTICS = _loaded_stats.get('norm_stats',
                                                     _loaded_stats)

# Persist action stats in model order (LA, RA, LH, RH, waist).
# External statistics must use the raw order emitted by
# compute_pi05_norm_stats.
_STATE_PERMUTATION = (
    list(range(7)) + list(range(13, 20)) + list(range(7, 13)) +
    list(range(20, 29)))
for _stats in _ROBOCASA_DATASET_STATISTICS.values():
    for _key in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
        if _stats.get('action', {}).get(_key) is not None:
            _stats['action'][_key] = [
                _stats['action'][_key][i] for i in _STATE_PERMUTATION
            ]

_ROBOCASA_TASKS = [
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


train_dataloader = dict(
    # DreamZeroCollator pads variable 1..4-block windows within each batch.
    per_device_batch_size=2,
    per_device_num_workers=0,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        dataset_statistics=_ROBOCASA_DATASET_STATISTICS,
        datasets=dict(
            type='LanguageChunkParquetDataset',
            max_chunk_size=_NUM_BLOCKS,
            data_root_path=[
                _robocasa_data_path(task_name) for task_name in _ROBOCASA_TASKS
            ],
            transforms=[
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
                    embodiment_id=0,
                    video_backend=os.environ.get('DREAMZERO_VIDEO_BACKEND',
                                                 'pyav'),
                ),
                dict(
                    type='ChunkRelativeActions',
                    action_chunk_size=_ACTION_HORIZON,
                    mask=_ROBOCASA_JOINT_DELTA_MASK,
                ),
                dict(
                    type='RobocasaGR1N15Bridge',
                    apply_state_sincos=True,
                    reorder_action_stats=False,
                ),
                dict(
                    type='ParquetPrompter',
                    use_conversation=False,
                    lowercase_task_description=True,
                    prompt_template=_PROMPT_TEMPLATE,
                ),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_TOKENIZER,
                    ),
                    max_len=512,
                ),
                dict(
                    type='ConsistentVideoTransform',
                    height=_IMAGE_SIZE,
                    width=_IMAGE_SIZE,
                    training=True),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max',
                    clip_norm=True,
                    normalization_epsilon=0.0,
                    preserve_input_dtype=True,
                    zero_constant_min_max_dims=True,
                    normalize_states=False,
                ),
                dict(
                    type='PrepareVideo',
                    num_views=_NUM_VIEWS,
                    frame_window_size=_TRAIN_FRAME_WINDOW_SIZE,
                ),
            ],
            action_window_size=_ACTION_HORIZON * _NUM_BLOCKS,
            action_key='action',
            action_dtype=None,
            use_delta=False,
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
            frame_window_size=_TRAIN_FRAME_WINDOW_SIZE,
            frame_sample_stride=_FRAME_SAMPLE_STRIDE,
        ),
    ),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    # 16 GPUs x 2 samples = effective batch 32.
    grad_accumulation_steps=1,
    # Static DDP fails on the first no_sync backward during accumulation.
    static_graph=False,
    optimizer=dict(lr=1e-5, type='AdamW', weight_decay=1e-5),
    max_grad_norm=1.0,
    save_iter_interval=5000,
    max_keep_ckpts=8,
    collator=dict(
        type='DreamZeroCollator',
        video_frames_per_block=8,
        actions_per_block=_ACTION_HORIZON,
        states_per_block=1,
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    # Frozen weights BF16; adapter/projector parameters and AdamW moments FP32.
    keep_lora_trainable_params_fp32=True,
)

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='dreamzero',
    task_list=[_robocasa_task_env(task_name) for task_name in _ROBOCASA_TASKS],
    total_tasks=24,
    eval_chunk_size=8,
    max_episode_steps=720,
    num_trials_per_task=50,
    episode_seed_stride=50,
    seed=7,
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    action_order='n15',
    action_keys={
        'action.left_arm': (0, 7),
        'action.right_arm': (7, 14),
        'action.left_hand': (14, 20),
        'action.right_hand': (20, 26),
        'action.waist': (26, 29),
    },
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    # Autocast alone leaves the 23B-parameter model in FP32 (>80 GiB).
    # Convert weights while placing them on the rank-local GPU.
    model_build_dtype='bf16',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            # Record all control steps: eight executed actions are only a
            # third of a training block. Rebuild history at video stride 3
            # instead of advancing a 24-action cache block on every replan.
            dict(
                type='ProcessRobocasaVideoHistoryInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=_IMAGE_SIZE,
                normalize=False,
                embodiment_id=0,
                frame_stride=_FRAME_SAMPLE_STRIDE,
                frames_per_block=8,
                max_history_blocks=_NUM_BLOCKS - 1,
            ),
            dict(
                type='ConsistentVideoTransform',
                key='pixel_values',
                height=_IMAGE_SIZE,
                width=_IMAGE_SIZE,
                training=False),
            dict(
                type='RobocasaGR1N15Bridge',
                apply_state_sincos=True,
                reorder_action_stats=False,
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='min_max',
                clip_norm=True,
                normalize_states=False,
            ),
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                lowercase_task_description=True,
                prompt_template=_PROMPT_TEMPLATE,
            ),
            dict(
                type='ProcessPrompts',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_TOKENIZER,
                ),
                max_len=512,
                negative_prompt=_NEGATIVE_PROMPT,
            ),
            dict(
                type='PrepareVideo',
                num_views=_NUM_VIEWS,
                frame_window_size=1,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        norm_type='min_max',
        action_dim=29,
        delta_action_mask=_ROBOCASA_N15_JOINT_DELTA_MASK,
        state_permutation=_STATE_PERMUTATION,
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        clip_normalized_action=True,
    ),
)
