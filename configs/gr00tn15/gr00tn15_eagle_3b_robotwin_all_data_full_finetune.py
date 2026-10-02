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
"""GR00T N1.5 full-data fine-tuning and evaluation on RoboTwin.

The data combines clean and randomized LeRobot v2.1 datasets for all 50
RoboTwin tasks (27,500 episodes), with three 480x640 (H x W) RGB cameras.
Training uses 14-D absolute joint/gripper targets from future observation
states, with state-derived mean/std statistics shared by training and
closed-loop evaluation.

Usage (16 GPUs):
    # Run on both nodes with NODE_RANK=0/1 and the same MASTER_ADDR.
    torchrun --nproc-per-node=8 --nnodes=2 \
        --node-rank=${NODE_RANK} --master-addr=${MASTER_ADDR} \
        --master-port=29500 scripts/train.py \
        --config \
        configs/gr00tn15/gr00tn15_eagle_3b_robotwin_all_data_full_finetune.py \
        --work-dir work_dirs/gr00tn15_eagle_3b_robotwin_all_data_full_finetune

Evaluation:
    torchrun --nproc-per-node=1 scripts/eval.py \
        --config \
        configs/gr00tn15/gr00tn15_eagle_3b_robotwin_all_data_full_finetune.py \
        --ckpt-path <checkpoint.safetensors>

Evaluation defaults to all 50 tasks in both clean and random suites.
"""

# All 50 RoboTwin tasks, matching the upstream RoboTwin benchmark.
_ROBOTWIN_TASK_LIST = [
    'adjust_bottle',
    'beat_block_hammer',
    'blocks_ranking_rgb',
    'blocks_ranking_size',
    'click_alarmclock',
    'click_bell',
    'dump_bin_bigbin',
    'grab_roller',
    'handover_block',
    'handover_mic',
    'hanging_mug',
    'lift_pot',
    'move_can_pot',
    'move_pillbottle_pad',
    'move_playingcard_away',
    'move_stapler_pad',
    'open_laptop',
    'open_microwave',
    'pick_diverse_bottles',
    'pick_dual_bottles',
    'place_a2b_left',
    'place_a2b_right',
    'place_bread_basket',
    'place_bread_skillet',
    'place_burger_fries',
    'place_can_basket',
    'place_cans_plasticbox',
    'place_container_plate',
    'place_dual_shoes',
    'place_empty_cup',
    'place_fan',
    'place_mouse_pad',
    'place_object_basket',
    'place_object_scale',
    'place_object_stand',
    'place_phone_stand',
    'place_shoe',
    'press_stapler',
    'put_bottles_dustbin',
    'put_object_cabinet',
    'rotate_qrcode',
    'scan_object',
    'shake_bottle',
    'shake_bottle_horizontally',
    'stack_blocks_three',
    'stack_blocks_two',
    'stack_bowls_three',
    'stack_bowls_two',
    'stamp_seal',
    'turn_switch',
]

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_inference_timesteps=4,
        num_steps=50,
        action_dim=32,
        ori_action_dim=14),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_steps=50,
        num_inference_timesteps=4,
        ori_action_dim=14,
        action_dim=32,
        max_input_seq_len=900,
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None)))

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=7,
        reshuffle_each_epoch=True,
        name_mappings={'observation.state': ['proprio', 'action']},
        statistic_keys=['observation.state', 'timestamp'],
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=[
                    'datasets/robotwin_clean_lerobotv2.1',
                    'datasets/robotwin_randomized_lerobotv2.1',
                ],
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=0,
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist'
                        ],
                        name_mappings={'observation.state': ['states']}),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=3,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'fluxvla/models/third_party_models/eagle2_hg_model',  # noqa: E501
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='NormalizeImages',
                        means=[[123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375]],
                        stds=[[58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062]],
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std')
                ],
                action_window_size=50)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=3,
    optimizer=dict(
        lr=1e-5,
        type='AdamW',
        weight_decay=0.0,
        paramwise_learning_rate=dict(vla_head=1e-4)),
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path='fluxvla/models/third_party_models/eagle2_hg_model'),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks', 'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.03),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    type='RobotwinEvalRunner',
    task_suite_name=['clean', 'random'],
    model_family='groot',
    task_list=_ROBOTWIN_TASK_LIST,
    eval_chunk_size=50,
    num_trials_per_task=100,
    seed=7,
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=0,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=3,
                tokenizer=dict(type='PretrainedTokenizer')),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375]] * 3,
                stds=[[58.27148438, 57.02636719, 57.27539062]] * 3),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction', norm_type='mean_std', action_dim=14),
)
