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

model = dict(
    type='OpenVLA',
    arch_specifier='no-align+fused-gelu-mlp',
    vision_backbone=dict(
        type='DinoSigLIPViTBackbone',
        vision_backbone_id='dinosiglip-vit-so-224px',
        dino_config=dict(
            model_id='dino',
            file=  # noqa: E251
            './checkpoints/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors'  # noqa: E501
        ),
        image_resize_strategy='resize-naive',
        siglip_config=dict(
            model_id='siglip_224',
            file=  # noqa: E251
            './checkpoints/ViT-SO400M-14-SigLIP/open_clip_model.safetensors'  # noqa: E501
        )),
    llm_backbone=dict(
        type='LLaMa2LLMBackbone',
        llm_backbone_id='llama2-7b-pure_causal',
        llm_family='llama',
        llm_path=  # noqa: E251
        './checkpoints/Llama-2-7b-hf',  # noqa: E501
        llm_max_length=2048,
        hf_token=None,
        inference_mode=False),
    projector=dict(
        type='FusedMLPProjector', fused_vision_dim=2176, llm_dim=4096),
    tokenizer=dict(
        type='ActionTokenizer',
        model_path=  # noqa: E251
        'checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
        bins=256,
        min_action=-1,
        max_action=1,
    ),
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
    vla_head=dict(type='OpenVLAHead', norm_stats=None, vocab_size=32000),
    freeze_vision_backbone=False,
    freeze_llm_backbone=False,
    freeze_projector=False,
    name_mapping={
        'llm_backbone.llm': 'language_model',
        'vision_backbone.siglip_featurizer':
        'vision_backbone.fused_featurizer',
        'vision_backbone.dino_featurizer': 'vision_backbone.featurizer'
    })

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='libero_10_no_noops',
        reshuffle_each_epoch=True,
        statistics_overrides=dict(
            libero_10_no_noops=dict(
                action=dict(
                    q01=[
                        -0.6348214149475098,
                        -0.7741071581840515,
                        -0.7633928656578064,
                        -0.09749999642372131,
                        -0.14819999992847435,
                        -0.2742857038974762,
                        0.0,
                    ],
                    q99=[
                        0.7714285850524902,
                        0.8464285731315613,
                        0.9375,
                        0.13928571343421936,
                        0.15964286029338837,
                        0.3246428668498993,
                        1.0,
                    ],
                ), ), ),
        datasets=dict(
            type='ParquetDataset',
            data_root_path='./datasets/libero_10_no_noops_lerobotv2.1',
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
                    video_keys=[
                        'observation.images.image',
                        'observation.images.image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    dataset_name='libero_10_no_noops',
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=7,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile',
                    state_norm_type='min_max',
                    action_norm_type='quantile',
                    clip_norm=True,
                    action_norm_mask=[
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                    ],
                ),
                dict(
                    type='ParquetPrompter',
                    lowercase_task_description=True,
                    action_tokenizer=dict(
                        type='ActionTokenizer',
                        model_path=  # noqa: E251
                        './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
                        bins=256,
                        min_action=-1,
                        max_action=1,
                    ),
                ),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
                        # special_tokens={'pad_token': '<PAD>'}
                    ),
                    max_len=None,
                    with_labels=True,
                ),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375],
                           [128, 128, 128]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [128, 128, 128]],
                ),
            ],
            action_window_size=1,
            action_key='action',
            use_delta=False,
            statistic_name='libero_10_no_noops',
            window_start_idx=0,
        ),
    ))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=24,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    collator=dict(
        type='PaddedCollatorForActionPrediction',
        model_max_length=2048,
        pad_token_id=0,
        padding_side='right',
        pixel_values_dtype='fp16',
        ignore_idx=-100),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='constant',
    warmup_ratio=0.0,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    eval=dict(
        type='LiberoEvalRunner',
        model_family='openvla',
        task_suite_name='libero_10',
        dataset=dict(
            type='LiberoParquetEvalDataset',
            transforms=[
                dict(
                    type='ProcessLiberoEvalInputs',
                    img_keys=['agentview_image', 'agentview_image'],
                ),
                dict(
                    type='TransformImage',
                    image_resize_strategy='resize-naive',
                    input_sizes=[[3, 224, 224], [3, 224, 224]],
                    means=[[123.515625, 116.04492188, 103.59375],
                           [128, 128, 128]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [128, 128, 128]],
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    prompt_suffix=' ',
                    max_len=None,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        'openvla/openvla-7b-finetuned-libero-10',  # noqa: E501
                        # special_tokens={'pad_token': '<PAD>'}
                    )),
            ]),
        denormalize_action=dict(
            type='DenormalizeLiberoAction',
            norm_type='quantile',
            action_norm_mask=[True, True, True, True, True, True, False],
        ),
        resize_size=224,
        num_trials_per_task=50,
        num_steps_wait=10,
        seed=7))

eval = dict(
    type='LiberoEvalRunner',
    model_family='openvla',
    task_suite_name='libero_10',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'agentview_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[123.515625, 116.04492188, 103.59375], [128, 128, 128]],
                stds=[[58.27148438, 57.02636719, 57.27539062], [128, 128,
                                                                128]],
            ),
            dict(
                type='LiberoPromptFromInputs',
                prompt_suffix=' ',
                max_len=None,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
                    # special_tokens={'pad_token': '<PAD>'}
                )),
        ]),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='quantile',
        action_norm_mask=[True, True, True, True, True, True, False],
    ),
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7)
