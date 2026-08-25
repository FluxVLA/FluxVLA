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
"""Full-data PI0 fine-tuning on the RoboCasa GR1 tabletop tasks.

The dataset mixture, 16-step action targets, optimizer, schedule, and
RoboCasa evaluation protocol are inherited from the aligned PI0.5 recipe.
PI0 keeps the state as a 32D padded continuous suffix token and uses the
plain task prompt expected by the PI0 base checkpoint.

Example:
    torchrun --nproc_per_node=8 scripts/train.py \
        --config \
        configs/pi0/pi0_paligemma_robocasa_full_data_full_finetune.py \
        --work-dir work_dirs/pi0_paligemma_robocasa_full_data_full_finetune
"""

import os

_base_ = '../pi05/pi05_paligemma_robocasa_full_data_full_finetune.py'

train_seed = int(os.environ.get('PI0_TRAIN_SEED', '42'))
_PI0_CHECKPOINT = os.environ.get('PI0_CHECKPOINT',
                                 './checkpoints/pi0_base/model.safetensors')
_PI0_TOKENIZER_ROOT = (
    './checkpoints/pi0_base'
    if os.path.isfile('./checkpoints/pi0_base/tokenizer_config.json') else
    './checkpoints/pi05_base')
_PI0_TOKENIZER = os.environ.get('PI0_TOKENIZER', _PI0_TOKENIZER_ROOT)

# Keep the PI0 checkpoint architecture while matching the RoboCasa recipe's
# 16-step horizon and 29D robot action contract. The learned projectors are
# horizon-independent, so the base checkpoint can be used with this horizon.
model = dict(
    _delete_=True,
    type='PI0FlowMatching',
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    loss_action_dim=32,
    openpi_fp32_flow=True,
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    projector=dict(type='LinearProjector', in_dim=1152, out_dim=2048),
    proj_width=1024,
    n_action_steps=16,
    state_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    action_time_mlp_in=dict(type='LinearProjector', in_dim=2048, out_dim=1024),
    action_time_mlp_out=dict(
        type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=None,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=False,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=_PI0_CHECKPOINT,
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'action_time_mlp_in.projector': 'action_time_mlp_in',
        'action_time_mlp_out.projector': 'action_time_mlp_out',
        'state_proj.projector': 'state_proj',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=29,
)

inference_model = model.copy()

# PI0 consumes continuous state through state_proj, rather than serializing it
# into the language prompt as PI0.5 does. Everything else remains inherited
# from the PI0.5 RoboCasa data recipe.
train_dataloader = dict(
    dataset=dict(
        seed=train_seed,
        datasets=dict(transforms=[
            dict(
                type='ProcessParquetInputs',
                parquet_keys=[
                    'observation.state', 'timestamp', 'actions', 'info',
                    'stats', 'action_masks'
                ],
                video_keys=['observation.images.ego_view'],
                name_mappings={
                    'observation.state': ['states'],
                    'actions': ['actions'],
                },
                video_backend='pyav'),
            dict(
                type='RelativeActions',
                mask=([True] * 7 + [False] * 6 + [True] * 7 + [False] * 6 +
                      [True] * 3),
                state_key='states',
                action_key='actions'),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=32,
                state_dim=32,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                add_new_line=True),
            dict(
                type='ProcessPrompts',
                max_len=180,
                tokenizer=dict(
                    type='PretrainedTokenizer', model_path=_PI0_TOKENIZER)),
            dict(type='RandomCropImages', scale=0.95),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='ColorJitterImages',
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08),
            dict(type='SimpleNormalizeImages'),
        ])))

runner = dict(
    seed=train_seed,
    tokenizer=dict(type='PretrainedTokenizer', model_path=_PI0_TOKENIZER),
)

eval = dict(
    dataset=dict(transforms=[
        dict(
            type='ProcessRobocasaEvalInputs',
            img_key='video.ego_view_bg_crop_pad_res256_freq20',
            resize_size=224,
            center_crop_scale=0.95,
            normalize=True,
            value_range='tanh'),
        dict(
            type='NormalizeStatesAndActions',
            state_dim=32,
            state_key='proprio',
            action_key='action',
            norm_type='quantile',
            output_dtype='float32'),
        dict(
            type='ParquetPrompter', use_conversation=False, add_new_line=True),
        dict(
            type='ProcessPrompts',
            max_len=180,
            tokenizer=dict(
                type='PretrainedTokenizer', model_path=_PI0_TOKENIZER)),
    ]), )
