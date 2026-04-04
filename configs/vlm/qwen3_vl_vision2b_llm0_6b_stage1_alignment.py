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

# Stage1: Alignment. Only Projection (merger) is trained; VisionEncoder and
# Backbone (LLM) are frozen. Use the output checkpoint as input for stage2.

# Paths (adjust to your env)
# _qwen3vl_2b = 'Qwen/Qwen3-VL-2B-Instruct'
# _qwen3_0_6b = 'Qwen/Qwen3-0.6B'
_qwen3vl_2b = 'Qwen/Qwen3-VL-2B-Instruct'
_qwen3_0_6b = 'Qwen/Qwen3-0.6B'

model = dict(
    type='VLMForSFT',
    vlm_backbone=dict(
        type='Qwen3VL',
        vlm_backbone_id='qwen3_2b_vl_pt',
        vlm_path=_qwen3vl_2b,
        vlm_config=None,
        use_projection=False,
        attn_implementation='flash_attention_2',
        vision_encoder_path=_qwen3vl_2b,
        llm_backbone_path=_qwen3_0_6b,
    ),
    # Stage1: only projection (merger) trainable
    freeze_vision_encoder=True,
    freeze_projection=False,
    freeze_backbone=True,
)

train_dataloader = dict(
    per_device_batch_size=12,
    per_device_num_workers=2,
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=[
            'datasets/Cambrian-10M/jsons/Cambrian737k.jsonl',
        ],
        data_root_path=[
            'datasets/Cambrian-10M/',
        ],
        processor_type='qwen3_vl',
        model_path=_qwen3vl_2b,
        image_max_pixels=32 * 32 * 1280,
        video_max_pixels=32 * 32 * 1280 * 2,
        truncation_max_length=1800,
        statistic_name='vlm_sft',
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=2,
    learning_rate=1e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler='distributed',
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_qwen3vl_2b,
    ),
    collator=dict(
        type='VLMSFTCollator',
        pad_token_id=151643,
        ignore_index=-100,
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.03,
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False,
)
