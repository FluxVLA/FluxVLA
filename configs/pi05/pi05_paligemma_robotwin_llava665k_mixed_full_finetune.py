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
"""Pi05 mixed RoboTwin action training and raw LLaVA-665K language SFT.

Defaults target two 80-GB GPUs: three VLA updates followed by one VLM update.
Each update accumulates eight local batches of one sample (global batch 16).
Robot evaluation settings are inherited from the existing RoboTwin recipe.
"""

_base_ = './pi05_paligemma_robotwin_all_data_full_finetune.py'

llava_root = './datasets/llava665k'

model = dict(
    freeze_llm_expert=False,
    pretrained_skip_prefixes=['llm_expert.embed_tokens.'],
    strict_mapping=True,
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True))

inference_model = dict(
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True))

# Reuse the robot pipeline and README's datasets/robotwin_lerobotv2.1 layout.
train_dataloader = _base_.train_dataloader
train_dataloader.per_device_batch_size = 1
train_dataloader.per_device_num_workers = 4

vlm_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=4,
    grad_accumulation_steps=8,
    collator=dict(type='VLMSFTCollator', pad_token_id=0, image_layout='views'),
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=llava_root + '/llava_v1_5_mix665k.json',
        data_root_path=llava_root,
        index_cache_dir='./datasets/.cache/vlm',
        seed=42,
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                type='ProcessPaliGemmaVLMSFTInputs',
                image_size=224,
                max_length=1024,
                tokenizer_path='./checkpoints/pi05_base'),
        ]))

runner = dict(
    train_task='mixed',
    mixed_training=dict(
        task_schedule=('vla', 'vla', 'vla', 'vlm'),
        sync_each_microbatch=True,
        loss_weights=dict(vla=1.0, vlm=1.0)),
    max_epochs=None,
    max_steps=1000,
    grad_accumulation_steps=8,
    # Mixed Pi05 shards MLP calls common to both execution paths.
    fsdp_wrap_policy='model',
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=True,
    ema_decay=None,
    save_iter_interval=250,
    max_keep_ckpts=2,
    optimizer=dict(lr=2e-5),
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robotwin_llava665k_mixed'))

# Keep the existing robot evaluator; language benchmark data is not required.
eval = dict(output_dir='work_dirs/pi05_robotwin_llava665k_mixed')
