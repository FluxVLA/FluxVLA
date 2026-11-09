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
"""Pi05 RoboCasa action training + raw LLaVA-665K language SFT.

40,000 total updates = 30,000 VLA + 10,000 VLM at a 3:1 task ratio.
Both tasks use global batch 16 on two 80-GB GPUs. Each task keeps its own
dataset, transforms and collator; no converted language annotations are needed.
--eval-after-train evaluates RoboCasa robot success, not language loss.
Robot data defaults to ./datasets/robocasa_lerobot_V2.1; set
ROBOCASA_DATA_ROOT to use another location, without changing legacy recipes.
"""

import os

_base_ = './pi05_paligemma_robocasa_full_data_full_finetune.py'

llava_root = './datasets/llava665k'
_ROBOCASA_DATA_ROOT = os.environ.get('ROBOCASA_DATA_ROOT',
                                     './datasets/robocasa_lerobot_V2.1')

model = dict(
    freeze_llm_expert=False,
    pretrained_skip_prefixes=['llm_expert.embed_tokens.'],
    strict_mapping=True,
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True))

# Inherit the existing 24-task robot pipeline and action/state statistics.
# Keep portable defaults in this new recipe, not in the legacy VLA base.
train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=4,
    dataset=dict(
        datasets=dict(data_root_path=[
            f'{_ROBOCASA_DATA_ROOT}/{os.path.basename(path)}'
            for path in _base_.train_dataloader.dataset.datasets.data_root_path
        ])))

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
        seed=_base_.runner.seed,
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                type='ProcessPaliGemmaVLMSFTInputs',
                image_size=224,
                max_length=1024,
                tokenizer_path=_base_.runner.tokenizer.model_path),
        ]))

runner = dict(
    train_task='mixed',
    mixed_training=dict(
        task_schedule=('vla', 'vla', 'vla', 'vlm'),
        sync_each_microbatch=True,
        loss_weights=dict(vla=1.0, vlm=1.0)),
    max_epochs=None,
    max_steps=40000,
    grad_accumulation_steps=8,
    fsdp_wrap_policy='model',
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=True,
    ema_decay=None,
    save_iter_interval=4000,
    max_keep_ckpts=3,
    optimizer=dict(lr=2e-5),
    lr_scheduler=dict(warmup_steps=1200, decay_steps=40000, min_lr=2e-6),
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robocasa_llava665k_mixed'))

eval = dict(
    output_dir='work_dirs/pi05_robocasa_llava665k_mixed',
    deterministic_env=True,
    deterministic_action_sampling=True,
    save_video=False)
