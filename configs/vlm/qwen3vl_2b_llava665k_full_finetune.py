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
"""Qwen3-VL-2B full SFT on the original LLaVA-665K dataset.

The dataset indexes the original JSON once; no JSONL preparation is needed.
Run from the repository root after following the README data preparation.
Training length matches Pi05 LLaVA SFT; TextVQA is evaluation-only.
"""

model_path = './checkpoints/Qwen3-VL-2B-Instruct'
media_root = './datasets/llava665k'
train_annotations = media_root + '/llava_v1_5_mix665k.json'

model = dict(
    type='VLMWrapper',
    vlm_backbone=dict(
        type='CausalVLMBackbone',
        model_path=model_path,
        attn_implementation='sdpa',
        torch_dtype='fp32',
        freeze_vision=False),
    freeze_vlm_backbone=False)

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=train_annotations,
        data_root_path=media_root,
        index_cache_dir='./datasets/.cache/vlm',
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                type='ProcessVLMSFTInputs',
                processor_path=model_path,
                processor_type='qwen3_vl',
                image_max_pixels=262144,
                video_max_pixels=131072,
                video_max_frames=16,
                truncation_max_length=4096),
        ],
        max_retry_times=10,
        seed=42))

runner = dict(
    type='FSDPTrainRunner',
    train_task='vlm',
    max_epochs=1,  # Match one full LLaVA-665K epoch in Pi05 SFT.
    max_steps=None,
    sharding_strategy='full-shard',
    fsdp_wrap_policy='model',
    max_grad_norm=1.0,
    grad_accumulation_steps=2,  # 4 samples x 2 GPUs x 2 = global batch 16.
    sampler='distributed',
    # Padding is masked by lengths; filler token 0 is never supervised.
    collator=dict(type='VLMSFTCollator', pad_token_id=0),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs/qwen3vl_2b_llava665k_full_finetune',
        window_size=1),
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.01),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.03),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    save_epoch_interval=1,
    max_keep_ckpts=2,
    seed=42)

# Standalone TextVQA generation evaluation, also used by --eval-after-train.
eval = dict(
    type='TextVQAEvalRunner',
    checkpoint_dir=runner['metric']['run_dir'] + '/checkpoints',
    annotation_path='./datasets/textvqa/TextVQA_0.5.1_val.json',
    image_root=media_root + '/textvqa/train_images',
    preprocessing=dict(
        type='ProcessVLMSFTInputs',
        processor_path=model_path,
        processor_type='qwen3_vl',
        image_max_pixels=262144,
        truncation_max_length=4096),
    prompt_suffix='\nAnswer the question using a single word or phrase.',
    compare_base=True,
    max_samples=None,
    max_new_tokens=32,
    keep_params_fp32=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    run_dir=runner['metric']['run_dir'] + '/textvqa')
