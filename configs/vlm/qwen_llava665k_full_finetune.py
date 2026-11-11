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
"""Native Qwen3-VL / Qwen3.5 SFT over LLaVA JSON or JSONL annotations.

Defaults use Qwen3-VL-2B and raw LLaVA-665K prepared as in the README.
Override FLUXVLA_VLM_PATH, FLUXVLA_VLM_TRAIN and FLUXVLA_VLM_MEDIA as needed.
Set FLUXVLA_VLM_PROCESSOR=qwen3_5 for Qwen3.5.
Use FLUXVLA_VLM_VAL for optional teacher-forced validation.
Training length matches Pi05 LLaVA SFT; TextVQA is evaluation-only.
The processor and model must refer to the same checkpoint/token vocabulary.
"""

import os

model_path = os.environ.get('FLUXVLA_VLM_PATH',
                            './checkpoints/Qwen3-VL-2B-Instruct')
processor_type = os.environ.get('FLUXVLA_VLM_PROCESSOR', 'qwen3_vl')
media_root = os.environ.get('FLUXVLA_VLM_MEDIA', './datasets/llava665k')
train_annotations = os.environ.get('FLUXVLA_VLM_TRAIN',
                                   media_root + '/llava_v1_5_mix665k.json')
val_annotations = os.environ.get('FLUXVLA_VLM_VAL')

model = dict(
    type='VLMWrapper',
    vlm_backbone=dict(
        type='CausalVLMBackbone',
        model_path=model_path,
        attn_implementation='sdpa',
        torch_dtype='fp32',
        freeze_vision=False),
    freeze_vlm_backbone=False)

transforms = [
    dict(type='BuildVLMMessages'),
    dict(
        type='ProcessVLMSFTInputs',
        processor_path=model_path,
        processor_type=processor_type,
        image_max_pixels=262144,
        video_max_pixels=131072,
        video_max_frames=16,
        truncation_max_length=4096),
]

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=train_annotations,
        data_root_path=media_root,
        index_cache_dir='./datasets/.cache/vlm',
        transforms=transforms,
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
    # The collator masks padding by lengths. ID 0 is only filler and is
    # never supervised; it need not be the tokenizer's EOS/PAD ID.
    collator=dict(type='VLMSFTCollator', pad_token_id=0),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs/qwen_llava665k_full_finetune',
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

if val_annotations:
    eval_dataset = dict(
        type='LLaVAFormatDataset',
        anno_json_path=val_annotations,
        data_root_path=media_root,
        index_cache_dir='./datasets/.cache/vlm',
        transforms=transforms,
        max_retry_times=0,
        seed=42)
    runner['evaluator'] = dict(
        type='VLMEvaluator', eval_every=100, batch_size=1, max_batches=100)

# This generation benchmark is separate from the optional loss evaluator.
eval = dict(
    type='TextVQAEvalRunner',
    checkpoint_dir=runner['metric']['run_dir'] + '/checkpoints',
    annotation_path='./datasets/textvqa/TextVQA_0.5.1_val.json',
    image_root=media_root + '/textvqa/train_images',
    preprocessing=dict(
        type='ProcessVLMSFTInputs',
        processor_path=model_path,
        processor_type=processor_type,
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
