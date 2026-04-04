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

# VLM SFT config: LLaVA-format data, Qwen3.5 (e.g. Qwen3.5-0.8B).
# Saved checkpoints are HuggingFace-compatible.

_qwen3_5_base = 'Qwen/Qwen3.5-0.8B'

model = dict(
    type='VLMForSFT',
    vlm_backbone=dict(
        type='Qwen3_5',
        vlm_backbone_id='qwen3_5_0.8b_pt',
        vlm_path=_qwen3_5_base,
        vlm_config=None,
        use_projection=False,
        attn_implementation='flash_attention_2',
    ),
)

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=2,
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=[
            'datasets/Cambrian-10M/jsons/Cambrian7M_withsystemprompt.jsonl',
        ],
        data_root_path=[
            'datasets/Cambrian-10M/',
        ],
        processor_type='qwen3_5',
        model_path=_qwen3_5_base,
        image_max_pixels=32 * 32 * 1280,
        video_max_pixels=32 * 32 * 1280 * 2,
        truncation_max_length=0,
        statistic_name='vlm_sft',
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler='distributed',
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_qwen3_5_base,
    ),
    collator=dict(
        type='VLMSFTCollator',
        # Qwen3.5 eos/pad (verify tokenizer.pad_token_id)
        pad_token_id=248044,
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

eval = dict(
    type='VLMEvalRunner',
    use_llm_judge=False,
    verbose=False,
    prompt_suffix=None,
    ckpt_path=None,  # set by --ckpt-path
    benchmarks=[
        'gqa',
        'realworldqa',
        'ai2d',
    ],
    processor_type='qwen3_vl',
    data_root=None,
    seed=42,
    batch_size=1,
    max_samples_per_benchmark=None,
    output_dir=None,
    max_new_tokens=128,
    attn_implementation='flash_attention_2',
)
