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

# Qwen3-VL-2B VLM SFT config.
# Reference:
# - configs/vlm/qwen3_vl_sft_vision2b_llm0_6b_stage2_sft.py (overall layout)
# - configs/gr00t/gr00t_qwen3vl_2b_libero_all_full_finetune.py
#   (2B backbone path/model scale)

_qwen3vl_2b_base = 'Qwen/Qwen3-VL-2B-Instruct'

model = dict(
    type='VLMForSFT',
    vlm_backbone=dict(
        type='Qwen3VL',
        vlm_backbone_id='qwen3_2b_vl_pt',
        vlm_path=_qwen3vl_2b_base,
        vlm_config=None,
        use_projection=False,
        attn_implementation='flash_attention_2',
    ),
    # Full SFT: train vision/projection/backbone together.
    freeze_vision_encoder=False,
    freeze_projection=False,
    freeze_backbone=False,
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=[
            'datasets/Cambrian-10M/jsons/Cambrian7M_withsystemprompt.jsonl',
        ],
        data_root_path=[
            'datasets/Cambrian-10M/',
        ],
        processor_type='qwen3_vl',
        model_path=_qwen3vl_2b_base,
        image_max_pixels=32 * 32 * 1280,
        video_max_pixels=32 * 32 * 1280 * 2,
        truncation_max_length=2000,
        statistic_name='vlm_sft',
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=4,
    learning_rate=1e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler='distributed',
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_qwen3vl_2b_base,
    ),
    collator=dict(
        type='VLMSFTCollator',
        # Qwen3-VL pad/eos id (verify with tokenizer.pad_token_id if changed).
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

# VLM evaluation (optional):
# python scripts/eval.py --config configs/vlm/qwen3_vl_2b_sft.py \
#   --ckpt-path <path_to_hf_checkpoint>
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
