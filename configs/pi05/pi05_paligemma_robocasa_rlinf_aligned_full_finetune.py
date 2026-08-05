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
"""RoboCasa PI0.5 source-parity baseline derived from RLinf/OpenPI.

This config keeps the full-data/evaluation contract from the score-oriented
recipe, but replaces its numerical and optimization boundaries with the
audited OpenPI JAX contract and RLinf's published SFT schedule.

For 16 ranks, ``4 samples/rank * 4 accumulation steps`` gives the published
global batch size of 256::

    torchrun --nproc_per_node=16 scripts/train.py \
        --config configs/pi05/\
pi05_paligemma_robocasa_rlinf_aligned_full_finetune.py \
        --work-dir work_dirs/pi05_robocasa_rlinf_aligned_full24k
"""

_base_ = ['./pi05_paligemma_robocasa_full_data_full_finetune.py']

model = dict(
    # Use the literal OpenPI grouped-query attention equation: FP32 QK logits,
    # finite Gemma mask constant, FP32 softmax, then BF16 probabilities.
    attention_implementation='jax',
    num_steps=10,
    # Keep actions/noise/time, action/time projections, velocity head, and loss
    # in FP32. Tokens are cast once at the Gemma boundary.
    openpi_fp32_flow=True,
    # OpenPI performs the SigLIP convolutional stem and position addition in
    # FP32, then casts the resulting tokens to the encoder compute dtype.
    vision_backbone=dict(openpi_stem_fp32=True),
)

runner = dict(
    max_steps=30000,
    grad_accumulation_steps=4,
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
    ),
    lr_scheduler=dict(
        _delete_=True,
        type='openpi-warmup+cosine-decay',
        warmup_steps=1000,
        decay_steps=30000,
        # RLinf's published BEHAVIOR PI0.5 config uses zero. OpenPI's generic
        # default is 2.5e-6, so that remains a clean one-field ablation.
        min_lr=0.0,
    ),
    save_iter_interval=5000,
    max_keep_ckpts=6,
    # RLinf wraps the whole policy once. Avoid FluxVLA's historical policy
    # that recursively wrapped many small Linear/LayerNorm modules.
    sharding_strategy='full-shard',
    use_fsdp_auto_wrap_policy=False,
    # FP32 masters are required for AdamW convergence. Keeping FSDP's forward
    # parameters FP32 lets local autocast exclusions preserve OpenPI's FP32
    # SigLIP stem and flow projections; transformer operators still run BF16
    # under the runner's ambient autocast context.
    master_weight_dtype='fp32',
    fsdp_param_dtype='fp32',
    cast_batch_to_mixed_precision=False,
    reduce_in_full_precision=True,
)

# Source-style closed-loop preprocessing/action execution. Keep the base
# config's 24 tasks, 50 deterministic scenes/task, stats, and denormalizer, but
# do not mix this number with the historical 0.95-crop/8-step-ensemble sheet.
eval = dict(
    eval_chunk_size=16,
    action_chunk_ensemble_weight=0.0,
    dataset=dict(transforms=[
        dict(
            type='ProcessRobocasaEvalInputs',
            img_key='video.ego_view_bg_crop_pad_res256_freq20',
            resize_size=224,
            center_crop_scale=None,
            normalize=True,
            value_range='tanh'),
        dict(
            type='NormalizeStatesAndActions',
            state_dim=29,
            state_key='proprio',
            action_key='action',
            norm_type='quantile'),
        dict(
            type='PreparePromptWithState',
            max_state_dim=29,
            lowercase_task_description=False,
            add_action_prefix=True),
        dict(
            type='ProcessPrompts',
            max_len=200,
            tokenizer=dict(
                type='PretrainedTokenizer',
                model_path='checkpoints/pi05_base')),
    ]),
)
