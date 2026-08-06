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
"""Low-change, BF16 PI0.5 RoboCasa score-target recipe.

This is the single production recipe selected after auditing the experiment
sheet, RLinf, OpenPI, and StarVLA. It deliberately starts from the official
PI0.5 base checkpoint instead of continuing the 31.58% RoboCasa checkpoint.

The public StarVLA 43.9% result is from QwenPI_v2, not OpenPI PI0.5, so 40% is
a target rather than a reproduced guarantee. Only its uniform 24-task mixture
and larger sample budget are transferred. The optimizer schedule follows the
RLinf/OpenPI values, while parameter and Adam state precision intentionally
retain the historical FluxVLA BF16 behavior.

Expected topology: 4 nodes x 8 A800-80GB GPUs. The effective batch is
``4 samples/GPU * 32 GPUs * 2 micro-batches = 256``. For a different world
size, set ``runner.grad_accumulation_steps = 256 // (4 * world_size)``.
"""

_base_ = ['./pi05_paligemma_robocasa_full_data_full_finetune.py']

runner = dict(
    # 100k global-256 updates expose 25.6M samples: the same sample count as
    # the full-data stage behind the strongest measured 31.58% lineage and
    # four times the intended 6c recipe's 6.4M-sample budget.
    max_steps=100000,
    grad_accumulation_steps=2,
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        # PyTorch otherwise auto-selects the foreach CUDA path, whose first
        # step needs roughly one additional model-sized tensor list. The
        # fused kernel keeps the same AdamW state while avoiding that peak.
        foreach=False,
        fused=True,
    ),
    lr_scheduler=dict(
        _delete_=True,
        type='openpi-warmup+cosine-decay',
        warmup_steps=5000,
        decay_steps=100000,
        min_lr=2.5e-6,
    ),
    save_iter_interval=10000,
    max_keep_ckpts=10,
    # Reuse the historical RoboCasa execution path: replicated BF16 model and
    # BF16 Adam states. This intentionally avoids any runner/FSDP code change.
    sharding_strategy='no-shard',
    master_weight_dtype='bf16',
    fsdp_param_dtype='bf16',
    # Before the RLinf runner refactor, NO_SHARD always reduced gradients and
    # stored FSDP buffers in BF16. Override the new FP32 default explicitly so
    # this recipe retains the historical NO_SHARD BF16 memory behavior.
    reduce_in_full_precision=False,
)

# Inherit the sheet-comparable formal protocol unchanged: 24 tasks x 50
# deterministic trials, 0.95 crop, execute 8 actions, and ensemble weight 0.5.
