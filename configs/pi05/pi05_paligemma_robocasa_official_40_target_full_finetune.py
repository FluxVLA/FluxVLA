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
RLinf/OpenPI values. Global SHARD_GRAD_OP keeps BF16 compute and globally
sharded FP32 state, but retains unsharded parameters between forward and
backward. This avoids both Hybrid's extra inter-node communicators and Full
Shard's second parameter all-gather during gradient-checkpoint recomputation.

Expected topology: 4 nodes x 8 RTX PRO 5000 72GB GPUs. The effective batch is
``8 samples/GPU * 32 GPUs * 1 micro-batch = 256``. For a different world
size, set ``runner.grad_accumulation_steps = 256 // (8 * world_size)``.
"""

_base_ = ['./pi05_paligemma_robocasa_full_data_full_finetune.py']

seed = 7

model = dict(
    # OpenPI uses a true Beta distribution and supervises all 32 padded action
    # dimensions. Keep these PI0.5-specific corrections local to this recipe.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    loss_action_dim=32,
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedBalancedRepeatingDataset',
        seed=seed,
        reshuffle_each_epoch=True,
        datasets=dict(supervise_terminal_padding=True),
    ),
)

runner = dict(
    # 100k global-256 updates expose 25.6M samples: the same sample count as
    # the full-data stage behind the strongest measured 31.58% lineage and
    # four times the intended 6c recipe's 6.4M-sample budget.
    max_steps=100000,
    grad_accumulation_steps=1,
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
    # This is the public PyTorch SHARD_GRAD_OP strategy on the default global
    # process group. Do not use this repo's legacy ``shard-grad-op`` spelling:
    # that maps to private _HYBRID_SHARD_ZERO2 and recreates the failing
    # ``_inter_node_pg`` communicator.
    sharding_strategy='global-shard-grad-op',
)

eval = dict(
    # Formal sheet-comparable protocol: 24 tasks x 50 deterministic trials.
    num_trials_per_task=50,
    episode_seed_stride=50,
)
