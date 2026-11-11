# Copyright 2026 Limx Dynamics
"""Qwen3.5 VLM SFT: current three-camera views + goal -> current subtask.

Uses dense RoboDojo labels, not robot actions and not candidate IDs. Labels
are assistant targets only; user prompts never include the target subtask.
Split whole episodes per task (95/5) before sampling up to 3 frames/segment.
No API, annotation export, action statistics, or simulator is required.

Provide a native Qwen3.5-4B HF checkpoint (weights AND processor) at
./checkpoints/Qwen3.5-4B, or override FLUXVLA_VLM_PATH. No base weights are
bundled. Other Qwen3.5 sizes can use the same native model/processor path.

Defaults: 2 GPUs x 2 samples/GPU x accumulation 1 = global batch 4,
3 epochs, full fine-tuning, full-shard FSDP, bf16 compute / fp32 masters.
Validation is teacher-forced language loss/token accuracy, not task success.

Run with the fluxvla environment from the repository root:
    torchrun --standalone --nproc_per_node=2 scripts/train.py \\
        --config \\
        configs/vlm/qwen3_5_4b_robodojo_subtask_full_finetune.py \\
        --work-dir work_dirs/qwen3_5_4b_robodojo_subtask
"""

import os

model_path = os.environ.get('FLUXVLA_VLM_PATH', './checkpoints/Qwen3.5-4B')
data_root = os.environ.get('ROBODOJO_SUBTASK_DATA_ROOT',
                           '/mnt/data/oss/datasets/RoboDojo_lerobot_v3')

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
    dict(
        type='BuildSubtaskMessages',
        video_keys=[
            'observation.images.cam_high', 'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist'
        ],
        backend='torchcodec'),
    dict(
        type='ProcessVLMSFTInputs',
        processor_path=model_path,
        processor_type='qwen3_5',
        enable_thinking=False,
        image_max_pixels=262144,
        truncation_max_length=4096),
]

dataset = dict(
    type='LeRobotSubtaskDataset',
    data_root=data_root,
    subtask_prefix='dense',
    frames_per_subtask=3,
    validation_fraction=0.05,
    seed=42,
    transforms=transforms)
train_dataloader = dict(
    per_device_batch_size=2,
    per_device_num_workers=4,
    dataset=dict(dataset, split='train'))
val_dataloader = dict(dataset=dict(dataset, split='val'))

runner = dict(
    type='FSDPTrainRunner',
    train_task='vlm',
    max_epochs=3,
    max_steps=None,
    sharding_strategy='full-shard',
    fsdp_wrap_policy='model',
    max_grad_norm=1.0,
    grad_accumulation_steps=1,
    sampler='distributed',
    collator=dict(type='VLMSFTCollator', pad_token_id=0),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs/qwen3_5_4b_robodojo_subtask',
        window_size=1),
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.01),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.03),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    ema_decay=None,
    evaluator=dict(
        type='VLMEvaluator', eval_every=500, batch_size=1, max_batches=100),
    save_iter_interval=1000,
    save_epoch_interval=1,
    max_keep_ckpts=2,
    seed=42)

# Language-only training. This does not replace the deployed GPT planner.
eval = None
themis = None
