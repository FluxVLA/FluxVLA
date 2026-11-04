# Copyright 2026 Limx Dynamics
"""Qwen3-VL-2B full SFT on the original LLaVA-665K dataset.

The dataset indexes the original JSON once; no JSONL preparation is needed.
Run from the repository root after following the README data preparation.
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
    per_device_batch_size=1,
    # Keep transforms in the training process so dataset breakpoints work.
    per_device_num_workers=0,
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
    max_epochs=None,
    max_steps=1000,
    sharding_strategy='full-shard',
    fsdp_wrap_policy='model',
    max_grad_norm=1.0,
    grad_accumulation_steps=8,  # Two GPUs give an effective batch size of 16.
    sampler='distributed',
    # Padding is masked by lengths; filler token 0 is never supervised.
    collator=dict(type='VLMSFTCollator', pad_token_id=0),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs/qwen3_vl_2b_sft',
        window_size=1),
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.01),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.03),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    save_iter_interval=250,
    max_keep_ckpts=2,
    seed=42)
