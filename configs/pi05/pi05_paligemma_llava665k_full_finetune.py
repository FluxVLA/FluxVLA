# Copyright 2026 Limx Dynamics
"""Pi05 VLM SFT on raw LLaVA-665K using the pretrained PaliGemma vocabulary.

Reuses the Pi05 base checkpoint and keeps action parameters for later VLA use.
Reads the original annotations and extracted images without preconversion.
"""

llava_root = './datasets/llava665k'

model = dict(
    action_in_proj=dict(in_dim=32, out_dim=1024, type='LinearProjector'),
    action_out_proj=dict(in_dim=1024, out_dim=32, type='LinearProjector'),
    freeze_llm_backbone=False,
    freeze_llm_expert=True,
    freeze_vision_backbone=False,
    # The action expert never embeds text; OpenPI omits this unused tensor.
    pretrained_skip_prefixes=['llm_expert.embed_tokens.'],
    llm_backbone=dict(
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        type='ConditionGemmaModel',
        use_cache=True,
        vocab_size=257152),
    llm_expert=dict(
        adarms_cond_dim=1024,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        type='ConditionGemmaModel',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    loss_action_dim=32,
    max_action_dim=32,
    n_action_steps=50,
    name_mapping=dict({
        'action_in_proj.projector':
        'action_in_proj',
        'action_out_proj.projector':
        'action_out_proj',
        'llm_backbone':
        'paligemma_with_expert.paligemma.model.language_model',
        'llm_backbone.embed_tokens':
        'paligemma_with_expert.paligemma.lm_head',
        'llm_expert':
        'paligemma_with_expert.gemma_expert.model',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'time_mlp_in.projector':
        'time_mlp_in',
        'time_mlp_out.projector':
        'time_mlp_out',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower'
    }),
    openpi_fp32_flow=True,
    ori_action_dim=14,
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
    ],
    preserve_fp32_residuals=True,
    pretrained_name_or_path='./checkpoints/pi05_base/model.safetensors',
    proj_width=1024,
    projector=dict(in_dim=1152, out_dim=2048, type='LinearProjector'),
    strict_mapping=True,
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    time_mlp_in=dict(in_dim=1024, out_dim=1024, type='LinearProjector'),
    time_mlp_out=dict(in_dim=1024, out_dim=1024, type='LinearProjector'),
    time_sampler='beta',
    type='PI05FlowMatching',
    vision_backbone=dict(
        openpi_stem_fp32=True,
        preserve_fp32_residuals=True,
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False)))
runner = dict(
    collator=dict(pad_token_id=0, type='Pi05VLMSFTCollator'),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    fsdp_wrap_policy='model',
    grad_accumulation_steps=8,
    keep_params_fp32=True,
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.03),
    max_epochs=None,
    max_grad_norm=1.0,
    max_keep_ckpts=2,
    max_steps=1000,
    metric=dict(
        active_trackers=('jsonl', ),
        run_dir='work_dirs/pi05_vlm_sft',
        type='VLAMetric',
        window_size=1),
    mixed_precision_dtype='bf16',
    optimizer=dict(lr=2e-05, type='AdamW', weight_decay=0.01),
    sampler='distributed',
    save_iter_interval=250,
    seed=42,
    sharding_strategy='full-shard',
    train_task='vlm',
    type='FSDPTrainRunner')
train_dataloader = dict(
    dataset=dict(
        anno_json_path=llava_root + '/llava_v1_5_mix665k.json',
        data_root_path=llava_root,
        index_cache_dir='./datasets/.cache/vlm',
        seed=42,
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                image_size=224,
                max_length=1024,
                tokenizer_path='./checkpoints/pi05_base',
                type='ProcessPi05VLMSFTInputs'),
        ],
        type='LLaVAFormatDataset'),
    per_device_batch_size=1,
    per_device_num_workers=0)
