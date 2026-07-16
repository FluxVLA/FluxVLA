# ============================================================
# FluxVLA PI0.5 Robocasa 微调配置
# ============================================================
#
# 用途: 在 Robocasa GR1 Tabletop 数据上微调 PI0.5 模型
# 基于: pi05_paligemma_aloha_full_finetune.py（同为非 LIBERO 数据集）
#
# 与 LIBERO 配置的差异:
#   - 数据集: Robocasa GR1 29 维关节 (vs LIBERO 7 维末端执行器)
#   - 归一化: PI0.5 quantile q01/q99
#   - 视频: 单相机 ego_view (vs LIBERO 双相机 agentview + wrist)
#   - 动作: 绝对关节位置 (vs LIBERO delta 末端位姿)
#   - state: 29D joint angles, quantile-normalized and discretized in prompt
#   - action_window_size: 16 (每次预测 16 步未来动作)
#
# 数据准备:
#   运行 scripts/convert_robocasa_for_fluxvla.py 将原始 44 维数据
#   裁剪为 29 维并生成 episodes_stats.jsonl
#
# 训练命令（2 台机器，每台 8 卡；两台使用相同 MASTER_ADDR/PORT）:
#   torchrun --nnodes=2 --nproc_per_node=8 \
#       --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
#       --master_port=${MASTER_PORT} scripts/train.py \
#       --config \
#       configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py \
#       --work-dir work_dirs/pi05_paligemma_robocasa_full_data_full_finetune
#
# 作者: yiming | 创建: 2026-04-13
# ============================================================

# ============================================================
# 模型配置 — PI0.5 原版架构，与 LIBERO/ALOHA 完全相同
# ============================================================
# PI0.5 架构:
#   - Vision: SigLIP ViT (224x224, patch=14, hidden=1152)
#   - LLM Backbone: PaliGemma Gemma (hidden=2048, 18层)
#   - LLM Expert: Gemma Expert (hidden=1024, 18层, ADaRMS)
#   - Action Head: Flow Matching (proj_width=1024)
# 模型内部维度固定为 max_action_dim=32，实际 29 维零填充到 32
model = dict(
    type='PI05FlowMatching',
    # --- PaliGemma 主干 LLM (处理图像+语言 token) ---
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,  # 主干 LLM 不使用 ADaRMS
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,  # PaliGemma Gemma 隐藏维度
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,  # 18 层 Transformer
        num_key_value_heads=1,  # GQA (Grouped Query Attention)
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    # --- SigLIP 视觉编码器 ---
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,  # SigLIP 输出维度
            image_size=224,  # 输入分辨率
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,  # 224/14 = 16x16 = 256 个 patch
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    # --- Vision → LLM 投影层 (1152 → 2048) ---
    projector=dict(
        type='LinearProjector',
        in_dim=1152,  # SigLIP hidden_size
        out_dim=2048,  # Gemma hidden_size
    ),
    # --- Flow Matching 动作生成相关 ---
    proj_width=1024,  # 内部投影维度
    n_action_steps=16,  # RoboCasa GR1 action horizon (对齐 StarVLA / GR00T)
    # Robocasa 单 episode ~300 步 @20fps
    # chunk=16 即覆盖约 0.8 秒的未来动作
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,  # 模型固定维度，实际 29 维 + 3 维零填充
    # --- Gemma Expert (处理 action/state 条件) ---
    # 通过 ADaRMS (Adaptive RMSNorm) 将时间步信息注入每层
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,  # ADaRMS 条件维度 = proj_width
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,  # Expert 隐藏维度 (比主干小)
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,  # 与主干层数一致 (layer-wise 对齐)
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,  # 启用 Adaptive RMSNorm
        use_cache=True,
        vocab_size=257152),
    # --- 训练设置 ---
    # PI0.5 injects the normalized 29D proprio state through discrete prompt
    # tokens, so the language backbone remains trainable during adaptation.
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    # --- 预训练权重 ---
    # 使用 PI0.5 base 预训练权重 (非 LIBERO 微调权重)
    # 这样模型从通用 base 开始，在 Robocasa 数据上从头微调
    pretrained_name_or_path='./checkpoints/pi05_base/model.safetensors',
    # --- 权重 key 映射 (预训练权重 → FluxVLA 模型) ---
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
        'llm_expert.embed_tokens':
        'paligemma_with_expert.gemma_expert.lm_head',
    },
    strict_mapping=True,
    # --- 需要转 bf16 的模块 (节省显存) ---
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=29,  # Robocasa 实际动作维度 (29 个活跃关节)
)

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_OFFICIAL_GR1_STATS_PATH = ('./datasets/robocasa_gr1_24tasks_first30ep/'
                            'official_groot_gr1_dataset_statistics.json')
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'


def _robocasa_data_path(task_name):
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name):
    return f'{_ROBOCASA_TASK_PREFIX}/{task_name}{_ROBOCASA_ENV_SUFFIX}'


# ============================================================
# 训练数据配置
# ============================================================
# Robocasa GR1 数据特点:
#   - 格式: LeRobot v2.0 Parquet (已通过 convert 脚本转为 v2.1 兼容)
#   - 观测: 单相机 ego_view (256x256 → resize 224x224)
#   - 状态: 29 维关节角度 (绝对值, 非 delta)
#   - 动作: 29 维目标关节位置 (绝对值)
#   - 归一化: quantile q01/q99 (映射到 [-1, 1])
#   - 24 个任务（6 Seen + 18 Novel）, 每任务约 1000 episodes
train_dataloader = dict(
    # 16×A800, global batch = 16 GPUs × 16/GPU = 256.
    per_device_batch_size=16,
    # 8 GPUs/node × 8 workers/GPU = 64 data workers per node.
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        # --- 列名映射: parquet 列名 → 统计量 key ---
        # state/action 分开统计；动作统计必须来自 action，而不是 observation.state。
        # 这是 GR00T 排查时修过的关键点，也避免后续改变 statistic_keys 顺序时误覆盖。
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        # --- 需要计算统计量的 key ---
        statistic_keys=['observation.state', 'timestamp', 'action'],
        # --- statistic_name: 用于 dataset_statistics.json 的命名空间 ---
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        # PI0.5 upstream uses q01/q99 quantile statistics. Reuse one fixed
        # robot/task statistics asset for both full-data training and eval.
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        # --- 数据集列表 (可指定多个任务目录) ---
        datasets=dict(
            type='ParquetDataset',
            # 转换后的数据路径 (由 convert_robocasa_for_fluxvla.py 生成)
            # 多个任务目录以列表形式传入
            data_root_path=[
                _robocasa_data_path('PnPBottleToCabinetClose'),
                _robocasa_data_path('PnPCanToDrawerClose'),
                _robocasa_data_path('PnPCupToDrawerClose'),
                _robocasa_data_path('PnPMilkToMicrowaveClose'),
                _robocasa_data_path('PnPPotatoToMicrowaveClose'),
                _robocasa_data_path('PnPWineToCabinetClose'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToBasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToPanSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToPotSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToTieredbasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToBasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToBowlSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToTieredshelfSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToBowlSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlate'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToPanSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTrayToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTrayToPotSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToTieredbasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToTieredshelfSplitA'),
            ],
            transforms=[
                # --- Step 1: 从 Parquet 提取原始数据 ---
                # 读取指定列，解码视频帧，应用列名映射
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',  # 29 维关节角度
                        'timestamp',  # 秒
                        'actions',  # 29 维目标关节位置
                        'info',  # 数据集元信息
                        'stats',  # 归一化统计量
                        'action_masks',  # 动作有效性掩码
                    ],
                    # 视频 key: Robocasa 只有 ego_view 单相机
                    # (LIBERO 有 agentview + wrist 双相机)
                    video_keys=[
                        'observation.images.ego_view',
                    ],
                    # Parquet 列名 → 模型内部 key 的映射
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    }),
                # --- Step 2: PI0.5 原生 quantile 归一化 ---
                # Keep the converted dataset's native ordering and tokenize
                # the actual 29D normalized state, as OpenPI does.
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,  # 零填充到模型维度
                    state_dim=29,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile'),
                # --- Step 3: 构造官方 PI0.5 state prompt ---
                dict(
                    type='PreparePromptWithState',
                    max_state_dim=29,
                    lowercase_task_description=False,
                    add_action_prefix=True),
                # --- Step 4: Tokenize prompt ---
                dict(
                    type='ProcessPrompts',
                    max_len=200,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path='checkpoints/pi05_base',
                    )),
                # --- Step 6: 图像增强与 resize ---
                # StarVLA RoboCasa 主要使用 224 resize；这里加入 GR00T 中有效的
                # crop + color jitter 作为 PI0.5 训练消融。
                dict(type='RandomCropImages', scale=0.95),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='ColorJitterImages',
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08),
                # --- Step 7: 图像归一化 ---
                # SimpleNormalizeImages: pixel/255.0*2-1 (与 OpenPI PI0.5 一致)
                dict(type='SimpleNormalizeImages'),
            ],
            # --- 动作窗口配置 ---
            action_window_size=16,  # 与 n_action_steps 一致
            action_key='action',
            use_delta=False,  # Robocasa 使用绝对关节位置，非增量
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
        )))

# ============================================================
# 训练 Runner 配置
# ============================================================
runner = dict(
    type='FSDPTrainRunner',  # Fully Sharded Data Parallel
    max_epochs=None,
    # Match OpenPI's full-data PI0.5 recipe: global batch 256 and 100k
    # optimizer updates. Do not shorten the LR horizon when changing the
    # sample/epoch budget; the previous 50k cosine run reached near-zero LR
    # at 40k and under-optimized the full-finetuned backbone.
    max_steps=100000,
    grad_accumulation_steps=1,
    # PI0.5 RoboCasa 需要 full-finetune 语言主干来学习离散 state prompt。
    # OpenPI full-data PI0.5 uses a 1k-step warmup followed by a constant
    # 5e-5 LR; decaying to zero over 50k steps materially reduced updates.
    optimizer=dict(lr=5e-5, type='AdamW', weight_decay=0.0),
    max_grad_norm=1.0,  # 梯度裁剪
    # --- checkpoint 保存策略 ---
    # 之前未设置, 走默认 max_keep_ckpts=2, 导致早期表现更好的 ckpt 被删除,
    # 无法回收。这里按 epoch + step 双触发保存, 并多保留几个以便按 eval 选最优。
    save_epoch_interval=1,
    save_iter_interval=5000,
    max_keep_ckpts=8,
    # --- 加速路径（对齐 mentor 149ad50 commit） ---
    # no-shard: DDP-风格的参数放置，每卡持有整份权重，关闭 500+ 子模块 FSDP wrap，
    #           配合 bf16 master weights, 达到 LeRobot-level 吞吐。
    sharding_strategy='no-shard',
    # --- Collator: 定义 batch 中的 key ---
    collator=dict(
        type='DictCollator',
        keys=[
            'states',  # (B, 29) quantile-normalized joint state
            'observation.eepose',  # 可能不存在，DictCollator 会跳过
            'timestamp',  # (B,) 时间戳
            'images',  # (B, N_views, C, H, W) 图像
            'img_masks',  # (B, N_views) 图像有效性掩码
            'lang_tokens',  # (B, max_len) 分词后的 prompt
            'lang_masks',  # (B, max_len) 注意力掩码
            'actions',  # (B, chunk_size, 32) 归一化+零填充后的动作
            'action_masks',  # (B, chunk_size) 动作有效性掩码
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path='checkpoints/pi05_base',
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+constant',
        warmup_steps=1000,
    ),
    enable_gradient_checkpointing=True,  # 省显存 (2×A800 开)
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

# ============================================================
# 评测配置 — Robocasa 24 任务完整仿真评测
# ============================================================
# 用法:
#   conda activate fluxvla && cd /root/projects/fluxvla
#   bash scripts/eval_robocasa.sh \
#       --config configs/pi05/pi05_paligemma_robocasa_finetune.py \
#       --ckpt-path \
#       work_dirs/pi05_robocasa_full_2n8g_20260421/checkpoints/\
#       latest-checkpoint.safetensors
#
# 可选覆盖:
#   --cfg-options eval.num_trials_per_task=20 eval.seed=7
#
# 注意: unnorm_key 必须和训练时的 statistic_name 保持一致
#      (训练 statistic_name: robocasa_gr1_24tasks_30ep)
eval = dict(
    type='RobocasaEvalRunner',
    model_family='pi0',
    task_list=[
        _robocasa_task_env('PnPBottleToCabinetClose'),
        _robocasa_task_env('PnPCanToDrawerClose'),
        _robocasa_task_env('PnPCupToDrawerClose'),
        _robocasa_task_env('PnPMilkToMicrowaveClose'),
        _robocasa_task_env('PnPPotatoToMicrowaveClose'),
        _robocasa_task_env('PnPWineToCabinetClose'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToBasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToPanSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToPotSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToTieredbasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToBasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToBowlSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToTieredshelfSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToBowlSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlate'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToPanSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTrayToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTrayToPotSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToTieredbasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToTieredshelfSplitA'),
    ],
    eval_chunk_size=16,  # 对齐训练 horizon / GR00T RoboCasa
    max_episode_steps=720,  # 单 episode 最大步数 (与 starVLA 默认一致)
    num_trials_per_task=20,  # 每任务 20 次，总 24×20=480 episode
    seed=7,  # 与 GR00T RoboCasa eval 保持同一批初始状态
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    action_order='fluxvla',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            # 关键: eval 图像预处理必须与训练完全一致, 否则 sim 帧分布与训练
            # 分布不匹配, 训练 loss 正常但闭环成功率塌到 0。
            #   - center_crop_scale=0.95 对齐训练 RandomCropImages(scale=0.95)
            #   - value_range='tanh' 对齐训练 SimpleNormalizeImages -> [-1, 1]
            #     (默认 normalize 只到 [0, 1], 与训练差一个 ×2-1 的缩放)
            #   - img_key 需与训练数据相机一致 (与 GR00T RoboCasa 配置同源,
            #     使用 bg_crop 变体)
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop_scale=0.95,
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
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='quantile',
        action_dim=29,  # Robocasa 29 维活跃关节
        clip_actions=False,
        stats_order='native',
    ),
)
