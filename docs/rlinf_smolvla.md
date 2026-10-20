# SmolVLA 接入 RLinf

依赖、公共接口及恢复语义见[接入总览](rlinf.md)，扩展规则见[接口约定](rlinf_extension.md)。
注册名为 `fluxvla_smolvla`，native 类型为 `SmolVLAFlowMatching`；RL policy 位于
`fluxvla/rl/models/smolvla/policy.py`，不修改原生 SmolVLA 或 RLinf 源码。

## 模型适配与 PI0.5 的区别

`FluxSmolVLARLPolicy` 继承共享 `FlowPPOPolicyMixin`、原生 SmolVLA 和 RLinf
`BasePolicy`。采样、Gaussian 复算、critic、bootstrap 和加载复用公共实现，
只保留本模型的 prefix/cache/velocity 逻辑。

- `_prefix` 按原生 SmolVLA 顺序执行交错 VLM/expert 层，构建相同 KV cache。
- `_velocity` 调用原生 `_denoise_step`；Eval 调用原生 `predict_action`。
- `model_action_horizon` 映射 `chunk_size`，`model_action_dim` 映射
  `max_action_dim`，不借用 PI0.5 的内部变量名。
- 冻结整个 VLM（含视觉 connector）与 `state_proj`，训练 action expert、
  动作投影、action-time MLP 和 value head。
- `state_proj` 属于缓存 prefix；当前契约冻结 prefix，不能只解除冻结就宣称
  支持训练 state projector，需要修改可微 prefix 复算契约。
- Value head 对最终有效 prefix token（图像、语言、state）做 masked mean。
  SFT 参数名不变，只新增 `value_head.*`。

FSDP wrapping 使用 `SmolVLMEncoderLayer`、`LinearProjector`、`ValueHead`。
SmolVLA 直接调用 decoder 子操作，绕过 `LlamaDecoderLayer.forward`；
若按 decoder 包装，FSDP unshard hook 不会执行。因此文本/expert 参数归 root
FSDP2 管理，不可复制 PI0.5 的 Gemma wrapping。该方式同时 gather 较多 root
参数，扩容前应实测显存。

actor/rollout 均使用 FP32 主参数与局部 BF16 autocast。训练 rollout microbatch
必须匹配 actor microbatch；ODE 评测保留环境 worker 的 batch 布局。

## 观测与初始权重

当前仅支持 LIBERO-10：两路已旋转 RGB、8 维原始 proprio；模型动作 32 维、
环境动作 7 维，horizon 50、执行 chunk 10、去噪 10 步、noise level 0.5。
没有验证 SmolVLA RoboTwin 双臂权重/预处理，配置会拒绝该组合。

复用 `LiberoObservationAdapter`，但图像 letterbox、prompt/tokenizer、mean/std
state 归一化及动作反归一化读取 SmolVLA 自己的配方和统计，不能混用 PI0.5 统计。
MMEngine 配方为 `configs/smolvla/smolvla_libero_10_finetune.py`。
须配套提供任务 SFT、tokenizer 目录、Flux dataset statistics JSON
（包含 `libero_10_no_noops` 下的 `proprio`/`action`）。支持原生及已有 LeRobot
`name_mapping` 权重，缺失/不匹配直接失败；`smolvla_base` 不能冒充 LIBERO SFT。

## 启动

先按总览安装环境，在 tmux 内设置资源根目录并准备 SFT：

```bash
tmux new -s smolvla-rl
# 在新 shell 中激活 RL 环境，进入仓库并设置总览中的环境变量。
export SMOLVLA_ARTIFACT_ROOT=/absolute/path/rlinf-smolvla
python scripts/rl/prepare_smolvla.py
```

下载工具固定 SFT revision 和校验值；`--endpoint` 可显式选择镜像。未指定资源
根目录时默认写入仓库 `work_dirs/rlinf-smolvla`。准备完成后设置通用启动器资源：

```bash
smolvla_bundle="$SMOLVLA_ARTIFACT_ROOT/weights/FluxVLA-SmolVLA-LIBERO10"
export FLUX_RL_MODEL_PATH="$smolvla_bundle/checkpoints/step-057096-epoch-36-loss=0.2340.safetensors"
export FLUX_RL_TOKENIZER_PATH="$smolvla_bundle/tokenizer"
export FLUX_RL_STATS_PATH="$smolvla_bundle/dataset_statistics.json"
bash scripts/rl/run.sh train benchmarks/libero/smolvla/ppo_8gpu
```

已有文件时可直接使用 module 入口：

```bash
python -m fluxvla.rl.train --config-name=benchmarks/libero/smolvla/ppo \
  actor.model.model_path=/absolute/path/sft.safetensors \
  actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer \
  actor.model.fluxvla.norm_stats_path=/absolute/path/dataset_statistics.json \
  runner.logger.experiment_name=smolvla_run \
  'runner.logger.logger_backends=[wandb]'
```

追加 `--cfg job --resolve` 仅解析配置。通用启动器会正确引用资源环境变量中
包含空格或 `=` 的路径。无需为特定训练轮数或单任务另建配置：

```bash
bash scripts/rl/run.sh train benchmarks/libero/smolvla/ppo_8gpu \
  runner.max_epochs=1000 runner.max_steps=1000 \
  runner.save_interval=25 runner.val_check_interval=25
```

八卡配方 actor 0–3、rollout 4–5、环境 6–7，32 个训练环境、global batch 128、
microbatch 2、update epoch 2、actor LR `1e-6`、value LR `1e-4`；不是最优超参。
轮数指 runner 外层轮次，不是 optimizer step。正式训练需已有 `WANDB_API_KEY`。

## 独立评测与验收

```bash
python -m fluxvla.rl.eval --config-name=benchmarks/libero/smolvla/eval \
  actor.model.model_path=/absolute/path/sft.safetensors \
  actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer \
  actor.model.fluxvla.norm_stats_path=/absolute/path/dataset_statistics.json \
  runner.ckpt_path=/absolute/path/checkpoints/global_step_200 \
  runner.logger.experiment_name=smolvla_eval
```

省略 `runner.ckpt_path` 则评测初始 SFT；续训设置 `runner.resume_dir`。
`rollout.model` 完整引用 `actor.model`，保证 eval-only workers 使用相同配置。
评测成功结束时输出 `FLUXVLA_EVALUATION_COMPLETED`；自动化串联任务时应检查
此标志，避免 RLinf/Ray 清理掩盖失败退出码后仍启动后续阶段。

原生预处理/cache/velocity/ODE 一致性、PPO 复算、冻结、trajectory、bootstrap、
严格加载和恢复均在 `test/test_rl/` 验证；命令与最新结果见总览。
八卡真实采样/训练/同步曾运行，双卡 FSDP2 更新与恢复也已测试。

默认评测不自动构成严格的 held-out 配对协议。正式对照需固定任务、episode/noise
seeds、布局及 checkpoint 选择规则，并检查恢复评测与训练内评测是否一致。
