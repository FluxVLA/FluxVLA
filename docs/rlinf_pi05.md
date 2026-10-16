# PI0.5 接入 RLinf

依赖、公共接口及恢复语义见[接入总览](rlinf.md)。注册名为 `fluxvla_pi05`，
native 类型为 `PI05FlowMatching`；RL policy 是
`fluxvla/rl/models/pi05/policy.py` 中的 `FluxPI05RLPolicy`。

## 模型适配

Policy 继承 `FlowPPOPolicyMixin`、原生 `PI05FlowMatching` 和 RLinf `BasePolicy`，
保留全部 SFT 参数名，只新增 `value_head.*`。

- `_prefix` 调用原生图像/语言 embedding 与 Gemma 前向，返回 hidden、有效 token
  mask 和 KV cache；prefix 冻结，actor 每次复算重新构建 cache。
- `_velocity` 调用原生 `denoise_step`；评测沿用原生 `predict_action`。
- `model_action_horizon` 映射原生 `n_action_steps`，
  `model_action_dim` 映射 `max_action_dim`。
- 冻结视觉/语言主干、多模态 projector 及 expert 未使用的词嵌入；
  训练 action expert、动作/时间投影及 value head。
- Value head 对有效 prefix token 做 masked mean。
- FSDP wrapping 为 `GemmaDecoderLayer`、`SiglipEncoderLayer`、
  `LinearProjector`、`ValueHead`。

OpenPI 共享 embedding/LM-head 别名及未使用 expert embedding 替代规则隔离在
`models/pi05/checkpoint.py`，由模型声明选择，不影响其他 VLA。
重复别名 tensor 必须一致。OpenPI 数值对齐由模型专属 `_configure_model_rl()`
和 `precision.py` 实现，不修改原生模型源码。

## 观测与动作适配

| 配方                      | LIBERO-10                                               | RoboTwin adjust_bottle                                       |
| ------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| Adapter                   | `LiberoObservationAdapter`                              | `RoboTwinObservationAdapter`                                 |
| 相机 / 原始 proprio       | 双相机 / 8 维                                           | 三相机 / 14 维双臂                                           |
| 预处理                    | Flux 图像、prompt/tokenizer、mean/std；state 补齐 32 维 | Aloha 坐标/夹爪转换、quantile、离散 state prompt、resize/pad |
| 模型 horizon / 执行 chunk | 10 / 10                                                 | 50 / 50                                                      |
| 模型 / 环境动作维度       | 32 / 7                                                  | 32 / 14                                                      |
| 去噪步数 / noise level    | 10 / 0.5                                                | 5 / 0.3                                                      |

LIBERO 图像由 RLinf 旋转，adapter 不重复旋转；拼接好的 proprio 直接归一化，
不重复做四元数转换。RoboTwin 动作反变换接收当前 `env_obs`，完成 delta-action
还原及夹爪转换。两者 chain 均留在模型归一化坐标。

RoboTwin SFT 元数据 horizon 为 10，而本配方按 RLinf 公开评测代码使用 50；
这是显式差异，不能把两种 horizon 的结果混为同一协议。

## LIBERO 启动

MMEngine 配方为 `configs/pi05/pi05_paligemma_libero_10_full_finetune.py`。
准备匹配的 SFT、tokenizer 和 Flux dataset statistics。统计顶层须包含
`statistic_name`（默认 `libero_10_no_noops`），不能替换为任意 OpenPI 统计。
在 tmux 内激活 RL 环境，从仓库根目录执行：

```bash
python -m fluxvla.rl.train --config-name=benchmarks/libero/pi05/ppo \
  actor.model.model_path=/absolute/path/sft.safetensors \
  actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer.model \
  actor.model.fluxvla.norm_stats_path=/absolute/path/dataset_statistics.json \
  runner.logger.experiment_name=pi05_libero_run \
  'runner.logger.logger_backends=[wandb]'
```

追加 `--cfg job --resolve` 只解析配置，不训练。结构不同时同时覆盖
`actor.model.fluxvla.config_path`，不得只替换权重。
LIBERO 原生推理对齐/PPO 接口已经测试，但 PI0.5 LIBERO 大模型训练不是本次发布
验收内容；不要用 RoboTwin 实测替代这一验证。

## RoboTwin 启动与环境准备

需要 RoboTwin `RLinf_support` 源码、资产、任务 seeds，以及
`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` 的完整权重、统计和 tokenizer。
资源准备工具为 `scripts/rl/prepare_robotwin.py`。先在独立环境中按 RLinf/RoboTwin
官方安装说明安装模拟器依赖（包括 SAPIEN、mplib、渲染和规划库）；不要安装
RLinf 的 `embodied` extra，以免覆盖本链路要求的 Transformers 版本。
仓库不自动修改系统库或其他虚拟环境。Blackwell 的 OIDN 兼容库可选用
`scripts/rl/setup_robotwin_oidn.sh` 下载；该工具需要 `aria2c`、`tar` 和 `sha256sum`。

```bash
export FLUX_ROBOTWIN_ROOT=/absolute/path/robotwin_experiment
export RLINF_ROOT=/absolute/path/RLinf
python scripts/rl/prepare_robotwin.py checkout --root "$FLUX_ROBOTWIN_ROOT"
python scripts/rl/prepare_robotwin.py assets --root "$FLUX_ROBOTWIN_ROOT"
python scripts/rl/prepare_robotwin.py weights --root "$FLUX_ROBOTWIN_ROOT"
python scripts/rl/prepare_robotwin.py protocol --root "$FLUX_ROBOTWIN_ROOT"
source scripts/rl/robotwin_env.sh

# 环境/模型/GPU 门禁通过后，在 tmux 内启动。
python -m fluxvla.rl.train \
  --config-name=benchmarks/robotwin/pi05/ppo_8gpu \
  runner.logger.experiment_name=pi05_robotwin_run

python -m fluxvla.rl.eval \
  --config-name=benchmarks/robotwin/pi05/eval_8gpu \
  runner.ckpt_path=/absolute/path/checkpoints/global_step_30 \
  runner.logger.experiment_name=pi05_robotwin_eval
```

配置从 `$FLUX_ROBOTWIN_ROOT` 的 `weights/`、`src/RoboTwin`、`protocol/` 读取资源，
以 YAML 为准。省略 `runner.ckpt_path` 可评测初始 SFT。
八卡配方 actor 0–3、rollout 4–5、环境 6–7；不是四卡配方的同批量复制，
轮数、global batch、学习率和采样量应查看解析后的配置。

通用入口也支持 RoboTwin，并自动加载环境辅助脚本：

```bash
bash scripts/rl/run.sh train benchmarks/robotwin/pi05/ppo_8gpu
bash scripts/rl/run.sh eval benchmarks/robotwin/pi05/eval_8gpu \
  runner.ckpt_path=/absolute/path/checkpoints/global_step_30
```

训练 rollout 与 actor 的计算 microbatch 必须一致，以控制 BF16 batch-shape
概率漂移。八卡配方关闭 FSDP 延迟梯度同步，但保留优化器 global batch 累积；
这是当前 Torch 2.8/FSDP2 栈的实测规避配置，不应随意删掉。
本次整理未在新机器重新验证模拟器、分布式训练或成功率；上线前应先用少量环境
和较短轮数验证，不能把接口测试结果等同于 GPU/模拟器验收。
