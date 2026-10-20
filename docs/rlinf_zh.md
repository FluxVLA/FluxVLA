# FluxVLA × RLinf

[English](rlinf.md)

FluxVLA 将 RLinf 集成为同步 PPO/GAE 后端。Policy 和仿真器 adapter 仍是
FluxVLA 原生的注册组件；RLinf 负责 Ray 资源放置、轨迹管理、PPO loss、
FSDP2、checkpoint 和权重同步。

## 环境安装

创建 Python 3.10 Conda 环境，将 `RLINF_ROOT` 指向 RLinf 源码目录，然后选择
一种安装模式：

```bash
export RLINF_ROOT=/path/to/RLinf
bash scripts/install_env.sh fluxvla-all
# 或者，仅开发后端和运行单元测试，不安装仿真器：
bash scripts/install_env.sh rlinf-only
```

只有 `fluxvla-all` 保证能够端到端运行 LIBERO 和 RoboTwin。RLinf 从
`RLINF_ROOT` 以 editable 模式安装，不使用 overlay venv，也不依赖机器特定的
`site-packages` 路径。

## 架构

```text
RLinf 环境 -> 已注册的 Flux adapter -> 已注册的 Flux RL policy
  -> 动作 + transition log-prob + value -> 轨迹/reward/GAE
  -> RLinf PPO/FSDP2 更新 -> 权重同步 -> rollout policy
```

`FluxPI05RLPolicy` 和 `FluxSmolVLARLPolicy` 都只训练各自的 action expert、
动作/时间投影层和 value head，同时冻结视觉/语言 prefix。训练时采样一次高斯
flow-SDE transition，并保存对应的 FP32 隐变量链；actor 会重算同一个
transition，而不会重新采样。评测时使用各模型原生的 ODE 路径。PI0.5 支持
LIBERO-10 和 RoboTwin `adjust_bottle`；SmolVLA 目前仅支持 LIBERO-10。

OpenPI 的精度行为直接实现在 PI0.5、SigLIP 和 Gemma 中，不使用仅供 RL 使用
的 monkey patch。Actor 和 rollout 保留 FP32 master weights，网络计算可以使用
BF16，隐变量和概率计算保持 FP32。

## 配置与命令

RL 配置由一个模型配置和一个仿真器配置组合而成：

```text
configs/rl/default.yaml
configs/rl/model/pi05.yaml
configs/rl/model/smolvla.yaml
configs/rl/simulator/libero10.yaml
configs/rl/simulator/robotwin_adjust_bottle.yaml
```

```bash
# 解析配置但不启动 Ray。
python scripts/rl_train.py model=pi05 simulator=libero10 \
  actor.model.model_path=/path/to/sft --cfg job --resolve

# SmolVLA 复用同一个 runner 和 LIBERO adapter。
python scripts/rl_train.py model=smolvla simulator=libero10 \
  actor.model.model_path=/path/to/smolvla_sft

# 集成检查：执行真实 rollout、PPO 更新、权重同步和 checkpoint 恢复。
python scripts/rl_train.py --preflight model=pi05 \
  simulator=robotwin_adjust_bottle

# 训练和独立的 held-out 评测。
python scripts/rl_train.py model=pi05 simulator=robotwin_adjust_bottle
python scripts/rl_eval.py model=pi05 simulator=robotwin_adjust_bottle \
  runner.ckpt_path=/path/to/global_step_N
```

GPU placement、batch size、checkpoint 路径和实验名称均使用普通 Hydra override
设置。增加新模型时，只需新增一个已注册且支持 RL 的 VLA builder 和一个模型
配置。增加新仿真器时，只需新增已注册的 observation/action processor 和一个
仿真器配置；不要为每种模型、仿真器及 train/eval 组合创建乘积配置。

配置校验会明确拒绝 SmolVLA + RoboTwin。在启用该组合前，需要单独实现并验证
它的 LeRobot 归一化、tokenizer、绝对动作以及图像预处理约定，不能直接复用
PI0.5 的 RoboTwin adapter。

## Checkpoint 与审计规则

- `actor.model.model_path` 始终表示初始 SFT checkpoint。
- `runner.resume_dir` 用于恢复完整的 RLinf 训练 checkpoint。
- `runner.ckpt_path` 用于选择独立评测所加载的权重。
- 训练 checkpoint 只能根据 validation seeds 选择；分数相同时选择更早的 step。
  test seeds 不能用于重新选择 checkpoint。
- RoboTwin 会为每个完成的 episode 记录 worker、slot、seed、checkpoint step、
  success 和 return。出现 seed 缺失、重复或计划外 seed 时，
  `scripts/rl_report.py` 会直接失败。
- 审计版 actor 检测到 rollout/actor 概率漂移或非有限梯度时会停止运行，而不是
  静默跳过更新。

## 已记录的 RoboTwin 结果

在 `adjust_bottle` 任务上，一次已完成的本地实验使用 150 个固定 test seeds
进行了评测：

| 模型                      |  成功数 | 成功率 | 相对 SFT 的差异 |
| ------------------------- | ------: | -----: | --------------: |
| SFT                       | 129/150 | 86.00% |               — |
| validation 选出的 step 30 | 135/150 | 90.00% |        +4.00 pp |
| 最终 step 200             | 136/150 | 90.67% |        +4.67 pp |

该结果只说明一个本地仿真任务和一个训练 seed 的表现，不能证明多 seed、跨任务
或外部 benchmark 上均有提升，也没有复现 RLinf 另行公布的 96.09% 结果。

## 测试

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest test/test_rl -q
```

测试覆盖 registry/config 组合、两个 adapter、固定 noise 的 ODE parity、FP32/
BF16 概率重算、PPO 梯度及冻结权重、FSDP wrapping、seed coverage、resume 语义
和仅使用 validation 选择 checkpoint 的约束。
