# GR00T N1.7 LIBERO

本文档说明如何在 FluxVLA 中训练和评测 native GR00T N1.7 LIBERO。
当前实现目标是：训练和评测都不依赖 NVIDIA Isaac-GR00T Python 包，
LIBERO Gymnasium env 注册由 FluxVLA 本地 wrapper 提供。

## 已包含内容

- native GR00T N1.7 model assembly；
- native Qwen3-VL / Cosmos backbone 加载；
- native GR00T N1.7 processor 和 collator；
- FluxVLA 本地 LIBERO Gymnasium env 注册：
  `fluxvla/envs/libero_gymnasium_env.py`；
- 四个 LIBERO full finetune config，每个 config 同时包含 train 和 eval：
  - `configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py`
  - `configs/gr00t/gr00t_n17_native_libero_goal_full_finetune.py`
  - `configs/gr00t/gr00t_n17_native_libero_object_full_finetune.py`
  - `configs/gr00t/gr00t_n17_native_libero_spatial_full_finetune.py`

## 环境安装

建议在干净 conda 环境中安装：

```bash
conda create -n fluxvla-n17 python=3.10 -y
conda activate fluxvla-n17
bash scripts/install_env.sh sim-only --skip-robocasa
```

说明：

- `sim-only` 会安装训练、LIBERO、MuJoCo、robosuite 等仿真依赖；
- `--skip-robocasa` 会跳过 Isaac-GR00T / RoboCasa source checkout；
- native N1.7 LIBERO 不需要安装 Isaac-GR00T Python 包；
- `requirements-base.txt` 固定 `transformers==5.3.0`；
- `requirements-sim.txt` 固定 LIBERO 可用的 robosuite 版本。

Headless LIBERO/MuJoCo 评测建议使用：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
export MPLCONFIGDIR=/tmp/matplotlib
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
```

## 资产准备

准备基础权重、Cosmos backbone、LIBERO processor/statistics metadata 和
LeRobot v2.1 parquet 数据：

```bash
python tools/prepare_groot_n17_libero.py --all-suites --download
python tools/prepare_groot_n17_libero.py --all-suites
```

默认目录：

```text
checkpoints/GR00T-N1.7-3B
checkpoints/GR00T-N1.7-LIBERO/libero_10
checkpoints/GR00T-N1.7-LIBERO/libero_goal
checkpoints/GR00T-N1.7-LIBERO/libero_object
checkpoints/GR00T-N1.7-LIBERO/libero_spatial
checkpoints/nvidia/Cosmos-Reason2-2B
datasets/libero_10_no_noops_lerobotv2.1
datasets/libero_goal_no_noops_lerobotv2.1
datasets/libero_object_no_noops_lerobotv2.1
datasets/libero_spatial_no_noops_lerobotv2.1
```

`N17_PROCESSOR_META` 应指向对应 suite 的
`GR00T-N1.7-LIBERO/<suite>` metadata/statistics 目录；它不是基础权重目录，
也不是训练数据目录。

### GR00T-N1.7-LIBERO 目录的用途

当前 native N1.7 LIBERO 训练和评测不需要加载官方 LIBERO 微调后的大权重作为
模型结果来源。我们自己的训练 checkpoint 由 `--ckpt-path` 或训练保存逻辑加载；
`GR00T-N1.7-LIBERO/<suite>` 在当前配置里主要作为 processor/statistics metadata
来源。

最小需要的文件是：

```text
processor_config.json
statistics.json
embodiment_id.json
```

其中 `statistics.json` 必须包含 `libero_sim`。base `GR00T-N1.7-3B`
checkpoint 自带的 statistics 不包含 `libero_sim`，所以不能只用 base
metadata 替代 suite metadata。`statistics.json` 会同时影响训练和评测：

- 训练时，native collator / processor 会用它归一化 state 和 action label；
- 评测时，processor 会用它把模型输出的归一化 action 反归一化回环境 action；
- 已训练 checkpoint 应使用训练时相同的 suite statistics，否则 action 尺度会变。

之前的低点评测排查里，我们验证的是 base config 与官方 LIBERO config 作为
`N17_INIT_CKPT` 时不会导致首动作差异；这不能说明 processor metadata/statistics
不重要。metadata/statistics 没有被判定为低点 bug 的根因，但它仍是训练和评测
必须固定的兼容条件。

后续可以把这三类轻量 metadata 固定到 FluxVLA 自带资源，或让准备脚本只下载
metadata 文件而不是完整官方微调权重目录。进入上游 PR 前如果要内置这些文件，
需要先确认来源、版本和许可证；在未内置前，`N17_PROCESSOR_META` 仍需要显式指向
对应 suite 的 metadata 目录。

## LIBERO10 训练

单机 8 卡、global batch size 640 示例：

```bash
export N17_INIT_CKPT=./checkpoints/GR00T-N1.7-3B
export N17_PROCESSOR_META=./checkpoints/GR00T-N1.7-LIBERO/libero_10
export N17_BACKBONE_MODEL_PATH=./checkpoints/nvidia/Cosmos-Reason2-2B
export LIBERO_DATA_ROOT=./datasets/libero_10_no_noops_lerobotv2.1

export N17_PER_DEVICE_BATCH_SIZE=10
export N17_GRAD_ACCUM_STEPS=8
export N17_MAX_STEPS=20000
export N17_LR=1e-4
export N17_WARMUP_RATIO=0.05
export N17_WEIGHT_DECAY=1e-5
export N17_SHARDING_STRATEGY=full-shard
export N17_ENABLE_GRAD_CKPT=1
export N17_ACTIVE_TRACKERS="('jsonl',)"

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --work-dir work_dirs/groot_n17_native_libero_10_full
```

训练参数默认对齐 GR00T N1.7 LIBERO recipe：

```text
max_steps=20000
lr=1e-4
weight_decay=1e-5
warmup_ratio=0.05
state_dropout=0.2
color jitter:
  brightness=0.3
  contrast=0.4
  saturation=0.5
  hue=0.08
```

global batch size 计算：

```text
global_batch = world_size * per_device_batch_size * grad_accum_steps
```

例如：

```text
1 机 8 卡:  8 * 10 * 8 = 640
2 机 16 卡: 16 * 10 * 4 = 640
```

## 自动评测

四个 N1.7 LIBERO config 都包含 `eval` 配置。训练完成后可以使用同一 config
对 checkpoint 评测：

```bash
export N17_INIT_CKPT=./checkpoints/GR00T-N1.7-3B
export N17_PROCESSOR_META=./checkpoints/GR00T-N1.7-LIBERO/libero_10
export N17_BACKBONE_MODEL_PATH=./checkpoints/nvidia/Cosmos-Reason2-2B
export LIBERO_DATA_ROOT=./datasets/libero_10_no_noops_lerobotv2.1
export N17_AUTO_EVAL_SEED=7

python scripts/eval.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --ckpt-path work_dirs/groot_n17_native_libero_10_full/checkpoints/<checkpoint>.safetensors
```

说明：

- 评测默认 `50 trials/task`；
- LIBERO10 完整评测为 `10 tasks * 50 trials = 500 attempts`；
- 评测我们自己训练出的完整 `.safetensors` checkpoint 时，`N17_INIT_CKPT`
  建议与训练一致，指向 base `GR00T-N1.7-3B`；
- `N17_PROCESSOR_META` 必须指向对应 suite 的 metadata/statistics 目录；
- 实际微调权重由 `--ckpt-path` 指定；
- native eval 默认使用 FluxVLA 本地 LIBERO env wrapper，不导入 Isaac-GR00T
  Python 包。
- 如果使用 `scripts/train.py --eval-after-train`，评测会复用训练时已经解析的
  config；这种情况下保持训练所需的 `N17_INIT_CKPT=GR00T-N1.7-3B` 即可。

快速 smoke：

```bash
python scripts/eval.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --ckpt-path work_dirs/groot_n17_native_libero_10_full/checkpoints/<checkpoint>.safetensors \
  --cfg-options \
    'eval.task_ids=[0]' \
    eval.num_trials_per_task=1 \
    eval.save_rollout_videos=False \
    eval.save_failed_rollout_videos=False \
    eval.result_output_dir=work_dirs/n17_native_libero_10_eval_smoke
```

## DLC 从零复线模板

在基础镜像中拉取仓库后，可以用以下流程复线环境、训练和自动评测。
路径根据实际挂载替换，不应写入代码或 config。

```bash
conda create -n fluxvla-n17 python=3.10 -y
conda activate fluxvla-n17

bash scripts/install_env.sh sim-only --skip-robocasa

python tools/prepare_groot_n17_libero.py --suite libero_10 --download
python tools/prepare_groot_n17_libero.py --suite libero_10

export N17_INIT_CKPT=./checkpoints/GR00T-N1.7-3B
export N17_PROCESSOR_META=./checkpoints/GR00T-N1.7-LIBERO/libero_10
export N17_BACKBONE_MODEL_PATH=./checkpoints/nvidia/Cosmos-Reason2-2B
export LIBERO_DATA_ROOT=./datasets/libero_10_no_noops_lerobotv2.1

export N17_PER_DEVICE_BATCH_SIZE=10
export N17_GRAD_ACCUM_STEPS=8
export N17_MAX_STEPS=20000
export N17_LR=1e-4
export N17_WARMUP_RATIO=0.05
export N17_WEIGHT_DECAY=1e-5
export N17_SHARDING_STRATEGY=full-shard
export N17_ENABLE_GRAD_CKPT=1
export N17_ACTIVE_TRACKERS="('jsonl',)"

export N17_AUTO_EVAL_TRIALS=50
export N17_AUTO_EVAL_SEED=7

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --work-dir work_dirs/groot_n17_native_libero_10_full \
  --eval-after-train
```

如果使用 2 机 16 卡并希望保持 gbs640，把
`N17_GRAD_ACCUM_STEPS` 改为 `4`。

## DSW 本地复线脚本

当前 DSW 上如果已经有 checkpoint 和 dataset，可以先跳过下载，用软链接复用已有
资产，验证“干净 conda 环境安装、资产路径检查、短步数训练、自动评测”这条流程。

脚本位置：

```bash
scripts/run_n17_libero_dsw_repro.sh
```

默认行为：

- 使用 conda 环境 `fluxvla-n17`，不存在则创建；
- 执行 `bash scripts/install_env.sh sim-only --skip-robocasa`；
- 跳过下载资产，通过软链接复用当前 DSW 上已有路径；
- 运行 LIBERO10 preflight；
- 训练 `1000` steps；
- 训练结束后自动评测 `task0 * 3 trials`，用于验证流程闭环，不作为最终成功率。

如果只是验证当前未提交 worktree：

```bash
FLUXVLA_REPRO_MODE=current \
MAX_STEPS=1000 \
AUTO_EVAL_TRIALS=3 \
AUTO_EVAL_TASK_IDS='[0]' \
bash scripts/run_n17_libero_dsw_repro.sh
```

如果要按 DLC 方式替换旧的 `/root/projects/fluxvla`，从远端重新拉取要提交 PR 的
分支并在该路径下安装和运行，使用：

```bash
cd /mnt/workspace/mnt/data/yiming/fluxvla-n17-libero-pr

FLUXVLA_REPRO_MODE=clone \
FLUXVLA_REPO_DIR=/root/projects/fluxvla \
FLUXVLA_REPLACE_EXISTING=1 \
FLUXVLA_REPO_URL=git@github.com:jzzzzzzzzzzzzzzzz/FluxVLA.git \
FLUXVLA_REPO_REF=gr00t-n17-native-libero-pr \
MAX_STEPS=1000 \
AUTO_EVAL_TRIALS=3 \
AUTO_EVAL_TASK_IDS='[0]' \
bash scripts/run_n17_libero_dsw_repro.sh
```

`FLUXVLA_REPLACE_EXISTING=1` 会先把旧目录整体改名为
`/root/projects/fluxvla.backup_<timestamp>`，再重新 clone 到
`/root/projects/fluxvla`。如果不设置该变量，脚本检测到旧仓库有本地修改时会直接
退出，避免覆盖旧实验内容。

如需改成 2000 steps：

```bash
MAX_STEPS=2000 bash scripts/run_n17_libero_dsw_repro.sh
```

当前 DSW 默认软链接来源：

```text
CKPT_SRC_ROOT=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints
COSMOS_SRC=/mnt/data/cpfs/mnt/data/yiming/fluxvla-n17-native-dev/checkpoints/nvidia/Cosmos-Reason2-2B
DATA_SRC_ROOT=/mnt/workspace/mnt/data/liyinhao/datasets
```

如果新机器路径不同，启动脚本前覆盖这些变量即可。脚本会在训练前检查：

```text
checkpoints/GR00T-N1.7-3B/config.json
checkpoints/GR00T-N1.7-LIBERO/libero_10/statistics.json
checkpoints/nvidia/Cosmos-Reason2-2B/config.json
datasets/libero_10_no_noops_lerobotv2.1/meta/info.json
```

注意：当前 PR worktree 有未提交或未推送修改时，远端 clone 模式不会包含这些本地
修复。这种情况下应先提交并推送到 `FLUXVLA_REPO_REF`，再用 clone 模式验证
`/root/projects/fluxvla` 复线。

## 本地 env wrapper 行为

FluxVLA 本地 LIBERO Gymnasium wrapper 注册 `libero_sim/<task.name>`，并输出
native N1.7 评测需要的 key：

```text
video.image
video.wrist_image
state.x
state.y
state.z
state.roll
state.pitch
state.yaw
state.gripper
annotation.human.action.task_description
```

wrapper 行为：

- 使用 `OffScreenRenderEnv` 创建 LIBERO env；
- 对 agentview 和 wrist image 做与 N1.7 processor 匹配的方向处理；
- 将 quaternion 转成 axis-angle；
- 将 `action.x/y/z/roll/pitch/yaw/gripper` 拼成 LIBERO 7D action；
- gripper action 按 LIBERO convention 做 invert；
- native N1.7 eval 不再从 Isaac-GR00T Python 包注册 env。

## 已验证结果

使用 gbs320 LIBERO10 checkpoint、本地 FluxVLA env wrapper、`seed=7`、
`50 trials/task`，完整 LIBERO10 结果恢复到旧高分范围：

```text
451 / 500 = 90.2%
```

逐 task 结果：

```text
task0 46/50 92.0%
task1 48/50 96.0%
task2 48/50 96.0%
task3 49/50 98.0%
task4 44/50 88.0%
task5 49/50 98.0%
task6 41/50 82.0%
task7 45/50 90.0%
task8 34/50 68.0%
task9 47/50 94.0%
```

本次回归曾从约 `90%` 降到 `82.2%`。最终根因不是 env wrapper 迁移，而是共享
action head DiT block 中 cross-attention mask 路径被改坏：
`BasicTransformerBlock` 在 cross-attention 场景下错误地使用了 query side
`attention_mask`，丢掉了 `encoder_attention_mask`。修复后恢复旧行为：

```python
attention_mask=(
    encoder_attention_mask
    if encoder_hidden_states is not None else attention_mask
)
```

同时恢复 `_sdpa_context()`，保持旧高分 worktree 的 SDPA 行为。修复后
first-action 对齐和完整 50t 都已验证通过。

## 提交前检查

建议提交前运行：

```bash
python -m py_compile \
  fluxvla/envs/libero_gymnasium_env.py \
  fluxvla/engines/runners/libero_eval_runner.py \
  fluxvla/models/blocks/cross_attention_dit.py \
  tools/prepare_groot_n17_libero.py

python tools/prepare_groot_n17_libero.py --suite libero_10
git diff --check
```
