# 用训练后的本地 VLM 替代 GPT Planner

## 运行配置

使用 `configs/hierarchical/robodojo_pi05_qwen35.py`。
它继承现有 RoboDojo Planner+PI05 配置，只替换 Planner 后端，并将候选上限
统一为 4096（当前元数据最多 2601 条/任务，不截断候选）。
原有 GPT 配置保留；两种 Planner 共用更新后的 PI05 权重和配套统计/tokenizer，
动作预处理、反归一化、Themis 协议均不变。

默认加载：

```text
/mnt/data/oss/users/liyinhao/projects/FluxVLA/work_dirs/
qwen35_4b_robodojo_subtask_d78c5a778_4node_bs64_old_20261116_dlcf6osrr2ymq9yd/
  checkpoints/step-007311-epoch-03-loss=0.0678.safetensors
  vlm_assets/  # 训练时保存的架构、tokenizer、processor 和 chat template

/mnt/data/oss/users/liyinhao/projects/FluxVLA/work_dirs/
pi05_robodojo_subtask_d78c5a778_8node_bs128_old_20261116_dlc1wn1n2jhosfyh/
  checkpoints/step-060000-epoch-04-loss=0.0011.safetensors
  tokenizer/
  dataset_statistics.json
```

以上是 2026-11-17 核验到的各目录最新 checkpoint，默认固定到这些文件。
模型架构从 `vlm_assets/config.json` 创建，然后严格加载 FluxVLA 完整权重。
**不加载原始 Qwen 基座权重，也不调用 GPT/API**。默认以 BF16 推理，权重只加载一次。
如果替换 checkpoint，必须同时使用该训练的 processor/tokenizer，不能随意改用其他版本。

## 调试入口

`.vscode/launch.json` 新增两个入口：

1. **RoboDojo Qwen3.5 + PI05: Debug VLA Server**
   - GPU 0，同一个模型 worker 内运行本地 Qwen3.5 和 PI05。
   - 不使用 `.env.planner`，无需 API key。
   - 可在 `fluxvla/planning/local_vlm.py` 的 `choose()` 中断点：
     `messages` 是实际 VLM 输入；`text` 是预测 subtask；`choice` 是候选 ID。
     返回前的 `self.last_prediction` 包含文本、ID、token 数及推理耗时。
     请求结束后 planner 会重置，预测原文仍在 serving runner 的
     `last_prediction_metadata['subtask']['instruction']` 中。
2. **RoboDojo Qwen3.5 + PI05: Themis Eval + Video**
   - GPU 1，使用原版 Themis 的 `scripts/distributed_eval.py`。
   - 仍是 `stack_blocks,stack_bowls,push_T,insert_tubes`，各 5 回合，550 步上限。

建议先等模型服务端打印 ready，再启动 Themis；原来先启动 Themis 等待服务端的方式
也不变。若 Themis 在另一个终端运行，只需将原命令的 `--config` 改为新配置，
`--video-dir` 改为需要的目录，其他任务/回合覆盖参数保持不变。
两种 Planner 使用相同端口，不要同时启动 GPT 和本地 VLM 两个服务端。

完整配置仍是全量评测，调试规模只在启动参数中覆盖；输出目录也只由启动参数指定。
新增调试项的报告根目录为 `work_dirs/robodojo_pi05_qwen35`，
视频目录为该目录下的 `videos`，其后仍按原服务端/Themis 规则创建运行或会话子目录。

## 输入、输出和候选约束

```text
当前三路 RGB + 总任务
  → 与训练完全相同的 BuildSubtaskMessages 提示词（不含答案/候选列表）
  → 保存的 Qwen3.5 processor，关闭 thinking，最大图像像素 262144
  → 本地 VLM 自回归预测 subtask 原文
  → 精确映射到当前任务的训练候选 ID
  → 现有 HierarchicalPolicy → PI05 tokenize → 机器人动作
```

默认 `constrained_decoding=True`：候选文本构成 token 前缀树，限制每一步允许生成的
下一 token，直到完整候选加 EOS。不是模糊相似度匹配，也不会在生成失败时随便选第一条。
这保证输出属于候选词表，**不保证它适合当前场景或任务一定成功**。
不把候选 ID 或答案放入 VLM 提示词，不读取当前样本的训练帧号、未来标注或环境成功信号。

该 checkpoint 只训练了 subtask 原文，没有训练成功/blocked/停止标签。
因此此后端只支持 `completion_mode='disabled'`；Themis 仍负责按原规则终止和评分。
每个请求独立观察/预测，不保留跨请求对话、图像或 KV cache；只复用模型和词表缓存。
现有 GPT 的 Agent 停止能力不受影响。

可用环境变量覆盖输入路径：

- `PI05_ROBODOJO_CHECKPOINT_DIR`：PI05 训练输出根目录，配套统计和 tokenizer 一同切换。
- `PI05_ROBODOJO_CHECKPOINT`：需要推理的 PI05 完整 checkpoint 文件。
- `ROBODOJO_VLM_PLANNER_DIR`：训练输出根目录。
- `ROBODOJO_VLM_PLANNER_CHECKPOINT`：完整 `.safetensors` 文件，也支持 checkpoint 目录。
  目录优先解析有效的 `latest-checkpoint.safetensors` 软链接，否则取最大 step。
- `ROBODOJO_VLM_PLANNER_ASSETS`：配套 `vlm_assets`。
- `ROBODOJO_VLM_PLANNER_DEVICE`：可选设备，如 `cuda:0`（逻辑编号）；默认当前模型 worker 的设备。

其他设置仍可用顶层 `--cfg-options policy.planner.*` 覆盖。
`constrained_decoding=False` 可观察自由生成，但结果仍须与当前任务某条候选原文精确一致，
否则明确报错，不把词表外指令交给 VLA。候选超过 token/数量上限也报错，不静默截断。
训练元数据中不存在的任务继续明确报错，不借用其他任务的候选。

两次新训练的配置都记录了 `/mnt/data/oss-wlcb/datasets/RoboDojo_lerobot_v3`，
本机该数据可读路径为 `/mnt/data/oss/datasets/RoboDojo_lerobot_v3`。
候选仍来自其 `meta/episodes` 中的 `dense_subtask_names`，每次启动重新读取，
不是手工维护的静态列表。当前可读元数据共 3500 个 episode、55003 段标注，
与新 PI05 的 `dataset_statistics_metadata.json` 记录一致。
35 个任务分别去重后合计 35600 条任务内候选；四个调试任务的候选数为：

| 任务           | 候选数 |
| -------------- | -----: |
| `stack_blocks` |    638 |
| `stack_bowls`  |   1041 |
| `push_T`       |    299 |
| `insert_tubes` |    810 |

若数据使用另一挂载路径，用 `ROBODOJO_SUBTASK_DATA_ROOT` 显式覆盖。

## 本次权重更新验证（2026-11-17）

- GPT/本地 VLM 两个配置均可由 FluxVLA 和 Themis 各自 Python 环境解析，
  仍为全量配置、stateless/per-request 模式；未修改 launch 中的调试任务和输出目录。
- 相关配置、候选、SFT 预处理和 subtask 数据回归：143 项通过，1 项旧 worker 测试排除。
  旧 worker 测试仍引用 `cfg.eval.policy`；更广范围试跑中的三个旧 serving 测试也
  仍依赖旧的 policy 位置/模式。本次没有修改这些测试或 serving 行为。
- 新 tokenizer 对全部 35493 条跨任务去重原文均可逐字往返编码；
  最长答案连同 EOS 为 52 token，不超过 `max_new_tokens=256`。
- 已实际严格加载本次指定的新 Qwen 和 PI05 权重，用真实三路图像、任务和状态
  跑通 subtask → PI05：输出有限值 `50×14` 动作，训练/生成提示词 token 前缀一致。
  本次样本调用约 3.83 秒，PyTorch 峰值 GPU 分配约 27.34 GiB。
- 该快照输出属于 638 条 `stack_blocks` 候选，但与快照标注不逐字相同。
  上述为接口运行检查，不代表预测准确率或任务成功率；未运行完整仿真 rollout。

## 先前后端接入验证（旧 checkpoint）

- 新后端、SFT 数据/预处理、原 serving runner 的相关测试：51 项通过。
- 扩展候选/控制器回归：合计 123 项通过，1 项已有测试失败。
  `test_worker_loads_source_once_and_filters_on_episode_reset` 仍读取旧位置
  `cfg.eval.policy`，而当前分支原配置已将它移到顶层 `cfg.policy`；本次未改该测试或原配置。
- 实际加载指定 Qwen checkpoint 与现有 PI05 checkpoint，使用真实验证集三路图像和状态：
  训练/生成 token 前缀一致，完成 subtask → PI05 推理，输出有限值 `50×14` 动作。
  单次调用约 3.4 秒，两模型进程的 PyTorch 峰值分配约 27.3 GiB；仅为这次样本/机器的测量值。
- 另外三个真实快照均生成了合法训练候选，但没有与对应标注逐字相同。
  这些是接口运行检查，**不是预测准确率或仿真成功率评测**；尚未跑完整 simulator rollout。
