# feat/lzh/orin-triton-optimise rebase 记录

日期：2026-07-28

## 目标

将 `feat/lzh/orin-triton-optimise` 更新到最新 `main` 之后，并保留该分支的最终功能内容。

## 关键提交

- 原分支备份：`backup/feat-lzh-orin-triton-before-main-rebase-20260728`
- 原分支 tip：`66c3af3 feat : restore pi0.5 fm inference`
- 旧分叉点：`7f9f774 [Fix] Remove stale model and operator config arguments`
- 最新 `main`：`9b6b969 [Fix] Correct PI0.5 RoboCasa checkpoint path`

## 尝试过的直接 rebase

先尝试执行：

```bash
git rebase main
```

该操作在重放早期提交 `ed9fccc feat: add Fluxvla Orin support and inference benchmarks` 时停下。因为这个提交本身覆盖了大量配置、runner、operator 和文档，而 `main` 在同一区域已有更新，逐提交 rebase 会在很早阶段出现大量冲突。

当时的冲突文件包括：

```text
README.md
configs/gr00t/fluxbisim/gr00t_eagle_3b_close_box_full_finetune.py
configs/gr00t/gr00t_eagle_3b_oli_full_finetune.py
configs/gr00t/gr00t_eagle_3b_robocasa_30_eps_full_finetune.py
configs/gr00t/gr00t_qwen3vl_0.6b_libero_object_full_finetune.py
configs/gr00t/gr00t_qwen3vl_0.6b_libero_spatial_full_finetune.py
fluxvla/engines/operators/__init__.py
fluxvla/engines/operators/ur_operator.py
fluxvla/engines/runners/__init__.py
fluxvla/engines/runners/base_train_runner.py
fluxvla/engines/runners/ddp_train_runner.py
fluxvla/engines/runners/fsdp_train_runner.py
fluxvla/engines/runners/serving/zmq_server.py
fluxvla/engines/utils/builder.py
fluxvla/optimizers/__init__.py
fluxvla/optimizers/lr_scheduler_policies.py
scripts/eval.py
```

其中 `configs/gr00t/gr00t_eagle_3b_oli_full_finetune.py` 和 `fluxvla/optimizers/lr_scheduler_policies.py` 是 modify/delete 类型冲突。

## 最终处理方式

由于逐提交 rebase 会反复处理历史中间态冲突，本次改为 squash-rebase 方式：

1. `git rebase --abort` 回到原分支 tip。
2. 从最新 `main` 创建临时结果分支。
3. 取旧分叉点到原分支 tip 的最终差异：

   ```bash
   ORIG=backup/feat-lzh-orin-triton-before-main-rebase-20260728
   BASE=$(git merge-base "$ORIG" main)
   git diff --binary "$BASE" "$ORIG" > /tmp/feat-lzh-orin-triton-final.patch
   ```

4. 将最终差异一次性应用到最新 `main`：

   ```bash
   git switch -C tmp/rebase-feat-lzh-orin-triton-20260728 main
   git apply --3way --index /tmp/feat-lzh-orin-triton-final.patch
   ```

这等价于将原分支最终内容作为一个 squashed commit 放到最新 `main` 后面。

## 冲突解决说明

最终 `git apply --3way --index` 后没有留下 unmerged 文件：

```bash
git diff --name-only --diff-filter=U
```

输出为空。

处理原则：

- 对最新 `main` 已经包含且没有语义冲突的内容，保留 `main` 当前基础再应用分支最终差异。
- 对 Orin Docker、Orin runtime 文档、HUD04/OLI/RTC 配置、Orin 测试脚本等 `main` 不存在的内容，作为分支新增内容保留。
- 对逐提交 rebase 中出现的早期历史冲突，不逐个保留中间态；以原分支 tip 的最终文件内容作为准入标准。

## 结果形态

最终结果是 squash-rebase 形态：

```text
main(9b6b969) -- <one squashed Orin Triton optimization commit>
```

原始提交历史仍保存在备份分支：

```bash
backup/feat-lzh-orin-triton-before-main-rebase-20260728
```

如需回到 rebase 前状态，可从该备份分支恢复。