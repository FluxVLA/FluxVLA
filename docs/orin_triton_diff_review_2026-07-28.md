# Orin Triton diff 清理 Review

日期：2026-07-28

基准：`main` -> `feat/lzh/orin-triton-optimise`

## 本次处理结论

### HUD04 configs

保留 OliOperator 路线：

- `configs/gr00t/gr00t_hud04_rtc_no_done_oli_full_finetune.py`
- `configs/gr00t/gr00t_hud04_rtc_no_done_oli_kernel_inference.py`
- `configs/gr00t/gr00t_hud04_rtc_no_done_oli_rtc_kernel_inference.py`

处理内容：

- 将原 `gr00t_hud04_rtc_no_done_full_finetune.py` 重命名为 `gr00t_hud04_rtc_no_done_oli_full_finetune.py`。
- 将该基础配置里的默认 inference 从 `Teleop02WbtRTCInferenceRunner` / `Teleop02WbtOperator` 改为 `OliInferenceRunner` / `OliOperator`。
- `gr00t_hud04_rtc_no_done_oli_kernel_inference.py` 改为依赖 Oli base，而不是依赖 Teleop02 RTC config。
- 删除 Teleop-only HUD04 inference config：
  - `gr00t_hud04_rtc_no_done_rtc_kernel_inference.py`
  - `gr00t_hud04_rtc_no_done_rtc_kernel_inference_residual.py`

说明：`gr00t_hud04_rtc_no_done_oli_rtc_kernel_inference.py` 仍基于 `oli_kernel`，用于 Oli RTC prefix 推理。

### Teleop02 WBT operator/runner

已删除 Teleop02 专用 operator/runner，并从注册入口移除：

- `fluxvla/engines/operators/teleop02_wbt_operator.py`
- `fluxvla/engines/runners/teleop02_wbt_inference_runner.py`
- `fluxvla/engines/runners/teleop02_wbt_rtc_inference_runner.py`
- `fluxvla/engines/operators/__init__.py` 中的 Teleop02 导入
- `fluxvla/engines/runners/__init__.py` 中的 Teleop02 导入

原因：当前希望只保留 OliOperator 路线，避免 HUD04/Teleop02 专线扩大 diff 面。

### `fluxvla/engines/utils/builder.py` `_print_build_log`

已恢复到 `main`。

这个 helper 原本用于规避 `mmengine.logging.print_log` 在某些缺失 `torch.distributed.ReduceOp` 的环境中打印 debug log 失败的问题。当前不是核心推理路径必需逻辑，且会改变通用 builder 的日志错误处理方式，因此按“非必要则 restore”处理。

### `fluxvla/engines/utils/registry.py` `_safe_print_log`

已恢复到 `main`。

这个 helper 会吞掉 registry 自动导入和查找过程里的日志异常，虽然能提高瘦身环境容错，但过于宽泛，可能隐藏真实日志/导入问题。当前没有明确运行时证据要求它存在，因此恢复。

### `fluxvla/engines/utils/name_map.py`

保留。

改动作用：把 `torch.distributed.fsdp.StateDictType` 改成可选导入。这样普通推理导入 `fluxvla.engines.utils` 时，不会因为 Orin/瘦身 torch 缺 FSDP 而在 import 阶段失败。只有真正调用 `state_dict_type_map()` 时才会报清晰错误。

### `fluxvla/models/backbones/vlms/configs.py`

保留。

改动作用：`Qwen3VLConfig` / `Qwen3VLForConditionalGeneration` 变成可选依赖。部分 Orin 镜像或旧 transformers 版本不包含 Qwen3VL 类；如果顶层强导入，会导致非 Qwen3 模型也无法 import。保留后，Qwen3 配置只在依赖存在时注册。

### `radio_model.py`

已恢复到 `main`。

原因：RADIO 当前不是本次 Oli/GR00T Orin 测试链路必需；之前改动主要是 FlashAttention 可选 fallback 和实例级 monkey patch，属于额外兼容/加速改动。为减少 diff 面，恢复。

### `fluxvla/ops/cuda/*` 三个算子 import

已恢复到 `main`。

涉及文件：

- `gemma_rotary_embedding.py`
- `matmul_bias.py`
- `rotary_pos_embedding.py`

之前改成 `import_module()` 不是解决 CUDA extension 缺失的根因。实际问题是 `.so` 没有在当前 Orin torch ABI 下编译。直接相对导入在扩展存在时可工作，因此恢复。

### `scripts/inference.py`

已恢复 `configure_inference_attention_defaults()`。

原因：这是推理入口的默认 attention backend 配置，删除会让不同环境走到不同 attention 默认行为，不利于稳定复现。

### `scripts/inference_real_robot.py`

已去掉 startup time profile 输出。

保留：

- `--cfg-options`
- 显式 import 相关 registry 模块
- `configure_inference_attention_defaults()`

原因：时间 profile 是临时调试输出；`--cfg-options` 和 registry 显式导入对真实推理配置覆盖与注册稳定性有用。

### PI0.5 / GR00T 测试脚本

保留两个离线 100 次 benchmark 入口：

- `test/test_models/test_gr00t_orin.py`
- `test/test_models/test_pi05_orin.py`

删除额外 profile/实验脚本：

- `scripts/profile_pi05_phases.py`
- `scripts/test_gr00t_with_embodiment.py`
- `scripts/test_gr00t_with_embodiment_fixed.py`
- `scripts/test_pi05_dummy_forward.py`
- `test/test_models/pi05_triton_bench_real_100.py`
- `test/test_models/test-gr00t-100times.py`
- `test/test_ops/test_triton_fused_kernels_orin.py`

`test_pi05_orin.py` 已改成自包含 CLI benchmark，支持：

```bash
--variant baseline
--variant accelerated
--predict-runs 100
```

### Orin Docker helper 文件位置

保留在 `docker/orin/`：

- `docker/orin/constraints_orin.txt`
- `docker/orin/patch_rosconsole.py`
- `docker/orin/ros_entrypoint.sh`
- `docker/orin/requirements_orin_notorch.txt`

删除根目录重复文件：

- `constraints_orin.txt`
- `patch_rosconsole.py`
- `ros_entrypoint.sh`

保留根目录 `.dockerignore`。

原因：Docker build context 是仓库根目录，`.dockerignore` 必须位于 context 根目录才会生效，不能移动到 `docker/orin/` 后继续起作用。

### 未主动处理项

- `fluxvla/engines/runners/base_train_runner.py` 当前保留删除重复局部 `build_evaluator_from_cfg` import 的改动。
- `fluxvla/models/third_party_models/dreamzero/modules/*.py` 当前仍保持工作区已有改动，本次未评估。

## 验证

已通过语法检查：

```bash
python -m py_compile \
  configs/gr00t/gr00t_hud04_rtc_no_done_oli_full_finetune.py \
  configs/gr00t/gr00t_hud04_rtc_no_done_oli_kernel_inference.py \
  configs/gr00t/gr00t_hud04_rtc_no_done_oli_rtc_kernel_inference.py \
  scripts/inference.py \
  scripts/inference_real_robot.py \
  test/test_models/test_gr00t_orin.py \
  test/test_models/test_pi05_orin.py
```
