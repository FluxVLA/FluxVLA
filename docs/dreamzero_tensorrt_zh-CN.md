# DreamZero TensorRT 构建与推理指南

本文说明如何从 FluxVLA DreamZero checkpoint 导出 ONNX、构建 TensorRT
engine，并在 LIBERO 评测中加载该 engine。所有路径均为占位符，不依赖开发者的
本地目录。

## 加速范围

当前 TensorRT engine 只替换 DreamZero DiT 的 cached denoise forward：

```text
update_cache=False -> TensorRT engine
update_cache=True  -> PyTorch model
```

cache fill 和新观测的 cache update 会修改 KV cache，因此仍由 PyTorch 执行；
TensorRT engine 读取已经建立的 KV cache，只输出 video/action velocity prediction。
它不包含文本、CLIP、VAE encoder，也不包含 LIBERO 数据预处理和环境执行。

## 前置条件

### 软件和硬件

构建和运行至少需要：

- NVIDIA GPU 和与其兼容的驱动。
- 可运行当前 FluxVLA 环境的 PyTorch/CUDA。
- TensorRT 完整发行包，包含 Python wheel、共享库和 `bin/trtexec`。
- 与当前 Python ABI 一致的 TensorRT wheel，例如 Python 3.10 使用 `cp310`。
- Python `onnx` 包，用于 `torch.onnx.export`。
- 足够的 GPU 显存、主机内存和磁盘空间。DreamZero 14B 的 ONNX、engine 和构建
  临时数据都很大。

以下组件作用不同：

| 组件                                  | 用途                                  | 是否必需                    |
| ------------------------------------- | ------------------------------------- | --------------------------- |
| `onnx` Python 包                      | PyTorch 导出 ONNX                     | 必需                        |
| TensorRT full Python wheel            | FluxVLA 反序列化并执行 engine         | 必需                        |
| `trtexec`                             | 将 ONNX 构建成 TensorRT engine        | 构建 engine 时必需          |
| ONNX-TensorRT 独立仓库                | TensorRT 的另一种 ONNX parser/backend | 本流程不需要                |
| `tensorrt_lean` / `tensorrt_dispatch` | TensorRT 的精简/分发 runtime          | 当前 wrapper 不要求单独安装 |

### 版本原则

建议使用同一份 TensorRT 发行包中的 Python wheel、共享库和 `trtexec`。构建 engine
和运行 engine 时应保持 TensorRT 版本一致。TensorRT engine 不保证跨以下条件可移植：

- TensorRT 版本；
- GPU 架构；
- CUDA/驱动能力；
- engine shape profile；
- DreamZero 模型结构和 checkpoint。

升级 TensorRT、换 GPU 型号或修改模型 shape 后，应重新导出并构建 engine。

### 使用 TAR 包隔离安装

推荐把 TensorRT TAR 包解压到独立目录，不修改系统级 CUDA。以下命令中的路径和
wheel 文件名需要替换为实际版本：

```bash
export TRT_ROOT=/absolute/path/to/TensorRT-<version>
export PATH="$TRT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}"

/path/to/fluxvla/python -m pip install \
  "$TRT_ROOT/python/tensorrt-<version>-cp310-none-linux_x86_64.whl"
```

验证：

```bash
which trtexec
trtexec --version

/path/to/fluxvla/python -c \
  'import tensorrt as trt; print("TensorRT", trt.__version__)'

/path/to/fluxvla/python -c \
  'import torch; print("PyTorch CUDA", torch.version.cuda); print(torch.cuda.get_device_name(0))'
```

Python wheel 只提供 Python API，并不保证 `trtexec` 已进入 `PATH`。如果 `which trtexec` 没有输出，应设置 `PATH`，或者在 builder 命令中显式传入
`--trtexec "$TRT_ROOT/bin/trtexec"`。

安装 ONNX：

```bash
/path/to/fluxvla/python -m pip install onnx
```

## Shape profile

`tools/build_dreamzero_trt_engine.py` 从 `model.vla_head` 配置读取 DiT 层数、head
维度、action/state 维度和 cache 上限。LIBERO 默认输入 shape 固定，只有
`kv_cache_packed` 的 cache length 动态：

```text
min: kv_cache_packed:40x2x1x128x40x128
opt: kv_cache_packed:40x2x1x384x40x128
max: kv_cache_packed:40x2x1x640x40x128
```

各维含义：

```text
[DiT layers, K/V slots, batch, cached tokens, attention heads, head dim]
```

先只检查 shape，不加载模型、不导出 ONNX，也不需要 `trtexec`：

```bash
python tools/build_dreamzero_trt_engine.py \
  --config configs/dreamzero/dreamzero_libero_10_full_finetune_w_cache.py \
  --print-shapes-only
```

以下变化会使旧 engine 不再适用，需要修改参数并重新构建：

- 图像分辨率、视角拼接方式或 VAE latent 高宽；
- `frame_seqlen`、`num_frame_per_block` 或 cache window；
- batch size；
- action horizon、最大 action/state 维度；
- DiT 层数、head 数或 hidden dimension；
- 模型类型或导出 forward 的输入输出契约。

当前 AR-14B 导出 forward 沿用模型实现中的 embodiment 处理。接入新的机器人构型时，
必须先确认 action/state padding、embodiment id 和模型配置与导出图一致，不能直接把
LIBERO engine 当作跨 embodiment 通用 engine。

## 导出 ONNX

在 FluxVLA 仓库根目录执行，并显式选择本仓库代码：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
```

只导出 ONNX：

```bash
python tools/build_dreamzero_trt_engine.py \
  --config configs/dreamzero/dreamzero_libero_10_full_finetune_w_cache.py \
  --ckpt-path /absolute/path/to/checkpoint.safetensors \
  --onnx-path work_dirs/dreamzero_trt/CausalWanModel.onnx \
  --strict-load \
  --export-only
```

builder 会在导出模型前启用 ONNX/TensorRT 兼容实现。日志中应确认：

```text
matched tensors: <non-zero>
missing tensors: 0
unexpected tensors: 0
ONNX exported: .../CausalWanModel.onnx
```

`TracerWarning` 表示 tracing 将部分 Python shape/control flow 固化。当前导出设计本来
就固定 video/action 等 shape，只允许 KV cache length 动态；但如果出现 tensor shape
不匹配或 unsupported operator，不能把它当作普通 warning 忽略。

## 构建 TensorRT engine

导出和构建可用同一个命令完成：

```bash
python tools/build_dreamzero_trt_engine.py \
  --config configs/dreamzero/dreamzero_libero_10_full_finetune_w_cache.py \
  --ckpt-path /absolute/path/to/checkpoint.safetensors \
  --onnx-path work_dirs/dreamzero_trt/CausalWanModel.onnx \
  --engine-path work_dirs/dreamzero_trt/CausalWanModel.trt \
  --trtexec "$TRT_ROOT/bin/trtexec" \
  --strict-load
```

默认命令会使用：

```text
--useCudaGraph
--memPoolSize=workspace:65536
--minShapes=<cache min>
--optShapes=<cache opt>
--maxShapes=<cache max>
```

`workspace:65536` 是 builder 可用的上限，不代表一定分配 64 GiB。如果机器资源不足，
可通过 `--workspace-mib` 调低；这可能改变 TensorRT tactic 选择和最终性能。

TensorRT 11 默认使用 strongly typed ONNX 网络。当前 ONNX 权重和输入按 FP16 导出，
通常不要额外传 precision flag。只有旧 TensorRT 明确需要时才使用
`--legacy-precision fp16` 或 `--legacy-precision bf16`，二者不能同时启用。需要完整
`trtexec` 诊断输出时使用 `--verbose`；默认不输出
逐 layer profile，避免把研究阶段的大量日志带入普通构建流程。

构建产物：

```text
work_dirs/dreamzero_trt/CausalWanModel.onnx
work_dirs/dreamzero_trt/CausalWanModel.trt
work_dirs/dreamzero_trt/CausalWanModel_build.log
```

## 在评测中加载 engine

运行环境需要能够 import 与构建版本一致的 TensorRT Python wheel，并能从
`LD_LIBRARY_PATH` 找到相同版本的共享库。两卡 CFG 模式下两个 rank 都必须能读取
同一 engine 文件。

运行时通过配置传递 engine：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$TRT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  scripts/eval.py \
  --config configs/dreamzero/dreamzero_libero_10_full_finetune_w_cache.py \
  --ckpt-path /absolute/path/to/checkpoint.safetensors \
  --cfg-options \
  model.pretrained_name_or_path=/absolute/path/to/DreamZero-AgiBot \
  model.vla_head.trt_engine_path=/absolute/path/to/CausalWanModel.trt \
  model.vla_head.dynamic_cache_schedule=True \
  model.vlm_backbone.use_torch_compile_encoders=True \
  eval.cfg_parallel=True \
  eval.result_output_dir=/absolute/path/to/eval_output
```

构建期和运行期的环境变量不要混淆：

```text
构建期：builder 内部设置 ENABLE_TENSORRT=true，选择可导出的兼容算子。
运行期：通过 model.vla_head.trt_engine_path 或 LOAD_TRT_ENGINE 加载 engine。
```

运行评测时不要额外设置 `ENABLE_TENSORRT=true`。该变量会把仍在 PyTorch 中执行的
cache update attention 切到导出兼容实现，并跳过 encoder `torch.compile`；它不是
“启用 TensorRT runtime”的开关。

## 验证

不需要 GPU 或 TensorRT 安装的 shape/command 单测：

```bash
python test/test_models/test_dreamzero_trt_engine_builder.py
```

TensorRT runtime 路由单测：

```bash
python test/test_models/test_dreamzero_trt_runtime.py
```

完整验收至少包括：

1. builder 输出的 checkpoint `missing/unexpected tensors` 为 0。
2. engine 能被当前 TensorRT Python runtime 反序列化。
3. cache update 调用仍进入 PyTorch，cached denoise 调用进入 TensorRT。
4. 相同 checkpoint、seed 和评测协议下比较成功率。
5. 预热后记录 action chunk latency；不要把 engine build 时间算入推理延迟。

## 常见问题

### `ModuleNotFoundError: No module named 'onnx'`

当前 Python 环境缺少 ONNX 导出包。安装 `onnx` 后重新导出，不需要额外安装
ONNX-TensorRT 仓库。

### `trtexec was not found`

Python TensorRT wheel 不等于 `trtexec`。检查：

```bash
find "$TRT_ROOT" -type f -name trtexec -print
```

然后把其 `bin` 目录加入 `PATH`，或传入绝对路径：

```text
--trtexec /absolute/path/to/TensorRT/bin/trtexec
```

### engine 无法反序列化

首先比较构建端和运行端：

```bash
trtexec --version
python -c 'import tensorrt as trt; print(trt.__version__)'
```

版本不一致、GPU 架构变化或 engine 文件损坏时应重新构建，不要强行复用旧 engine。

### 构建或加载时 OOM

确保没有其他模型占用 GPU；必要时调低 `--workspace-mib`。ONNX 导出、TensorRT
构建和完整两卡 eval 是三个不同阶段，可以分别执行，避免同时保留不需要的模型进程。

### 导入了其他 FluxVLA checkout

每次运行前检查：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -c 'import fluxvla; print(fluxvla.__file__)'
```

输出必须指向当前准备导出或评测的仓库。
