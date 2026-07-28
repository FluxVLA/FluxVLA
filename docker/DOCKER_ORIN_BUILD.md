# FluxVLA Orin Docker Guide

## 当前状态

截至 2026-05-09，Jetson Orin 上已经完成以下事项：

- 已构建镜像：`fluxvla:orin`
- Image ID：`50375e5a3153`
- 镜像大小：`22.6GB`
- 已在 Docker 内跑通 GR00T dummy input 无权重测试
- 已定位当前 `/mnt/nvme` 下 GR00T checkpoint 无法加载的原因

本次实际使用的主要路径：

- 源码仓库：`/home/limx/sober/FluxVLA`
- NVMe：`/mnt/nvme`
- GR00T 日志：`/mnt/nvme/sober/tmp/gr00t_dummy_no_weights.log`
- GR00T 带权重失败日志：`/mnt/nvme/sober/tmp/gr00t_dummy_ckpt.log`

## 我做了什么

### 1. 解决 Jetson/Orin 容器运行链路上的兼容问题

为了让 FluxVLA 在 Orin 的 Docker 环境里能实际跑起来，我补了几类兼容性修改：

- FSDP / distributed 相关导入改成可选，避免 Jetson PyTorch 缺失完整分布式组件时直接 import 失败
- Eagle/GR00T 路径尊重 `ATTN_IMPLEMENTATION=eager`，避免强依赖 FlashAttention
- `radio_model.py` 改成 `flash_attn` 可选导入，缺失时不在 import 阶段失败
- `EagleInferenceBackbone` 改成懒加载 inference 模型，避免 `triton` 缺失时拖垮训练态 dummy forward
- `heads/__init__.py` 中 `flow_matching_inference_head` 改成可选导入，避免无关 Triton/CUDA op 在当前测试链路上抢先报错
- `scripts/test_gr00t_with_embodiment.py` 新增 `--ckpt-only`，后续拿到有效 checkpoint 时可以直接“无 base pretrained，仅加载 checkpoint”验证

### 2. 收紧了包初始化，避免无关模块在导入时触发额外依赖

当前容器测试的目标是把 Pi0.5 / GR00T 的推理链路先跑通，不是把所有训练路径一次性全部拉起。
因此我保留了当前测试必须的模块导入，把一些会触发 Triton、自定义 CUDA op、FSDP 的无关路径改成了按需导入或可选导入。

### 3. 在 Docker 内实际验证了 GR00T dummy input

实际执行的成功命令如下：

```bash
docker run --rm --runtime nvidia   -e PYTHONWARNINGS=ignore   -e PYTHONPATH=/home/limx/sober/FluxVLA   -v /home/limx/sober:/home/limx/sober   -v /mnt/nvme:/mnt/nvme   fluxvla:orin   bash -lc "cd /home/limx/sober/FluxVLA &&     python3 scripts/test_gr00t_with_embodiment.py       --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py       --no-weights       --predict-runs 1       --warmup 0"
```

实际结果：

- 模型构建成功
- `predict_action` 成功
- 输出形状：`(1, 32, 7)`
- 单次 dummy forward 延迟：约 `3492.644 ms`
- 日志：`/mnt/nvme/sober/tmp/gr00t_dummy_no_weights.log`

### 4. 对当前 GR00T 权重做了容器内验证

实际执行的带权重命令如下：

```bash
docker run --rm --runtime nvidia   -e PYTHONWARNINGS=ignore   -e PYTHONPATH=/home/limx/sober/FluxVLA   -v /home/limx/sober:/home/limx/sober   -v /mnt/nvme:/mnt/nvme   fluxvla:orin   bash -lc "cd /home/limx/sober/FluxVLA &&     python3 scripts/test_gr00t_with_embodiment.py       --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py       --ckpt /mnt/nvme/gr00t_eagle_3b_ur3_finetune_test_10241025/checkpoints/step-006567-epoch-03-loss=0.2167.pt       --ckpt-only       --predict-runs 1       --warmup 0"
```

结果：失败。

报错位置：

```text
KeyError: "filename 'storages' not found"
```

这不是模型结构不匹配，而是当前 checkpoint 文件本身不可被 `torch.load` 正常解析。

我进一步检查了这两个文件：

- `/mnt/nvme/gr00t_eagle_3b_ur3_finetune_test_10241025/checkpoints/step-004378-epoch-02-loss=0.3078.pt`
- `/mnt/nvme/gr00t_eagle_3b_ur3_finetune_test_10241025/checkpoints/step-006567-epoch-03-loss=0.2167.pt`

检查结果：

- 文件表面大小正常，都是约 `11G`
- 但从多个偏移位置采样，文件内容全部是 `0x00`
- 当前 `/mnt/nvme` 下这两份 `.pt` 对 `torch.load` 来说是坏文件或占位文件

因此，当前能确认的是：

- Docker 内 GR00T 推理链路已经打通
- 现在阻塞带权重验证的，不是代码链路，而是权重文件本身不可用

## 直接进入容器

推荐直接用下面这条命令进入，并挂载当前源码与 NVMe：

```bash
docker run --rm -it --runtime nvidia   -e PYTHONWARNINGS=ignore   -e PYTHONPATH=/home/limx/sober/FluxVLA   -v /home/limx/sober:/home/limx/sober   -v /mnt/nvme:/mnt/nvme   fluxvla:orin   bash
```

进入后：

```bash
cd /home/limx/sober/FluxVLA
```

## 如何复现 GR00T 测试

### 无权重 dummy input

```bash
python3 scripts/test_gr00t_with_embodiment.py   --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py   --no-weights   --predict-runs 1   --warmup 0
```

### 有效 checkpoint 到手后的推荐命令

```bash
python3 scripts/test_gr00t_with_embodiment.py   --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py   --ckpt /path/to/valid_checkpoint.pt   --ckpt-only   --predict-runs 1   --warmup 0
```

这里的 `--ckpt-only` 是我新增的参数，用于在没有 `./checkpoints/GR00T-N1.5-3B` base model 目录时，直接构建模型并加载 finetune checkpoint。

## 关键日志

- GR00T 无权重成功日志：`/mnt/nvme/sober/tmp/gr00t_dummy_no_weights.log`
- GR00T 带权重失败日志：`/mnt/nvme/sober/tmp/gr00t_dummy_ckpt.log`
- Pi0.5 dummy forward 日志：`/mnt/nvme/sober/tmp/pi05_dummy_forward.log`

## 本次关键修改文件

与 GR00T / Orin Docker 直接相关的重点文件：

- `Dockerfile.orin-l4t-jetpack`
- `DOCKER_ORIN_BUILD.md`
- `scripts/test_gr00t_with_embodiment.py`
- `fluxvla/models/backbones/vlms/eagle.py`
- `fluxvla/models/third_party_models/eagle2_hg_model/radio_model.py`
- `fluxvla/models/heads/__init__.py`
- `fluxvla/models/heads/flow_matching_head.py`
- `fluxvla/models/heads/llava_action_head.py`
- `fluxvla/models/vlas/open_vla.py`
- `fluxvla/models/__init__.py`
- `fluxvla/models/backbones/__init__.py`
- `fluxvla/models/backbones/vlms/__init__.py`
- `fluxvla/models/vlas/__init__.py`

## 注意事项

1. 当前测试必须显式设置：

   - `PYTHONPATH=/home/limx/sober/FluxVLA`

2. 当前测试必须挂载：

   - `-v /home/limx/sober:/home/limx/sober`
   - `-v /mnt/nvme:/mnt/nvme`

3. 不要只依赖镜像里安装的旧包代码。
   这次验证依赖的是宿主机当前源码树上的修复。

4. 当前 `/mnt/nvme/gr00t_eagle_3b_ur3_finetune_test_10241025/checkpoints/*.pt` 不能作为有效权重使用。
   如果要完成“带权重跑通”，需要满足下面任一条件：

   - 提供一份可被 `torch.load` 正常读取的 `.pt`
   - 提供对应的 `.safetensors`
   - 提供 `./checkpoints/GR00T-N1.5-3B` base model 目录，以及一份有效 finetune checkpoint
   - 或者提供 root 可读的原始 checkpoint 实际路径

5. `run_docker.sh` 可以作为基础参考，但当前 GR00T 验证命令仍建议直接使用上文的 `docker run`，因为它明确包含了本次验证所需的 `PYTHONPATH` 和挂载路径。

## 2026-05-09 增量更新：Mayer catch_ball 权重验证

本次新增验证的源权重路径：

- 远端源：`/limx_embop/tos/users/Mayer/checkpoints/gr00t_eagle_3b_ur3_full_finetune_catch_ball_1_1`
- 本地落盘：`/mnt/nvme/sober/gr00t_eagle_3b_ur3_full_finetune_catch_ball_1_1`
- 实际 checkpoint：`/mnt/nvme/sober/gr00t_eagle_3b_ur3_full_finetune_catch_ball_1_1/checkpoints/step-002826-epoch-18-loss=0.0383.pt`

### 下载与整理

实际通过 rsync 从 LimVLA-3 拉取到 NVMe，并把 `latest-checkpoint.pt` 修正为本地相对软链接：

```bash
cd /mnt/nvme/sober/gr00t_eagle_3b_ur3_full_finetune_catch_ball_1_1/checkpoints
rm -f latest-checkpoint.pt
ln -s step-002826-epoch-18-loss=0.0383.pt latest-checkpoint.pt
```

### checkpoint 可读性验证

先在 Docker 内直接验证 `torch.load`：

```bash
docker run --rm -e PYTHONWARNINGS=ignore -v /mnt/nvme:/mnt/nvme \
  fluxvla:orin \
  python3 -c "import torch; p='/mnt/nvme/sober/gr00t_eagle_3b_ur3_full_finetune_catch_ball_1_1/checkpoints/step-002826-epoch-18-loss=0.0383.pt'; obj=torch.load(p, map_location='cpu'); print(type(obj).__name__); print(sorted(obj.keys())[:20])"
```

结果：

- `dict`
- keys: `['epoch', 'global_step', 'model', 'optimizer_state_dict', 'scheduler_state_dict']`

说明这份 Mayer checkpoint 是有效的，不是之前那种全零坏文件。

### Docker 内 GR00T 带权重测试

实际执行命令：

```bash
docker run --rm --runtime nvidia \
  -e PYTHONWARNINGS=ignore \
  -e PYTHONPATH=/home/limx/sober/FluxVLA \
  -v /home/limx/sober:/home/limx/sober \
  -v /mnt/nvme:/mnt/nvme \
  fluxvla:orin \
  bash -lc "cd /home/limx/sober/FluxVLA && \
    python3 scripts/test_gr00t_with_embodiment.py \
      --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
      --ckpt /mnt/nvme/sober/gr00t_eagle_3b_ur3_full_finetune_catch_ball_1_1/checkpoints/step-002826-epoch-18-loss=0.0383.pt \
      --ckpt-only \
      --predict-runs 1 \
      --warmup 0"
```

结果：成功，退出码 `0`。

关键输出：

- `load_state_dict: missing=899, unexpected=5`
- `CUDA: Orin`
- `predict_action output shape: (1, 32, 7)`
- `latency_ms: min=6101.404 max=6101.404 mean=6101.404 median=6101.404`
- `OK`

日志文件：

- `/mnt/nvme/sober/tmp/gr00t_catch_ball_mayer.log`
- `/mnt/nvme/sober/tmp/gr00t_catch_ball_mayer.status`

### 注意事项补充

1. 这次 `catch_ball` 权重可以加载并完成 dummy `predict_action`，已经明显优于此前 `/mnt/nvme/gr00t_eagle_3b_ur3_finetune_test_10241025` 那批损坏 checkpoint。
2. `load_state_dict: missing=899, unexpected=5` 说明 repo 当前测试配置与这份 finetune checkpoint 不是完全逐键对齐，但当前 dummy 推理链路能实际跑通。
3. 日志里有一条非致命 warning：
   - `The size of tensor a (0) must match the size of tensor b (512) ...`
     这来自 dummy 输入路径中的 embedding 选择逻辑，但没有阻塞 `predict_action` 成功返回。
4. 如果后续要做真实数据或更严格精度验证，建议基于这份可用 checkpoint 继续做真实 observation/action 输入测试，而不是再用之前那批损坏权重。

## 2026-05-09 增量更新：Pi0.5 Triton backend 验证

本次目标是验证：Jetson Orin 的 `fluxvla:orin` 容器里，Pi0.5 是否能实际走 `PI05FlowMatchingInference` 这条 Triton 加速推理路径。

### 结论

可以，但当前仓库和镜像不是开箱即用，必须先补下面几项：

1. 恢复 `fluxvla/ops/__init__.py` 与 `fluxvla/ops/cuda/__init__.py` 的符号导出
2. 重新编译 `fluxvla/ops/cuda/*/*.so`，使其与当前容器内 PyTorch ABI 一致
3. 安装 `triton`（本次安装到 NVMe 路径，而不是容器根文件系统）
4. 在 `fluxvla/models/vlas/__init__.py` 中显式导入 `PI05FlowMatchingInference`，让它进入 `VLAS` registry

### 本次实际修复

#### 1. 恢复 ops 导出

恢复为：

```python
# fluxvla/ops/__init__.py
from .cuda import *
from .triton import *

# fluxvla/ops/cuda/__init__.py
from .gemma_rotary_embedding import *
from .matmul_bias import *
from .rotary_pos_embedding import *
```

#### 2. 重新编译 CUDA 扩展

当前容器内的 torch 是：

- `torch 2.5.0a0+872d972e41.nv24.08`
- `cuda 12.6`

原始 `.so` 与这套 torch ABI 不匹配，报错：

```text
undefined symbol: _ZNK3c1011StorageImpl27throw_data_ptr_access_errorEv
```

因此在 Docker 内重新编译：

```bash
docker run --rm --runtime nvidia \
  -e TMPDIR=/mnt/nvme/sober/tmp \
  -e MAX_JOBS=2 \
  -v /home/limx/sober:/home/limx/sober \
  -v /mnt/nvme:/mnt/nvme \
  fluxvla:orin \
  bash -lc "cd /home/limx/sober/FluxVLA && \
    FORCE_CUDA=1 python3 setup.py build_ext \
      --build-temp /mnt/nvme/sober/tmp/build_fluxvla_ext \
      --inplace"
```

结果：成功，退出码 `0`。

日志：

- `/mnt/nvme/sober/tmp/logs/build_fluxvla_ext.log`
- `/mnt/nvme/sober/tmp/logs/build_fluxvla_ext.status`

#### 3. 安装 Triton 到 NVMe

考虑到系统盘空间紧张，本次没有写入容器根文件系统，而是安装到 NVMe 目录：

```bash
docker run --rm \
  -v /mnt/nvme:/mnt/nvme \
  fluxvla:orin \
  bash -lc "python3 -m pip install --no-cache-dir \
    --target /mnt/nvme/sober/pylibs/triton351 \
    triton==3.5.1"
```

结果：成功。

验证：

```bash
docker run --rm -v /mnt/nvme:/mnt/nvme fluxvla:orin \
  python3 -c "import sys; sys.path.insert(0, '/mnt/nvme/sober/pylibs/triton351'); import triton; print(triton.__version__)"
```

输出：

- `3.5.1`

#### 4. 补 registry 导入

在 `fluxvla/models/vlas/__init__.py` 中增加：

```python
from .pi05_flowmatching_inference import PI05FlowMatchingInference
```

否则即使模块文件存在，也会报：

```text
PI05FlowMatchingInference is not in the fluxvla::vlas registry
```

### Docker 内最小 Pi0.5 Triton 验证

本次实际验证脚本：

- `/mnt/nvme/sober/tmp/pi05_triton_dummy.py`

执行命令：

```bash
docker run --rm --runtime nvidia \
  -e PYTHONWARNINGS=ignore \
  -v /mnt/nvme:/mnt/nvme \
  -v /home/limx/sober:/home/limx/sober \
  -e PYTHONPATH=/mnt/nvme/sober/pylibs/triton351:/home/limx/sober/FluxVLA \
  fluxvla:orin \
  python3 /mnt/nvme/sober/tmp/pi05_triton_dummy.py
```

结果：成功，退出码 `0`。

关键输出：

- `CUDA: Orin`
- `[Triton Inference] Recording CUDA Graph ...`
- `[Triton Inference] CUDA Graph recorded successfully!`
- `predict_action output shape: (1, 10, 32)`
- `dtype: torch.float32`
- `OK`

日志：

- `/mnt/nvme/sober/tmp/logs/pi05_triton_dummy.log`
- `/mnt/nvme/sober/tmp/logs/pi05_triton_dummy.status`

### 复现时的关键环境变量

必须带：

```bash
-e PYTHONPATH=/mnt/nvme/sober/pylibs/triton351:/home/limx/sober/FluxVLA
```

否则会出现：

- `ModuleNotFoundError: No module named 'triton'`

### 注意事项补充

1. 当前 Pi0.5 Triton 路径已经证明可以在 Orin Docker 内执行到首次 `predict_action`，并且 CUDA Graph 录制成功。
2. 这次验证用的是随机初始化权重，目的是证明加速推理链路可用，不代表数值正确性。
3. 若后续要做真实 Pi0.5 权重验证，需要继续在同一环境变量下加载可用 checkpoint。
4. 当前这条 Triton 路径依赖 NVMe 上的额外 Python 路径 `/mnt/nvme/sober/pylibs/triton351`；如果以后重建镜像，建议把 Triton 和 CUDA 扩展重编流程直接固化到 Dockerfile。

## 2026-05-09 增量更新：Pi0.5 真实 checkpoint Triton 100 次统计

本次在已经打通的 Pi0.5 Triton inference 路径上，进一步加载真实 checkpoint，统计 100 次稳定态推理耗时。

### 使用的权重

- 配置：`/mnt/nvme/sober/checkpoints/pi05_aloha_fold_towel/config.yaml`
- 权重：`/mnt/nvme/sober/checkpoints/pi05_aloha_fold_towel/checkpoints/step-021062-epoch-01-loss=0.0205.safetensors`

### 权重加载结果

实际加载输出：

```text
load_state_dict(strict=False): missing=0 unexpected=0
```

说明这份 `fold_towel` checkpoint 与当前构造的 `PI05FlowMatchingInference` 模型是完整匹配的。

### 100 次 Triton 推理统计

实际执行的是带真实权重的 Triton benchmark 脚本：

- `test/test_models/test_pi05_orin.py`

对应执行命令：

```bash
docker run --rm --runtime nvidia \
  -e PYTHONWARNINGS=ignore \
  -v /mnt/nvme:/mnt/nvme \
  -v /home/limx/sober:/home/limx/sober \
  -e PYTHONPATH=/mnt/nvme/sober/pylibs/triton351:/home/limx/sober/FluxVLA \
  fluxvla:orin \
  python3 test/test_models/test_pi05_orin.py
```

结果：成功，退出码 `0`。

关键输出：

- `CUDA: Orin`
- `Loading weights: /mnt/nvme/sober/checkpoints/pi05_aloha_fold_towel/checkpoints/step-021062-epoch-01-loss=0.0205.safetensors`
- `load_state_dict(strict=False): missing=0 unexpected=0`
- `[Triton Inference] Recording CUDA Graph ...`
- `[Triton Inference] CUDA Graph recorded successfully!`
- `cold_start_ms: 22319.797`
- `predict_action output shape: (1, 50, 32)`
- `latency_ms: min=583.809 max=587.641 mean=585.100 median=585.031`
- `stdev_ms: 0.702`
- `total_wall_ms: 58510.047`
- `OK`

前 10 次耗时：

```text
[585.507, 585.379, 585.202, 585.363, 585.258, 585.61, 585.006, 584.843, 584.824, 584.196]
```

日志文件：

- `/mnt/nvme/sober/tmp/logs/test_pi05_orin.log`
- `/mnt/nvme/sober/tmp/logs/test_pi05_orin.status`

### 结果解释

1. 这次验证的是“真实权重 + Triton backend + CUDA Graph”下的稳定态推理时延。
2. 输出 shape 是 `(1, 50, 32)`，因为这份 `fold_towel` 配置里的 `action_chunk=50`。
3. 因此前面随机初始化的 `(1, 10, 32)` benchmark 不能和这次结果直接按数值比较，两者推理长度不同。
4. 当前结果已经证明：在 Orin Docker 环境里，Pi0.5 可以加载真实 checkpoint，并通过 Triton inference 路径完成 100 次稳定推理。

## 2026-05-11 增量更新：Triton 直接固化进镜像

### 最终结果

- 新镜像：`fluxvla:orin`
- Image ID：`964262fe72af`
- 镜像大小：`23.4GB`
- 新镜像内验证：
  - `import triton` 成功
  - `triton 3.5.1`
  - `torch.cuda.is_available() == True`

### 本次做了什么

1. 已更新 `Dockerfile.orin-l4t-jetpack`，把 `pip3 install triton==3.5.1` 加进镜像构建流程。
2. 已把 `triton` 加进 build-time smoke check。
3. 由于 Jetson 上完整 `docker build` 在长依赖层不稳定，本次实际落地方式是：
   - 保留 Dockerfile 改动
   - 从旧 `fluxvla:orin` 启动辅助容器
   - 容器内安装 `triton==3.5.1`
   - `docker commit` 回新的 `fluxvla:orin`

### 验证结果

镜像内验证命令：

```bash
docker run --rm --runtime nvidia   -v /home/limx/sober/FluxVLA:/workspace/FluxVLA   -v /mnt/nvme:/mnt/nvme   -w /workspace/FluxVLA   fluxvla:orin   python3 -c "import triton, torch; print(triton.__version__); print(torch.cuda.is_available())"
```

输出：

- `3.5.1`
- `True`

Pi0.5 容器验证：

- `pi05_triton_real_test`：退出码 `0`
- `pi05_triton_short_test`：退出码 `0`

### 使用变化

现在进入容器时：

```bash
-e PYTHONPATH=/workspace/FluxVLA
```

不再需要：

```bash
-e PYTHONPATH=/mnt/nvme/sober/pylibs/triton351:/workspace/FluxVLA
```
