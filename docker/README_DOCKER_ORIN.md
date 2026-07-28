# FluxVLA Orin Docker

## Orin 前置系统配置工作

挂载 NVMe SSD

```bash
# 分区
sudo fdisk /dev/nvme0n1
# 交互输入: n → 回车 → 回车 → 回车 → 回车 → w

# 格式化
sudo mkfs.ext4 /dev/nvme0n1p1

# 挂载
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme
sudo chown <user_name>:<user_name> /mnt/nvme

# 开机自动挂载
echo '/dev/nvme0n1p1 /mnt/nvme ext4 defaults 0 2' | sudo tee -a /etc/fstab

# 创建工作目录
mkdir -p /mnt/nvme/{checkpoints,datasets,work_dirs}
```

将 Docker 数据目录放到 NVMe

Docker build 产生的镜像 layer 默认存放在 Docker image store 中，通常是系统盘：

```bash
docker info | grep "Docker Root Dir"
```

如果输出是 `/var/lib/docker`，建议在构建 FluxVLA 镜像前把 Docker 数据目录迁移到 NVMe，避免占满系统盘：

```bash
sudo apt update
sudo apt install -y rsync
sudo systemctl stop docker
sudo mkdir -p /mnt/nvme/docker
sudo rsync -aHAX /var/lib/docker/ /mnt/nvme/docker/

sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "data-root": "/mnt/nvme/docker"
}
EOF

sudo systemctl start docker
docker info | grep "Docker Root Dir"
```

期望输出：

```text
Docker Root Dir: /mnt/nvme/docker
```

> 这个步骤会改变 Docker 后续 build / pull / run 使用的存储位置。建议在第一次构建 FluxVLA Docker 前完成。

## 配置 pip 国内源

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

## 0.3 安装系统依赖

```bash
sudo apt update
sudo apt install -y cmake libglew-dev libosmesa6-dev ninja-build
```

## 当前镜像标签

下面是仓库中推荐的 Orin 分层镜像和它们的配置/用途摘要：

| 镜像标签              |                             基础镜像 |   是否包含 ROS   |          FlashAttention          | 主要用途                                                       | 备注                                      |
| --------------------- | -----------------------------------: | :--------------: | :------------------------------: | -------------------------------------------------------------- | ----------------------------------------- |
| `fluxvla:orin-base`   | `nvcr.io/nvidia/l4t-jetpack:r36.4.0` |        否        |       无（仅系统/Runtime）       | 基础层：系统依赖、PyTorch/Triton、FluxVLA 通用依赖             | 用于对外发布的最小运行时基线              |
| `fluxvla:orin-fa`     |                     基于 `orin-base` |        否        | `flash-attn==2.5.5`（SM87 编译） | 包含为 Orin 预编译的 FlashAttention，供极速推理使用            | 优先从预编译 wheel 安装，避免现场源码编译 |
| `fluxvla:orin-ros`    |                     基于 `orin-base` | 是（ROS Noetic） |         无（或可选安装）         | 包含 ROS Noetic 运行时及 Python ROS 绑定，用于机器人集成与部署 | 不包含 flash‑attn 编译，体积以 ROS 为主   |
| `fluxvla:orin-ros-fa` |     基于 `orin-fa` + `orin-ros` 组合 | 是（ROS Noetic） | `flash-attn==2.5.5`（SM87 编译） | 集成 ROS 与 FlashAttention 的完整运行镜像                      | 适合在真机上同时运行 ROS 节点与高性能推理 |

### 镜像配置说明

- `fluxvla:orin-base`:

  - 包含系统包（CMake、ninja、libosmesa、glew 等）、Python 3.10 运行时、经过验证的 PyTorch / Triton 安装以及 FluxVLA 的 Python 依赖。
  - 目的：作为其它衍生镜像的稳定基线，减少重复构建成本。

- `fluxvla:orin-fa`:

  - 在 `orin-base` 基础上安装或复制预编译的 `flash_attn` wheel（针对 SM87 编译的二进制）。
  - 环境变量：推荐在运行时保留 `ATTN_IMPLEMENTATION=flash_attention_2` 和 `TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2`。
  - 优点：避免每次构建都从源码长时间编译 flash‑attn；运行时能显著提升 attention-heavy 模型的性能（前提是使用 flash‑attn 支持的 attention 实现）。

- `fluxvla:orin-ros`:

  - 单独构建 ROS Noetic（/opt/ros/noetic）和 Python ROS 绑定（`rospy`、`cv_bridge` 等），并复制到镜像的 Python site-packages 中。
  - 目的：在需要 ROS 通信、消息与传感器接口的真机部署场景使用。
  - 注意：ROS 层会显著增加镜像体积，但对 GPU 内核性能没有直接影响。

- `fluxvla:orin-ros-fa`:

  - 复用 `orin-fa` 的 flash‑attn 二进制与 `orin-ros` 的 ROS 运行时，形成面向真机部署且具备高性能 attention kernel 的镜像。

### 选择指南

- 如果只关心推理性能且需要 FlashAttention 优化：使用 `fluxvla:orin-fa`（或 `orin-ros-fa` 若同时需要 ROS）。
- 如果只需要最小运行时（更小镜像、快速分发）：使用 `fluxvla:orin-base` 并在容器内按需安装额外包。
- 如果需要机器人集成（ROS 节点 + 推理）：使用 `fluxvla:orin-ros-fa`。

查看本机已有镜像：

```bash
docker images fluxvla
```

查看本机已有镜像：

```bash
docker images fluxvla
```

基础镜像：

```text
nvcr.io/nvidia/l4t-jetpack:r36.4.0
```

## 构建镜像

请在仓库根目录执行：

```bash
cd FluxVLA
```

统一构建入口是：

```bash
docker/build_docker.sh
```

默认 target 是 `all`，会依次构建推荐的分层镜像：

```text
fluxvla:orin-base
fluxvla:orin-fa
fluxvla:orin-ros
fluxvla:orin-ros-fa
```

为了避免 `flash-attn` 与 ROS Noetic 每次都一起重编，日常只需要记住这一个脚本，并按 target 选择构建范围：

```bash
docker/build_docker.sh all        # 推荐完整分层构建
docker/build_docker.sh base       # 只构建 fluxvla:orin-base
docker/build_docker.sh wheel      # 只编译 flash-attn SM87 wheel
docker/build_docker.sh fa         # 只构建 fluxvla:orin-fa
docker/build_docker.sh ros        # 只构建 fluxvla:orin-ros
docker/build_docker.sh ros-fa     # 只构建 fluxvla:orin-ros-fa
```

这些 target 由同一个 multi-stage Dockerfile 管理：`docker/Dockerfile.orin`。

发布版本号作为第二个参数：

```bash
docker/build_docker.sh all 1.0.0
docker/build_docker.sh ros-fa 1.0.0
```

对应版本标签示例：

```text
fluxvla:orin-base-1.0.0
fluxvla:orin-fa-1.0.0
fluxvla:orin-ros-1.0.0
fluxvla:orin-ros-fa-1.0.0
```

## 推荐的分层构建流程

`docker/build_docker.sh all` 等价于依次完成下面这些 target；底层脚本保留用于兼容和排障，但日常不需要直接调用：

```bash
docker/build_docker.sh base
docker/build_docker.sh wheel
docker/build_docker.sh fa
docker/build_docker.sh ros
docker/build_docker.sh ros-fa
```

对应镜像标签：

```text
fluxvla:orin-base
fluxvla:orin-fa
fluxvla:orin-ros
fluxvla:orin-ros-fa
```

如果只改了 flash-attn：

```bash
docker/build_docker.sh wheel
docker/build_docker.sh fa
docker/build_docker.sh ros-fa
```

如果只改了 ROS：

```bash
docker/build_docker.sh ros
docker/build_docker.sh ros-fa
```

重构过程与设计说明见 [docs/orin_docker_refactor_2026-06.md](docs/orin_docker_refactor_2026-06.md)。

flash-attn 在 Orin 上编译耗时较长。默认只使用 1 个编译 job，避免内存不足；如果确认内存足够，可以提高并发：

```bash
FLUXVLA_FLASH_ATTN_MAX_JOBS=2 docker/build_docker.sh all
```

如果构建时 `ports.ubuntu.com` 或 PyPI 网络不稳定，可以临时启用国内源。该开关只影响构建过程，不改变运行容器的网络配置：

```bash
FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh all
FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh ros
```

默认国内源为清华源：

```text
Ubuntu ports: https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports
pip: https://pypi.tuna.tsinghua.edu.cn/simple
```

也可以手动指定镜像源：

```bash
FLUXVLA_UBUNTU_PORTS_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports \
FLUXVLA_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
FLUXVLA_PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
docker/build_docker.sh all
```

发布版本同理：

```bash
docker/build_docker.sh all 1.0.0
```

对应产出：

```text
fluxvla:orin-base-1.0.0
fluxvla:orin-fa-1.0.0
fluxvla:orin-ros-1.0.0
fluxvla:orin-ros-fa-1.0.0
```

验证 ROS 镜像：

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros docker/run_docker.sh \
  python3 -c "import rospy, cv_bridge; print('ROS OK')"
```

验证 FlashAttention：

```bash
docker/run_docker.sh \
  python3 -c "import flash_attn; print(flash_attn.__version__)"
```

验证 ROS + FlashAttention：

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros docker/run_docker.sh \
  python3 -c "import rospy, cv_bridge, flash_attn; print('ROS + FA OK')"
```

如果确实需要手动构建，使用当前 Dockerfile：

```bash
docker build \
  -f docker/Dockerfile.orin \
  --build-arg INSTALL_ROS=1 \
  -t fluxvla:orin-ros \
  .
```

## 快速使用（推荐）

所有 Docker 操作统一通过 `run_docker.sh`，它会自动设置 GPU runtime、环境变量、挂载路径：

```bash
# 进入容器交互 shell
docker/run_docker.sh

# 在容器内执行任意命令
docker/run_docker.sh python3 -c "import torch; print(torch.cuda.is_available())"

# 验证 Triton
docker/run_docker.sh python3 -c "import triton; print(triton.__version__)"

# 验证 FlashAttention
docker/run_docker.sh python3 -c "import flash_attn; print(flash_attn.__version__)"

# 运行 ROS 镜像
FLUXVLA_IMAGE=fluxvla:orin-ros docker/run_docker.sh
```

如果还没有构建 ROS 镜像，先执行：

```bash
docker/build_docker.sh ros
```

进入 ROS 镜像：

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros docker/run_docker.sh
```

运行 ROS 发布版：

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-1.0.0 docker/run_docker.sh
```

验证 ROS：

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros docker/run_docker.sh \
  python3 -c "import rospy, cv_bridge; print('ROS OK')"
```

`run_docker.sh` 自动处理的参数：

| 参数                               | 值                   | 用途                            |
| ---------------------------------- | -------------------- | ------------------------------- |
| `--runtime=nvidia`                 | -                    | GPU 支持                        |
| `--ipc=host`                       | -                    | 避免共享内存不足                |
| `--network=host`                   | -                    | 支持 ROS / 机器人网络通信       |
| `--shm-size`                       | `16g`                | 增大共享内存                    |
| `PYTHONPATH`                       | `/workspace/FluxVLA` | 源码导入路径                    |
| `ATTN_IMPLEMENTATION`              | `flash_attention_2`  | 默认使用 FlashAttention         |
| `TRANSFORMERS_ATTN_IMPLEMENTATION` | `flash_attention_2`  | Transformers attention 后端     |
| `-v FluxVLA:/workspace/FluxVLA`    | -                    | 挂载源码（宿主机实时同步）      |
| `-v /mnt/nvme:/mnt/nvme`           | -                    | 挂载 NVMe（checkpoint、日志等） |
| `WANDB_MODE`                       | `disabled`           | 禁用 wandb                      |
| `ROS_MASTER_URI`                   | 宿主机环境变量       | 如果设置则自动透传              |
| `ROS_IP` / `ROS_HOSTNAME`          | 宿主机环境变量       | 如果设置则自动透传              |

如果需要临时回退 eager attention：

```bash
ATTN_IMPLEMENTATION=eager TRANSFORMERS_ATTN_IMPLEMENTATION=eager \
  docker/run_docker.sh
```

验证 GR00T 使用 FlashAttention：

```bash
docker/run_docker.sh python3 scripts/test_gr00t_with_embodiment.py \
  --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
  --no-weights \
  --warmup 1 \
  --predict-runs 3
```

期望日志中包含：

```text
Attention backend: flash_attention_2
OK
```

## GR00T 测试

快速验证链路：

```bash
./run_docker.sh python3 test/test_models/test_gr00t_orin.py \
  --variant baseline \
    --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
    --ckpt /mnt/nvme/gr00t/gr00t_eagle_3b_ur3_full_finetune_orin/checkpoints/step-000100-epoch-00-loss=1.2629.pt \
    --warmup 0 --predict-runs 1
```

100 次 benchmark：

```bash
./run_docker.sh python3 test/test_models/test_gr00t_orin.py \
  --variant baseline \
    --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
    --ckpt /mnt/nvme/gr00t/gr00t_eagle_3b_ur3_full_finetune_orin/checkpoints/step-000100-epoch-00-loss=1.2629.pt \
    --warmup 5 --predict-runs 100
```

后台运行（长时间任务）：

```bash
nohup ./run_docker.sh python3 test/test_models/test_gr00t_orin.py \
  --variant baseline \
    --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
    --ckpt /mnt/nvme/gr00t/gr00t_eagle_3b_ur3_full_finetune_orin/checkpoints/step-000100-epoch-00-loss=1.2629.pt \
    --warmup 5 --predict-runs 100 \
    > /mnt/nvme/gr00t/gr00t_ur3_100run.log 2>&1 &
```

测试脚本会自动识别训练 checkpoint 格式并加载 `checkpoint['model']`，正常输出：

```text
Detected training checkpoint; loading checkpoint['model']
load_state_dict: missing=0, unexpected=0
```

## GR00T 加速推理测试

使用 `FlowMatchingInferenceHead` + `EagleInferenceBackbone`（Triton kernel + CUDA Graph）：

```bash
./run_docker.sh python3 test/test_models/test_gr00t_orin.py \
  --variant accelerated \
    --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
    --ckpt /mnt/nvme/gr00t/gr00t_eagle_3b_ur3_full_finetune_orin/checkpoints/step-000100-epoch-00-loss=1.2629.pt \
    --warmup 5 --predict-runs 100
```

## Pi0.5 Triton 测试

```bash
./run_docker.sh python3 test/test_models/test_pi05_orin.py
```

首次运行会录制 CUDA Graph，100 次统计耗时较长。

## 已知验证结果

### Pi0.5

- 2026-05-09 已验证：
  - `load_state_dict(strict=False): missing=0 unexpected=0`
  - `cold_start_ms: 22319.797`
  - `100-run mean latency: 585.100 ms`
  - `output shape: (1, 50, 32)`
- 2026-05-11 再次验证：
  - 新镜像内 `triton` 可直接导入
  - `pi05_triton_real_test` 容器退出码 `0`
  - `pi05_triton_short_test` 容器退出码 `0`

### GR00T

- GR00T dummy input 在 Docker 内已跑通
- Mayer catch_ball checkpoint 在 Docker 内可加载并完成推理
- 2026-05-11 已验证 UR3 full finetune checkpoint：
  - 权重路径：`/mnt/nvme/gr00t/gr00t_eagle_3b_ur3_full_finetune_orin/checkpoints/step-000100-epoch-00-loss=1.2629.pt`
  - `load_state_dict: missing=0, unexpected=0`
  - `predict_action output shape: (1, 32, 7)`
  - 100-run 延迟：`min=287.162 ms`, `max=570.103 ms`, `mean=371.316 ms`, `median=300.021 ms`, `stdev=113.016 ms`
  - `total_wall_predict=37131.573 ms`
  - 日志：`/mnt/nvme/gr00t/gr00t_ur3_100run.log`

### GR00T Baseline vs Accelerated 对比（Orin MODE_30W）

| 方案                          | 注意力                         | 模型组件                                               | 中位延迟                            | 频率         | 日志                           |
| ----------------------------- | ------------------------------ | ------------------------------------------------------ | ----------------------------------- | ------------ | ------------------------------ |
| **Baseline**                  | eager                          | `EagleBackbone` + `FlowMatchingHead`                   | 190.6 ms                            | ~5.24 Hz     | terminal                       |
| **Accelerated (eager)**       | eager + math SDPA              | `EagleInferenceBackbone` + `FlowMatchingInferenceHead` | 305.6 ms                            | ~3.27 Hz     | `gr00t_ur3_accel_100run.log`   |
| **Accelerated (eager, 分段)** | eager + math SDPA              | 同上，分段计时                                         | VLM: 152ms + Head: 150ms = 302.4 ms | ~3.31 Hz     | `gr00t_accel_split_100.log`    |
| **Accelerated (flash)**       | flash_attention_2 + flash SDPA | `EagleInferenceBackbone` + `FlowMatchingInferenceHead` | **151.1 ms**                        | **~6.62 Hz** | `gr00t_accel_flash_100run.log` |

> **结论**：
>
> 1. **eager 路径下**，FluxVLA 的 Triton kernel + CUDA Graph 加速版（305.6 ms）反而比 baseline（190.6 ms）慢约 1.6x —— 因为脚本强制 `enable_flash_sdp(False)` / `enable_mem_efficient_sdp(False)`，注意力退回 math 后端，且 Eagle backbone 走 eager。
> 2. **接入 flash-attn 2.5.5 (SM87) 后**，加速版降到 **151.1 ms（6.62 Hz）**，相比 eager 加速版提速 **2.02x**，也优于 baseline。提速来自两处：Eagle backbone 走 `flash_attention_2`，DiT head 的 SDPA 从 math 后端切到 flash/mem-efficient 后端。

### GR00T Baseline vs Accelerated 对比（Orin MAXN，2026-06-05 复测）

四种组合，均 warmup=5 / runs=100；baseline 用 lang-len=48，accelerated 用 lang-len=600（各自脚本默认）。

| 方案            | 注意力                             | 中位延迟      | 频率         | 日志                   |
| --------------- | ---------------------------------- | ------------- | ------------ | ---------------------- |
| Baseline        | eager                              | 198.05 ms     | ~5.05 Hz     | terminal               |
| Baseline        | flash                              | 191.97 ms     | ~5.21 Hz     | terminal               |
| Accelerated     | eager + math SDPA                  | 158.15 ms     | ~6.32 Hz     | terminal               |
| **Accelerated** | **flash_attention_2 + flash SDPA** | **150.66 ms** | **~6.64 Hz** | `maxn_accel_flash.log` |

> **MAXN 下的关键发现**：
>
> 1. **加速版对 GPU 频率极敏感**：accelerated-eager 从 30W 的 305.6 ms 暴降到 MAXN 的 158.15 ms（~1.9x）。因为 eager 走 math-SDPA，是 compute-bound 的 O(n²) 运算，随时钟线性提速。
> 2. **flash 版几乎不受功耗影响**：accelerated-flash 在 30W（151.1 ms）与 MAXN（150.66 ms）基本一致，说明它已不再被注意力计算卡住，瓶颈转移到访存 / kernel 启动 / CUDA Graph replay。
> 3. **baseline 基本不随 MAXN 变化**（190→198 ms）：lang-len 仅 48，注意力占比小，非 compute-bound；flash 仅带来约 3% 提升。
> 4. **最优组合：Accelerated + flash = 150.66 ms / 6.64 Hz**。在 MAXN 下 flash 相比 eager 加速版仅再快约 5%（因 eager 已被 MAXN 拉起），但 flash 的优势在于**不依赖高功耗即可达到该速度**（30W 下同样 ~151 ms）。

#### flash-attn 镜像与运行方式

- flash-attn 已编译进镜像 **`fluxvla:orin-fa`**（基于 `fluxvla:orin`，`fluxvla:orin` 保留为回退）。
- flash-attn 2.5.5 的 `setup.py` 默认只生成 `sm_80`/`sm_90`，**不含 SM87**，直接 pip 安装会报 `no kernel image is available`。本镜像通过源码修改 `cc_flag` 为 `arch=compute_87,code=sm_87` 后重新编译解决（构建日志 `/mnt/nvme/gr00t/flash_attn_build_2.5.5_sm87.log`）。
- 运行 flash 版测试（注意 `PYTHONPATH` 必须指向挂载源码，否则 VLA 注册表缺失）：

```bash
docker run --rm --runtime=nvidia --ipc=host --shm-size=16g \
    -e PYTHONPATH=/workspace/FluxVLA \
    -v /home/limx/sober/FluxVLA:/workspace/FluxVLA \
    -v /mnt/nvme:/mnt/nvme \
    -w /workspace/FluxVLA \
    fluxvla:orin-fa \
    python3 /mnt/nvme/gr00t/test_gr00t_accel_flash_100run.py \
      --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
      --ckpt /mnt/nvme/gr00t/gr00t_eagle_3b_ur3_full_finetune_orin/checkpoints/step-000100-epoch-00-loss=1.2629.pt \
      --warmup 5 --predict-runs 100
```

## 注意事项

1. 当前根盘 `/` 很紧张，后续如果要继续改文档或构建镜像，建议先清理系统盘。
2. `Dockerfile.orin-l4t-jetpack` 已经同步更新为包含 Triton，但 Jetson 上完整重建过程不稳定；当前本地可用镜像是通过“容器内安装 Triton + docker commit”落地的。
3. 根 README 入口仍在：`README.md` 与 `README_zh-CN.md`。
