# FluxVLA引擎：专为具身智能打造的“一站式”VLA 工程平台

<p align="center">
  <img src="assets/fluxvla.png" alt="FluxVLA" width="600">
</p>

<div align="center">
<a href="https://huggingface.co/limxdynamics/FluxVLAEngine"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://fluxvla.limxdynamics.com"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://fluxvla.limxdynamics.com/zh/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://github.com/limxdynamics/FluxVLA/issues/1"><img src="https://img.shields.io/badge/微信-green?logo=wechat"></a>
<a href="https://github.com/limxdynamics/FluxVLA/issues/1"><img src="https://img.shields.io/badge/飞书-3370FF?logo=lark&logoColor=white"></a>
</div>

<div align="center">

[English](README.md) | 简体中文 | [日本語](README_ja.md)

</div>

FluxVLA Engine是面向具身智能落地应用的全链路一体化工程平台，以统一配置、标准接口、模块解耦、可部署为核心设计理念，构建从数据到真机部署的完整工程闭环，并以“标准化产学研基座”为目标，显著降低 VLA 研究与开发的工程门槛。

## 框架

<p align="center">
  <img src="assets/framework.png" alt="Framework Architecture" width="800">
</p>

## 性能

| Codebase                    |                                                     Libero-Spatial                                                      |                                                      Libero-Object                                                      |                                                      Libero-Goal                                                      |                                                     Libero-Long                                                     | Libero-Average |
| --------------------------- | :---------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: | :------------: |
| FluxVLA(SmolVLA)            |      [86.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_spatial_full_finetune_bs64)      |      [92.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_object_full_finetune_bs64)       |      [91.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_goal_full_finetune_bs64)       |      [68.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_10_full_finetune_bs64)       |      84.7      |
| FluxVLA(GR00T)              |  [97.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_spatial_full_finetune_bs64)   |   [96.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_object_full_finetune_bs64)   |   [94.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_goal_full_finetune_bs64)   | [93.0±1.5](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_10_full_finetune_bs64) |      95.3      |
| FluxVLA(DreamZero)          | [98.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_spatial_full_finetune_w_cache_bs64) | [98.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_object_full_finetune_w_cache_bs64)  | [93.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_goal_full_finetune_w_cache_bs64)  | [94.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_10_full_finetune_w_cache_bs64)  |     96.25      |
| FluxVLA(Qwen3VL 0.6B+GR00T) | [96.0](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_object_full_finetune_bs64) | [99.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_object_full_finetune_bs64) | [95.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_goal_full_finetune_bs64) | [94.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_10_full_finetune_bs64) |     96.20      |
| FluxVLA(PI0)                |   [98.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_spatial_full_finetune_bs64)   |   [98.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_object_full_finetune_bs64)    |   [96.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_goal_full_finetune_bs64)    |   [93.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_10_full_finetune_bs64)    |     96.85      |
| FluxVLA(PI0.5)              |  [98.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_spatial_full_finetune_bs64)   |   [99.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_object_full_finetune_bs64)   |   [98.0](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_goal_full_finetune_bs64)   | [95.6±1.0](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_10_full_finetune_bs64) |     97.95      |

*带链接的分数可跳转到对应 checkpoint。*

#### RoboCasa GR1

| 模型           | 训练数据             | Cabinet | Drawer | Microwave | Generalization | Average                                                                                                                        |
| -------------- | -------------------- | ------- | ------ | --------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| FluxVLA(GR00T) | 24 个任务，30 条演示 | 22.7%   | 35.7%  | 32.5%     | 48.9%          | [44.3%(50trials)](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_robocasa_gr1_24x30_finetune_bs64) |

#### 说明

- `Cabinet`：`PnPBottleToCabinetClose` + `PnPWineToCabinetClose`。
- `Drawer`：`PnPCanToDrawerClose` + `PnPCupToDrawerClose`。
- `Microwave`：`PnPMilkToMicrowaveClose` + `PnPPotatoToMicrowaveClose`。
- `Generalization`：剩余的 18 个后训练新任务。
- RoboCasa GR00T 结果使用每个任务 50 次试验评测。

## 📢 最新动态

**\[2026/08/10\]** 🔥 FluxVLA 现已支持在 NVIDIA Jetson Orin 上部署，GR00T-N1.5 端侧加速推理可达到 7.4 Hz。Orin 初始刷机流程见 [docs/orin_flashing_zh-CN.md](docs/orin_flashing_zh-CN.md)，FluxVLA Docker 环境启动与运行测试见 [docs/orin_docker_runtime_zh-CN.md](docs/orin_docker_runtime_zh-CN.md)。

**\[2026/06/30\]** 🔥 现已支持 Franka 单臂与双臂真机推理，包括 joint/eepose 控制配置与部署指南。详见 [docs/franka.md](docs/franka.md)。

**\[2026/06/25\]** 🔥 现已支持 GR00T-RTC 加速版本，在 RTX 5090 设备上可达到 45 Hz。

**\[2026/06/22\]** 🔥 现已提供 Oli 人形机器人全身（移动操作）真机推理最小链路，包括 operator、runner 与示例配置。详见 [docs/oli_whole_body.md](docs/oli_whole_body.md)。

**\[2026/06/17\]** 🔥 现已支持 ARM 奖励建模与 RA-BC/AW-BC 重加权。配置与使用方法见 [docs/arm.md](docs/arm.md)。

**\[2026/06/10\]** 🔥 现已支持基于 GR00T 的 RoboCasa GR1 仿真任务。

**\[2026/06/04\]** 🔥 现已支持 Pi0.5-RTC 的 Triton 后端，详见 [inference_acceleration](docs/inference_acceleration.md)。

**\[2026/05/28\]** 🔥 正式发布面向双臂操作的模型解耦 DAgger 流水线 [FluxDAgger](https://github.com/FluxVLA/FluxDAgger)，便于接入不同 VLA 与奖励模型。

**\[2026/05/28\]** 🔥 正式发布具身操作仿真 Benchmark [FluxBisim](https://github.com/FluxVLA/FluxBisim)。

**\[2026/05/09\]** 🔥 现已支持 SmolVLA。

**\[2026/04/24\]** 🔥 现已支持 Pi0.5-RTC。

**\[2026/04/22\]** 🔥 现已支持基于 ZMQ 的远程推理框架。

**\[2026/04/15\]** 🔥 现已支持世界动作模型 DreamZero。

**\[2026/04/08\]** 🔥 FluxVLA开源了。

## 🛠️ 安装

请选择以下安装路径之一：

- **推荐：一键安装脚本**：适用于常规训练、仿真评估和真机推理环境。
- **更新已有 FluxVLA 环境**：适用于已经安装过早期 FluxVLA，只需要刷新变更依赖的环境。
- **从零手动安装**：仅在你需要完全控制每一步包安装时使用。

### 推荐：一键安装脚本

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla

# 选择一种模式：sim-only、real-only 或 full。
bash scripts/install_env.sh sim-only
# bash scripts/install_env.sh real-only
# bash scripts/install_env.sh full
```

<details>
<summary><b>如果安装脚本遇到问题：检查安装模式和 CUDA profile 选择</b></summary>

`sim-only` 安装仿真 / LIBERO / LIBERO-Plus / RoboCasa 相关依赖，并默认把固定版本的 RoboCasa 源码 checkout 放到 `./src`；`real-only` 安装真机和远程推理依赖，`full` 两者都安装。`requirements-sim.txt` 将标准 LIBERO 安装为 distribution / Python namespace `libero`，将兼容版 LIBERO-Plus 安装为独立的 distribution `libero-plus` 和 namespace `libero_plus`。二者可在同一个 Conda 环境中共存：普通评测命令使用 `libero`，`scripts/eval_libero_plus_manager.sh` 使用 `libero_plus`。如果不需要 RoboCasa checkout，可以加 `--skip-robocasa`。安装器默认还会下载并校验 LIBERO-Plus 的 6.4 GB benchmark 资产；如需延后下载，可传 `--skip-libero-plus-assets`。

只要安装器安装 RoboCasa 源码 checkout（`sim-only`、`full` 或 `real-only --with-robocasa`），默认也会下载 RoboCasa 仿真资产。安装器会调用 `scripts/download_robocasa_assets.py`，并使用 `FLUXVLA_ROBOCASA_ASSET_ENDPOINT`（默认依次取 `HF_ENDPOINT`、`https://hf-mirror.com`）。如只想跳过资产下载，可使用 `--skip-robocasa-assets`；如需同时跳过源码 checkout 和资产，可使用 `--skip-robocasa`。

安装器会优先根据当前 CUDA toolkit / `nvcc` 版本自动选择 PyTorch CUDA profile：CUDA >= 12.8 选择 `cu128`，否则选择 `cu124`。如果没有检测到 toolkit，则回退到驱动报告的 CUDA 版本和 GPU 代际。也可以通过 `--profile cu128` 或 `--profile cu124` 手动覆盖。

PyTorch 安装完成后，FlashAttention wheel 会根据实际 Python tag、PyTorch 版本、CUDA 主版本、C++ ABI 和 CPU 架构选择。如果当前平台没有匹配的预编译 wheel，可显式设置 `FLASH_ATTN_WHEEL_URL`，或传入 `--skip-flash-attn`。

`av` 默认优先从 pip wheel 安装，以避免 conda 依赖解析过慢；如果没有可用 wheel，安装器会回退到 conda。如需强制使用 conda-forge 包，可设置 `FLUXVLA_AV_INSTALLER=conda`。

真机 runner 仍然依赖系统 ROS 安装本身。在 ROS Noetic 机器上，启动推理前请先 source ROS：

```bash
source /opt/ros/noetic/setup.bash
```

</details>

<details>
<summary><b>如果安装脚本遇到问题：使用缓存或镜像的 FlashAttention wheel</b></summary>

FlashAttention wheel 文件较大，GitHub release 下载可能成为慢网络下的主要耗时。重复安装时，可把精确匹配的 wheel 文件放到 `./wheelhouse/`、`./wheels/` 或 `~/.cache/fluxvla/wheels/`；安装器会优先使用这些本地缓存。也可以指向本地文件或内部镜像：

```bash
FLASH_ATTN_WHEEL_FILE=/path/to/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
bash scripts/install_env.sh sim-only --profile cu128

FLASH_ATTN_WHEEL_BASE_URLS="https://your-mirror.example.com/fluxvla/wheels" \
bash scripts/install_env.sh sim-only --profile cu128
```

</details>

<details>
<summary><b>如果安装脚本遇到问题：自定义 pip 镜像和超时时间</b></summary>

安装器会优先尊重你已有的 pip 配置。如果该索引缺包，或没有配置 pip 索引，安装器会探测 PyPI 和几个常见镜像，并按响应情况重试，而不是全局固定到某一个镜像。对于慢速或不稳定网络，可以自定义候选索引和超时时间：

```bash
PIP_INDEX_CANDIDATES="https://mirrors.aliyun.com/pypi/simple https://mirrors.cloud.tencent.com/pypi/simple https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple" \
PIP_INSTALL_TIMEOUT=7200 \
PIP_NETWORK_TIMEOUT=900 \
GH_PROXY=https://ghfast.top \
bash scripts/install_env.sh full
```

</details>

### 更新已有 FluxVLA 环境

如果你已经 clone 并安装过 FluxVLA(v0.1.0)，通常不需要重建 conda 环境。拉取最新代码后，仅更新当前仿真 / 模型栈里版本发生变化的包即可：

```bash
bash scripts/update_env.sh
```

如果你已经手动更新了代码，可传入 `--skip-pull`；如果不想重新以 editable 模式安装 FluxVLA，可传入 `--skip-project`。

<details>
<summary><b>等价的手动命令</b></summary>

```bash
git pull
python -m pip install --upgrade "transformers==5.3.0" "datasets==4.0.0"
python -m pip install "numpy==1.26.4" "mujoco==3.2.6" gymnasium lxml bddl==1.0.1 hydra-core==1.2.0 robomimic==0.2.0 "Wand==0.7.2" "scikit-image==0.25.2"
python -m pip install --force-reinstall --no-deps "libero @ git+https://github.com/yinchimaoliang/LIBERO.git@058fda1ddebe92918af091cb6816759ca6d003f0"
python -m pip install --no-deps -e "git+https://github.com/hqr-robotic/LIBERO-plus.git@223cb493e7046698545a0b71bf539d2abfc211f4#egg=libero-plus"
python -m pip install --force-reinstall --no-deps "robosuite @ git+https://github.com/yinchimaoliang/robosuite.git@e293cc32ff3c48957a4ebcad09952432b0dc9049"
python scripts/download_libero_plus_assets.py
python -m pip install --no-build-isolation -e .
python -c "import transformers; print(transformers.__version__)"
```

</details>

RoboCasa GR00T 支持仍然是可选项。安装器会在 `sim-only` 和 `full` 模式下管理 `./src` 下的 Isaac-GR00T 与 RoboCasa GR1 本地 checkout；如果不使用 RoboCasa 配置，可传入 `--skip-robocasa`。

环境更新脚本不会重新安装 PyTorch 或 FlashAttention。已有 `flash-attn==2.5.5` 的环境，只有在它仍能匹配当前 PyTorch/CUDA 构建并成功导入时才建议继续使用：

```bash
python - <<'PY'
import torch, flash_attn
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash-attn", flash_attn.__version__)
PY
```

如果你用当前安装器或下面的手动命令升级了 PyTorch，也需要重新安装匹配的 FlashAttention wheel。安装器当前对支持的 PyTorch profile 默认使用 `flash-attn==2.8.3.post1`。

### 从零手动安装

仅当你不使用 `scripts/install_env.sh` 时，才建议走手动路径。请先安装 PyTorch，再安装 FlashAttention，最后安装 FluxVLA 的其余依赖。

<details>
<summary><b>1. 创建 conda 环境</b></summary>

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
```

</details>

<details>
<summary><b>2. 安装 PyTorch（CUDA 版本）</b></summary>

> **重要**：在执行 `pip install -r requirements.txt` 之前，**必须**先从官方 CUDA 索引安装 PyTorch。默认 PyPI 索引无法获取 CUDA 版本构建。

```bash
# CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

对于其他 CUDA 版本，请将 `cu128` 替换为对应值（例如 `cu118`、`cu121`）。详见 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 和 [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)。

</details>

<details>
<summary><b>3. 安装 flash-attention</b></summary>

一键安装器会从官方 release assets 下载预编译 FlashAttention wheel。手动安装时，请安装与你的 Python、PyTorch 和 C++ ABI 匹配的 wheel，而不是从源码构建：

```bash
PYTAG=$(python - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)
ABI=$(python - <<'PY'
import torch
print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())
PY
)

pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
```

如果安装的是 PyTorch 2.6，请把 wheel URL 中的 `torch2.8` 替换为 `torch2.6`。

FlashAttention wheel 与已安装的 Python、PyTorch、CUDA 和 C++ ABI 强绑定。`flash-attn==2.5.5` 并非禁止使用，但只有在它是为当前 PyTorch/CUDA 栈精确构建时才安全。任何 PyTorch 升级之后，都应重新安装匹配的 FlashAttention wheel。

</details>

<details>
<summary><b>4. 安装 av</b></summary>

```bash
conda install -c conda-forge av=14.4.0
```

</details>

<details>
<summary><b>5. 安装 fluxvla 及其余依赖</b></summary>

```bash
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

> **说明**：`requirements.txt` 当前组合了 `requirements-base.txt`、`requirements-sim.txt` 和 `requirements-real.txt`，不会安装 PyTorch；请先安装 CUDA PyTorch，或直接使用 `scripts/install_env.sh`。

> **LIBERO-Plus**：`requirements-sim.txt` 会在当前 FluxVLA 环境中同时安装
> `libero`、固定版本的 `libero-plus`，以及 `Wand` / `scikit-image`。两者的
> Python namespace 分别为 `libero` 与 `libero_plus`，不会互相覆盖。不要安装
> LIBERO-Plus 上游的 `requirements.txt`，否则其旧版本依赖会覆盖 FluxVLA 栈。

</details>

<details>
<summary><b>Jetson Orin Docker 配置</b></summary>

Jetson Orin 从刷机到 JetPack 初始化见 [docs/orin_flashing_zh-CN.md](docs/orin_flashing_zh-CN.md)，FluxVLA Docker 环境启动与运行测试见 [docs/orin_docker_runtime_zh-CN.md](docs/orin_docker_runtime_zh-CN.md)。

当前验证过的 Orin 运行环境已发布为 Docker 镜像：

```bash
docker pull fluxvla/fluxvla:fluxvla-orin-1.0.0
scripts/run_docker.sh
```

`scripts/run_docker.sh` 默认使用该镜像，并把当前仓库挂载到容器内 `/workspace/FluxVLA`。详细运行方式见 [docs/orin_docker_runtime_zh-CN.md](docs/orin_docker_runtime_zh-CN.md)。

</details>

<details>
<summary><b>RoboCasa GR00T 源码 checkout（可选）</b></summary>

RoboCasa GR00T 配置（如 `configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py`）需要固定版本的 Isaac-GR00T 与 RoboCasa GR1 任务 checkout。一键安装器会在 `sim-only` 和 `full` 模式下默认处理这些源码，并放到 `./src`：

```bash
bash scripts/install_env.sh sim-only
```

可用 `FLUXVLA_ROBOCASA_SRC_ROOT=/path/to/src` 指定其他 checkout 根目录，用 `--skip-robocasa` 跳过这些源码安装，用 `--with-robocasa` 在 `real-only` 模式下强制安装。运行时依赖与打过补丁的 robosuite 会从 `requirements-sim.txt` 安装。

如果不使用安装器，等价的手动命令如下：

```bash
pip install "mujoco==3.2.6" gymnasium lxml
pip install "robosuite @ git+https://github.com/yinchimaoliang/robosuite.git@e293cc32ff3c48957a4ebcad09952432b0dc9049"

git clone https://github.com/NVIDIA/Isaac-GR00T.git ./src/Isaac-GR00T
git -C ./src/Isaac-GR00T checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
pip install --no-deps -e ./src/Isaac-GR00T

git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git \
  ./src/robocasa-gr1-tabletop-tasks
git -C ./src/robocasa-gr1-tabletop-tasks checkout 4840e671596f93ca03651524b9f72ffb1aadfeff
pip install --no-deps -e ./src/robocasa-gr1-tabletop-tasks
```

可编辑安装建议加 `--no-deps`，避免 RoboCasa 相关包替换掉 FluxVLA 模型栈已固定的依赖。RoboCasa 的资产与数据集准备见[数据与资产准备](#数据与资产准备)。

</details>

<details>
<summary><b>在线评估环境（LIBERO / EGL）</b></summary>

如果你要在不支持光线追踪的设备（如 A100）上评估 LIBERO，请参考 [EGL Device GPU Rendering Configuration](https://github.com/google-deepmind/mujoco/issues/572#issuecomment-2419965230)。

`scripts/install_env.sh sim-only` 与 `scripts/install_env.sh full` 现在会自动探测 MuJoCo EGL。如果不可见 EGL 设备，安装器会尝试安装下方系统包、创建 NVIDIA GLVND vendor 文件，并写入用于设置 `MUJOCO_GL=egl` 的 conda activation hook。使用 `FLUXVLA_EGL_SETUP=always` 可让该检查变为严格模式，也可用 `--skip-egl-setup` 跳过。

**安装系统依赖**

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
sudo apt-get update
sudo apt-get install -y libegl1 libglvnd0 libopengl0 libegl-dev libgl1-mesa-dev libx11-dev libglew-dev libosmesa6-dev
```

**环境检查**

确认 `/proc/1/environ` 中包含以下环境变量：

- `NVIDIA_DRIVER_CAPABILITIES=all`
- `NVARCH=x86_64`
- `NVIDIA_REQUIRE_CUDA=cuda>=12.4`
- `brand=tesla` 且 `driver>=470`

**创建 EGL 配置文件**

创建文件 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`，内容如下：

```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
```

除非环境中已经导出了该变量，否则启动评测时请设置 `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`。

</details>

<details>
<summary><b>配置 pre-commit 钩子（可选但推荐）</b></summary>

为保证代码质量与一致性（尤其是 C++/CUDA 代码），建议安装 pre-commit 钩子：

```bash
pip install pre-commit
pre-commit install
```

这样会在每次提交前自动检查并格式化代码。

</details>

<details>
<summary><b>配置 Weights & Biases（wandb）</b></summary>

[Weights & Biases](https://wandb.ai/) 用于实验跟踪与可视化。配置方式如下：

1. 安装 wandb（已包含在 requirements.txt 中）：

```bash
pip install wandb
```

2. 登录你的 wandb 账号：

```bash
wandb login
```

3. 设置环境变量：

```bash
export WANDB_PROJECT=fluxvla        # 项目名（默认：fluxvla）
export WANDB_ENTITY=your-team-name  # 团队名或用户名（默认：None）
export WANDB_MODE=online            # online、offline 或 disabled（默认：online）
```

4. 如需在训练时禁用 wandb 日志，请设置：

```bash
export WANDB_MODE=disabled
```

说明：所有 wandb 配置都通过环境变量读取，无需在配置文件中额外设置。

</details>

<details>
<summary><b>配置 TensorBoard（可选）</b></summary>

[TensorBoard](https://www.tensorflow.org/tensorboard) 作为可选的日志后端，用于实验指标可视化。配置方式如下：

1. 在配置文件中将 `'tensorboard'` 添加到 `active_trackers`：

```python
metric=dict(
    type='VLAMetric',
    active_trackers=('jsonl', 'wandb', 'tensorboard'),
    ...
)
```

也可以不修改配置文件，通过命令行参数启用：

```bash
--cfg-options 'runner.metric.active_trackers=[jsonl,wandb,tensorboard]'
```

2. 训练完成后，启动 TensorBoard 查看指标：

```bash
tensorboard --logdir work_dirs/tensorboard
```

说明：每次实验的事件文件保存在 `{work_dir}/tensorboard/{run_id}/` 目录下，多次实验可自动对比。若设置了 `TENSORBOARD_LOG_PATH` 环境变量，将直接使用该路径作为日志目录。

</details>

## 数据与资产准备

<details>
<summary><b>直接使用我们准备好的数据</b></summary>

下载所需数据集并放到 `./datasets` 目录。请根据配置仅下载你需要的数据集。

| 数据集                  | 下载链接                                                                                                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| libero-object           | [limxdynamics/FluxVLAData/libero_object_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_object_no_noops_lerobotv2.1)         |
| libero-spatial          | [limxdynamics/FluxVLAData/libero_spatial_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_spatial_no_noops_lerobotv2.1)       |
| libero-10               | [limxdynamics/FluxVLAData/libero_10_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_10_no_noops_lerobotv2.1)                 |
| libero-goal             | [limxdynamics/FluxVLAData/libero_goal_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_goal_no_noops_lerobotv2.1)             |
| RoboCasa GR1 (30 demos) | [limxdynamics/FluxVLAData/robocasa_gr1_24tasks_first30ep](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/robocasa_gr1_24tasks_first30ep)                 |
| RoboCasa GR1            | [limxdynamics/FluxVLAData/robocasa_lerobot_V2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/robocasa_lerobot_V2.1)                                   |
| ARM manual test         | [limxdynamics/FluxVLAData/ARM_manual_test_10Episodes_lerobotv3.0](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/ARM_manual_test_10Episodes_lerobotv3.0) |
| RealRobot_AgileX_aloha  | [limxdynamics/FluxVLAData/RealRobot_AgileX_aloha_lerobot_v2](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/RealRobot_AgileX_aloha_lerobot_v2)           |
| RealRobot_UR3_Chem      | [limxdynamics/FluxVLAData/RealRobot_UR3_Chem_lerobot_v2](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/RealRobot_UR3_Chem_lerobot_v2)                   |

例如，下载 libero-10 数据集：

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "libero_10_no_noops_lerobotv2.1/*" --local-dir ./datasets
```

将 `libero_10_no_noops_lerobotv2.1` 替换为其他数据集对应的文件夹名即可下载。

如需用已发布的 30 条演示子集训练 RoboCasa GR00T，将数据集下载到 `./datasets`：

```bash
huggingface-cli download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include "robocasa_gr1_24tasks_first30ep/*" \
  --local-dir ./datasets
```

如需使用全量 RoboCasa GR1 数据训练，将 include 模式替换为 `robocasa_lerobot_V2.1/*`。

</details>

<details>
<summary><b>LIBERO-Plus benchmark 资产</b></summary>

`scripts/install_env.sh sim-only` 和 `scripts/install_env.sh full` 默认会安装固定的
`libero-plus` editable distribution，并下载官方 6.4 GB `assets.zip`。下载脚本会
自动定位已安装 checkout、校验 SHA-256、提取嵌套资产目录、检查物体 / 场景 /
纹理文件，并在 checkout 旁写入 `.libero/config.yaml`。Plus manager 会自动选择
该配置，标准 LIBERO 命令仍使用原来的 `~/.libero/config.yaml`。

如果不使用一键安装器，请先安装 ImageMagick/Wand 系统库：

```bash
sudo apt-get update
sudo apt-get install -y \
  libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev unzip
```

如果安装时传了 `--skip-libero-plus-assets`，可稍后执行：

```bash
python scripts/download_libero_plus_assets.py
```

已有 checkout 或缓存时可显式指定：

```bash
python scripts/download_libero_plus_assets.py \
  --libero-plus-root /path/to/LIBERO-plus \
  --cache-dir /path/to/cache
```

仅校验已安装资产、不下载：

```bash
python scripts/download_libero_plus_assets.py --validate-only
```

最终的 `libero_plus/libero/assets` 应包含 `articulated_objects/`、`new_objects/`、
`scenes/`、`textures/`、`serving_region.xml` 等官方 benchmark 资产。

</details>

<details>
<summary><b>ARM 数据集</b></summary>

内置 ARM 示例配置 `configs/arm/arm_clip_aloha_example.py` 期望带有 progress 标签的 LeRobot v3.x 数据位于 `./datasets/ARM_manual_test_10Episodes_lerobotv3.0`。

可通过以下命令下载到对应位置：

```bash
huggingface-cli download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include "ARM_manual_test_10Episodes_lerobotv3.0/*" \
  --local-dir ./datasets
```

ARM 训练会直接读取该数据集中的 `progress` 列。若需要在没有 `progress` 的 policy / DAgger 数据集上做 RA-BC / AW-BC，请先训练或加载 ARM checkpoint，再用 `scripts/compute_arm_awbc_progress.py` 生成 `arm_progress.parquet`。更多说明见 [docs/arm.md](docs/arm.md) 和 [tools/arm_awbc/README.md](tools/arm_awbc/README.md)。

</details>

<details>
<summary><b>准备资产</b></summary>

请使用下面的 FluxVLA 资产下载器作为 RoboCasa GR1 tabletop tasks 的受支持路径。表中列出了脚本使用的上游压缩包；仅手动下载和解压这些压缩包并不足够，因为脚本还会修正目录布局，并为固定版本的 RoboCasa GR1 checkout 规范化 Objaverse XML 元数据。

| 资产压缩包                                                 | 下载链接                                                                                                         | 本地目录                                                   |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `objaverse.zip`, `textures.zip`, `generative_textures.zip` | [robocasa/robocasa-assets](https://huggingface.co/datasets/robocasa/robocasa-assets)                             | `./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets` |
| `fixtures.zip`                                             | [jianzhang96/robocasa-assets](https://huggingface.co/datasets/jianzhang96/robocasa-assets)                       | `./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets` |
| `sketchfab.zip`, `lightwheel.zip`                          | [nvidia/PhysicalAI-DigitalCousin-Assets](https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets) | `./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets` |

使用 `scripts/install_env.sh` 时，除非传入 `--skip-robocasa` 或 `--skip-robocasa-assets`，否则该下载器会随 RoboCasa 源码 checkout 一起默认运行。手动安装或刷新资产时，请在 FluxVLA 仓库根目录运行以下命令。它会通过指定 Hugging Face endpoint 下载所需压缩包、解压到 RoboCasa 资产目录，并规范化 Objaverse XML 元数据：

```bash
python scripts/download_robocasa_assets.py --endpoint https://hf-mirror.com
```

即使压缩包或解压后的资产已经存在本地，也建议运行该脚本，以便应用 XML 兼容性处理。如果资产已经解压到 `./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets`，可以只运行验证和 XML 规范化步骤：

```bash
python scripts/download_robocasa_assets.py --normalize-only
```

软链接不是必须的；只有当资产已经位于其他本地磁盘或共享存储时，软链接才是一种便利手段。

</details>

<details>
<summary><b>SARM 数据集</b></summary>

FluxVLA 的 SARM 工作流支持标准 LeRobot v2.1 与 v3.x 数据集。除常规 observation / action 字段外，数据集还需要在 episodes 元信息里带有 SARM subtask 标注列。

已发布到 Hugging Face 的 SARM 示例数据集：

- LeRobot v3.x 版、用于训练 / 推理的完整人工 sparse+dense 标注数据：[limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv3.0](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv3.0)
- LeRobot v3.x 版、供手工或 VLM 继续标注的无标注数据：[limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv3.0](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv3.0)
- 新增的 LeRobot v2.1 manual 转换版，可直接用于训练 / 推理，也适合需要旧版目录结构的工具链：[limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv2.1)
- 新增的 LeRobot v2.1 vlm 转换版，作为手工补 stage 或 VLM 自动标注的干净起点：[limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv2.1)

可通过以下命令下载到 `./datasets`：

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv2.1/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv2.1/*" --local-dir ./datasets
```

`manual_*` 两份数据可以直接接训练 / 推理；`vlm_*` 两份数据适合作为手工写 stage 或 VLM 自动标注的起点。如果下游工具假定存在 `meta/episodes.jsonl` 和逐集视频，优先使用 v2.1；如果你要保留原生 LeRobot v3.x 的元信息布局，优先使用 v3.0。

在使用 LeRobot v3.x 的 SARM 数据集前，建议先做一次视频元信息自检：

- LeRobot v3.x 既允许多个 episode 共用一个 MP4，也允许一个 episode 对应一个 MP4。

- 如果多个 episode 共用同一个 MP4，那么每个 episode 的 `from_timestamp` / `to_timestamp` 必须正确描述它在该视频中的片段区间。

- 如果视频本身已经拆成 `file-000.mp4`、`file-001.mp4` 这样的逐集文件，那么每个 episode 就应该指向各自的 `file_index`，且 `from_timestamp` 通常应回到 `0.0`。

- 如果目录里明明有多个 MP4，但所有 episode 仍都指向 `file-000.mp4`，那就是错误的 metadata，应先修正再使用。

- SARM 数据集目录、标注列契约与 progress 推理说明见 [docs/sarm.md](docs/sarm.md)。

- 手动写入 stage 或使用 VLM 自动标注见 [tools/sarm_annotate/README.md](tools/sarm_annotate/README.md)。

</details>

<details>
<summary><b>私有数据集目录结构</b></summary>

若使用 fluxvla 在私有数据集上训练，需要先将原始数据（如 ALOHA 双臂机器人采集的 HDF5 文件）转换为 LeRobot Dataset v2.1 格式。详细的转换步骤请参考 [数据转换指南](docs/data_convert.md)。

对 SARM 而言，只要补齐所需的 SARM 标注列，FluxVLA 同时兼容 LeRobot v2.1 与 v3.x 数据集。SARM 需要的元信息格式见 [docs/sarm.md](docs/sarm.md)。

转换后的数据集目录结构如下：

```
├── data
│   └── chunk-000
│   │   └── episode_000000.parquet
│   │   └── episode_000001.parquet
│   │   └── ... (更多 parquet 文件)
│   │   └── episode_00000N.parquet
│   └── chunk-001
│   └── ... (更多 chunk)
│   └── chunk-00N
├── meta
│   └── episodes.jsonl
│   └── episodes_stats.jsonl
│   └── info.json
│   └── tasks.jsonl
├── videos
│   └── chunk-000
│   │   └── camera name 0
│   │   │   └── episode_000000.mp4
│   │   │   └── episode_000001.mp4
│   │   │   └── ...(更多 mp4 文件)
│   │   │   └── episode_00000N.mp4
│   │   └── camera name 1
│   │   └── ...(更多相机)
│   │   └── camera name N
│   └── chunk-001
│   └── ... (更多 chunk)
│   └── chunk-00N
```

</details>

## 🤗 Checkpoint 准备

下载所需预训练 checkpoint 并放到 `./checkpoints` 目录。请根据配置仅下载你需要的 checkpoint。

如果使用 ARM 或 SARM 工作流，通常至少需要一个 CLIP checkpoint 用于训练 / 推理；如果要用 SARM VLM 自动标注，还需要官方 SARM 使用的 Qwen3-VL checkpoint。详细用法见 [docs/arm.md](docs/arm.md) 和 [docs/sarm.md](docs/sarm.md)。

<details>
<summary><b>VLA 模型</b></summary>

| 模型                      | 大小 | 下载链接                                                                                                                                       |
| ------------------------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| GR00T N1.5                | 3B   | [🤗 Hugging Face](https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/main)                                                                       |
| OpenVLA                   | 7B   | [🤗 Hugging Face](https://huggingface.co/openvla/openvla-7b)                                                                                   |
| PI0_base                  | 3B   | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_base)                                                        |
| PI05_base                 | 3B   | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_base)                                                       |
| PI05_libero               | 3B   | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_libero)                                                     |
| PI05 RoboCasa（全量数据） | 3B   | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_robocasa_full_data_full_finetune_21aa5e82a_bs256) |
| SmolVLA                   | 450M | [🤗 Hugging Face](https://huggingface.co/lerobot/smolvla_base)                                                                                 |

按 config 预期的目录结构下载 PI0.5 RoboCasa 全量数据 checkpoint：

```bash
hf download limxdynamics/FluxVLAEngine \
  --include "pi05_paligemma_robocasa_full_data_full_finetune_21aa5e82a_bs256/*" \
  --local-dir ./checkpoints
```

</details>

<details>
<summary><b>视觉语言模型（VLM）</b></summary>

| 模型       | 大小 | 下载链接                                                                             |
| ---------- | ---- | ------------------------------------------------------------------------------------ |
| Qwen2.5-VL | 3B   | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)                |
| Qwen3-VL   | 30B  | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)             |
| SmolVLM2   | 500M | [🤗 Hugging Face](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) |

</details>

<details>
<summary><b>大语言模型（LLM）</b></summary>

| 模型     | 大小 | 下载链接                                                                     |
| -------- | ---- | ---------------------------------------------------------------------------- |
| Qwen 2.5 | 3B   | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B)                    |
| Qwen 2.5 | 7B   | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B)                    |
| Llama 2  | 7B   | [🤗 Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) |

</details>

<details>
<summary><b>视觉主干网络</b></summary>

| 模型                | 下载链接                                                                             |
| ------------------- | ------------------------------------------------------------------------------------ |
| CLIP ViT-B/32       | [🤗 Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)               |
| ViT-Large (DINOv2)  | [🤗 Hugging Face](https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m) |
| ViT-SO400M (SigLIP) | [🤗 Hugging Face](https://huggingface.co/timm/ViT-SO400M-14-SigLIP)                  |
| SigLIP2             | [🤗 Hugging Face](https://huggingface.co/google/siglip2-base-patch16-224)            |
| paligemma           | [🤗 Hugging Face](https://huggingface.co/google/paligemma-3b-pt-224)                 |

> **提示**：可使用 `huggingface-cli download <model-name> --local-dir ./checkpoints/<model-name>` 加速下载。

对于内置的 ARM 和 SARM 配置，请将 CLIP 文件放到 `./checkpoints/clip-vit-base-patch32`：

```bash
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./checkpoints/clip-vit-base-patch32
```

如果使用 VLM 自动标注，请将官方 SARM VLM 放到 `./checkpoints/Qwen3-VL-30B-A3B-Instruct`。

</details>

## 🌟 特性

<details>
<summary><b>All-in-one：单配置文件管理全流程</b></summary>

- 支持通过一个配置文件统一管理数据、模型、训练、评测、推理与部署所需的关键参数（便于复现与部署）。

</details>

<details>
<summary><b>支持不同 VLA 模型</b></summary>

- 支持 OpenVLA、LlavaVLA、Gr00t、Pi0 与 Pi0.5。

</details>

<details>
<summary><b>支持不同模块</b></summary>

- 支持 Llama、Gemma 与 Qwen 系列 LLM 主干。
- 支持 DINOv2、SigLIP 视觉主干。
- 支持 PaliGemma 与 Qwen-VL VLM 主干。

</details>

<details>
<summary><b>支持奖励建模工作流</b></summary>

- 支持 [SARM](https://github.com/xdofai/opensarm) 的训练、标注与 progress 推理，并兼容 LeRobot v2.1/v3.x 数据集。详情见 [docs/sarm.md](docs/sarm.md)。
- 支持 [ARM](https://arxiv.org/abs/2604.03037) 奖励建模、progress 重建以及 RA-BC / AW-BC 样本重加权。详情见 [docs/arm.md](docs/arm.md)。

</details>

<details>
<summary><b>支持不同训练策略</b></summary>

- 支持同时使用 FSDP 与 DDP，支持 LoRA 训练模式。
- 支持 train 后立即 eval（eval-after-train）。
- 支持从 checkpoint 恢复训练。

</details>

<details>
<summary><b>数据与权重格式</b></summary>

- 支持 Parquet 数据集，并支持加载 LeRobot 格式数据。
- 支持 safetensors 格式模型权重。

</details>

<details>
<summary><b>评估与推理能力</b></summary>

- 支持多 GPU 在无光追设备上评估 libero。
- 支持可持久化、可恢复的 LIBERO-Plus 评测，并对全部 10,030 个鲁棒性任务执行严格聚合。
- 支持将 LIBERO、LIBERO-Plus 和 RoboCasa 评估汇总自动写入飞书电子表格。详见 [Feishu Evaluation Reporting](docs/feishu_eval_reporting.md)。
- 支持基于 ZMQ 通信框架的远程推理设施，利用 server/client 架构将模型推理负载装卸到服务器端，适用于算力受限的边缘设备部署。详见 [远程推理服务](docs/remote_inference_serving.md)。
- 支持 [RTC (Real-Time Chunking)](docs/rtc.md)，提升跨 chunk 轨迹连续性。
- 支持 GR00T 与 PI0.5 推理加速；详见 [Inference Acceleration](docs/inference_acceleration.md)，包含 Triton 融合核、CUDA Graph 捕获与 CUDA 自定义算子。
- 提供 Oli 人形机器人全身（移动操作）真机推理最小链路（rospy 传感器输入 + WebSocket 控制；底盘 / 手部命令为机器人 SDK 集成点）。详见 [docs/oli_whole_body.md](docs/oli_whole_body.md)。

</details>

<p align="center">
  <img src="assets/VLA_speedup.png" alt="VLA Speedup" width="800">
</p>

## 使用方式

<details>
<summary><b>本地调试</b></summary>

```
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/train.py --config [CONFIG_PATH] --work-dir [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE]
```

例如：

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py --work-dir ./checkpoints/pi05_paligemma_libero_10_full_finetune --cfg-options train_dataloader.per_device_batch_size=2
```

RoboCasa GR00T 冒烟训练示例：

```bash
WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --work-dir work_dirs/smoke_groot_robocasa_train \
  --cfg-options \
    runner.type=FSDPTrainRunner \
    runner.sharding_strategy=no-shard \
    train_dataloader.per_device_batch_size=1 \
    runner.enable_gradient_checkpointing=False \
    runner.max_steps=2 \
    runner.save_iter_interval=1 \
    runner.max_keep_ckpts=2 \
    "runner.metric.active_trackers=('jsonl',)"
```

</details>

<details>
<summary><b>本地评估</b></summary>

```
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/eval.py --config [CONFIG_PATH] --ckpt-path [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

例如：

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/eval.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors
```

RoboCasa GR00T 评估示例：

```bash
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
PYTHONHASHSEED=7 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --ckpt-path work_dirs/gr00t_eagle_3b_robocasa_gr1_24x30_finetune_bs64/checkpoints/step-010000.safetensors \
  --cfg-options \
    eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
    eval.output_dir=work_dirs/gr00t_eagle_3b_robocasa_eval \
    eval.num_trials_per_task=50 \
    eval.seed=7
```

`eval.seed` 控制 RoboCasa episode seeds 以及评估过程中 GR00T 随机 action sampling seeds。`PYTHONHASHSEED` 与其相互独立，必须在 Python 启动前设置；复现已报告的 RoboCasa 结果时，建议使用相同数值。

</details>

<details>
<summary><b>LIBERO-Plus 鲁棒性评测</b></summary>

LIBERO-Plus 在相机视角、机器人初始状态、语言、光照、背景、传感器噪声和
物体布局七个扰动维度上提供 10,030 个任务。`requirements-sim.txt` 会在同一个
FluxVLA 环境中安装标准 `libero` namespace 与独立的 `libero_plus` namespace。
确认资产完整后，每次启动一个 suite：

|                  | 标准 LIBERO                                          | LIBERO-Plus                           |
| ---------------- | ---------------------------------------------------- | ------------------------------------- |
| pip distribution | `libero`                                             | `libero-plus`                         |
| Python namespace | `libero`                                             | `libero_plus`                         |
| 任务数           | 四个 suite 共 40 个                                  | 四个 suite 共 10,030 个               |
| 每任务试验次数   | 由 config 决定，通常为 50                            | 固定为 1                              |
| 入口             | `scripts/eval.py` / `scripts/eval_libero_manager.sh` | `scripts/eval_libero_plus_manager.sh` |
| 额外运行依赖     | 无                                                   | `Wand`、`scikit-image`、ImageMagick   |

```bash
python scripts/download_libero_plus_assets.py --validate-only

export OUTPUT_DIR=work_dirs/libero_plus_pi05
export CUDA_VISIBLE_DEVICES=0,1
export WORKERS_PER_GPU=1
export RESUME=1
export WANDB_MODE=disabled

CONFIG=configs/pi05/pi05_paligemma_libero_spatial_full_finetune.py \
CKPT=/path/to/pi05-libero-spatial-checkpoint.safetensors \
SUITE=libero_spatial \
bash scripts/eval_libero_plus_manager.sh
```

使用同一个 `OUTPUT_DIR`，分别用匹配的 config 与 checkpoint 继续运行
`libero_object`、`libero_goal` 和 `libero_10`。manager 会强制执行官方的
每任务一次试验协议，按 worker 原子保存结果，并在 `RESUME=1` 时跳过已完成任务。
冒烟测试可设置 `TASK_IDS=0,1,2`；完整评测时不设置。`DRY_RUN=1` 只检查任务
分配，不加载模型。

worker 结果和日志位于
`$OUTPUT_DIR/<suite>/{worker_results,worker_logs,worker_status,worker_tasks}`。
中断后使用相同 config、checkpoint、suite 和 `OUTPUT_DIR` 再次启动即可恢复。

四个 suite 全部结束后，生成严格的 leaderboard 汇总：

```bash
python tools/summarize_libero_plus_results.py \
  --run-root "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --title PI0.5
```

汇总工具会从已安装的 `libero_plus` 自动读取 `task_classification.json`，并拒绝
重复任务、未知任务、缺失任务及非单次试验结果。除标准 LIBERO 汇总外，还会
生成 `libero_plus_summary.{csv,json,txt}`、`libero_plus_by_suite.csv` 和
`libero_plus_missing_tasks.csv`。leaderboard 列为
`Camera/Robot/Language/Light/Background/Noise/Layout/Total`，其中 `Total` 对全部
task trial 做微平均。正式对比要求全部 10,030 个任务完整；
`--allow-incomplete` 仅用于冒烟测试和进度检查，partial 结果不会上报飞书。

</details>

<details>
<summary><b>集群训练</b></summary>

```
export WANDB_MODE=disabled
bash scripts/train.sh [CONFIG] [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] train_dataloader.batch_size=[GLOBAL_BATCH_SIZE] runner.max_steps=[MAX_STEPS] runner.save_interval=[SAVE_INTERVAL] runner.max_keep_ckpts=[MAX_KEEP_CKPTS] --eval-after-train
```

</details>

<details>
<summary><b>从 checkpoint 恢复训练</b></summary>

要从 checkpoint 恢复训练，可使用 `--resume-from` 参数指定 checkpoint 文件路径。训练会从已保存的 global step、epoch、模型状态与优化器状态继续。

**本地训练示例：**

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --work-dir ./work_dirs/pi05_paligemma_libero_10_full_finetune \
  --resume-from ./work_dirs/pi05_paligemma_libero_10_full_finetune/checkpoints/checkpoint_epoch_5.pt \
  --cfg-options train_dataloader.per_device_batch_size=2
```

**集群训练示例：**

```
export WANDB_MODE=disabled
bash scripts/train.sh [CONFIG] [WORK_DIR] \
  --resume-from [CHECKPOINT_PATH] \
  --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] runner.max_steps=[MAX_STEPS]
```

</details>

<details>
<summary><b>集群评估</b></summary>

```
export WANDB_MODE=disabled
bash scripts/eval.sh [CONFIG] [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

</details>

<details>
<summary><b>真机推理</b></summary>

在真实机器人上运行推理时，请先在机器人端安装好环境，然后执行以下命令：

```
python scripts/inference_real_robot.py --config [CONFIG] -- ckpt-path [CKPT_PATH]
```

</details>

## 常见问题（FAQ）

<details>
<summary><b>Q：下载模型或数据集时，连接 Hugging Face 有问题。</b></summary>

A：如果遇到 Hugging Face 连接问题（如下载慢、超时、连接被拒绝），可以在执行命令前设置以下环境变量，使用 [hf-mirror](https://hf-mirror.com)：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

</details>

<details>
<summary><b>Q：<code>conda install av</code> 解析环境很慢。</b></summary>

A：可使用 `libmamba` 求解器加速依赖解析：

```bash
conda install -c conda-forge av=14.4.0 --solver=libmamba
```

</details>

<details>
<summary><b>Q：GR00T 在 LIBERO 上评估结果不稳定。</b></summary>

A：这是预期现象。GR00T 在 LIBERO 上的表现对随机种子、硬件环境和训练 epoch 数都较敏感。这些因素的小变化都可能导致评估结果明显波动。建议使用多个随机种子进行实验，并依据评估表现选择最优 checkpoint。

</details>

<details>
<summary><b>Q：执行 <code>pip install -r requirements.txt</code> 时构建 <code>egl_probe</code> 失败，报错 <code>RuntimeError: CMake must be installed</code>。</b></summary>

A：`egl_probe` 需要 CMake 才能构建。可通过 conda（推荐）或 apt 安装：

```bash
conda install -c conda-forge cmake
# 或
sudo apt install cmake
```

> **说明**：不要使用 `pip install cmake`，pip 版本是 Python 封装，在 pip 隔离构建环境中可能失败。

</details>

<details>
<summary><b>Q：<code>egl_probe</code> 构建失败，提示 <code>Compatibility with CMake < 3.5 has been removed from CMake</code>。</b></summary>

A：这通常是因为你的 CMake 版本对 `egl_probe` 的 CMakeLists.txt 来说过新。安装前先设置以下环境变量：

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -r requirements.txt
```

</details>

<details>
<summary><b>Q：安装后出现 NumPy 版本错误（如 <code>RuntimeError: Numpy is not available</code> 或版本不兼容警告）。</b></summary>

A：安装过程中某些依赖可能覆盖了固定的 NumPy 版本。直接重装正确版本即可：

```bash
pip install numpy==1.26.4
```

</details>

## 贡献指南

贡献流程与规范请见：[贡献指南](docs/CONTRIBUTING.md#简体中文)。

快速约定：

- **先讨论再动手**：新功能/新模型/较大改动，优先在 GitHub Issue 里沟通设计与范围。
- **从上游主分支开新分支**：基于 `upstream/main` 创建分支，命名建议 `feat/`、`fix/`、`docs/` 等前缀（详见贡献指南）。
- **提交前跑检查**：确保本地 pre-commit 通过、CI 为绿后再提 PR。
- **提交信息规范**：建议使用 Conventional Commits（示例见贡献指南）。

## 支持

如果你在使用本仓库时遇到问题，欢迎联系我们。你可以直接联系 [mason@limxdynamics.com](mason@limxdynamics.com) 和 [wayne@limxdynamics.com](wayne@limxdynamics.com)，或在 Github 提交 issue 获取帮助。

## 🙏 引用与致谢

如果你在学术研究或工程项目中使用了 FluxVLA，欢迎引用以下工作：

```bibtex
@software{FluxVLA2026,
  author  = {Li, Yinhao and Mao, Weixin and Lan, Zihan and Rong, Jikun and Zhu, Minzhao and Mao, Yiming and Shen, Bowen and Huang, Xu},
  title   = {{FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence}},
  year    = {2026},
  month   = apr,
  version = {1.0.0},
  doi     = {10.5281/zenodo.20049506},
  url     = {https://github.com/FluxVLA/FluxVLA},
  license = {Apache-2.0},
}

@InProceedings{Mao_2026_CVPR,
    author    = {Mao, Yiming and Yu, Zixi and Mao, Weixin and Li, Yinhao and Hu, Qirui and Lan, Zihan and Zhu, Minzhao and Chen, Hua},
    title     = {ARM: Advantage Reward Modeling for Long-Horizon Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {4468-4477}
}

@article{Huang_2026_IROS,
  title={Long-Term Memory for VLA-based Agents in Open-World Task Execution},
  author={Huang, Xu and Mao, Weixin and Li, Yinhao and Chen, Hua and Zhao, Jiabao},
  journal={arXiv preprint arXiv:2604.15671},
  year={2026}
}
```

**致谢**：本项目受益于以下开源项目与社区工作，在此一并致谢：[LeRobot](https://github.com/huggingface/lerobot)、[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main)、[DreamZero](https://arxiv.org/abs/2602.15922)（[代码](https://github.com/dreamzero0/dreamzero)）、[OpenVLA](https://github.com/openvla/openvla)、[OpenPI (pi0)](https://github.com/Physical-Intelligence/openpi)、[LLaVA](https://github.com/haotian-liu/LLaVA)、[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)、[Qwen](https://github.com/QwenLM)、[Triton](https://github.com/triton-lang/triton)、[RTC](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)、[Training RTC](https://arxiv.org/pdf/2512.05964)、[Realtime-VLA](https://github.com/Dexmal/realtime-vla)。如果我们不慎遗漏了您的项目或贡献，请提交 issue 或 pull request，以便我们能够给予您应有的致谢。

## 路线图

- 支持更多视觉主干网络。
- 支持更多 VLM 主干。
- 支持更多 VLA 方法。
- 支持使用 VLM 数据或思维链（CoT）数据进行训练。
- logger 功能将完整实现。
- 支持 Isaac Sim。
