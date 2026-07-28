# Jetson Orin 刷机指南：从 Recovery Mode 到 JetPack 初始化

本文说明如何将 NVIDIA Jetson Orin 设备从零刷机到可用的 JetPack 6.2.x 系统状态，为下一步创建与进入 FluxVLA Docker 作准备。

适用设备：

- Jetson AGX Orin Developer Kit
- Jetson AGX Orin 模组 + 官方或兼容载板

> 本文以 JetPack 6.x / Jetson Linux r36.x 为主。不同 JetPack 小版本、不同载板的按键位置和刷机配置名可能不同，量产板请同时参考载板厂商 BSP 文档。

## 1. 准备刷机主机

刷机需要一台独立的 Linux x86_64 主机，不是在 Jetson 自己上执行。

> 刷机会重写目标存储上的系统分区。执行前请备份 Jetson 内部 eMMC、NVMe 或 USB 存储中的重要数据。

推荐主机环境：

- Ubuntu 20.04 或 Ubuntu 22.04 x86_64
- 至少 80 GB 可用磁盘空间
- 稳定网络
- 优先使用原装或高质量 USB-C 数据线连接 Orin 与主机
- 如果刷到外部 NVMe 或 USB 存储，推荐容量不低于 64 GB

安装基础工具：

```bash
sudo apt update
```

确认主机能看到 USB 设备：

```bash
lsusb
```

## 2. 刷机方式选择

推荐优先使用 **SDK Manager 刷机**，适合首次配置、开发板验证和安装 JetPack 组件。

SDK Manager Web : HTTPS://developer.nvidia.com/sdk-manager

<img src="../assets/SDK-Manager.png" alt="Jetson Orin" width="400">

选择 Ubuntu .deb x86_64 进行下载安装，方便后续使用 SDK Manager Cli 进行进一步系统配置。


## 3. 硬件连接

刷机前连接：

1. Jetson 电源。
2. Jetson 与主机之间的 USB 数据线。
3. 有线网络 / 无线网络，方便首次启动后安装组件。

终端连接无线网络：
```bash
nmcli device wifi list
nmcli device wifi connect "WiFi_name" password "WiFi_password"
# 检测连接状态
nmcli connection show
```
4. 主机访问 Jetson 测试：
- 串口连接：
```bash
sudo screen /dev/ttyACM0 115200
```
- ssh :
配置无线网络后，配置 sshd server 并设置开机时开启，将 Orin 作为 server 端使用。


### 3.1 Jetson AGX Orin Developer Kit

AGX Orin Developer Kit 刷机时，主机应连接到开发套件上用于 flashing/recovery 的 USB-C 口（靠近 40-pin header 的 USB-C 口）。

## 4. 进入 Force Recovery Mode

Force Recovery Mode 是 Jetson 接受刷机命令的模式。进入该模式后，Jetson 不会正常启动系统，而是通过 USB 暴露为 NVIDIA recovery 设备。

### 4.1 AGX Orin Developer Kit：按键方式

<img src="../assets/orin.png" alt="Jetson Orin" width="400">

图示中，'2' 为 Force Recovery 键，'3' 为 reset 键。

分为‘未开机’ 与 ‘开机’ 两种 Recovery 方式：
#### 4.1.1 未开机
1. 先长按 2 键（Force Recovery）
2. 然后给 Orin 接上电源线通电
3. 此时白色灯亮起，进入 Recovery 模式
#### 4.1.2 开机
在开机状态下，先长按 '2' 键（Force Recovery），再按下 '3' 键（Reset），先松开 '3' 键，再松开 '2' 键。

在主机上执行：

```bash
lsusb | grep -i nvidia
```

如果看到 7023 代码，说明已进入 Recovery Mode 且连接正常：

```text
Bus 001 Device 006: ID 0955:7023 NVIDIA Corp.
```

如果没有任何 NVIDIA 设备：

1. 换一根确认支持数据传输的 USB 线。
2. 换主机 USB 口，避免 USB hub。
3. 确认连接的是 recovery/device USB 口，不是普通 host USB 口。
4. 重新执行 Recovery Mode 步骤。

## 5. 使用 SDK Manager 刷机（推荐）

### 5.1 安装 SDK Manager

在 Ubuntu 主机下载并安装 NVIDIA SDK Manager：

```bash
sudo apt install ./sdkmanager_*_amd64.deb
```

启动：

```bash
sdkmanager --cli
```

登录 NVIDIA Developer 账号。

### 5.2 STEP 01：选择设备与 JetPack

在 SDK Manager 中选择：

1. Product Category：`Jetson`
2. Target Hardware：选择你的 Orin 设备，例如 `Jetson AGX Orin Developer Kit`
3. SDK Version：选择目标 JetPack 版本，例如 JetPack 6.x
4. Host Machine：如果只刷 Jetson，可按需取消 Host 侧组件
5. Additional SDKs：通常保持默认即可

    <img src="../assets/jetpack-version.png" alt="Jetson Orin" width="500">

点击 **Continue**。

### 5.3 STEP 02：选择组件并开始刷机

<img src="../assets/sdk-manager-components.png" alt="Jetson Orin" width="500">

建议至少安装：

- Jetson Linux
- CUDA
- cuDNN
- TensorRT
- VPI
- Jetson Runtime Components


接受 license 后继续，开始刷机。

### 5.4 首次启动

刷机完成后，Jetson 会自动重启。首次启动需要在显示器或串口控制台上完成：

1. 接受系统许可，选择语言、键盘布局、时区。
3. 创建用户名与密码。
4. 配置网络。

如果 SDK Manager 还需要继续安装 CUDA、TensorRT 等 target 组件，请确保 Jetson 与主机在同一网络，或按 SDK Manager Cli 提示完成连接。


## 6. 刷机完成后的系统检查

首次进入 Jetson 后执行：

```bash
cat /etc/nv_tegra_release
dpkg-query --show nvidia-l4t-core
uname -a
```

检查 CUDA：

```bash
nvcc --version
```

检查 GPU/系统状态：

```bash
sudo tegrastats
```

期望结果：
- /etc/nv_tegra_release 显示 R36.4.x，当前目标记录为 R36.4.4。
- Ubuntu 为 22.04。
- nvcc 为 CUDA 12.6 系列。

检查 Docker 与 NVIDIA runtime：

```bash
docker --version
sudo docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.4.0 \
  bash -lc "cat /etc/nv_tegra_release"
```

## 7. 安装 jtop （类似于 NVITOP ）

```bash
#  Python install
sudo apt update
sudo apt install python3
sudo apt install python3-pip

#  jtop install
sudo pip3 install -U pip
sudo pip3 install jetson-stats
```

## 8. 下一步配置 FluxVLA Orin Docker

刷机完成并确认 JetPack 正常后，可以继续进行 FluxVLA Orin Docker 配置：

```bash
git clone https://github.com/limxdynamics/FluxVLA.git
cd FluxVLA
docker/build_docker.sh
docker/run_docker.sh
```

详细 Docker 使用见：

- `docker/README_DOCKER_ORIN.md`
- `docker/DOCKER_VERSIONING.md`

## 9. Q&A

### Q1：`lsusb` 看不到 NVIDIA 设备怎么办？

A：优先检查 USB 线是否支持数据传输、是否连接到 recovery/device USB 口、Recovery Mode 操作顺序是否正确、是否使用了不稳定的 USB hub。若使用第三方载板，还需要确认 RECOVERY 引脚是否真的被拉低到 GND。

### Q2：SDK Manager 卡在等待设备怎么办？

A：保持 SDK Manager 不关闭，重新让 Jetson 进入 Recovery Mode，然后在主机终端确认：

```bash
lsusb | grep -i nvidia
```

确认设备出现后，回到 SDK Manager 点击 Retry。

### Q3：命令行刷机提示空间不足怎么办？

A：Jetson Linux BSP、rootfs、镜像生成过程会占用大量磁盘，建议主机保留至少 80 GB 可用空间。可以先检查磁盘空间：

```bash
df -h
```

### Q4：第三方载板刷机失败怎么办？

A：第三方载板通常需要厂商提供的 BSP、设备树、pinmux、board config。不要直接套用官方 devkit 配置，除非厂商明确说明兼容。

## 10. 参考资料

- [NVIDIA Jetson Linux Developer Guide: Quick Start](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/IN/QuickStart.html)
- [NVIDIA Jetson Linux Developer Guide: Flashing Support](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/FlashingSupport.html)
- [NVIDIA SDK Manager Jetson Install Guide](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)
