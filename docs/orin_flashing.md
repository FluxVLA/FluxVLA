# Jetson Orin Flashing Guide: From Recovery Mode to JetPack Setup

This guide explains how to flash an NVIDIA Jetson Orin device from scratch to a usable JetPack 6.2.x system, preparing it for the next step: creating and entering the FluxVLA Docker environment.

Applicable devices:

- Jetson AGX Orin Developer Kit
- Jetson AGX Orin module with an official or compatible carrier board

> This guide focuses on JetPack 6.x / Jetson Linux r36.x. Button locations and flashing configuration names may vary across JetPack minor versions and carrier boards. For production carrier boards, also refer to the BSP documentation from the carrier board vendor.

## 1. Prepare the Flashing Host

Flashing requires a separate Linux x86_64 host. Do not run the flashing process on the Jetson device itself.

> Flashing rewrites the system partitions on the target storage. Before proceeding, back up important data from the Jetson internal eMMC, NVMe, or USB storage.

Recommended host environment:

- Ubuntu 20.04 or Ubuntu 22.04 x86_64
- At least 80 GB of free disk space
- Stable network connection
- Prefer the original or a high-quality USB-C data cable to connect the Orin device to the host
- If flashing to external NVMe or USB storage, use a device with at least 64 GB capacity

Install basic tools:

```bash
sudo apt update
```

Confirm that the host can see USB devices:

```bash
lsusb
```

## 2. Choose a Flashing Method

We recommend using **SDK Manager flashing** first. It is suitable for initial setup, developer kit validation, and installing JetPack components.

SDK Manager website: [https://developer.nvidia.com/sdk-manager](https://developer.nvidia.com/sdk-manager)

<img src="../assets/SDK-Manager.png" alt="Jetson Orin" width="400">

Download and install the Ubuntu .deb x86_64 package. This makes it convenient to use the SDK Manager CLI for further system configuration.

## 3. Hardware Connections

Connect the following before flashing:

1. Jetson power supply.
2. USB data cable between the Jetson and the host.
3. Wired or wireless network, which is useful for installing components after the first boot.

Connect to a wireless network from the terminal:

```bash
nmcli device wifi list
nmcli device wifi connect "WiFi_name" password "WiFi_password"
# Check connection status
nmcli connection show
```

4. Test host access to the Jetson:

- Serial connection:

```bash
sudo screen /dev/ttyACM0 115200
```

- SSH:
  After configuring the wireless network, configure the SSH server and enable it at boot so the Orin can be used as a server.

### 3.1 Jetson AGX Orin Developer Kit

When flashing the AGX Orin Developer Kit, connect the host to the USB-C port used for flashing/recovery on the developer kit. This is the USB-C port near the 40-pin header.

## 4. Enter Force Recovery Mode

Force Recovery Mode is the mode in which the Jetson accepts flashing commands. After entering this mode, the Jetson will not boot into the normal system. Instead, it exposes itself over USB as an NVIDIA recovery device.

### 4.1 AGX Orin Developer Kit: Button Method

<img src="../assets/orin.png" alt="Jetson Orin" width="400">

In the image, `2` is the Force Recovery button and `3` is the reset button.

There are two ways to enter Recovery Mode depending on whether the device is powered off or already powered on.

#### 4.1.1 Device Powered Off

1. Press and hold button `2` first (Force Recovery).
2. Connect the Orin power cable to power it on.
3. When the white LED turns on, the device has entered Recovery Mode.

#### 4.1.2 Device Powered On

When the device is already powered on, press and hold button `2` (Force Recovery), then press button `3` (Reset). Release button `3` first, then release button `2`.

Run the following on the host:

```bash
lsusb | grep -i nvidia
```

If you see code `7023`, the device has entered Recovery Mode and the connection is working:

```text
Bus 001 Device 006: ID 0955:7023 NVIDIA Corp.
```

If no NVIDIA device appears:

1. Replace the USB cable with one confirmed to support data transfer.
2. Try another USB port on the host, avoiding USB hubs.
3. Confirm that the cable is connected to the recovery/device USB port, not a normal host USB port.
4. Repeat the Recovery Mode steps.

## 5. Flash with SDK Manager (Recommended)

### 5.1 Install SDK Manager

Download and install NVIDIA SDK Manager on the Ubuntu host:

```bash
sudo apt install ./sdkmanager_*_amd64.deb
```

Start it:

```bash
sdkmanager --cli
```

Log in with your NVIDIA Developer account.

### 5.2 STEP 01: Select Device and JetPack

In SDK Manager, select:

1. Product Category: `Jetson`
2. Target Hardware: select your Orin device, for example `Jetson AGX Orin Developer Kit`
3. SDK Version: select the target JetPack version, for example JetPack 6.x
4. Host Machine: if you only need to flash the Jetson, you can deselect host-side components as needed
5. Additional SDKs: the default selection is usually sufficient

<img src="../assets/jetpack-version.png" alt="Jetson Orin" width="500">

Click **Continue**.

### 5.3 STEP 02: Select Components and Start Flashing

<img src="../assets/sdk-manager-components.png" alt="Jetson Orin" width="500">

We recommend installing at least:

- Jetson Linux
- CUDA
- cuDNN
- TensorRT
- VPI
- Jetson Runtime Components

Accept the license and continue to start flashing.

### 5.4 First Boot

After flashing completes, the Jetson will reboot automatically. On the first boot, complete the following on a monitor or serial console:

1. Accept the system license, then choose language, keyboard layout, and time zone.
2. Create a username and password.
3. Configure the network.

If SDK Manager still needs to install target components such as CUDA or TensorRT, make sure the Jetson and the host are on the same network, or follow the SDK Manager CLI prompts to complete the connection.

## 6. System Checks After Flashing

After logging into the Jetson for the first time, run:

```bash
cat /etc/nv_tegra_release
dpkg-query --show nvidia-l4t-core
uname -a
```

Check CUDA:

```bash
nvcc --version
```

Check GPU and system status:

```bash
sudo tegrastats
```

Expected results:

- `/etc/nv_tegra_release` shows `R36.4.x`; the currently recorded target is `R36.4.4`.
- Ubuntu is 22.04.
- `nvcc` is from the CUDA 12.6 series.

Check Docker and the NVIDIA runtime:

```bash
docker --version
sudo docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.4.0 \
  bash -lc "cat /etc/nv_tegra_release"
```

## 7. Install jtop (Similar to NVITOP)

```bash
# Python install
sudo apt update
sudo apt install python3
sudo apt install python3-pip

# jtop install
sudo pip3 install -U pip
sudo pip3 install jetson-stats
```

## 8. Next Step: Configure FluxVLA Orin Docker

After flashing is complete and JetPack has been verified, continue with the FluxVLA Orin Docker setup:

```bash
git clone https://github.com/limxdynamics/FluxVLA.git
cd FluxVLA
docker/build_docker.sh
docker/run_docker.sh
```

For detailed Docker usage, see:

- `docker/README_DOCKER_ORIN.md`
- `docker/DOCKER_VERSIONING.md`

## 9. Q&A

### Q1: What should I do if `lsusb` does not show an NVIDIA device?

A: First check whether the USB cable supports data transfer, whether it is connected to the recovery/device USB port, whether the Recovery Mode sequence is correct, and whether an unstable USB hub is being used. If using a third-party carrier board, also confirm that the RECOVERY pin is actually pulled down to GND.

### Q2: What should I do if SDK Manager is stuck waiting for the device?

A: Keep SDK Manager open, put the Jetson into Recovery Mode again, then confirm from the host terminal:

```bash
lsusb | grep -i nvidia
```

After the device appears, return to SDK Manager and click Retry.

### Q3: What should I do if command-line flashing reports insufficient disk space?

A: The Jetson Linux BSP, rootfs, and image generation process consume a large amount of disk space. We recommend keeping at least 80 GB of free space on the host. You can check disk space first:

```bash
df -h
```

### Q4: What should I do if flashing fails on a third-party carrier board?

A: Third-party carrier boards usually require a vendor-provided BSP, device tree, pinmux, and board config. Do not directly reuse the official devkit configuration unless the vendor explicitly states that it is compatible.

## 10. References

- [NVIDIA Jetson Linux Developer Guide: Quick Start](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/IN/QuickStart.html)
- [NVIDIA Jetson Linux Developer Guide: Flashing Support](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/FlashingSupport.html)
- [NVIDIA SDK Manager Jetson Install Guide](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)
