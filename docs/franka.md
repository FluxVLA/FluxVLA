# Franka Real-Robot Deployment and Environment Setup Guide

This guide describes a reference deployment workflow for single-arm and dual-arm Franka robots on Ubuntu 20.04 with ROS1 Noetic. It covers the realtime kernel, Franka FCI, `libfranka` / `franka_ros`, the FluxVLA Franka controller package, RealSense cameras, Franka grippers, ROS topic conventions, and FluxVLA real-robot inference.

______________________________________________________________________

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
   - [2.1 Realtime Kernel and NVIDIA Driver](#21-realtime-kernel-and-nvidia-driver)
   - [2.2 Franka FCI and Network Setup](#22-franka-fci-and-network-setup)
   - [2.3 Install libfranka and franka_ros](#23-install-libfranka-and-franka_ros)
   - [2.4 Install FluxVLA Franka Controllers](#24-install-fluxvla-franka-controllers)
   - [2.5 RealSense Cameras](#25-realsense-cameras)
   - [2.6 Franka Gripper](#26-franka-gripper)
3. [ROS Topic Contract for FluxVLA](#3-ros-topic-contract-for-fluxvla)
4. [Real-Robot Startup Workflow](#4-real-robot-startup-workflow)
5. [FluxVLA Inference](#5-fluxvla-inference)
6. [Troubleshooting](#6-troubleshooting)

______________________________________________________________________

## 1. System Requirements

- **OS:** Ubuntu 20.04 LTS
- **ROS:** ROS1 Noetic
- **Robot:** Franka Panda / Franka Research 3, single-arm or dual-arm
- **Cameras:** Intel RealSense cameras with stable ROS topic names
- **Control modes:**
  - Joint mode publishes `sensor_msgs/JointState`
  - Cartesian / eepose mode publishes `geometry_msgs/PoseStamped`
- **Recommended:** PREEMPT_RT realtime kernel, especially for high-rate control and dual-arm synchronization

______________________________________________________________________

## 2. Environment Setup

> If the machine already has a realtime kernel, NVIDIA driver, Franka FCI, and a working ROS workspace, you can skip to [3. ROS Topic Contract for FluxVLA](#3-ros-topic-contract-for-fluxvla) and verify topic names.

### 2.1 Realtime Kernel and NVIDIA Driver

Franka FCI control is expected to run on a realtime-capable system. The steps below follow the same PREEMPT_RT setup used in the UR3 guide.

<details>
<summary><strong>Click to expand: PREEMPT_RT kernel and NVIDIA setup</strong></summary>

#### 2.1.1 Install build dependencies

```bash
sudo apt update
sudo apt install build-essential bc ca-certificates gnupg2 libssl-dev wget gawk flex bison libelf-dev
sudo apt install -y libncurses5-dev liblz4-tool dwarves rsync kmod cpio libudev-dev libpci-dev libiberty-dev autoconf automake zstd
```

#### 2.1.2 Download Linux kernel source and RT patch

```bash
uname -r
mkdir -p ${HOME}/rt_kernel_build && cd ${HOME}/rt_kernel_build

wget https://cdn.kernel.org/pub/linux/kernel/projects/rt/5.15/older/patch-5.15.179-rt84.patch.xz
wget https://www.kernel.org/pub/linux/kernel/v5.x/linux-5.15.179.tar.xz

xz -cd linux-5.15.179.tar.xz | tar xvf -
cd linux-5.15.179/
xzcat ../patch-5.15.179-rt84.patch.xz | patch -p1
```

#### 2.1.3 Build and install the kernel

```bash
make oldconfig
# Select: Fully Preemptible Kernel (RT)
# Keep defaults for other options unless your system requires otherwise.

# If signing keys cause build issues, clear these in .config:
# CONFIG_SYSTEM_TRUSTED_KEYS=""
# CONFIG_SYSTEM_REVOCATION_KEYS=""

make -j `getconf _NPROCESSORS_ONLN` deb-pkg
cd ~/rt_kernel_build
sudo dpkg -i *.deb
```

Grant realtime scheduling permissions:

```bash
sudo groupadd realtime
sudo usermod -aG realtime $(whoami)
```

Append the following to `/etc/security/limits.conf`:

```text
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
```

Configure GRUB:

```bash
awk -F\' '/menuentry |submenu / {print $1 $2}' /boot/grub/grub.cfg
sudo vim /etc/default/grub
# Example:
# GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 5.15.179-rt84"
sudo update-grub
```

After reboot, verify:

```bash
uname -v | cut -d" " -f1-4
```

Expected output should include `#1 SMP PREEMPT RT`.

#### 2.1.4 NVIDIA driver on an RT kernel

If the NVIDIA driver stops working after switching kernels, reinstall the `.run` driver while bypassing the RT check:

```bash
sudo apt purge nvidia*
sudo rm -rf /usr/lib/nvidia-* /usr/bin/nvidia-* /etc/modprobe.d/nvidia.conf
sudo IGNORE_PREEMPT_RT_PRESENCE=1 bash <NVIDIA_DRIVER>.run
```

If the machine boots into a black screen, enter a TTY and remove the driver:

```bash
sudo nvidia-uninstall
sudo rm -rf /usr/lib/nvidia-* /usr/bin/nvidia-* /etc/modprobe.d/nvidia.conf
sudo rm -f /etc/X11/xorg.conf
sudo reboot
```

</details>

### 2.2 Franka FCI and Network Setup

Franka real-robot control uses FCI. Before launching ROS controllers, enable FCI in Franka Desk and place the control PC and robot on the same subnet. See the Franka getting-started guide for the official setup flow: <https://frankarobotics.github.io/docs/doc/libfranka/docs/getting_started.html>.

Example network layout:

- Robot IP: `172.16.0.2`
- Control PC IP: `172.16.0.1`
- Netmask: `255.255.255.0`

After assigning a static IP to the control PC, verify connectivity:

```bash
ping 172.16.0.2
```

If ping fails, check:

- The Ethernet cable or switch path
- The static IP configuration on the PC
- Whether FCI is enabled in Franka Desk
- Whether the robot is free of errors, protective stop, or emergency stop

### 2.3 Install libfranka and franka_ros

Use `libfranka` and `franka_ros` versions compatible with the robot firmware. Version mismatches can cause FCI connection failures or controller loading errors. Refer to the official compatibility page: <https://frankarobotics.github.io/docs/compatibility.html>.

#### 2.3.1 Install from apt

If the packaged versions match your robot:

```bash
sudo apt update
sudo apt install ros-noetic-libfranka ros-noetic-franka-ros
```

#### 2.3.2 Build franka_ros from source

Source builds are useful when you need to pin or debug a specific version:

```bash
source /opt/ros/noetic/setup.bash

mkdir -p ~/franka_ws/src && cd ~/franka_ws/src
git clone --recursive https://github.com/frankaemika/franka_ros.git

cd ~/franka_ws
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

If `libfranka` must be built separately:

```bash
git clone --recursive https://github.com/frankaemika/libfranka.git
cd libfranka
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo make install
```

### 2.4 Install FluxVLA Franka Controllers

FluxVLA Franka inference is designed to work with the companion controller package `fluxvla_franka_controllers`: <https://github.com/hhuang-xu/fluxvla_franka_controllers.git>.

The package provides:

- `cartesian_impedance_controller`, which accepts `PoseStamped` equilibrium pose targets
- `ruckig_joint_position_controller`, a Ruckig-smoothed joint position controller
- `ruckig_joint_impedance_controller`, an effort-based joint impedance controller accepting `sensor_msgs/JointState` targets
- Launch files for single-arm and dual-arm joint/eepose replay:
  - `single_joint.launch`
  - `single_eepose.launch`
  - `dual_joint.launch`
  - `dual_eepose.launch`

#### 2.4.1 Install Ruckig

Prefer the ROS package when available:

```bash
sudo apt update
sudo apt install ros-noetic-ruckig
```

If the package is unavailable, build Ruckig from source:

```bash
cd /tmp
git clone https://github.com/pantor/ruckig.git
cd ruckig
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PYTHON_MODULE=OFF
cmake --build build -j"$(nproc)"
sudo cmake --install build
sudo ldconfig
```

For a custom install prefix:

```bash
export CMAKE_PREFIX_PATH=/path/to/ruckig/install:$CMAKE_PREFIX_PATH
```

#### 2.4.2 Build the controller package

Place the package in the same catkin workspace as the Franka stack:

```bash
source /opt/ros/noetic/setup.bash

mkdir -p ~/franka_ws/src
cd ~/franka_ws/src
git clone https://github.com/hhuang-xu/fluxvla_franka_controllers.git

cd ~/franka_ws
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src -r -y
catkin_make --pkg fluxvla_franka_controllers
source devel/setup.bash
```

Verify the package:

```bash
rospack find fluxvla_franka_controllers
roscd fluxvla_franka_controllers
```

#### 2.4.3 Controller launch entry points

Single-arm joint control:

```bash
roslaunch fluxvla_franka_controllers single_joint.launch \
    robot_ip:=<RobotIP> \
    load_gripper:=false
```

Single-arm eepose / Cartesian control:

```bash
roslaunch fluxvla_franka_controllers single_eepose.launch \
    robot_ip:=<RobotIP> \
    load_gripper:=false
```

Dual-arm joint control:

```bash
roslaunch fluxvla_franka_controllers dual_joint.launch \
    left_robot_ip:=<LeftIP> \
    right_robot_ip:=<RightIP> \
    load_gripper:=false
```

Dual-arm eepose / Cartesian control:

```bash
roslaunch fluxvla_franka_controllers dual_eepose.launch \
    left_robot_ip:=<LeftIP> \
    right_robot_ip:=<RightIP> \
    load_gripper:=false
```

After launch, FluxVLA publishes targets to:

```text
/<arm_namespace>/ruckig_joint_impedance_controller/target_joint_state
/<arm_namespace>/cartesian_impedance_controller/equilibrium_pose
```

If a single-arm launch does not use a `/left_arm` namespace, the actual topics may not have the `/left_arm` prefix. In that case, update `joint_cmd_topic`, `cartesian_cmd_topic`, and state topics in the FluxVLA config.

#### 2.4.4 Basic controller verification

Check loaded controllers:

```bash
rosservice call /controller_manager/list_controllers
```

For namespaced dual-arm setups:

```bash
rosservice call /left_arm/controller_manager/list_controllers
rosservice call /right_arm/controller_manager/list_controllers
```

You can also use the controller package CSV replay script for a basic smoke test.

> **Safety notice**
>
> The example CSV shipped with the controller package is intended to demonstrate the replay script and controller API. It was collected with two Franka robots placed side by side on a table, with approximately 1 meter between the bases.
>
> Do not directly execute the example trajectory if your robot placement, base distance, workspace, or end-effector setup differs from the example. Use a single-arm test first, or collect replay data for your own robot layout.
>
> Before and during execution, confirm that the robot is inside a safe workspace, the area is clear, and an emergency stop is immediately available.

Joint replay:

```bash
rosrun fluxvla_franka_controllers replay_joint_impedance_csv.py \
    --csv ./examples/replay.csv \
    --control_mode joint \
    --execute
```

Eepose replay:

```bash
rosrun fluxvla_franka_controllers replay_joint_impedance_csv.py \
    --csv ./examples/replay.csv \
    --control_mode eepose \
    --execute
```

> On non-realtime development machines, `franka_ros` may reject realtime control. Some lab setups set `realtime_config: ignore` in `franka_control/config/franka_control_node.yaml`, but only use that workaround if it matches your lab safety policy.

### 2.5 RealSense Cameras

Franka inference configs use the following image keys:

- Single-arm: `cam_front`, `cam_wrist_left`
- Dual-arm: `cam_front`, `cam_wrist_left`, `cam_wrist_right`

The underlying ROS image topics should match:

- `/camera_front/color/image_raw`
- `/camera_left_wrist/color/image_raw`
- `/camera_right_wrist/color/image_raw`

Install dependencies:

```bash
sudo apt update
sudo apt install -y nlohmann-json3-dev
```

Install `librealsense` following the Intel RealSense guide:

- <https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md>

Validate the camera:

```bash
realsense-viewer
```

Install and launch the ROS1 driver:

- <https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy>

Single-camera example:

```bash
roslaunch realsense2_camera rs_camera.launch \
    camera:=camera_front \
    serial_no:=<FRONT_CAMERA_SERIAL> \
    align_depth:=true
```

For multiple cameras, put all camera launches in one launch file and keep the names stable. For example, a camera named `camera_front` should publish:

```text
/camera_front/color/image_raw
```

### 2.6 Franka Gripper

FluxVLA Franka operators publish gripper commands to Franka gripper move action goal topics:

```text
/left_arm/franka_gripper/move/goal
/right_arm/franka_gripper/move/goal
```

The message type is `franka_gripper.msg.MoveActionGoal`. Before inference, check:

- The Franka gripper is installed and connected
- The `franka_gripper` package is available
- Homing has been completed

Example:

```bash
rosrun franka_gripper franka_gripper_node _robot_ip:=172.16.0.2 __ns:=/left_arm/franka_gripper
rosservice call /left_arm/franka_gripper/homing "{}"
```

For dual-arm setups, start the gripper stack for both arms and keep namespaces consistent with the FluxVLA config.

______________________________________________________________________

## 3. ROS Topic Contract for FluxVLA

FluxVLA Franka operators do not launch the low-level robot stack themselves. They assume that camera topics, robot state topics, and controller command topics already exist.

Also note that the Franka Hand control and command-receiving frequency is usually around 10Hz. To keep model-side observations aligned with camera frames, FluxVLA fills each 30Hz synchronized observation frame with the latest available `gripper_width`, rather than waiting for the lower-frequency gripper control path.

### 3.1 Single-arm topics

Single-arm inference uses `FrankaOperator`, normally controlling the left arm:

```python
operator=dict(
    type='FrankaOperator',
    command_mode='joint',
    img_left_topic='/camera_left_wrist/color/image_raw',
    img_front_topic='/camera_front/color/image_raw',
    puppet_arm_left_topic='/left_arm/joint_states',
    puppet_franka_state_left_topic=(
        '/left_arm/franka_state_controller/franka_states'),
    joint_cmd_topic=(
        '/left_arm/ruckig_joint_impedance_controller/target_joint_state'),
    cartesian_cmd_topic=(
        '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
    gripper_left_topic='/left_arm/franka_gripper/move/goal',
)
```

Required observation topics:

- `/camera_front/color/image_raw`
- `/camera_left_wrist/color/image_raw`
- `/left_arm/joint_states`
- `/left_arm/franka_state_controller/franka_states`

Required command topics:

- Joint mode: `/left_arm/ruckig_joint_impedance_controller/target_joint_state`
- Cartesian mode: `/left_arm/cartesian_impedance_controller/equilibrium_pose`
- Gripper: `/left_arm/franka_gripper/move/goal`

If you replace the real-robot control interface used by FluxVLA, you may change the topic names or make your custom controller subscribe to these topics, but the ROS message types and field semantics must remain compatible:

- `joint_cmd_topic` uses `sensor_msgs/JointState`. FluxVLA fills `header.stamp`, `name`, and `position`; `name` defaults to the seven Franka joint names, and `position` contains the 7D target joint angles. A custom joint controller should read the target joint positions from `position[0:7]`.
- `cartesian_cmd_topic` uses `geometry_msgs/PoseStamped`. FluxVLA fills `header.stamp`, `header.frame_id`, and `pose`; `pose.position` is the target end-effector position `[x, y, z]`, and `pose.orientation` is the target quaternion `[qx, qy, qz, qw]`. A custom Cartesian / eepose controller should drive the end effector according to this pose command.
- The gripper topic `gripper_left_topic` uses `franka_gripper/MoveActionGoal`, where `goal.width` is the target gripper opening width and `goal.speed` is the gripper speed.

FluxVLA also supports the optional `gripper_control_mode` setting, which selects how gripper commands are post-processed:

- `move`: stream a continuous target width at every step, suitable for smoother opening and closing.
- `grasp`: threshold the continuous width into binary open/close states, then send `franka_gripper` `move` / `grasp` actions only when the state changes; internally, the operator remembers the previous binary state for each hand to avoid repeated commands.

> **Note:** Continuous `move` commands keep updating the gripper width and are smoother, but the closing response may lag. In grasping tasks, this can lead to the arm moving away before the gripper has fully closed. `grasp` executes the `franka_gripper` grasp action in a blocking way and only sends subsequent arm motion commands after that action finishes, so the grasp timing is more explicit, but the arm motion stream pauses until the grasp completes.

### 3.2 Dual-arm topics

Dual-arm inference uses `FrankaDualOperator`:

```python
operator=dict(
    type='FrankaDualOperator',
    command_mode='joint',
    img_left_topic='/camera_left_wrist/color/image_raw',
    img_right_topic='/camera_right_wrist/color/image_raw',
    img_front_topic='/camera_front/color/image_raw',
    puppet_arm_left_topic='/left_arm/joint_states',
    puppet_arm_right_topic='/right_arm/joint_states',
    puppet_franka_state_left_topic=(
        '/left_arm/franka_state_controller/franka_states'),
    puppet_franka_state_right_topic=(
        '/right_arm/franka_state_controller/franka_states'),
    joint_cmd_left_topic=(
        '/left_arm/ruckig_joint_impedance_controller/target_joint_state'),
    joint_cmd_right_topic=(
        '/right_arm/ruckig_joint_impedance_controller/target_joint_state'),
    cartesian_cmd_left_topic=(
        '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
    cartesian_cmd_right_topic=(
        '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
    gripper_left_topic='/left_arm/franka_gripper/move/goal',
    gripper_right_topic='/right_arm/franka_gripper/move/goal',
)
```

Dual-arm command topics use the same message types as single-arm topics:

- `joint_cmd_left_topic`, `joint_cmd_right_topic`: `sensor_msgs/JointState`
- `cartesian_cmd_left_topic`, `cartesian_cmd_right_topic`: `geometry_msgs/PoseStamped`
- `gripper_left_topic`, `gripper_right_topic`: `franka_gripper/MoveActionGoal`

When replacing the dual-arm controllers, provide compatible subscription interfaces for both arms. FluxVLA only publishes commands to the configured topics; it does not adapt custom message types.

Dual-arm operators also support `gripper_control_mode='move'|'grasp'`. In `grasp` mode, left and right hands are binarized and tracked independently.

The runner concatenates state and actions in the order:

```text
[left_8d, right_8d]
```

Joint mode uses one 8D block per arm:

```text
[joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper_width]
```

Cartesian / eepose mode uses:

```text
[x, y, z, qx, qy, qz, qw, gripper_width]
```

______________________________________________________________________

## 4. Real-Robot Startup Workflow

Run the robot stack, cameras, and inference in separate terminals. The recommended Franka controller stack is `fluxvla_franka_controllers`, whose joint/eepose launch files are aligned with the FluxVLA Franka operator topic contract.

### 4.0 Franka Desk Preparation

Before launching ROS controllers, complete these steps in Franka Desk:

1. Enable **FCI** for the target robot so the external control PC can command the robot.
2. Switch the lower-right **Operations** state to **Execution**.
3. Confirm that the robot is not in emergency stop, collision, error, or protective-stop state, and that the workspace is clear.

### 4.1 Single-arm startup

#### Terminal A: Launch the Franka controller

For a joint checkpoint:

```bash
source ~/franka_ws/devel/setup.bash
roslaunch fluxvla_franka_controllers single_joint.launch \
    robot_ip:=172.16.0.2 \
    load_gripper:=true
```

For an eepose / Cartesian checkpoint:

```bash
source ~/franka_ws/devel/setup.bash
roslaunch fluxvla_franka_controllers single_eepose.launch \
    robot_ip:=172.16.0.2 \
    load_gripper:=true
```

> Prefer `load_gripper:=true` when using the official Franka gripper. If you use another gripper stack or an independently managed gripper node, set `load_gripper:=false` and make sure `gripper_left_topic` in the FluxVLA config points to the actual gripper command topic.

> If the single-arm launch does not use the `/left_arm` namespace, actual topics may be `/joint_states`, `/franka_state_controller/franka_states`, and `/ruckig_joint_impedance_controller/target_joint_state`. Update `puppet_arm_left_topic`, `puppet_franka_state_left_topic`, `joint_cmd_topic`, `cartesian_cmd_topic`, and `gripper_left_topic` in `configs/pi05/pi05_paligemma_franka_single_inference.py` accordingly.

#### Terminal B: Launch cameras

```bash
roslaunch <your_camera_package> franka_single_cameras.launch
```

Verify:

```bash
rostopic hz /camera_front/color/image_raw
rostopic hz /camera_left_wrist/color/image_raw
```

#### Terminal C: Check required topics

```bash
rostopic hz /left_arm/joint_states
rostopic hz /left_arm/franka_state_controller/franka_states
rostopic info /left_arm/ruckig_joint_impedance_controller/target_joint_state
rostopic info /left_arm/cartesian_impedance_controller/equilibrium_pose
rostopic info /left_arm/franka_gripper/move/goal
```

### 4.2 Dual-arm startup

The dual-arm launch files use separate `left_arm` and `right_arm` namespaces.

For a dual-arm joint checkpoint:

```bash
source ~/franka_ws/devel/setup.bash
roslaunch fluxvla_franka_controllers dual_joint.launch \
    left_robot_ip:=172.16.0.2 \
    right_robot_ip:=172.16.0.3 \
    load_gripper:=true
```

For a dual-arm eepose / Cartesian checkpoint:

```bash
source ~/franka_ws/devel/setup.bash
roslaunch fluxvla_franka_controllers dual_eepose.launch \
    left_robot_ip:=172.16.0.2 \
    right_robot_ip:=172.16.0.3 \
    load_gripper:=true
```

> Use `load_gripper:=true` for the official Franka grippers. If either arm uses an external gripper stack, set `load_gripper:=false` and verify `gripper_left_topic` and `gripper_right_topic` in the FluxVLA config.

Extra checks for dual-arm setups:

```bash
rostopic hz /right_arm/joint_states
rostopic hz /right_arm/franka_state_controller/franka_states
rostopic hz /camera_right_wrist/color/image_raw
rostopic info /right_arm/ruckig_joint_impedance_controller/target_joint_state
rostopic info /right_arm/cartesian_impedance_controller/equilibrium_pose
```

### 4.3 Synchronization and safety checks

Before starting FluxVLA inference:

```bash
rostopic list | grep camera
rostopic list | grep franka_states
rostopic list | grep target_joint_state
rostopic list | grep equilibrium_pose
rostopic list | grep franka_gripper
```

Check stream rates:

```bash
rostopic hz /camera_front/color/image_raw
rostopic hz /left_arm/joint_states
rostopic hz /left_arm/franka_state_controller/franka_states
```

If `FrankaInferenceRunner` keeps waiting for synchronized observations, check:

- Image topics exist and publish frames
- `joint_states` and `franka_states` are publishing continuously
- `sync_slop` is not too small
- Camera timestamps are stable across all views

______________________________________________________________________

## 5. FluxVLA Inference

Recommended CLI entry point:

```bash
python scripts/inference.py \
    --config <CONFIG_PATH> \
    --ckpt-path <CHECKPOINT_PATH>
```

You can also use VSCode `debugpy` with `scripts/inference_real_robot.py`.

### 5.1 Single-arm joint inference

Use this for checkpoints trained on single-arm joint-state data.

Config:

```text
configs/pi05/pi05_paligemma_franka_single_inference.py
```

Example:

```bash
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_franka_single_inference.py \
    --ckpt-path /path/to/single_arm_checkpoint/model.safetensors
```

Required consistency:

```python
action_mode='joint'
active_arms=('left', )
denormalize_action=dict(action_dim=8)
operator=dict(type='FrankaOperator', command_mode='joint')
```

### 5.2 Dual-arm joint inference

Config:

```text
configs/pi05/pi05_paligemma_franka_dual_full_finetune.py
```

Example:

```bash
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_franka_dual_full_finetune.py \
    --ckpt-path /path/to/dual_arm_joint_checkpoint/model.safetensors
```

Required consistency:

```python
action_mode='joint'
active_arms=('left', 'right')
denormalize_action=dict(action_dim=16)
operator=dict(type='FrankaDualOperator', command_mode='joint')
```

### 5.3 Dual-arm eepose inference

Config:

```text
configs/pi05/pi05_paligemma_franka_dual_eepose_full_finetune.py
```

Example:

```bash
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_franka_dual_eepose_full_finetune.py \
    --ckpt-path /path/to/dual_arm_eepose_checkpoint/model.safetensors
```

Required consistency:

```python
action_mode='cartesian'
active_arms=('left', 'right')
denormalize_action=dict(action_dim=16)
operator=dict(type='FrankaDualOperator', command_mode='cartesian')
```

### 5.4 VSCode launch.json example

```jsonc
{
    "name": "Inference Franka single",
    "type": "debugpy",
    "request": "launch",
    "program": "{CONDA_ENV_PATH}/bin/torchrun",
    "python": "{CONDA_ENV_PATH}/bin/python",
    "args": [
        "scripts/inference_real_robot.py",
        "--config", "configs/pi05/pi05_paligemma_franka_single_inference.py",
        "--ckpt-path", "{CHECKPOINT_PATH}/checkpoints/model.safetensors"
    ],
    "console": "integratedTerminal",
    "justMyCode": false,
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "WANDB_MODE": "disabled",
        "LD_PRELOAD": "/lib/x86_64-linux-gnu/libffi.so.7"
    }
}
```

> Start inference only after the robot is controllable and camera, joint-state, FrankaState, and gripper topics are all publishing correctly.

______________________________________________________________________

## 6. Troubleshooting

### 6.1 The runner waits forever for synchronized Franka observations

Check that every observation topic required by the selected FluxVLA config is
ready. All camera, joint-state, FrankaState, and gripper state topics used by
the operator must exist, have active publishers, and publish at a stable rate
before the runner can build synchronized frames.

```bash
rostopic list
rostopic info <required_topic>
rostopic hz <required_topic>
```

### 6.2 Joint and eepose modes are mixed

Training, inference, and operator mode must match:

- Joint checkpoints: `action_mode='joint'`, `command_mode='joint'`
- Eepose / Cartesian checkpoints: `action_mode='cartesian'`, `command_mode='cartesian'`

Do not run an eepose checkpoint through joint control, or a joint checkpoint through Cartesian control.

### 6.3 Single-arm and dual-arm checkpoints are mixed

Do not mix them for normal deployment:

- Single-arm checkpoint: `action_dim=8`, `FrankaOperator`
- Dual-arm checkpoint: `action_dim=16`, `FrankaDualOperator`

If you need to temporarily test a dual-arm checkpoint against a single-arm operator, add an explicit adapter runner. Do not use that as the production deployment path.

### 6.4 Protective stop or controller error

Stop inference first, then recover from Franka Desk or the ROS controller stack. Before restarting inference, confirm:

- The robot is not in collision or emergency stop
- The gripper is not holding an object unexpectedly
- The prepare pose is inside a safe workspace
- Predicted joint/eepose commands are in the same range as the training data

______________________________________________________________________

## Quick Config Reference

- Normal single-arm joint inference: `configs/pi05/pi05_paligemma_franka_single_inference.py`
- Dual-arm joint inference: `configs/pi05/pi05_paligemma_franka_dual_full_finetune.py`
- Dual-arm eepose inference: `configs/pi05/pi05_paligemma_franka_dual_eepose_full_finetune.py`

For deployment, keep the checkpoint data type, `action_mode`, `active_arms`, `denormalize_action.action_dim`, and operator type consistent.
