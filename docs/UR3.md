# UR3 Real-Robot Deployment and Environment Setup Guide

This document provides a step-by-step setup guide for deploying a **UR3** robot system on **Ubuntu 20.04**, including RealSense cameras, a Robotiq gripper, and FluxVLA real-robot inference.

______________________________________________________________________

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
   - [2.1 RT Kernel + NVIDIA Driver (Optional)](#21-rt-kernel--nvidia-driver-optional)
   - [2.2 Build and Install UR ROS Driver](#22-build-and-install-ur-ros-driver)
   - [2.3 Install UR RTDE](#23-install-ur-rtde)
   - [2.4 RealSense Camera](#24-realsense-camera)
   - [2.5 Robotiq Gripper](#25-robotiq-gripper)
3. [Real-Robot Startup Workflow](#3-real-robot-startup-workflow)
4. [FluxVLA Inference](#4-fluxvla-inference)

______________________________________________________________________

## 1. System Requirements

- **OS:** Ubuntu 20.04 LTS
- **ROS:** ROS1 Noetic ([installation guide](https://wiki.ros.org/noetic/Installation/Ubuntu))
- **Hardware:**
  - UR3 robot (CB3 / e-Series)
  - Intel RealSense depth camera(s)
  - Robotiq gripper

______________________________________________________________________

## 2. Environment Setup

> If your machine is already configured with a real-time kernel and NVIDIA driver, you can start from [2.2 Build and Install UR ROS Driver](#22-build-and-install-ur-ros-driver).

### 2.1 RT Kernel + NVIDIA Driver (Optional)

<details>
<summary><strong>Click to expand: PREEMPT_RT kernel and NVIDIA setup</strong></summary>

For high-frequency tracking and servo control, installing a **PREEMPT_RT** kernel is strongly recommended.

References:

- UR: [Setting up Ubuntu with a PREEMPT_RT kernel](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/real_time.md)
- Franka: [Realtime kernel setup](https://franka.cn/FCI/installation_linux.html) (for reference)

#### 2.1.1 Install build dependencies

```bash
sudo apt update
sudo apt install build-essential bc ca-certificates gnupg2 libssl-dev wget gawk flex bison libelf-dev
sudo apt install -y libncurses5-dev liblz4-tool dwarves rsync kmod cpio libudev-dev libpci-dev libiberty-dev autoconf automake zstd
```

#### 2.1.2 Download kernel source and RT patch

```bash
uname -r
mkdir -p ${HOME}/rt_kernel_build && cd ${HOME}/rt_kernel_build

wget https://cdn.kernel.org/pub/linux/kernel/projects/rt/5.15/older/patch-5.15.179-rt84.patch.xz
wget https://www.kernel.org/pub/linux/kernel/v5.x/linux-5.15.179.tar.xz

xz -cd linux-5.15.179.tar.xz | tar xvf -
cd linux-5.15.179/
xzcat ../patch-5.15.179-rt84.patch.xz | patch -p1
```

#### 2.1.3 Build kernel package

```bash
make oldconfig
# Select: Fully Preemptible Kernel (RT)
# Keep default for other options

# If needed, disable key verification in .config:
# CONFIG_SYSTEM_TRUSTED_KEYS=""
# CONFIG_SYSTEM_REVOCATION_KEYS=""

make -j `getconf _NPROCESSORS_ONLN` deb-pkg
```

If you encounter `cannot represent change to vmlinux-gdb.py`:

```bash
rm vmlinux-gdb.py
# then re-run make
```

#### 2.1.4 Install and configure realtime permissions

```bash
cd ~/rt_kernel_build
sudo dpkg -i *.deb
sudo groupadd realtime
sudo usermod -aG realtime $(whoami)
```

Append to `/etc/security/limits.conf`:

```text
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
```

#### 2.1.5 Configure GRUB default kernel

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

You should see `#1 SMP PREEMPT RT`.

#### 2.1.6 NVIDIA driver on RT kernel

```bash
sudo apt purge nvidia*
sudo rm -rf /usr/lib/nvidia-* /usr/bin/nvidia-* /etc/modprobe.d/nvidia.conf
```

Install `.run` package while bypassing RT check:

```bash
sudo IGNORE_PREEMPT_RT_PRESENCE=1 bash <NVIDIA_DRIVER>.run
```

**Black-screen recovery (if driver install fails):**

1. Press `Ctrl + Alt + F3` to enter TTY and log in.
2. Remount root as writable:
   ```bash
   mount -n -o remount,rw /
   ```
3. Uninstall NVIDIA driver:
   - If installed via `.run`:
     ```bash
     sudo nvidia-uninstall
     sudo rm -rf /usr/lib/nvidia-*
     sudo rm -rf /usr/bin/nvidia-*
     sudo rm -rf /etc/modprobe.d/nvidia.conf
     ```
   - If installed via `apt`:
     ```bash
     sudo apt purge 'nvidia*'
     ```
4. Reset Xorg and reboot:
   ```bash
   sudo rm -f /etc/X11/xorg.conf
   sudo reboot
   ```

</details>

### 2.2 Build and Install UR ROS Driver

Use source build for easier debugging and customization.

Official repo: <https://github.com/UniversalRobots/Universal_Robots_ROS_Driver>

```bash
source /opt/ros/noetic/setup.bash

mkdir -p ~/catkin_ws/src && cd ~/catkin_ws
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver

rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src -y

catkin_make
source devel/setup.bash
```

> Notes:
>
> 1. You must install and run `External Control` URCap on the teach pendant for external PC control.
> 2. For accurate kinematics, extract and apply robot-specific calibration.
>    - URCap doc: <https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/install_urcap_cb3.md>
>    - Calibration doc: <https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_calibration/README.md>

### 2.3 Install UR RTDE

RTDE provides low-latency data exchange and motion control APIs.

```bash
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt update
sudo apt install librtde librtde-dev
pip install --user ur_rtde
```

<details>
<summary><strong>Click to expand: UR RTDE ROS node example rtde_ros.py (for control and state publishing)</strong></summary>

```python
#!/usr/bin/env python3
import sys
import rospy
import rtde_receive
import rtde_control

from geometry_msgs.msg import Pose, PoseStamped, Twist
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation


class URRTDEInterface:
    def __init__(self):
        rospy.init_node('ur_rtde_interface', anonymous=True)

        # Robot connection
        self.robot_ip = rospy.get_param('~robot_ip', '192.168.8.202')
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            rospy.loginfo(f"Connected to UR robot: {self.robot_ip}")
        except Exception as e:
            rospy.logerr(f"Failed to connect {self.robot_ip}: {str(e)}")
            sys.exit(1)

        # Motion parameters
        self.servo_velocity = rospy.get_param('~servo_velocity', 1.5)
        self.servo_acceleration = rospy.get_param('~servo_acceleration', 4.0)
        self.servo_dt = rospy.get_param('~servo_dt', 1.0 / 125.0)
        self.servo_lookahead_time = rospy.get_param('~servo_lookahead_time', 0.2)
        self.servo_gain = rospy.get_param('~servo_gain', 100)

        self.move_velocity = rospy.get_param('~move_velocity', 1.0)
        self.move_acceleration = rospy.get_param('~move_acceleration', 1.0)

        self.servo_mode = False

        # State publishers
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.pose_pub = rospy.Publisher('/arm/tcp_pose', PoseStamped, queue_size=10)
        # self.twist_pub = rospy.Publisher('/arm/tcp_twist', Twist, queue_size=10)

        # Command subscribers
        rospy.Subscriber('/cmd/movej', JointState, self.movej_callback)
        rospy.Subscriber('/cmd/movel', Pose, self.movel_callback)
        rospy.Subscriber('/cmd/servoj', JointState, self.servoj_callback)
        rospy.Subscriber('/cmd/servol', Pose, self.servol_callback)
        rospy.Subscriber('/cmd/speedj', JointState, self.servoj_callback)
        # rospy.Subscriber('/cmd/speedl', Twist, self.speedl_callback)

        # Publish rate for robot state topics
        self.rate = rospy.Rate(125)

    def movej_callback(self, msg: JointState):
        if self.servo_mode:
            self.rtde_c.servoStop()
            self.servo_mode = False

        ur_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        missing_joints = [j for j in ur_joint_names if j not in msg.name]
        if missing_joints:
            rospy.logerr(f"JointState missing joints: {missing_joints}")
            return

        qpos = [msg.position[msg.name.index(joint)] for joint in ur_joint_names]

        if self.rtde_c.moveJ(qpos, self.move_velocity, self.move_acceleration):
            rospy.logdebug(f"MoveJ executed: {[round(v, 4) for v in qpos]}")
        else:
            rospy.logerr('MoveJ command failed')

    def servoj_callback(self, msg: JointState):
        self.servo_mode = True

        ur_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        missing_joints = [j for j in ur_joint_names if j not in msg.name]
        if missing_joints:
            rospy.logerr(f"JointState missing joints: {missing_joints}")
            return

        try:
            qpos = [msg.position[msg.name.index(joint)] for joint in ur_joint_names]
            t_start = self.rtde_c.initPeriod()
            ok = self.rtde_c.servoJ(
                qpos,
                self.servo_velocity,
                self.servo_acceleration,
                self.servo_dt,
                self.servo_lookahead_time,
                self.servo_gain,
            )
            self.rtde_c.waitPeriod(t_start)

            if ok:
                rospy.logdebug(f"ServoJ sent: {[round(v, 4) for v in qpos]}")
            else:
                rospy.logwarn('ServoJ command failed')
        except Exception as e:
            rospy.logerr(f"Error in servoj_callback: {str(e)}")

    def movel_callback(self, msg: Pose):
        try:
            if self.servo_mode:
                self.rtde_c.servoStop()
                self.servo_mode = False

            pose_rtde = self.pose_to_rtde(msg)

            if not self.rtde_c.isPoseWithinSafetyLimits(pose_rtde):
                rospy.logerr(f"TCP pose out of limits: {[round(v, 4) for v in pose_rtde]}")
                return

            if self.rtde_c.moveL(pose_rtde, self.move_velocity, self.move_acceleration):
                rospy.loginfo(f"MoveL executed: {[round(v, 4) for v in pose_rtde]}")
            else:
                rospy.logerr('MoveL command failed')
        except Exception as e:
            rospy.logerr(f"Error in movel_callback: {str(e)}")

    def servol_callback(self, msg: Pose):
        try:
            self.servo_mode = True
            pose_rtde = self.pose_to_rtde(msg)

            t_start = self.rtde_c.initPeriod()
            ok = self.rtde_c.servoL(
                pose_rtde,
                self.servo_velocity,
                self.servo_acceleration,
                self.servo_dt,
                self.servo_lookahead_time,
                self.servo_gain,
            )
            self.rtde_c.waitPeriod(t_start)

            if ok:
                rospy.logdebug(f"ServoL sent: {[round(v, 4) for v in pose_rtde]}")
            else:
                rospy.logerr('ServoL command failed')
        except Exception as e:
            rospy.logerr(f"Error in servol_callback: {str(e)}")

    def pose_to_rtde(self, pose_msg: Pose) -> list:
        # Convert ROS Pose -> [x, y, z, rx, ry, rz]
        position = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        quat = [
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ]
        rot = Rotation.from_quat(quat)
        rotvec = rot.as_rotvec()
        return position + rotvec.tolist()

    def publish_robot_state(self):
        # Publish /joint_states
        q = self.rtde_r.getActualQ()
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]
        joint_msg.position = q
        self.joint_pub.publish(joint_msg)

        # Publish /arm/tcp_pose
        tcp = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'base'
        pose_msg.pose.position.x = tcp[0]
        pose_msg.pose.position.y = tcp[1]
        pose_msg.pose.position.z = tcp[2]

        quat = Rotation.from_rotvec([tcp[3], tcp[4], tcp[5]]).as_quat()  # [x, y, z, w]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

    def run(self):
        rospy.loginfo('UR RTDE ROS interface started')
        while not rospy.is_shutdown():
            self.publish_robot_state()
            self.rate.sleep()

        self.rtde_c.stopScript()
        rospy.loginfo('RTDE connection closed')


if __name__ == '__main__':
    try:
        node = URRTDEInterface()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node crashed: {str(e)}")
        sys.exit(1)
```

</details>

#### RTDE ROS topics overview

| Category | Topic           | Message Type                | Purpose                              |
| -------- | --------------- | --------------------------- | ------------------------------------ |
| Control  | `/cmd/movej`    | `sensor_msgs/JointState`    | Joint-space MoveJ (point-to-point)   |
| Control  | `/cmd/movel`    | `geometry_msgs/Pose`        | Cartesian MoveL (linear motion)      |
| Control  | `/cmd/servoj`   | `sensor_msgs/JointState`    | Joint-space ServoJ (high-rate servo) |
| Control  | `/cmd/servol`   | `geometry_msgs/Pose`        | Cartesian ServoL (high-rate servo)   |
| Control  | `/cmd/speedj`   | `sensor_msgs/JointState`    | Joint speed command entry            |
| State    | `/joint_states` | `sensor_msgs/JointState`    | Robot joint state feedback           |
| State    | `/arm/tcp_pose` | `geometry_msgs/PoseStamped` | TCP pose feedback                    |

Recommended loop rate:

```python
self.rate = rospy.Rate(125)
```

### 2.4 RealSense Camera

1. Install dependencies:

```bash
sudo apt update
sudo apt install -y nlohmann-json3-dev
```

2. Install `librealsense` by official guide:

- <https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md>

3. Validate camera device:

```bash
realsense-viewer
```

4. Install/start `realsense-ros` (ROS1 branch):

- <https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy>

```bash
roslaunch realsense2_camera rs_camera.launch
```

5. D405 detection fix (common issue):

If `realsense-viewer` works but ROS cannot detect D405, patch PID in:
`realsense-ros/realsense2_camera/include/constants.h`

```cpp
const uint16_t RS405_PID = 0x0b5b; // DS5U
```

Then rebuild and re-source workspace:

```bash
catkin_make
source devel/setup.bash
```

### 2.5 Robotiq Gripper

1. Hardware and URCap:

- Connect RS485-to-USB to the control PC.
- Install [Robotiq URCap](https://blog.robotiq.com/hubfs/support-files/UCG-1.8.13_20230720.zip) on the teach pendant.
- Official support: <https://robotiq.com/support>

2. Python socket control code:

- Use `robotiq_gripper.py` for direct socket-based control.
- A minimal usage example is shown below.

```python
from robotiq_gripper import RobotiqGripper

g = RobotiqGripper()
g.connect("<gripper_ip>", 63352)
g.activate(auto_calibrate=True)
g.move_and_wait_for_pos(position=0, speed=128, force=64)    # open
g.move_and_wait_for_pos(position=255, speed=128, force=64)  # close
g.disconnect()
```

> Recommended: validate communication with low speed/force first, then increase parameters gradually.

______________________________________________________________________

## 3. Real-Robot Startup Workflow

For real-robot inference, use separate terminals for system bring-up and RTDE control so motion can be stopped immediately when needed.

### 3.1 Terminal A: Start the main system

Save the following as `ur_control/launch/ur_bringup.launch` (adjust package paths if needed):

<details>
<summary><strong>Click to expand: ur_bringup.launch</strong></summary>

```xml
<launch>
  <!-- Use RTDE-only mode: UR ROS Driver can be omitted -->
  <!--
  <include file="$(find ur_robot_driver)/launch/ur3_bringup.launch">
    <arg name="robot_ip" value="192.168.8.202" />
    <arg name="kinematics_config" default="$(find ur_control)/etc/ex-ur3-1_calibration.yaml"/>
    <arg name="controllers" value="joint_state_controller twist_controller" />
  </include>
  -->

  <!-- Realsense cameras -->
  <include file="$(find realsense2_camera)/launch/rs_camera.launch">
    <arg name="serial_no" value="218622279630" />
    <arg name="camera" value="wrist_camera" />
    <arg name="align_depth" value="true"/>
  </include>

  <include file="$(find realsense2_camera)/launch/rs_camera.launch">
    <arg name="serial_no" value="338522301403" />
    <arg name="camera" value="front_camera" />
    <arg name="align_depth" value="true"/>
  </include>

  <!-- Robotiq control -->
  <node pkg="robotiq" type="robotiq_server.py" name="robotiq_server" output="screen" />

  <!-- Visualization -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find ur_control)/configs/show.rviz" />
</launch>
```

</details>

Start it:

```bash
roslaunch ur_control ur_bringup.launch
```

### 3.2 Terminal B (separate): Start RTDE control node

```bash
rosrun ur_control rtde_ros.py
```

______________________________________________________________________

## 4. FluxVLA Inference

Use VSCode `debugpy` for inference in debug mode.

Copy the following content into `launch.json`:

```jsonc
{
    "name": "Inference gr00t",
    "type": "debugpy",
    "request": "launch",
    "program": "{CONDA_ENV_PATH}/bin/torchrun",
    "python": "{CONDA_ENV_PATH}/bin/python",
    "args": [
        "scripts/inference_real_robot.py",
        "--config", "{PROJECT_PATH}/configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py",
        "--ckpt-path", "{CHECKPOINT_PATH}/checkpoints/step-001000-epoch-00-loss=0.0072.pt"
    ],
    "console": "integratedTerminal",
    "justMyCode": false,
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "WANDB_MODE": "disabled"
    }
}
```

> Start FluxVLA inference only after the real-robot stack is fully ready.
> To switch model, update `--config` and `--ckpt-path`.
