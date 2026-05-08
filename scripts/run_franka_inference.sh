#!/bin/bash
# Franka 双臂推理启动脚本
# 解决 libffi 版本冲突问题

# 设置环境变量，优先使用系统的 libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7

# 确保 ROS 环境已加载
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/noetic/setup.bash
fi

# 运行推理
cd /home/franka/FluxVLA

python scripts/inference_real_robot.py \
    --config configs/pi05/pi05_paligemma_franka_dual_inference.py \
    --ckpt-path "$@"
