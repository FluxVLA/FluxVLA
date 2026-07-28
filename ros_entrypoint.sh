#!/bin/bash
set -e

if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
    if [ -n "${ROS_MASTER_URI:-}" ]; then
        echo "[ROS] ROS_MASTER_URI = ${ROS_MASTER_URI}"
        echo "[ROS] ROS_IP         = ${ROS_IP:-<not set>}"
    fi
fi

exec "$@"
