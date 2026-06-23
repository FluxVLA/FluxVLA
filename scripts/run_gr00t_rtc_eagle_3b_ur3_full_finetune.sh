#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${USE_EXISTING_ROS_ENV:-0}" != "1" ]]; then
  unset ROS_MASTER_URI
  unset ROS_IP
fi

CONFIG_PATH="${CONFIG_PATH:-configs/gr00t/gr00t_rtc_eagle_3b_ur3_full_finetune_kernel_inference.py}"
CKPT_PATH="${CKPT_PATH:-/home/ur3/sober/ur_checkpoint/gr00t_rtc_eagle_3b_ur3_full_finetune_orin/checkpoints/step-006000-epoch-00-loss=0.0946.safetensors}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" scripts/inference_real_robot.py \
  --config "${CONFIG_PATH}" \
  --ckpt-path "${CKPT_PATH}" \
  "$@"
