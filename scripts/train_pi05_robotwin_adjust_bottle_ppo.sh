#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /mnt/data/cpfs/users/danny/env/bin/activate
cd "$PROJECT_ROOT"

export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Hydra overrides (including --cfg job --resolve) are forwarded unchanged.
exec python -u scripts/rl_train.py \
  model=pi05 simulator=pi05_robotwin_adjust_bottle_ppo "$@"
