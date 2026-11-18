#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FLUX_ENV_ACTIVATE="${FLUX_ENV_ACTIVATE:-/mnt/data/cpfs/users/danny/env/bin/activate}"
source "$FLUX_ENV_ACTIVATE"
cd "$PROJECT_ROOT"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PROFILE=smolvla_libero_10_ppo
ENTRYPOINT=scripts/rl_train.py
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) PROFILE=smolvla_libero_10_ppo_smoke; shift ;;
    --eval) ENTRYPOINT=scripts/rl_eval.py; shift ;;
    *) break ;;
  esac
done

# Forward Hydra overrides, including --cfg job --resolve and resume_dir.
exec python -u "$ENTRYPOINT" model=smolvla simulator="$PROFILE" "$@"
