#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

CONFIG="${CONFIG:-${REPO_DIR}/configs/pi05/pi05_paligemma_libero_spatial_full_finetune.py}"
CKPT="${CKPT:-${REPO_DIR}/work_dirs/pi05_base_eval/checkpoints/pi05_base_fluxvla.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/ceph/teamGolden/andyzeng/VLA/outputs/rl/fluxvla_e2e/baseline_smoke}"
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-40000}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
PYTHON_BIN="${PYTHON_BIN:-/tmp/fluxvla-e2e-venv/bin/python}"

for path in "${CONFIG}" "${CKPT}" "${PYTHON_BIN}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] Required path does not exist: ${path}" >&2
        exit 2
    fi
done

STATS_PATH="$(dirname "$(dirname "${CKPT}")")/dataset_statistics.json"
if [[ ! -f "${STATS_PATH}" ]]; then
    echo "[ERROR] Missing checkpoint dataset statistics: ${STATS_PATH}" >&2
    exit 2
fi

GPU_STATE="$(${PYTHON_BIN} - "${GPU_ID}" <<'PY'
import subprocess
import sys

gpu_id = sys.argv[1]
output = subprocess.check_output(
    [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip()
free_memory, utilization = (int(value.strip()) for value in output.split(","))
print(f"{free_memory} {utilization}")
PY
)"
read -r FREE_MEMORY_MB GPU_UTILIZATION <<<"${GPU_STATE}"

if (( FREE_MEMORY_MB < MIN_FREE_MEMORY_MB || GPU_UTILIZATION > MAX_GPU_UTILIZATION )); then
    echo "[BLOCKED] GPU ${GPU_ID} is busy: free=${FREE_MEMORY_MB} MiB, util=${GPU_UTILIZATION}%." >&2
    echo "[BLOCKED] Required: free>=${MIN_FREE_MEMORY_MB} MiB and util<=${MAX_GPU_UTILIZATION}%." >&2
    exit 3
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TASK_IDS="${TASK_IDS:-0}"
export NUM_GPUS=1
export MAX_TASKS_PER_GPU=1
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
export SAVE_ROLLOUT_VIDEOS=True
export SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS:-True}"
export MODEL_BUILD_DEVICE=cuda
export MODEL_BUILD_DTYPE=bf16
export OUTPUT_DIR

echo "[RUN] FluxVLA Gate 2 LIBERO smoke"
echo "[RUN] config=${CONFIG}"
echo "[RUN] checkpoint=${CKPT}"
echo "[RUN] output=${OUTPUT_DIR}"
echo "[RUN] gpu=${GPU_ID}, free=${FREE_MEMORY_MB} MiB, util=${GPU_UTILIZATION}%"

bash scripts/eval_libero_manager.sh "${CONFIG}" "${CKPT}" \
    --cfg-options eval.save_action_traces=True
