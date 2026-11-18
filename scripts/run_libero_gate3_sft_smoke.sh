#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
WORKSPACE_DIR="$(dirname "${REPO_DIR}")"

CONFIG="${CONFIG:-${REPO_DIR}/configs/pi05/pi05_paligemma_libero_10_full_finetune.py}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${REPO_DIR}/checkpoints/pi05_base/model.safetensors}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE_DIR}/datasets/libero_10_no_noops_lerobotv2.1}"
WORK_DIR="${WORK_DIR:-${WORKSPACE_DIR}/outputs/rl/fluxvla_e2e/sft_smoke}"
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-60000}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
PYTHON_BIN="${PYTHON_BIN:-/tmp/fluxvla-e2e-venv/bin/python}"

for path in "${CONFIG}" "${PRETRAINED_CKPT}" "${DATASET_DIR}" "${PYTHON_BIN}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] Required path does not exist: ${path}" >&2
        exit 2
    fi
done

for path in "${DATASET_DIR}/meta/info.json" "${DATASET_DIR}/meta/episodes.jsonl"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Dataset contract file does not exist: ${path}" >&2
        exit 2
    fi
done

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

mkdir -p "${WORK_DIR}"
cd "${REPO_DIR}"

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NPROC_PER_NODE=1

echo "[RUN] FluxVLA Gate 3 two-step LIBERO-10 SFT smoke"
echo "[RUN] config=${CONFIG}"
echo "[RUN] pretrained=${PRETRAINED_CKPT}"
echo "[RUN] dataset=${DATASET_DIR}"
echo "[RUN] work_dir=${WORK_DIR}"
echo "[RUN] gpu=${GPU_ID}, free=${FREE_MEMORY_MB} MiB, util=${GPU_UTILIZATION}%"

bash scripts/train.sh "${CONFIG}" "${WORK_DIR}" \
    --cfg-options \
    model.pretrained_name_or_path="${PRETRAINED_CKPT}" \
    train_dataloader.dataset.datasets.data_root_path="${DATASET_DIR}" \
    train_dataloader.per_device_batch_size=1 \
    train_dataloader.per_device_num_workers=1 \
    runner.max_epochs=None \
    runner.max_steps=2 \
    runner.save_iter_interval=2 \
    runner.max_keep_ckpts=1
