#!/usr/bin/env bash
# Train FluxVLA with the read-only Megatron-LM checkout on PYTHONPATH.
set -euo pipefail

MEGATRON_ROOT="${MEGATRON_ROOT:-/ceph/teamGolden/andyzeng/VLA/Megatron-LM}"
export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH:-}"

CONFIG="${1:-configs/pi05/pi05_paligemma_libero_10_mcore.py}"
WORK_DIR="${2:-work_dirs/pi05_libero_mcore}"
shift 2 || true

exec bash scripts/train.sh "${CONFIG}" "${WORK_DIR}" "$@"
