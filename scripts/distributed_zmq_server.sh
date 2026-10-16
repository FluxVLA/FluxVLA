#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${MLP_WORKER_GPU:-gpu}}"
WORLD_SIZE="${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec torchrun \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --nnodes="${WORLD_SIZE}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    scripts/distributed_zmq_server.py \
    "$@"
