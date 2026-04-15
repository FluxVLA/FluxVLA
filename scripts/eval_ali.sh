#!/bin/bash

CONFIG=$1
CKPT_PATH=$2

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "scripts/eval.py" \
  --config "${CONFIG}" \
  --ckpt-path "${CKPT_PATH}" \
  ${@:3}
