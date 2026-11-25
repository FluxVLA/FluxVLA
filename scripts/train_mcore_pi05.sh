#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLA_ROOT=${VLA_ROOT:-/ceph/teamGolden/andyzeng/VLA}
export PYTHONPATH="${VLA_ROOT}/pi05_training/src:${ROOT}:${MEGATRON_ROOT:-${VLA_ROOT}/Megatron-LM}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-1}" \
  "${ROOT}/scripts/train_mcore_pi05.py" \
  --tensor-model-parallel-size "${TP:-1}" \
  --pipeline-model-parallel-size "${PP:-1}" \
  --micro-batch-size "${MB:-1}" \
  --global-batch-size "${GB:-1}" \
  --train-iters "${STEPS:-10}" \
  --hidden-size 2048 --ffn-hidden-size 16384 --num-layers 18 \
  --num-attention-heads 8 --seq-length 512 --max-position-embeddings 8192 \
  --bf16 --no-gradient-accumulation-fusion --use-distributed-optimizer \
  "$@"
