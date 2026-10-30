#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BACKEND=${BACKEND:-mcore}
STRATEGY=${STRATEGY:-dp4}
MB=${MB:-1}
STEPS=${STEPS:-100}
RECOMPUTE=${RECOMPUTE:-0}
GPUS=${GPUS:-0,1,2,3}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/work_dirs/lingbot_gb64_4gpu_100step_20261026}
MANIFEST=${LINGBOT_MANIFEST:-${OUTPUT_ROOT}/gb64_100step_manifest.json}

case "${STRATEGY}" in
  dp4) TP=1; EP=1; SP=0 ;;
  tp2) TP=2; EP=1; SP=0 ;;
  tp2_sp) TP=2; EP=1; SP=1 ;;
  ep2) TP=1; EP=2; SP=0 ;;
  tp2_ep2) TP=2; EP=2; SP=0 ;;
  *) echo "unsupported STRATEGY=${STRATEGY}" >&2; exit 2 ;;
esac

DEFAULT_MOE_IMPLEMENTATION=fused
if [[ "${EP}" != 1 || "${SP}" == 1 ]]; then
  DEFAULT_MOE_IMPLEMENTATION=eager
fi
MOE_IMPLEMENTATION=${LINGBOT_MOE_IMPLEMENTATION:-${DEFAULT_MOE_IMPLEMENTATION}}

if [[ "${BACKEND}" == native && "${STRATEGY}" != dp4 ]]; then
  echo "Native FSDP only accepts STRATEGY=dp4" >&2
  exit 2
fi
if (( 64 % (4 * MB) != 0 )); then
  echo "GB64 is not divisible by 4 GPUs * MB=${MB}" >&2
  exit 2
fi

MODE=norecompute
RECOMPUTE_ARG=(--no-lingbot-gradient-checkpointing)
if [[ "${RECOMPUTE}" == 1 ]]; then
  MODE=recompute
  RECOMPUTE_ARG=()
fi
CASE_NAME="${BACKEND}_${STRATEGY}_mb${MB}_${MODE}"
OUT=${OUT:-${OUTPUT_ROOT}/${CASE_NAME}}
mkdir -p "${OUT}/tmp"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export TMPDIR="${OUT}/tmp"
export WANDB_MODE=disabled

monitor_pid=
cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
(
  while true; do
    date +%s
    nvidia-smi -i "${GPUS}" --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 2
  done
) >"${OUT}/gpu.csv" 2>&1 &
monitor_pid=$!

cd "${ROOT}"
if [[ "${BACKEND}" == native ]]; then
  export PYTHONPATH="${ROOT}:${ROOT}/../Megatron-LM:${PYTHONPATH:-}"
  # Keep the production 100-step scheduler horizon for short capacity/perf
  # runs. The runner stop hook exits after STEPS without changing LR decay.
  export FLUXVLA_PARITY_STOP_AFTER_STEP="${STEPS}"
  NATIVE_RECOMPUTE=False
  if [[ "${RECOMPUTE}" == 1 ]]; then
    NATIVE_RECOMPUTE=True
  fi
  GA=$((64 / (4 * MB)))
  NPROC_PER_NODE=4 MASTER_PORT="${MASTER_PORT:-29650}" \
    bash scripts/train.sh \
      configs/lingbot_vla_v2/lingbot_vla_v2_gb64_4gpu_parity.py \
      "${OUT}" \
      --cfg-options \
        train_dataloader.per_device_batch_size="${MB}" \
        runner.grad_accumulation_steps="${GA}" \
        runner.enable_gradient_checkpointing="${NATIVE_RECOMPUTE}" \
        runner.metric.run_dir="${OUT}" \
        runner.max_steps=100 2>&1 | tee "${OUT}/console.log"
elif [[ "${BACKEND}" == mcore ]]; then
  NPROC_PER_NODE=4 MASTER_PORT="${MASTER_PORT:-29650}" \
  TP="${TP}" EP="${EP}" SP="${SP}" MB="${MB}" GB=64 STEPS="${STEPS}" \
  LR_DECAY_ITERS=100 LINGBOT_MANIFEST="${MANIFEST}" \
  LINGBOT_TRAIN_SCOPE=all LINGBOT_MOE_IMPLEMENTATION="${MOE_IMPLEMENTATION}" \
  LINGBOT_FSDP2_STRICT_DP_REDUCTION=0 \
    bash scripts/train_mcore_lingbot_vla_v2.sh \
      --lingbot-replay-order official_step_rank \
      --lingbot-sampler-seed 42 --log-interval 1 --eval-iters 0 \
      "${RECOMPUTE_ARG[@]}" 2>&1 | tee "${OUT}/console.log"
else
  echo "unsupported BACKEND=${BACKEND}" >&2
  exit 2
fi
