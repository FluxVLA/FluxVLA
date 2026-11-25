#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLA_ROOT=${VLA_ROOT:-${ROOT}/..}
MEGATRON_ROOT=${MEGATRON_ROOT:-${VLA_ROOT}/Megatron-LM}
: "${LINGBOT_NATIVE_CONFIG:?Set LINGBOT_NATIVE_CONFIG to the matching model/training YAML}"
: "${LINGBOT_CHECKPOINT:?Set LINGBOT_CHECKPOINT to the verified pretrained checkpoint}"
: "${LINGBOT_TOKENIZER_PATH:?Set LINGBOT_TOKENIZER_PATH to the matching tokenizer directory}"
[[ -d "${MEGATRON_ROOT}/megatron" ]] || { echo "Invalid MEGATRON_ROOT: ${MEGATRON_ROOT}" >&2; exit 2; }
[[ -f "${LINGBOT_NATIVE_CONFIG}" ]] || { echo "Model YAML not found: ${LINGBOT_NATIVE_CONFIG}" >&2; exit 2; }
for input_dir in "${LINGBOT_CHECKPOINT}" "${LINGBOT_TOKENIZER_PATH}"; do
  [[ -d "${input_dir}" ]] || { echo "Input directory not found: ${input_dir}" >&2; exit 2; }
done
export PYTHONPATH="${ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLUXVLA_MCORE_PRESERVE_FP32_PARAMS="${FLUXVLA_MCORE_PRESERVE_FP32_PARAMS:-0}"

PRECISION_OPT_ARGS=()
if [[ "${LINGBOT_USE_PRECISION_AWARE_OPTIMIZER:-0}" == "1" ]]; then
  # Transformer Engine's precision-aware Adam keeps the parameter remainder
  # state in the optimizer.  Keep this opt-in: it requires a matching
  # Transformer Engine extension and changes optimizer-state dtypes relative
  # to the baseline.
  PRECISION_OPT_ARGS=(
    --use-precision-aware-optimizer
    --main-params-dtype "${LINGBOT_PRECISION_MAIN_PARAMS_DTYPE:-fp16}"
    --main-grads-dtype "${LINGBOT_PRECISION_MAIN_GRADS_DTYPE:-bf16}"
    --exp-avg-dtype "${LINGBOT_PRECISION_EXP_AVG_DTYPE:-bf16}"
    --exp-avg-sq-dtype "${LINGBOT_PRECISION_EXP_AVG_SQ_DTYPE:-bf16}"
  )
fi

MOE_ARGS=()
if [[ "${EP:-1}" != "1" ]]; then
  MOE_ARGS+=(--num-experts "${NUM_EXPERTS:-32}" --moe-router-topk "${MOE_TOPK:-4}" --moe-router-pre-softmax --disable-bias-linear)
fi

DATA_MODE=${LINGBOT_DATA_MODE:-online}
DATA_ARGS=(--lingbot-data-mode "${DATA_MODE}")
if [[ "${DATA_MODE}" == "online" ]]; then
  : "${LINGBOT_DATA_PATH:?Set LINGBOT_DATA_PATH for online training}"
  [[ -d "${LINGBOT_DATA_PATH}" ]] || { echo "Dataset not found: ${LINGBOT_DATA_PATH}" >&2; exit 2; }
  NORM_STATS=${LINGBOT_NORM_STATS:-${ROOT}/configs/lingbot_vla_v2/assets/robotwin_norm_stats.json}
  [[ -f "${NORM_STATS}" ]] || { echo "Normalization statistics not found: ${NORM_STATS}" >&2; exit 2; }
  DATA_ARGS+=(
    --lingbot-data-path "${LINGBOT_DATA_PATH}"
    --lingbot-norm-stats "${NORM_STATS}"
    --lingbot-video-backend "${LINGBOT_VIDEO_BACKEND:-torchcodec}"
    --lingbot-resize-dtype "${LINGBOT_RESIZE_DTYPE:-float32}"
    # Zero isolates parity diagnostics; tune workers on real online workloads.
    --lingbot-num-workers "${LINGBOT_NUM_WORKERS:-0}"
    --lingbot-online-teacher-mode "${LINGBOT_ONLINE_TEACHER_MODE:-disabled}"
  )
elif [[ "${DATA_MODE}" == "replay" ]]; then
  : "${LINGBOT_MANIFEST:?LINGBOT_MANIFEST is required when LINGBOT_DATA_MODE=replay}"
  [[ -f "${LINGBOT_MANIFEST}" ]] || { echo "Replay manifest not found: ${LINGBOT_MANIFEST}" >&2; exit 2; }
  DATA_ARGS+=(--lingbot-manifest "${LINGBOT_MANIFEST}")
else
  echo "Unsupported LINGBOT_DATA_MODE: ${DATA_MODE}" >&2
  exit 2
fi

python -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --master_port="${MASTER_PORT:-29500}" \
  "${ROOT}/scripts/train_mcore_lingbot_vla_v2.py" \
  --tensor-model-parallel-size "${TP:-1}" \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size "${EP:-1}" \
  $(if [[ "${SP:-0}" == "1" ]]; then echo --sequence-parallel; fi) \
  --micro-batch-size "${MB:-1}" \
  --global-batch-size "${GB:-2}" \
  --train-iters "${STEPS:-2}" \
  --eval-iters "${EVAL_ITERS:-0}" \
  --hidden-size 2560 --ffn-hidden-size 9728 --num-layers 36 \
  --num-attention-heads 32 --seq-length 2048 --max-position-embeddings 32768 \
  "${MOE_ARGS[@]}" \
  --bf16 --no-gradient-accumulation-fusion \
  $(if [[ "${LINGBOT_DDP_AVERAGE_IN_COLLECTIVE:-1}" == "1" ]]; then echo --ddp-average-in-collective; fi) \
  $(if [[ "${LINGBOT_USE_DISTRIBUTED_OPTIMIZER:-1}" == "1" ]]; then echo --use-distributed-optimizer; fi) \
  --optimizer adam --lr 1e-4 --min-lr 5e-5 \
  --lr-decay-style cosine --lr-decay-iters "${LR_DECAY_ITERS:-100}" \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0 --clip-grad 1.0 --seed 42 \
  "${PRECISION_OPT_ARGS[@]}" \
  --lingbot-native-config "${LINGBOT_NATIVE_CONFIG}" \
  --lingbot-checkpoint "${LINGBOT_CHECKPOINT}" \
  --lingbot-tokenizer-path "${LINGBOT_TOKENIZER_PATH}" \
  "${DATA_ARGS[@]}" \
  --lingbot-train-scope "${LINGBOT_TRAIN_SCOPE:-all}" \
  --lingbot-expert-lr-multiplier "${LINGBOT_EXPERT_LR_MULTIPLIER:-2.8284271247461903}" \
  --lingbot-moe-implementation "${LINGBOT_MOE_IMPLEMENTATION:-auto}" \
  --lingbot-flow-rng-seed "${LINGBOT_FLOW_RNG_SEED:-42}" \
  --lingbot-sampler-seed "${LINGBOT_SAMPLER_SEED:-42}" \
  $(if [[ "${LINGBOT_TP_ATTENTION:-1}" == "0" ]]; then echo --no-lingbot-tp-attention; fi) \
  $(if [[ "${LINGBOT_TP_MLP:-1}" == "0" ]]; then echo --no-lingbot-tp-mlp; fi) \
  $(if [[ "${LINGBOT_TP_ROUTED_EXPERTS:-1}" == "0" ]]; then echo --no-lingbot-tp-routed-experts; fi) \
  $(if [[ "${LINGBOT_TP_SHARED_EXPERT:-1}" == "0" ]]; then echo --no-lingbot-tp-shared-expert; fi) \
  $(if [[ "${LINGBOT_TP_MLP_COLUMN:-1}" == "0" ]]; then echo --no-lingbot-tp-mlp-column; fi) \
  $(if [[ "${LINGBOT_TP_MLP_ROW:-1}" == "0" ]]; then echo --no-lingbot-tp-mlp-row; fi) \
  $(if [[ "${LINGBOT_TP_PARITY_REFERENCE:-0}" == "1" ]]; then echo --lingbot-tp-parity-reference; fi) \
  $(if [[ "${LINGBOT_PRESERVE_FSDP2_FP32_MODULES:-1}" == "0" ]]; then echo --no-lingbot-preserve-fsdp2-fp32-modules; fi) \
  $(if [[ "${LINGBOT_FSDP2_STRICT_DP_REDUCTION:-0}" == "1" ]]; then echo --lingbot-fsdp2-strict-dp-reduction; fi) \
  "$@"
