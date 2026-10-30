#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLA_ROOT=${VLA_ROOT:-/ceph/teamGolden/andyzeng/VLA}
MEGATRON_ROOT=${MEGATRON_ROOT:-${VLA_ROOT}/Megatron-LM}
export PYTHONPATH="${ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLUXVLA_MCORE_PRESERVE_FP32_PARAMS="${FLUXVLA_MCORE_PRESERVE_FP32_PARAMS:-0}"

PRECISION_OPT_ARGS=()
if [[ "${LINGBOT_USE_PRECISION_AWARE_OPTIMIZER:-0}" == "1" ]]; then
  # Transformer Engine's precision-aware Adam keeps the parameter remainder
  # state in the optimizer.  Keep this opt-in: it requires a matching TE
  # extension and changes optimizer-state dtypes relative to the baseline.
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
  DATA_ARGS+=(
    --lingbot-data-path "${LINGBOT_DATA_PATH:-/ceph/teamGolden/datasets/huggingface/lerobot/lingbot_beat_block_hammer_reduced}"
    --lingbot-norm-stats "${LINGBOT_NORM_STATS:-${ROOT}/fluxvla/datasets/lingbot_vla_v2/assets/robotwin_norm_stats.json}"
    --lingbot-num-workers "${LINGBOT_NUM_WORKERS:-0}"
    --lingbot-online-teacher-mode "${LINGBOT_ONLINE_TEACHER_MODE:-disabled}"
  )
else
  : "${LINGBOT_MANIFEST:?LINGBOT_MANIFEST is required when LINGBOT_DATA_MODE=replay}"
  DATA_ARGS+=(--lingbot-manifest "${LINGBOT_MANIFEST}")
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
  --lingbot-native-config "${LINGBOT_NATIVE_CONFIG:-${VLA_ROOT}/lingbot_runs/reduced_beat_block_hammer/lingbotvla_cli.yaml}" \
  --lingbot-checkpoint "${LINGBOT_CHECKPOINT:-${VLA_ROOT}/models/lingbot-vla-v2-6b}" \
  --lingbot-tokenizer-path "${LINGBOT_TOKENIZER_PATH:-${VLA_ROOT}/models/Qwen3-VL-4B-Instruct}" \
  "${DATA_ARGS[@]}" \
  --lingbot-train-scope "${LINGBOT_TRAIN_SCOPE:-all}" \
  --lingbot-expert-lr-multiplier "${LINGBOT_EXPERT_LR_MULTIPLIER:-2.8284271247461903}" \
  --lingbot-moe-implementation "${LINGBOT_MOE_IMPLEMENTATION:-auto}" \
  --lingbot-flow-rng-seed "${LINGBOT_FLOW_RNG_SEED:-42}" \
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
