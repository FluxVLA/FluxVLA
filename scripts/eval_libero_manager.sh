#!/usr/bin/env bash
# Dynamic LIBERO eval manager for configs that use LiberoEvalRunner.
#
# The manager builds a suite/task queue and launches one single-task eval
# worker per task. Each worker runs with NPROC_PER_NODE=1 and writes
# manager artifacts directly under OUTPUT_DIR.
#
# Usage:
#   CONFIG=configs/model/my_libero_eval.py CKPT=/path/to/ckpt.safetensors \
#     bash scripts/eval_libero_manager.sh
#
# Positional CONFIG and CKPT are also accepted:
#   bash scripts/eval_libero_manager.sh configs/model/my_libero_eval.py /path/to/ckpt.safetensors
#
# Common overrides:
#   SUITES="libero_10 libero_goal libero_spatial libero_object"
#   NUM_GPUS=8
#   MAX_TASKS_PER_GPU=2
#   NUM_TRIALS_PER_TASK=50
#   MODEL_BUILD_DEVICE=cuda
#   MODEL_BUILD_DTYPE=bf16
#   PREPROCESS_EVERY_STEP=False
#   SAVE_ROLLOUT_VIDEOS=True
#   SAVE_MULTI_VIEW_ROLLOUT_VIDEOS=True
#   ROLLOUT_DIR=work_dirs/libero_eval_manager/videos
#   OUTPUT_DIR=work_dirs/libero_eval_manager/my_run
#   FEISHU_SHEET_URL=https://.../sheets/...
#   FEISHU_APP_ID=cli_xxx
#   FEISHU_APP_SECRET=xxx
#
# Defaults are resolved as: environment override -> config value -> built-in
# fallback. Manager-only defaults should be set in config via
# ``eval = dict(runner=dict(...), manager=dict(...))``.
# If OUTPUT_DIR is unset, the manager uses eval.manager.output_dir, then
# eval.runner.result_output_dir, then a timestamped work_dirs fallback.
#
# OUTPUT_DIR layout:
#   summary.{csv,txt,json}, suite_success_rates.csv,
#   task_success_rates.csv, failed_tasks.txt,
#   task_logs/, task_status/, <suite>/gpuX_taskY_results.json,
#   <suite>/videos/*.mp4 unless ROLLOUT_DIR/eval.rollout_dir overrides it
#
# Extra arguments are forwarded to scripts/eval.py after --cfg-options.
set -euo pipefail

if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG="$1"
  shift
fi
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CKPT="$1"
  shift
fi

CONFIG="${CONFIG:?set CONFIG or pass it as the first argument}"
CKPT="${CKPT:?set CKPT or pass it as the second argument}"
MAX_STEPS="${MAX_STEPS:-}"
EVAL_SHARD_STRATEGY="${EVAL_SHARD_STRATEGY:-}"
TASK_IDS="${TASK_IDS:-}"
PREPROCESS_EVERY_STEP="${PREPROCESS_EVERY_STEP:-}"
SAVE_FAILED_ROLLOUT_VIDEOS="${SAVE_FAILED_ROLLOUT_VIDEOS:-}"
SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS:-}"
ROLLOUT_DIR="${ROLLOUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
FEISHU_SHEET_URL="${FEISHU_SHEET_URL:-}"
FEISHU_APP_ID="${FEISHU_APP_ID:-}"
FEISHU_APP_SECRET="${FEISHU_APP_SECRET:-}"
FEISHU_TIMEOUT="${FEISHU_TIMEOUT:-10}"
USER_CFG_OPTIONS=()
WORKER_USER_CFG_OPTIONS=()
FORWARDED_EXTRA_ARGS=()

split_user_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --cfg-options)
        shift
        while [[ $# -gt 0 && "$1" != --* ]]; do
          USER_CFG_OPTIONS+=("$1")
          case "$1" in
            eval.manager.*) ;;
            *) WORKER_USER_CFG_OPTIONS+=("$1") ;;
          esac
          shift
        done
        ;;
      *)
        FORWARDED_EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

# Manager-only overrides control scheduling and must not reach eval.py workers.
split_user_args "$@"

CFG_PARSE_ARGS=("${CONFIG}")
if [[ "${#USER_CFG_OPTIONS[@]}" -gt 0 ]]; then
  CFG_PARSE_ARGS+=("--cfg-options" "${USER_CFG_OPTIONS[@]}")
fi

DEFAULT_SUITES="libero_10 libero_goal libero_spatial libero_object"
DEFAULT_NUM_GPUS="8"
DEFAULT_MAX_TASKS_PER_GPU="2"
DEFAULT_MASTER_PORT_BASE="${MASTER_PORT:-29690}"
DEFAULT_MONITOR_INTERVAL="5"
DEFAULT_STATUS_INTERVAL="30"
DEFAULT_LAUNCH_DELAY="0.5"
DEFAULT_SUMMARY_TOOL="tools/summarize_libero_eval_results.py"
DEFAULT_NUM_TRIALS_PER_TASK="50"
DEFAULT_MODEL_BUILD_DEVICE="cuda"
DEFAULT_MODEL_BUILD_DTYPE="bf16"
DEFAULT_EVAL_SHARD_STRATEGY="task"
DEFAULT_SAVE_ROLLOUT_VIDEOS="True"
DEFAULT_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="False"

CFG_EVAL_RUNNER_PREFIX=""
CFG_SUITES=""
CFG_NUM_GPUS=""
CFG_MAX_TASKS_PER_GPU=""
CFG_NUM_TRIALS_PER_TASK=""
CFG_MODEL_BUILD_DEVICE=""
CFG_MODEL_BUILD_DTYPE=""
CFG_MASTER_PORT_BASE=""
CFG_MONITOR_INTERVAL=""
CFG_STATUS_INTERVAL=""
CFG_LAUNCH_DELAY=""
CFG_SUMMARY_TOOL=""
CFG_TASK_FILE=""
CFG_TASK_IDS=""
CFG_OUTPUT_DIR=""
CFG_SAVE_ROLLOUT_VIDEOS=""
CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS=""
CFG_ROLLOUT_DIR=""
CFG_FEISHU_SHEET_URL=""
CFG_FEISHU_APP_ID=""
CFG_FEISHU_APP_SECRET=""

CFG_VALUES="$(python - "${CFG_PARSE_ARGS[@]}" <<'PY'
import argparse

from mmengine import Config, DictAction


def get_path(obj, path):
    cur = obj
    for key in path.split('.'):
        if isinstance(cur, dict):
            if key not in cur:
                return None
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return None
            cur = getattr(cur, key)
    return cur


def first_path(obj, *paths):
    for path in paths:
        value = get_path(obj, path)
        if value is not None:
            return value
    return None


def format_value(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        return ' '.join(str(item) for item in value)
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--cfg-options', nargs='+', action=DictAction)
args, _ = parser.parse_known_args()

cfg = Config.fromfile(args.config)
if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)
eval_runner_prefix = 'eval.runner' if hasattr(cfg.eval, 'runner') else 'eval'
fields = {
    'CFG_SUITES': ('eval.runner.task_suite_name', 'eval.task_suite_name'),
    'CFG_NUM_GPUS': ('eval.manager.num_gpus', ),
    'CFG_MAX_TASKS_PER_GPU': ('eval.manager.max_tasks_per_gpu', ),
    'CFG_NUM_TRIALS_PER_TASK': (
        'eval.runner.num_trials_per_task', 'eval.num_trials_per_task'),
    'CFG_MODEL_BUILD_DEVICE': (
        'eval.runner.model_build_device', 'eval.model_build_device'),
    'CFG_MODEL_BUILD_DTYPE': (
        'eval.runner.model_build_dtype', 'eval.model_build_dtype'),
    'CFG_MASTER_PORT_BASE': ('eval.manager.master_port_base', ),
    'CFG_MONITOR_INTERVAL': ('eval.manager.monitor_interval', ),
    'CFG_STATUS_INTERVAL': ('eval.manager.status_interval', ),
    'CFG_LAUNCH_DELAY': ('eval.manager.launch_delay', ),
    'CFG_SUMMARY_TOOL': ('eval.manager.summary_tool', ),
    'CFG_TASK_FILE': ('eval.manager.task_file', ),
    'CFG_TASK_IDS': (
        'eval.manager.task_ids', 'eval.runner.task_ids', 'eval.task_ids'),
    'CFG_OUTPUT_DIR': (
        'eval.manager.output_dir', 'eval.runner.result_output_dir',
        'eval.output_dir', 'eval.result_output_dir'),
    'CFG_SAVE_ROLLOUT_VIDEOS': (
        'eval.runner.save_rollout_videos', 'eval.save_rollout_videos'),
    'CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS': (
        'eval.runner.save_multi_view_rollout_videos',
        'eval.save_multi_view_rollout_videos'),
    'CFG_ROLLOUT_DIR': ('eval.runner.rollout_dir', 'eval.rollout_dir'),
    'CFG_FEISHU_SHEET_URL': (
        'eval.manager.feishu_sheet_url', 'eval.feishu_sheet_url'),
    'CFG_FEISHU_APP_ID': (
        'eval.manager.feishu_app_id', 'eval.feishu_app_id'),
    'CFG_FEISHU_APP_SECRET': (
        'eval.manager.feishu_app_secret', 'eval.feishu_app_secret'),
}
print(f'CFG_EVAL_RUNNER_PREFIX\t{eval_runner_prefix}')
for name, paths in fields.items():
    print(f'{name}\t{format_value(first_path(cfg, *paths))}')
PY
)"

while IFS=$'\t' read -r key value; do
  case "${key}" in
    CFG_EVAL_RUNNER_PREFIX) CFG_EVAL_RUNNER_PREFIX="${value}" ;;
    CFG_SUITES) CFG_SUITES="${value}" ;;
    CFG_NUM_GPUS) CFG_NUM_GPUS="${value}" ;;
    CFG_MAX_TASKS_PER_GPU) CFG_MAX_TASKS_PER_GPU="${value}" ;;
    CFG_NUM_TRIALS_PER_TASK) CFG_NUM_TRIALS_PER_TASK="${value}" ;;
    CFG_MODEL_BUILD_DEVICE) CFG_MODEL_BUILD_DEVICE="${value}" ;;
    CFG_MODEL_BUILD_DTYPE) CFG_MODEL_BUILD_DTYPE="${value}" ;;
    CFG_MASTER_PORT_BASE) CFG_MASTER_PORT_BASE="${value}" ;;
    CFG_MONITOR_INTERVAL) CFG_MONITOR_INTERVAL="${value}" ;;
    CFG_STATUS_INTERVAL) CFG_STATUS_INTERVAL="${value}" ;;
    CFG_LAUNCH_DELAY) CFG_LAUNCH_DELAY="${value}" ;;
    CFG_SUMMARY_TOOL) CFG_SUMMARY_TOOL="${value}" ;;
    CFG_TASK_FILE) CFG_TASK_FILE="${value}" ;;
    CFG_TASK_IDS) CFG_TASK_IDS="${value}" ;;
    CFG_OUTPUT_DIR) CFG_OUTPUT_DIR="${value}" ;;
    CFG_SAVE_ROLLOUT_VIDEOS) CFG_SAVE_ROLLOUT_VIDEOS="${value}" ;;
    CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS) CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${value}" ;;
    CFG_ROLLOUT_DIR) CFG_ROLLOUT_DIR="${value}" ;;
    CFG_FEISHU_SHEET_URL) CFG_FEISHU_SHEET_URL="${value}" ;;
    CFG_FEISHU_APP_ID) CFG_FEISHU_APP_ID="${value}" ;;
    CFG_FEISHU_APP_SECRET) CFG_FEISHU_APP_SECRET="${value}" ;;
  esac
done <<< "${CFG_VALUES}"

EVAL_RUNNER_PREFIX="${CFG_EVAL_RUNNER_PREFIX:-eval}"

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR_SOURCE="env"
elif [[ -n "${CFG_OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${CFG_OUTPUT_DIR}"
  OUTPUT_DIR_SOURCE="config"
else
  OUTPUT_DIR="work_dirs/libero_eval_manager/$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR_SOURCE="default"
fi

if [[ -n "${SUITES:-}" ]]; then
  SUITES_SOURCE="env"
elif [[ -n "${CFG_SUITES}" ]]; then
  SUITES="${CFG_SUITES}"
  SUITES_SOURCE="config"
else
  SUITES="${DEFAULT_SUITES}"
  SUITES_SOURCE="default"
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
  NUM_GPUS_SOURCE="env"
elif [[ -n "${CFG_NUM_GPUS}" ]]; then
  NUM_GPUS="${CFG_NUM_GPUS}"
  NUM_GPUS_SOURCE="config"
else
  NUM_GPUS="${DEFAULT_NUM_GPUS}"
  NUM_GPUS_SOURCE="default"
fi

if [[ -n "${MAX_TASKS_PER_GPU:-}" ]]; then
  MAX_TASKS_PER_GPU_SOURCE="env"
elif [[ -n "${CFG_MAX_TASKS_PER_GPU}" ]]; then
  MAX_TASKS_PER_GPU="${CFG_MAX_TASKS_PER_GPU}"
  MAX_TASKS_PER_GPU_SOURCE="config"
else
  MAX_TASKS_PER_GPU="${DEFAULT_MAX_TASKS_PER_GPU}"
  MAX_TASKS_PER_GPU_SOURCE="default"
fi

if [[ -n "${MASTER_PORT_BASE:-}" ]]; then
  :
elif [[ -n "${MASTER_PORT:-}" ]]; then
  MASTER_PORT_BASE="${MASTER_PORT}"
elif [[ -n "${CFG_MASTER_PORT_BASE}" ]]; then
  MASTER_PORT_BASE="${CFG_MASTER_PORT_BASE}"
else
  MASTER_PORT_BASE="${DEFAULT_MASTER_PORT_BASE}"
fi

if [[ -n "${MONITOR_INTERVAL:-}" ]]; then
  :
elif [[ -n "${CFG_MONITOR_INTERVAL}" ]]; then
  MONITOR_INTERVAL="${CFG_MONITOR_INTERVAL}"
else
  MONITOR_INTERVAL="${DEFAULT_MONITOR_INTERVAL}"
fi

if [[ -n "${STATUS_INTERVAL:-}" ]]; then
  :
elif [[ -n "${CFG_STATUS_INTERVAL}" ]]; then
  STATUS_INTERVAL="${CFG_STATUS_INTERVAL}"
else
  STATUS_INTERVAL="${DEFAULT_STATUS_INTERVAL}"
fi

if [[ -n "${LAUNCH_DELAY:-}" ]]; then
  :
elif [[ -n "${CFG_LAUNCH_DELAY}" ]]; then
  LAUNCH_DELAY="${CFG_LAUNCH_DELAY}"
else
  LAUNCH_DELAY="${DEFAULT_LAUNCH_DELAY}"
fi

if [[ -n "${SUMMARY_TOOL:-}" ]]; then
  :
elif [[ -n "${CFG_SUMMARY_TOOL}" ]]; then
  SUMMARY_TOOL="${CFG_SUMMARY_TOOL}"
else
  SUMMARY_TOOL="${DEFAULT_SUMMARY_TOOL}"
fi

if [[ -n "${TASK_FILE:-}" ]]; then
  :
elif [[ -n "${CFG_TASK_FILE}" ]]; then
  TASK_FILE="${CFG_TASK_FILE}"
else
  TASK_FILE=""
fi

if [[ -n "${TASK_IDS:-}" ]]; then
  TASK_IDS_SOURCE="env"
elif [[ -n "${CFG_TASK_IDS}" ]]; then
  TASK_IDS="${CFG_TASK_IDS}"
  TASK_IDS_SOURCE="config"
else
  TASK_IDS_SOURCE="default"
fi

if [[ -n "${NUM_TRIALS_PER_TASK:-}" ]]; then
  NUM_TRIALS_PER_TASK_SOURCE="env"
elif [[ -n "${CFG_NUM_TRIALS_PER_TASK}" ]]; then
  NUM_TRIALS_PER_TASK="${CFG_NUM_TRIALS_PER_TASK}"
  NUM_TRIALS_PER_TASK_SOURCE="config"
else
  NUM_TRIALS_PER_TASK="${DEFAULT_NUM_TRIALS_PER_TASK}"
  NUM_TRIALS_PER_TASK_SOURCE="default"
fi

if [[ -n "${MODEL_BUILD_DEVICE:-}" ]]; then
  MODEL_BUILD_DEVICE_SOURCE="env"
elif [[ -n "${CFG_MODEL_BUILD_DEVICE}" ]]; then
  MODEL_BUILD_DEVICE="${CFG_MODEL_BUILD_DEVICE}"
  MODEL_BUILD_DEVICE_SOURCE="config"
else
  MODEL_BUILD_DEVICE="${DEFAULT_MODEL_BUILD_DEVICE}"
  MODEL_BUILD_DEVICE_SOURCE="default"
fi

if [[ -n "${MODEL_BUILD_DTYPE:-}" ]]; then
  MODEL_BUILD_DTYPE_SOURCE="env"
elif [[ -n "${CFG_MODEL_BUILD_DTYPE}" ]]; then
  MODEL_BUILD_DTYPE="${CFG_MODEL_BUILD_DTYPE}"
  MODEL_BUILD_DTYPE_SOURCE="config"
else
  MODEL_BUILD_DTYPE="${DEFAULT_MODEL_BUILD_DTYPE}"
  MODEL_BUILD_DTYPE_SOURCE="default"
fi

if [[ -n "${EVAL_SHARD_STRATEGY:-}" ]]; then
  EVAL_SHARD_STRATEGY_SOURCE="env"
else
  EVAL_SHARD_STRATEGY="${DEFAULT_EVAL_SHARD_STRATEGY}"
  EVAL_SHARD_STRATEGY_SOURCE="default"
fi

if [[ -n "${SAVE_ROLLOUT_VIDEOS:-}" ]]; then
  SAVE_ROLLOUT_VIDEOS_SOURCE="env"
elif [[ -n "${CFG_SAVE_ROLLOUT_VIDEOS}" ]]; then
  SAVE_ROLLOUT_VIDEOS="${CFG_SAVE_ROLLOUT_VIDEOS}"
  SAVE_ROLLOUT_VIDEOS_SOURCE="config"
else
  SAVE_ROLLOUT_VIDEOS="${DEFAULT_SAVE_ROLLOUT_VIDEOS}"
  SAVE_ROLLOUT_VIDEOS_SOURCE="default"
fi

if [[ -n "${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS:-}" ]]; then
  SAVE_MULTI_VIEW_ROLLOUT_VIDEOS_SOURCE="env"
elif [[ -n "${CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS}" ]]; then
  SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS}"
  SAVE_MULTI_VIEW_ROLLOUT_VIDEOS_SOURCE="config"
else
  SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${DEFAULT_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS}"
  SAVE_MULTI_VIEW_ROLLOUT_VIDEOS_SOURCE="default"
fi

if [[ -n "${ROLLOUT_DIR:-}" ]]; then
  ROLLOUT_DIR_SOURCE="env"
elif [[ -n "${CFG_ROLLOUT_DIR}" ]]; then
  ROLLOUT_DIR="${CFG_ROLLOUT_DIR}"
  ROLLOUT_DIR_SOURCE="config"
else
  ROLLOUT_DIR_SOURCE="default"
fi

if [[ -z "${FEISHU_SHEET_URL}" && -n "${CFG_FEISHU_SHEET_URL}" ]]; then
  FEISHU_SHEET_URL="${CFG_FEISHU_SHEET_URL}"
fi
if [[ -z "${FEISHU_APP_ID}" && -n "${CFG_FEISHU_APP_ID}" ]]; then
  FEISHU_APP_ID="${CFG_FEISHU_APP_ID}"
fi
if [[ -z "${FEISHU_APP_SECRET}" && -n "${CFG_FEISHU_APP_SECRET}" ]]; then
  FEISHU_APP_SECRET="${CFG_FEISHU_APP_SECRET}"
fi

bool_cfg() {
  case "${1}" in
    1|true|True|TRUE|yes|Yes|YES) echo "True" ;;
    *) echo "False" ;;
  esac
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#GPU_ARRAY[@]}"
else
  GPU_ARRAY=()
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    GPU_ARRAY+=("${gpu}")
  done
fi

ckpt_abs="$(readlink -f "${CKPT}")"
ckpt_stem="$(basename "${ckpt_abs}")"
ckpt_stem="${ckpt_stem%.*}"
run_tag="manager_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}/task_logs" "${OUTPUT_DIR}/task_status"
: > "${OUTPUT_DIR}/failed_tasks.txt"
: > "${OUTPUT_DIR}/task_gpu_map.txt"

task_file="${OUTPUT_DIR}/tasks.txt"
if [[ -n "${TASK_FILE}" ]]; then
  cp "${TASK_FILE}" "${task_file}"
else
  python - "${task_file}" "${CONFIG}" "${TASK_IDS:-}" ${SUITES} <<'PY'
import sys

from libero.libero import benchmark
from mmengine import Config


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def get_eval_runner_cfg(cfg):
    if hasattr(cfg.eval, 'runner'):
        return cfg.eval.runner
    return cfg.eval


def parse_task_ids(raw_value, num_tasks):
    if raw_value is None:
        return list(range(num_tasks))
    value = str(raw_value).strip()
    if value == '' or value.lower() == 'none':
        return list(range(num_tasks))
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    value = value.replace(',', ' ')
    task_ids = [
        int(item.strip()) for item in value.split()
        if item.strip() != ''
    ]
    if len(task_ids) == 0:
        raise SystemExit('TASK_IDS did not contain any valid task ids.')
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit(f'Duplicate TASK_IDS are not supported: {task_ids}')
    invalid = [task for task in task_ids if task < 0 or task >= num_tasks]
    if invalid:
        raise SystemExit(
            f'Invalid TASK_IDS {invalid}; expected range [0, '
            f'{num_tasks - 1}].')
    return task_ids


output_file = sys.argv[1]
config_path = sys.argv[2]
raw_task_ids = sys.argv[3]
suite_names = sys.argv[4:]
if len(suite_names) == 0:
    cfg = Config.fromfile(config_path)
    suite_names = as_list(get_eval_runner_cfg(cfg).task_suite_name)

benchmark_dict = benchmark.get_benchmark_dict()
with open(output_file, 'w', encoding='utf-8') as f:
    for suite_name in suite_names:
        task_suite = benchmark_dict[suite_name]()
        for task_id in parse_task_ids(raw_task_ids, int(task_suite.n_tasks)):
            f.write(f'{suite_name},{task_id}\n')
PY
fi

declare -A GPU_LOAD
for gpu in "${GPU_ARRAY[@]}"; do
  GPU_LOAD["${gpu}"]=0
done

write_gpu_load_file() {
  local gpu_id
  : > "${OUTPUT_DIR}/gpu_load.txt"
  for gpu_id in "${GPU_ARRAY[@]}"; do
    echo "${gpu_id}:${GPU_LOAD[${gpu_id}]}" >> "${OUTPUT_DIR}/gpu_load.txt"
  done
}

write_task_gpu_map_file() {
  local idx
  : > "${OUTPUT_DIR}/task_gpu_map.txt"
  for idx in "${!running_tasks[@]}"; do
    echo "${running_tasks[$idx]}:${running_gpus[$idx]}" >> "${OUTPUT_DIR}/task_gpu_map.txt"
  done
}

write_manager_config() {
  {
    echo "config: ${CONFIG}"
    echo "ckpt: ${ckpt_abs}"
    echo "suites: ${SUITES:-config default}"
    echo "gpus: ${GPU_ARRAY[*]}"
    echo "max_tasks_per_gpu: ${MAX_TASKS_PER_GPU}"
    echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK:-config default}"
    echo "model_build_device: ${MODEL_BUILD_DEVICE:-config default}"
    echo "model_build_dtype: ${MODEL_BUILD_DTYPE:-config default}"
    echo "eval_shard_strategy: ${EVAL_SHARD_STRATEGY}"
    echo "task_ids: ${TASK_IDS:-all}"
    echo "preprocess_every_step: ${PREPROCESS_EVERY_STEP:-config default}"
    echo "save_rollout_videos: ${SAVE_ROLLOUT_VIDEOS:-config default}"
    echo "save_failed_rollout_videos: ${SAVE_FAILED_ROLLOUT_VIDEOS:-config default}"
    echo "save_multi_view_rollout_videos: ${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS:-config default}"
    echo "rollout_dir: ${ROLLOUT_DIR:-default}"
    echo "output_dir: ${OUTPUT_DIR}"
    echo "output_dir_source: ${OUTPUT_DIR_SOURCE}"
    echo "task_file: ${task_file}"
  } > "${OUTPUT_DIR}/manager_config.yaml"
}

find_least_loaded_gpu() {
  local best_gpu=""
  local best_load=999999
  local gpu_id
  for gpu_id in "${GPU_ARRAY[@]}"; do
    local load="${GPU_LOAD[${gpu_id}]}"
    if [[ "${load}" -lt "${best_load}" && "${load}" -lt "${MAX_TASKS_PER_GPU}" ]]; then
      best_gpu="${gpu_id}"
      best_load="${load}"
    fi
  done
  echo "${best_gpu}"
}

readarray -t TASK_QUEUE < "${task_file}"
total_tasks="${#TASK_QUEUE[@]}"
if [[ "${total_tasks}" -eq 0 ]]; then
  echo "[manager] no tasks found in ${task_file}" >&2
  exit 1
fi

declare -A SEEN_SUITES
declare -A SUITE_TOTAL_TASKS
SUITE_ORDER=()
for task_entry in "${TASK_QUEUE[@]}"; do
  suite_name="${task_entry%%,*}"
  SUITE_TOTAL_TASKS["${suite_name}"]=$((${SUITE_TOTAL_TASKS[${suite_name}]:-0} + 1))
  if [[ -z "${SEEN_SUITES[${suite_name}]:-}" ]]; then
    SUITE_ORDER+=("${suite_name}")
    SEEN_SUITES["${suite_name}"]=1
  fi
done

next_task_idx=0
completed_tasks=0
failed_tasks=0
launch_count=0
overall_eval_successes=0
overall_eval_episodes=0

declare -A SUITE_EVAL_SUCCESSES
declare -A SUITE_EVAL_EPISODES
declare -A SUITE_COMPLETED_TASKS

running_pids=()
running_gpus=()
running_status=()
running_tasks=()
running_logs=()

kill_process_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_process_tree "${child}"
  done
  kill "${pid}" 2>/dev/null || true
}

cleanup_children() {
  for pid in "${running_pids[@]:-}"; do
    kill_process_tree "${pid}"
  done
}
trap cleanup_children INT TERM

build_worker_cfg_options() {
  local suite="$1"
  local task_id="$2"
  local gpu="$3"
  local suffix="$4"
  local cfg_options=(
    "${EVAL_RUNNER_PREFIX}.task_suite_name=${suite}"
    "${EVAL_RUNNER_PREFIX}.task_ids=[${task_id}]"
    "${EVAL_RUNNER_PREFIX}.eval_shard_strategy=${EVAL_SHARD_STRATEGY}"
    "${EVAL_RUNNER_PREFIX}.run_id_suffix=${suffix}"
    "${EVAL_RUNNER_PREFIX}.result_output_dir=${OUTPUT_DIR}"
    "${EVAL_RUNNER_PREFIX}.result_gpu_id=${gpu}"
  )

  if [[ "${NUM_TRIALS_PER_TASK_SOURCE}" != "config" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.num_trials_per_task=${NUM_TRIALS_PER_TASK}")
  fi
  if [[ -n "${MAX_STEPS}" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.max_steps=${MAX_STEPS}")
  fi
  if [[ "${MODEL_BUILD_DEVICE_SOURCE}" != "config" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.model_build_device=${MODEL_BUILD_DEVICE}")
  fi
  if [[ "${MODEL_BUILD_DTYPE_SOURCE}" != "config" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.model_build_dtype=${MODEL_BUILD_DTYPE}")
  fi
  if [[ -n "${PREPROCESS_EVERY_STEP}" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.preprocess_every_step=$(bool_cfg "${PREPROCESS_EVERY_STEP}")")
  fi
  if [[ "${SAVE_ROLLOUT_VIDEOS_SOURCE}" != "config" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.save_rollout_videos=$(bool_cfg "${SAVE_ROLLOUT_VIDEOS}")")
  fi
  if [[ -n "${SAVE_FAILED_ROLLOUT_VIDEOS}" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.save_failed_rollout_videos=$(bool_cfg "${SAVE_FAILED_ROLLOUT_VIDEOS}")")
  fi
  if [[ "${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS_SOURCE}" != "config" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.save_multi_view_rollout_videos=$(bool_cfg "${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS}")")
  fi
  if [[ "${ROLLOUT_DIR_SOURCE}" == "env" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.rollout_dir=${ROLLOUT_DIR}")
  fi

  # Per-task overrides come last so manager task assignment always wins.
  worker_cfg_options=("${WORKER_USER_CFG_OPTIONS[@]}" "${cfg_options[@]}")
}

launch_task() {
  local suite="$1"
  local task_id="$2"
  local gpu="$3"
  local port="$4"
  local suffix="${run_tag}_${suite}_task${task_id}_gpu${gpu}"
  local log_file="${OUTPUT_DIR}/task_logs/${suite}_task${task_id}_gpu${gpu}.log"
  local status_file="${OUTPUT_DIR}/task_status/${suite}_task${task_id}.status"
  local result_file="${OUTPUT_DIR}/${suite}/gpu${gpu}_task${task_id}_results.json"

  rm -f "${status_file}"
  GPU_LOAD["${gpu}"]=$((GPU_LOAD["${gpu}"] + 1))
  echo "[manager] launch ${suite},task${task_id} -> GPU ${gpu} port ${port} load ${GPU_LOAD[${gpu}]}/${MAX_TASKS_PER_GPU}"
  write_gpu_load_file

  (
    set +e
    build_worker_cfg_options "${suite}" "${task_id}" "${gpu}" "${suffix}"

    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
      FLUXVLA_ALLOW_LIBERO_TASK_SHARDING=1 \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      NPROC_PER_NODE=1 \
      WORLD_SIZE=1 \
      RANK=0 \
      MASTER_ADDR="${MASTER_ADDR:-localhost}" \
      MASTER_PORT="${port}" \
      torchrun \
        --nproc-per-node=1 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="${MASTER_ADDR:-localhost}" \
        --master_port="${port}" \
        "scripts/eval.py" \
        --config "${CONFIG}" \
        --ckpt-path "${ckpt_abs}" \
        --cfg-options "${worker_cfg_options[@]}" \
        "${FORWARDED_EXTRA_ARGS[@]}" \
        > "${log_file}" 2>&1
    rc=$?
    if [[ "${rc}" -eq 0 && -f "${result_file}" ]]; then
      echo "SUCCESS|${gpu}|${rc}|$(date +%s)|${log_file}" > "${status_file}"
    else
      echo "FAILED|${gpu}|${rc}|$(date +%s)|${log_file}" > "${status_file}"
    fi
    exit "${rc}"
  ) &

  running_pids+=("$!")
  running_gpus+=("${gpu}")
  running_status+=("${status_file}")
  running_tasks+=("${suite},${task_id}")
  running_logs+=("${log_file}")
  launch_count=$((launch_count + 1))
  write_task_gpu_map_file
}

process_finished_without_status() {
  local pid="$1"
  local stat
  stat="$(ps -p "${pid}" -o stat= 2>/dev/null || true)"
  [[ -z "${stat}" || "${stat}" == *Z* ]]
}

format_success_rate() {
  local successes="$1"
  local episodes="$2"
  if [[ "${episodes}" -eq 0 ]]; then
    echo "0.00"
    return
  fi
  python - "${successes}" "${episodes}" <<'PY'
import sys

successes = int(sys.argv[1])
episodes = int(sys.argv[2])
print(f'{successes / max(episodes, 1) * 100:.2f}')
PY
}

write_suite_success_rates_file() {
  local output_file="${OUTPUT_DIR}/suite_success_rates.csv"
  {
    echo "Suite,Completed Tasks,Total Tasks,Successes,Episodes,Success Rate (%)"
    local suite_name
    for suite_name in "${SUITE_ORDER[@]}"; do
      local completed="${SUITE_COMPLETED_TASKS[${suite_name}]:-0}"
      local total="${SUITE_TOTAL_TASKS[${suite_name}]:-0}"
      local successes="${SUITE_EVAL_SUCCESSES[${suite_name}]:-0}"
      local episodes="${SUITE_EVAL_EPISODES[${suite_name}]:-0}"
      local success_rate
      success_rate="$(format_success_rate "${successes}" "${episodes}")"
      echo "${suite_name},${completed},${total},${successes},${episodes},${success_rate}"
    done
  } > "${output_file}"
}

format_eval_success_rate() {
  format_success_rate "${overall_eval_successes}" \
    "${overall_eval_episodes}"
}

format_suite_eval_summary() {
  local summaries=()
  local suite_name
  for suite_name in "${SUITE_ORDER[@]}"; do
    local episodes="${SUITE_EVAL_EPISODES[${suite_name}]:-0}"
    if [[ "${episodes}" -eq 0 ]]; then
      continue
    fi
    local successes="${SUITE_EVAL_SUCCESSES[${suite_name}]:-0}"
    local success_rate
    success_rate="$(format_success_rate "${successes}" "${episodes}")"
    summaries+=("${suite_name}=${successes}/${episodes} (${success_rate}%)")
  done
  if [[ "${#summaries[@]}" -eq 0 ]]; then
    echo "none"
    return
  fi
  local IFS=', '
  echo "${summaries[*]}"
}

record_eval_result() {
  local suite_name="$1"
  local result_file="$2"
  RECORDED_TASK_SUCCESSES=0
  RECORDED_TASK_EPISODES=0
  if [[ ! -f "${result_file}" ]]; then
    return
  fi
  local counts
  counts="$(python - "${result_file}" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    result = json.load(f)
print(f"{int(result.get('successes', 0))} {int(result.get('total_episodes', 0))}")
PY
)"
  local successes episodes
  read -r successes episodes <<< "${counts}"
  RECORDED_TASK_SUCCESSES="${successes}"
  RECORDED_TASK_EPISODES="${episodes}"
  overall_eval_successes=$((overall_eval_successes + successes))
  overall_eval_episodes=$((overall_eval_episodes + episodes))
  SUITE_EVAL_SUCCESSES["${suite_name}"]=$((${SUITE_EVAL_SUCCESSES[${suite_name}]:-0} + successes))
  SUITE_EVAL_EPISODES["${suite_name}"]=$((${SUITE_EVAL_EPISODES[${suite_name}]:-0} + episodes))
}

poll_finished_tasks() {
  local idx
  local new_pids=()
  local new_gpus=()
  local new_status=()
  local new_tasks=()
  local new_logs=()

  for idx in "${!running_pids[@]}"; do
    local pid="${running_pids[$idx]}"
    local gpu="${running_gpus[$idx]}"
    local status_file="${running_status[$idx]}"
    local task_info="${running_tasks[$idx]}"
    local log_file="${running_logs[$idx]}"

    if [[ -f "${status_file}" ]]; then
      local status_line
      status_line="$(cat "${status_file}")"
      local status
      status="${status_line%%|*}"
      wait "${pid}" 2>/dev/null || true
      GPU_LOAD["${gpu}"]=$((GPU_LOAD["${gpu}"] - 1))
      local suite_name="${task_info%,*}"
      completed_tasks=$((completed_tasks + 1))
      if [[ "${status}" == "FAILED" ]]; then
        failed_tasks=$((failed_tasks + 1))
        SUITE_COMPLETED_TASKS["${suite_name}"]=$((${SUITE_COMPLETED_TASKS[${suite_name}]:-0} + 1))
        write_suite_success_rates_file
        echo "[manager] failed ${task_info}: ${status_line}" | tee -a "${OUTPUT_DIR}/failed_tasks.txt"
      else
        record_eval_result "${suite_name}" \
          "${OUTPUT_DIR}/${suite_name}/gpu${gpu}_task${task_info#*,}_results.json"
        SUITE_COMPLETED_TASKS["${suite_name}"]=$((${SUITE_COMPLETED_TASKS[${suite_name}]:-0} + 1))
        write_suite_success_rates_file
        local task_success_rate
        local suite_success_rate
        local overall_success_rate
        task_success_rate="$(format_success_rate \
          "${RECORDED_TASK_SUCCESSES}" "${RECORDED_TASK_EPISODES}")"
        suite_success_rate="$(format_success_rate \
          "${SUITE_EVAL_SUCCESSES[${suite_name}]:-0}" \
          "${SUITE_EVAL_EPISODES[${suite_name}]:-0}")"
        overall_success_rate="$(format_eval_success_rate)"
        echo "[manager] done ${task_info} on GPU ${gpu} (${completed_tasks}/${total_tasks}) task=${RECORDED_TASK_SUCCESSES}/${RECORDED_TASK_EPISODES} (${task_success_rate}%) suite=${SUITE_EVAL_SUCCESSES[${suite_name}]:-0}/${SUITE_EVAL_EPISODES[${suite_name}]:-0} (${suite_success_rate}%) overall=${overall_eval_successes}/${overall_eval_episodes} (${overall_success_rate}%)"
      fi
    elif process_finished_without_status "${pid}"; then
      local rc=1
      if wait "${pid}" 2>/dev/null; then
        rc=0
      else
        rc=$?
      fi
      GPU_LOAD["${gpu}"]=$((GPU_LOAD["${gpu}"] - 1))
      completed_tasks=$((completed_tasks + 1))
      local suite_name="${task_info%,*}"
      SUITE_COMPLETED_TASKS["${suite_name}"]=$((${SUITE_COMPLETED_TASKS[${suite_name}]:-0} + 1))
      write_suite_success_rates_file
      failed_tasks=$((failed_tasks + 1))
      echo "[manager] failed ${task_info}: child exited before writing status (rc=${rc}, log=${log_file})" \
        | tee -a "${OUTPUT_DIR}/failed_tasks.txt"
    else
      new_pids+=("${pid}")
      new_gpus+=("${gpu}")
      new_status+=("${status_file}")
      new_tasks+=("${task_info}")
      new_logs+=("${log_file}")
    fi
  done

  running_pids=("${new_pids[@]}")
  running_gpus=("${new_gpus[@]}")
  running_status=("${new_status[@]}")
  running_tasks=("${new_tasks[@]}")
  running_logs=("${new_logs[@]}")
  write_gpu_load_file
  write_task_gpu_map_file
}

show_status() {
  local running_count="${#running_pids[@]}"
  local pending_count=$((total_tasks - next_task_idx))
  local gpu_id
  local success_rate
  local suite_summary
  success_rate="$(format_eval_success_rate)"
  suite_summary="$(format_suite_eval_summary)"
  echo "[manager] status completed=${completed_tasks}/${total_tasks} running=${running_count} pending=${pending_count} failed=${failed_tasks} overall=${overall_eval_successes}/${overall_eval_episodes} (${success_rate}%) suites=${suite_summary}"
  for gpu_id in "${GPU_ARRAY[@]}"; do
    echo "[manager]   GPU ${gpu_id}: ${GPU_LOAD[${gpu_id}]}/${MAX_TASKS_PER_GPU}"
  done
}

write_gpu_load_file
write_manager_config
write_suite_success_rates_file

echo "[manager] config=${CONFIG}"
echo "[manager] ckpt=${ckpt_abs}"
echo "[manager] suites=${SUITES:-config default}"
echo "[manager] gpus=${GPU_ARRAY[*]}"
echo "[manager] max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
echo "[manager] num_trials_per_task=${NUM_TRIALS_PER_TASK:-config default}"
echo "[manager] model_build_device=${MODEL_BUILD_DEVICE:-config default}"
echo "[manager] model_build_dtype=${MODEL_BUILD_DTYPE:-config default}"
echo "[manager] eval_shard_strategy=${EVAL_SHARD_STRATEGY}"
echo "[manager] task_ids=${TASK_IDS:-all}"
echo "[manager] output=${OUTPUT_DIR}"
echo "[manager] task_file=${task_file}"

last_status_time=0
while [[ "${completed_tasks}" -lt "${total_tasks}" ]]; do
  while [[ "${next_task_idx}" -lt "${total_tasks}" ]]; do
    gpu="$(find_least_loaded_gpu)"
    if [[ -z "${gpu}" ]]; then
      break
    fi
    IFS=',' read -r suite task_id <<< "${TASK_QUEUE[${next_task_idx}]}"
    port=$((MASTER_PORT_BASE + launch_count))
    launch_task "${suite}" "${task_id}" "${gpu}" "${port}"
    next_task_idx=$((next_task_idx + 1))
    sleep "${LAUNCH_DELAY}"
  done

  sleep "${MONITOR_INTERVAL}"
  poll_finished_tasks

  now="$(date +%s)"
  if [[ $((now - last_status_time)) -ge "${STATUS_INTERVAL}" ]]; then
    show_status
    last_status_time="${now}"
  fi

  if [[ "${failed_tasks}" -gt 0 ]]; then
    echo "[manager] stopping because at least one task failed. See ${OUTPUT_DIR}/failed_tasks.txt"
    cleanup_children
    exit 1
  fi
done

summary_args=(
  --run-dir "${OUTPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --title "${ckpt_stem}"
)
if [[ -n "${FEISHU_SHEET_URL}${FEISHU_APP_ID}${FEISHU_APP_SECRET}" ]]; then
  summary_args+=(
    --feishu-sheet-url "${FEISHU_SHEET_URL}"
    --feishu-app-id "${FEISHU_APP_ID}"
    --feishu-app-secret "${FEISHU_APP_SECRET}"
    --feishu-timeout "${FEISHU_TIMEOUT}"
  )
fi

CKPT="${ckpt_abs}" CONFIG="${CONFIG}" python "${SUMMARY_TOOL}" \
  "${summary_args[@]}"

echo "[manager] summary: ${OUTPUT_DIR}/summary.csv"
