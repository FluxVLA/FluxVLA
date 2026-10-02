#!/usr/bin/env bash
# Dynamic RoboTwin eval manager for configs that use RobotwinEvalRunner.
#
# The manager builds a task queue and launches one single-task eval worker per
# task. Each worker runs as a single-process python scripts/eval.py pinned with
# CUDA_VISIBLE_DEVICES, writes full eval artifacts under
# OUTPUT_DIR/workers/<task>/eval_runs/<ckpt>/<run_id>, and the manager
# aggregates the per-worker summary.json files into one summary.
#
# Usage:
#   CONFIG=configs/gr00t/gr00t_eagle_3b_robotwin_all_finetune_starvla_lr.py \
#   CKPT=/path/to/checkpoint.safetensors \
#     bash scripts/eval_robotwin_manager.sh
#
# Positional CONFIG and CKPT are also accepted:
#   bash scripts/eval_robotwin_manager.sh configs/...py /path/to/ckpt
#
# Common overrides:
#   TASK_NAMES="adjust_bottle click_bell"
#   TASK_SUITE_NAME=random
#   TASK_SUITE_NAME="clean random"  # one shared GPU queue
#   NUM_GPUS=8
#   MAX_TASKS_PER_GPU=2
#   NUM_TRIALS_PER_TASK=100
#   SAVE_VIDEO=True
#   OUTPUT_DIR=work_dirs/robotwin_eval_manager/my_run
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
# Multi-condition runs write each condition under OUTPUT_DIR/<task_suite_name>,
# plus an outer summary with separate Easy/Hard rates (no combined rate).
#
# Single-condition OUTPUT_DIR layout:
#   summary.{csv,txt,json}, failed_tasks.txt,
#   task_logs/, task_status/,
#   workers/<task>/eval_runs/<ckpt>/EVAL-robotwin-*/summary.json
#
# Rerunning with the same OUTPUT_DIR skips tasks whose worker summary is
# complete and retries the rest.
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
TASK_NAMES="${TASK_NAMES:-}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SAVE_VIDEO="${SAVE_VIDEO:-}"
FEISHU_SHEET_URL="${FEISHU_SHEET_URL:-}"
FEISHU_APP_ID="${FEISHU_APP_ID:-}"
FEISHU_APP_SECRET="${FEISHU_APP_SECRET:-}"
FEISHU_TIMEOUT="${FEISHU_TIMEOUT:-10}"
USER_CFG_OPTIONS=()
WORKER_USER_CFG_OPTIONS=()
FORWARDED_EXTRA_ARGS=()

# Distributed-launcher variables that must not leak into single-process
# workers: with WORLD_SIZE set, overwatch would attempt an env:// rendezvous.
WORKER_ENV_UNSET=(
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK
  -u ROLE_RANK -u NODE_RANK -u NPROC_PER_NODE -u MASTER_ADDR -u MASTER_PORT
)

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

DEFAULT_NUM_GPUS="8"
DEFAULT_MAX_TASKS_PER_GPU="2"
DEFAULT_MONITOR_INTERVAL="10"
DEFAULT_STATUS_INTERVAL="60"
DEFAULT_LAUNCH_DELAY="2"
DEFAULT_SUMMARY_TOOL="tools/summarize_robotwin_eval_results.py"
DEFAULT_NUM_TRIALS_PER_TASK="100"
DEFAULT_TASK_SUITE_NAME="clean"
DEFAULT_SAVE_VIDEO="False"

CFG_EVAL_RUNNER_PREFIX=""
CFG_NUM_GPUS=""
CFG_MAX_TASKS_PER_GPU=""
CFG_NUM_TRIALS_PER_TASK=""
CFG_MONITOR_INTERVAL=""
CFG_STATUS_INTERVAL=""
CFG_LAUNCH_DELAY=""
CFG_SUMMARY_TOOL=""
CFG_TASK_NAMES=""
CFG_TASK_SUITE_NAME=""
CFG_OUTPUT_DIR=""
CFG_SAVE_VIDEO=""
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
    'CFG_NUM_GPUS': ('eval.manager.num_gpus', ),
    'CFG_MAX_TASKS_PER_GPU': ('eval.manager.max_tasks_per_gpu', ),
    'CFG_NUM_TRIALS_PER_TASK': (
        'eval.runner.num_trials_per_task', 'eval.num_trials_per_task'),
    'CFG_MONITOR_INTERVAL': ('eval.manager.monitor_interval', ),
    'CFG_STATUS_INTERVAL': ('eval.manager.status_interval', ),
    'CFG_LAUNCH_DELAY': ('eval.manager.launch_delay', ),
    'CFG_SUMMARY_TOOL': ('eval.manager.summary_tool', ),
    'CFG_TASK_NAMES': (
        'eval.manager.tasks', 'eval.runner.task_list', 'eval.task_list'),
    'CFG_TASK_SUITE_NAME': (
        'eval.manager.task_suite_name', 'eval.runner.task_suite_name',
        'eval.task_suite_name'),
    'CFG_OUTPUT_DIR': (
        'eval.manager.output_dir', 'eval.runner.result_output_dir',
        'eval.output_dir', 'eval.result_output_dir'),
    'CFG_SAVE_VIDEO': ('eval.runner.save_video', 'eval.save_video'),
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
    CFG_NUM_GPUS) CFG_NUM_GPUS="${value}" ;;
    CFG_MAX_TASKS_PER_GPU) CFG_MAX_TASKS_PER_GPU="${value}" ;;
    CFG_NUM_TRIALS_PER_TASK) CFG_NUM_TRIALS_PER_TASK="${value}" ;;
    CFG_MONITOR_INTERVAL) CFG_MONITOR_INTERVAL="${value}" ;;
    CFG_STATUS_INTERVAL) CFG_STATUS_INTERVAL="${value}" ;;
    CFG_LAUNCH_DELAY) CFG_LAUNCH_DELAY="${value}" ;;
    CFG_SUMMARY_TOOL) CFG_SUMMARY_TOOL="${value}" ;;
    CFG_TASK_NAMES) CFG_TASK_NAMES="${value}" ;;
    CFG_TASK_SUITE_NAME) CFG_TASK_SUITE_NAME="${value}" ;;
    CFG_OUTPUT_DIR) CFG_OUTPUT_DIR="${value}" ;;
    CFG_SAVE_VIDEO) CFG_SAVE_VIDEO="${value}" ;;
    CFG_FEISHU_SHEET_URL) CFG_FEISHU_SHEET_URL="${value}" ;;
    CFG_FEISHU_APP_ID) CFG_FEISHU_APP_ID="${value}" ;;
    CFG_FEISHU_APP_SECRET) CFG_FEISHU_APP_SECRET="${value}" ;;
  esac
done <<< "${CFG_VALUES}"

EVAL_RUNNER_PREFIX="${CFG_EVAL_RUNNER_PREFIX:-eval}"

if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR_SOURCE="env"
elif [[ -n "${CFG_OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${CFG_OUTPUT_DIR}"
  OUTPUT_DIR_SOURCE="config"
else
  OUTPUT_DIR="work_dirs/robotwin_eval_manager/$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR_SOURCE="default"
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

if [[ -n "${NUM_TRIALS_PER_TASK:-}" ]]; then
  NUM_TRIALS_PER_TASK_SOURCE="env"
elif [[ -n "${CFG_NUM_TRIALS_PER_TASK}" ]]; then
  NUM_TRIALS_PER_TASK="${CFG_NUM_TRIALS_PER_TASK}"
  NUM_TRIALS_PER_TASK_SOURCE="config"
else
  NUM_TRIALS_PER_TASK="${DEFAULT_NUM_TRIALS_PER_TASK}"
  NUM_TRIALS_PER_TASK_SOURCE="default"
fi

if [[ -n "${SAVE_VIDEO}" ]]; then
  SAVE_VIDEO_SOURCE="env"
elif [[ -n "${CFG_SAVE_VIDEO}" ]]; then
  SAVE_VIDEO="${CFG_SAVE_VIDEO}"
  SAVE_VIDEO_SOURCE="config"
else
  SAVE_VIDEO="${DEFAULT_SAVE_VIDEO}"
  SAVE_VIDEO_SOURCE="default"
fi

if [[ -n "${TASK_NAMES}" ]]; then
  TASK_NAMES_SOURCE="env"
else
  TASK_NAMES="${CFG_TASK_NAMES}"
  TASK_NAMES_SOURCE="config"
fi

if [[ -n "${TASK_SUITE_NAME}" ]]; then
  TASK_SUITE_NAME_SOURCE="env"
elif [[ -n "${CFG_TASK_SUITE_NAME}" ]]; then
  TASK_SUITE_NAME="${CFG_TASK_SUITE_NAME}"
  TASK_SUITE_NAME_SOURCE="config"
else
  TASK_SUITE_NAME="${DEFAULT_TASK_SUITE_NAME}"
  TASK_SUITE_NAME_SOURCE="default"
fi
read -r -a CONDITIONS <<< "${TASK_SUITE_NAME}"
MULTI_CONDITION=0
[[ "${#CONDITIONS[@]}" -le 1 ]] || MULTI_CONDITION=1

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

read -r -a TASK_NAMES_ARRAY <<< "${TASK_NAMES}"
if [[ "${#TASK_NAMES_ARRAY[@]}" -eq 0 ]]; then
  echo "[manager] no tasks resolved (set TASK_NAMES or eval.task_list)" >&2
  exit 1
fi

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

declare -A JOB_CONFIG JOB_ROOT
TASK_QUEUE=()
mkdir -p "${OUTPUT_DIR}"
for condition in "${CONDITIONS[@]}"; do
  condition_root="${OUTPUT_DIR}"
  if [[ "${MULTI_CONDITION}" -eq 1 ]]; then
    condition_root="${OUTPUT_DIR}/${condition}"
  fi
  mkdir -p "${condition_root}/task_logs" "${condition_root}/task_status" "${condition_root}/workers"
  : > "${condition_root}/failed_tasks.txt"
  printf '%s\n' "${TASK_NAMES_ARRAY[@]}" > "${condition_root}/tasks.txt"
done
# Interleave conditions so both can use the shared pool immediately.
for task in "${TASK_NAMES_ARRAY[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    job="${task}"
    [[ "${MULTI_CONDITION}" -eq 0 ]] || job="${condition}/${task}"
    TASK_QUEUE+=("${job}")
    JOB_CONFIG[${job}]="${condition}"
    JOB_ROOT[${job}]="${OUTPUT_DIR}"
    [[ "${MULTI_CONDITION}" -eq 0 ]] || JOB_ROOT[${job}]+="/${condition}"
  done
done
total_tasks="${#TASK_QUEUE[@]}"
: > "${OUTPUT_DIR}/failed_tasks.txt"
: > "${OUTPUT_DIR}/task_gpu_map.txt"

task_file="${OUTPUT_DIR}/tasks.txt"
printf '%s\n' "${TASK_QUEUE[@]}" > "${task_file}"

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
    echo "tasks: ${total_tasks}"
    echo "tasks_source: ${TASK_NAMES_SOURCE}"
    echo "task_suite_name: ${TASK_SUITE_NAME}"
    echo "task_suite_name_source: ${TASK_SUITE_NAME_SOURCE}"
    echo "gpus: ${GPU_ARRAY[*]}"
    echo "num_gpus_source: ${NUM_GPUS_SOURCE}"
    echo "max_tasks_per_gpu: ${MAX_TASKS_PER_GPU}"
    echo "max_tasks_per_gpu_source: ${MAX_TASKS_PER_GPU_SOURCE}"
    echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK}"
    echo "num_trials_per_task_source: ${NUM_TRIALS_PER_TASK_SOURCE}"
    echo "save_video: ${SAVE_VIDEO}"
    echo "save_video_source: ${SAVE_VIDEO_SOURCE}"
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

next_task_idx=0
completed_tasks=0
failed_tasks=0
skipped_tasks=0
launch_count=0

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
  local task="$1"
  local worker_dir="$2"
  local condition="$3"
  local cfg_options=(
    "${EVAL_RUNNER_PREFIX}.task_list=[${task}]"
    "${EVAL_RUNNER_PREFIX}.result_output_dir=${worker_dir}"
    "${EVAL_RUNNER_PREFIX}.run_id_suffix=${task}"
    "${EVAL_RUNNER_PREFIX}.resume=True"
    "${EVAL_RUNNER_PREFIX}.task_suite_name=${condition}"
    "${EVAL_RUNNER_PREFIX}.num_trials_per_task=${NUM_TRIALS_PER_TASK}"
    # Workers evaluate one task each; only the merged summary is reported.
    "${EVAL_RUNNER_PREFIX}.feishu_report=False"
  )

  if [[ "${SAVE_VIDEO_SOURCE}" != "config" ]]; then
    cfg_options+=("${EVAL_RUNNER_PREFIX}.save_video=$(bool_cfg "${SAVE_VIDEO}")")
  fi

  # Per-task overrides come last so manager task assignment always wins.
  worker_cfg_options=("${WORKER_USER_CFG_OPTIONS[@]}" "${cfg_options[@]}")
}

launch_task() {
  local task="$1"
  local gpu="$2"
  local task_name="${task##*/}"
  local condition="${JOB_CONFIG[$task]}"
  local root="${JOB_ROOT[$task]}"
  local log_file="${root}/task_logs/${task_name}_gpu${gpu}.log"
  local status_file="${root}/task_status/${task_name}.status"
  local worker_dir="${root}/workers/${task_name}"

  rm -f "${status_file}"
  GPU_LOAD["${gpu}"]=$((GPU_LOAD["${gpu}"] + 1))
  echo "[manager] launch ${task} -> GPU ${gpu} load ${GPU_LOAD[${gpu}]}/${MAX_TASKS_PER_GPU}"
  write_gpu_load_file

  (
    set +e
    # Isolate each worker's Warp cache to avoid concurrent NVRTC/PCH
    # corruption.
    local warp_cache_dir
    warp_cache_dir="$(mktemp -d "${TMPDIR:-/tmp}/warp_cache_${task_name}_gpu${gpu}.XXXXXX")"
    build_worker_cfg_options "${task_name}" "${worker_dir}" "${condition}"

    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      WARP_CACHE_PATH="${warp_cache_dir}" \
      env "${WORKER_ENV_UNSET[@]}" \
      python scripts/eval.py \
        --config "${CONFIG}" \
        --ckpt-path "${ckpt_abs}" \
        --cfg-options "${worker_cfg_options[@]}" \
        "${FORWARDED_EXTRA_ARGS[@]}" \
        > "${log_file}" 2>&1
    rc=$?
    rm -rf "${warp_cache_dir}"
    if [[ "${rc}" -eq 0 && -n "$(worker_complete_counts "${task}")" ]]; then
      echo "SUCCESS|${gpu}|${rc}|$(date +%s)|${log_file}" > "${status_file}"
    else
      echo "FAILED|${gpu}|${rc}|$(date +%s)|${log_file}" > "${status_file}"
    fi
    exit "${rc}"
  ) &

  running_pids+=("$!")
  running_gpus+=("${gpu}")
  running_status+=("${status_file}")
  running_tasks+=("${task}")
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

worker_summary_path() {
  local task="$1"
  find "${JOB_ROOT[$task]}/workers/${task##*/}" -name summary.json 2>/dev/null | head -n 1
}

# Print "<successes> <episodes>" when task_results holds a completed task
# with the expected trial count; print nothing otherwise.
worker_complete_counts() {
  local task="$1"
  local path
  path="$(worker_summary_path "${task}")"
  [[ -n "${path}" ]] || return 0
  python - "${path}" "${task##*/}" "${NUM_TRIALS_PER_TASK}" <<'PY'
import json
import sys

path, task, trials = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    payload = json.load(open(path, encoding='utf-8'))
except Exception:
    raise SystemExit(0)
task_results = payload['task_results']
stats = task_results.get(task)
if (stats and stats['status'] == 'COMPLETED'
        and int(stats['total_episodes']) == trials):
    successes = int(stats['successes'])
    if trials > 0 and 0 <= successes <= trials:
        print(successes, int(stats['total_episodes']))
PY
}

record_task_failure() {
  local task="$1"
  local message="$2"
  echo "${message}" | tee -a "${OUTPUT_DIR}/failed_tasks.txt"
  if [[ "${MULTI_CONDITION}" -eq 1 ]]; then
    echo "${message}" >> "${JOB_ROOT[$task]}/failed_tasks.txt"
  fi
}

record_eval_result() {
  local task="$1"
  RECORDED_TASK_SUCCESSES=0
  RECORDED_TASK_EPISODES=0
  local counts
  counts="$(worker_complete_counts "${task}")"
  if [[ -z "${counts}" ]]; then
    return 1
  fi
  local successes episodes
  read -r successes episodes <<< "${counts}"
  RECORDED_TASK_SUCCESSES="${successes}"
  RECORDED_TASK_EPISODES="${episodes}"
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
    local task="${running_tasks[$idx]}"
    local log_file="${running_logs[$idx]}"

    if [[ -f "${status_file}" ]]; then
      local status_line
      status_line="$(cat "${status_file}")"
      local status
      status="${status_line%%|*}"
      wait "${pid}" 2>/dev/null || true
      GPU_LOAD["${gpu}"]=$((GPU_LOAD["${gpu}"] - 1))
      completed_tasks=$((completed_tasks + 1))
      if [[ "${status}" == "FAILED" ]]; then
        failed_tasks=$((failed_tasks + 1))
        record_task_failure "${task}" "[manager] failed ${task}: ${status_line}"
      else
        record_eval_result "${task}" || true
        local task_success_rate
        task_success_rate="$(format_success_rate \
          "${RECORDED_TASK_SUCCESSES}" "${RECORDED_TASK_EPISODES}")"
        echo "[manager] done ${task} on GPU ${gpu} (${completed_tasks}/${total_tasks}) task=${RECORDED_TASK_SUCCESSES}/${RECORDED_TASK_EPISODES} (${task_success_rate}%)"
      fi
    elif process_finished_without_status "${pid}" && [[ ! -f "${status_file}" ]]; then
      local rc=1
      if wait "${pid}" 2>/dev/null; then
        rc=0
      else
        rc=$?
      fi
      GPU_LOAD["${gpu}"]=$((GPU_LOAD["${gpu}"] - 1))
      completed_tasks=$((completed_tasks + 1))
      failed_tasks=$((failed_tasks + 1))
      record_task_failure "${task}" "[manager] failed ${task}: child exited before writing status (rc=${rc}, log=${log_file})"
    else
      new_pids+=("${pid}")
      new_gpus+=("${gpu}")
      new_status+=("${status_file}")
      new_tasks+=("${task}")
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
  echo "[manager] status completed=${completed_tasks}/${total_tasks} (skipped=${skipped_tasks}) running=${running_count} pending=${pending_count} failed=${failed_tasks}"
  for gpu_id in "${GPU_ARRAY[@]}"; do
    echo "[manager]   GPU ${gpu_id}: ${GPU_LOAD[${gpu_id}]}/${MAX_TASKS_PER_GPU}"
  done
}

write_gpu_load_file
write_manager_config

echo "[manager] config=${CONFIG}"
echo "[manager] ckpt=${ckpt_abs}"
echo "[manager] runner_prefix=${EVAL_RUNNER_PREFIX}"
echo "[manager] tasks=${total_tasks}"
echo "[manager] task_suite_name=${TASK_SUITE_NAME}"
echo "[manager] gpus=${GPU_ARRAY[*]}"
echo "[manager] max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
echo "[manager] num_trials_per_task=${NUM_TRIALS_PER_TASK}"
echo "[manager] save_video=${SAVE_VIDEO}"
echo "[manager] summary_tool=${SUMMARY_TOOL}"
echo "[manager] output=${OUTPUT_DIR}"
echo "[manager] task_file=${task_file}"

last_status_time=0
while [[ "${completed_tasks}" -lt "${total_tasks}" ]]; do
  while [[ "${next_task_idx}" -lt "${total_tasks}" ]]; do
    task="${TASK_QUEUE[${next_task_idx}]}"
    # Resume: skip tasks whose worker summary is already complete.
    if record_eval_result "${task}"; then
      completed_tasks=$((completed_tasks + 1))
      skipped_tasks=$((skipped_tasks + 1))
      next_task_idx=$((next_task_idx + 1))
      echo "[manager] skip ${task}: already complete (${RECORDED_TASK_SUCCESSES}/${RECORDED_TASK_EPISODES})"
      continue
    fi
    gpu="$(find_least_loaded_gpu)"
    if [[ -z "${gpu}" ]]; then
      break
    fi
    launch_task "${task}" "${gpu}"
    next_task_idx=$((next_task_idx + 1))
    sleep "${LAUNCH_DELAY}"
  done

  if [[ "${completed_tasks}" -ge "${total_tasks}" ]]; then
    break
  fi
  sleep "${MONITOR_INTERVAL}"
  poll_finished_tasks

  now="$(date +%s)"
  if [[ $((now - last_status_time)) -ge "${STATUS_INTERVAL}" ]]; then
    show_status
    last_status_time="${now}"
  fi
done

summary_args=(
  --run-dir "${OUTPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --title "${ckpt_abs##*/}"
  --ckpt "${ckpt_abs}"
)
if [[ -n "${FEISHU_SHEET_URL}${FEISHU_APP_ID}${FEISHU_APP_SECRET}" ]]; then
  summary_args+=(
    --feishu-sheet-url "${FEISHU_SHEET_URL}"
    --feishu-app-id "${FEISHU_APP_ID}"
    --feishu-app-secret "${FEISHU_APP_SECRET}"
    --feishu-timeout "${FEISHU_TIMEOUT}"
  )
fi

# Failed tasks are reported after the summary so partial results are kept;
# rerunning the same command retries only the failed tasks.
summary_rc=0
CONFIG="${CONFIG}" python "${SUMMARY_TOOL}" "${summary_args[@]}" || summary_rc=$?

echo "[manager] summary: ${OUTPUT_DIR}/summary.json"

if [[ "${failed_tasks}" -gt 0 ]]; then
  echo "[manager] ${failed_tasks} task(s) failed; see ${OUTPUT_DIR}/failed_tasks.txt. Rerun the same command to retry only those."
  exit 1
fi
exit "${summary_rc}"
