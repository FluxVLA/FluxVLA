#!/usr/bin/env bash
# Dynamic RoboCasa eval manager for configs that use RobocasaEvalRunner.
#
# The manager builds a task queue and launches one single-task eval worker per
# task. Each worker runs with NPROC_PER_NODE=1, writes full eval artifacts under
# OUTPUT_DIR/eval_runs/<ckpt>/<run_id>, mirrors per-task JSON files under
# OUTPUT_DIR/robocasa, and the manager aggregates them into one summary.
#
# Usage:
#   CONFIG=configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
#   CKPT=/path/to/checkpoint.safetensors \
#     bash scripts/eval_robocasa_manager.sh
#
# Positional CONFIG and CKPT are also accepted:
#   bash scripts/eval_robocasa_manager.sh configs/...py /path/to/ckpt
#
# Common overrides:
#   TASK_IDS="0 1 2"
#   NUM_GPUS=8
#   MAX_TASKS_PER_GPU=1
#   NUM_TRIALS_PER_TASK=50
#   MAX_EPISODE_STEPS=720
#   SAVE_VIDEO=False
#   OUTPUT_DIR=work_dirs/robocasa_eval_manager/my_run
#   FEISHU_SHEET_URL=https://.../sheets/...
#   FEISHU_APP_ID=cli_xxx
#   FEISHU_APP_SECRET=xxx
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
TASK_IDS="${TASK_IDS:-}"
TASK_FILE="${TASK_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
SAVE_VIDEO="${SAVE_VIDEO:-}"
EVAL_SHARD_STRATEGY="${EVAL_SHARD_STRATEGY:-task}"
FEISHU_SHEET_URL="${FEISHU_SHEET_URL:-}"
FEISHU_APP_ID="${FEISHU_APP_ID:-}"
FEISHU_APP_SECRET="${FEISHU_APP_SECRET:-}"
FEISHU_TIMEOUT="${FEISHU_TIMEOUT:-10}"
EXTRA_ARGS=("$@")

DEFAULT_NUM_GPUS="8"
DEFAULT_MAX_TASKS_PER_GPU="1"
DEFAULT_MASTER_PORT_BASE="${MASTER_PORT:-29790}"
DEFAULT_MONITOR_INTERVAL="5"
DEFAULT_STATUS_INTERVAL="30"
DEFAULT_LAUNCH_DELAY="0.5"
DEFAULT_SUMMARY_TOOL="tools/summarize_robocasa_eval_results.py"
DEFAULT_NUM_TRIALS_PER_TASK="50"
DEFAULT_SAVE_VIDEO="False"

CFG_EVAL_RUNNER_PREFIX=""
CFG_NUM_GPUS=""
CFG_MAX_TASKS_PER_GPU=""
CFG_NUM_TRIALS_PER_TASK=""
CFG_MASTER_PORT_BASE=""
CFG_MONITOR_INTERVAL=""
CFG_STATUS_INTERVAL=""
CFG_LAUNCH_DELAY=""
CFG_SUMMARY_TOOL=""
CFG_TASK_FILE=""
CFG_TASK_IDS=""
CFG_OUTPUT_DIR=""
CFG_SAVE_VIDEO=""
CFG_FEISHU_SHEET_URL=""
CFG_FEISHU_APP_ID=""
CFG_FEISHU_APP_SECRET=""

while IFS=$'\t' read -r key value; do
  case "${key}" in
    CFG_EVAL_RUNNER_PREFIX) CFG_EVAL_RUNNER_PREFIX="${value}" ;;
    CFG_NUM_GPUS) CFG_NUM_GPUS="${value}" ;;
    CFG_MAX_TASKS_PER_GPU) CFG_MAX_TASKS_PER_GPU="${value}" ;;
    CFG_NUM_TRIALS_PER_TASK) CFG_NUM_TRIALS_PER_TASK="${value}" ;;
    CFG_MASTER_PORT_BASE) CFG_MASTER_PORT_BASE="${value}" ;;
    CFG_MONITOR_INTERVAL) CFG_MONITOR_INTERVAL="${value}" ;;
    CFG_STATUS_INTERVAL) CFG_STATUS_INTERVAL="${value}" ;;
    CFG_LAUNCH_DELAY) CFG_LAUNCH_DELAY="${value}" ;;
    CFG_SUMMARY_TOOL) CFG_SUMMARY_TOOL="${value}" ;;
    CFG_TASK_FILE) CFG_TASK_FILE="${value}" ;;
    CFG_TASK_IDS) CFG_TASK_IDS="${value}" ;;
    CFG_OUTPUT_DIR) CFG_OUTPUT_DIR="${value}" ;;
    CFG_SAVE_VIDEO) CFG_SAVE_VIDEO="${value}" ;;
    CFG_FEISHU_SHEET_URL) CFG_FEISHU_SHEET_URL="${value}" ;;
    CFG_FEISHU_APP_ID) CFG_FEISHU_APP_ID="${value}" ;;
    CFG_FEISHU_APP_SECRET) CFG_FEISHU_APP_SECRET="${value}" ;;
  esac
done < <(python - "${CONFIG}" <<'PY'
import sys

from mmengine import Config


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


cfg = Config.fromfile(sys.argv[1])
eval_runner_prefix = 'eval.runner' if hasattr(cfg.eval, 'runner') else 'eval'
fields = {
    'CFG_NUM_GPUS': ('eval.manager.num_gpus', ),
    'CFG_MAX_TASKS_PER_GPU': ('eval.manager.max_tasks_per_gpu', ),
    'CFG_NUM_TRIALS_PER_TASK': (
        'eval.runner.num_trials_per_task', 'eval.num_trials_per_task'),
    'CFG_MASTER_PORT_BASE': ('eval.manager.master_port_base', ),
    'CFG_MONITOR_INTERVAL': ('eval.manager.monitor_interval', ),
    'CFG_STATUS_INTERVAL': ('eval.manager.status_interval', ),
    'CFG_LAUNCH_DELAY': ('eval.manager.launch_delay', ),
    'CFG_SUMMARY_TOOL': ('eval.manager.summary_tool', ),
    'CFG_TASK_FILE': ('eval.manager.task_file', ),
    'CFG_TASK_IDS': ('eval.manager.task_ids', ),
    'CFG_OUTPUT_DIR': (
        'eval.manager.output_dir', 'eval.runner.result_output_dir',
        'eval.output_dir', 'eval.result_output_dir'),
    'CFG_SAVE_VIDEO': (
        'eval.runner.save_video', 'eval.save_video',
        'eval.runner.save_rollout_videos', 'eval.save_rollout_videos'),
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
)

EVAL_RUNNER_PREFIX="${CFG_EVAL_RUNNER_PREFIX:-eval}"

if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR_SOURCE="env"
elif [[ -n "${CFG_OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${CFG_OUTPUT_DIR}"
  OUTPUT_DIR_SOURCE="config"
else
  OUTPUT_DIR="work_dirs/robocasa_eval_manager/$(date +%Y%m%d_%H%M%S)"
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

if [[ -z "${TASK_FILE}" && -n "${CFG_TASK_FILE}" ]]; then
  TASK_FILE="${CFG_TASK_FILE}"
fi
if [[ -z "${TASK_IDS}" && -n "${CFG_TASK_IDS}" ]]; then
  TASK_IDS="${CFG_TASK_IDS}"
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
  IFS=',' read -r -a RAW_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  GPU_ARRAY=()
  for gpu in "${RAW_GPU_ARRAY[@]}"; do
    if [[ -n "${gpu}" ]]; then
      GPU_ARRAY+=("${gpu}")
    fi
  done
  NUM_GPUS="${#GPU_ARRAY[@]}"
else
  GPU_ARRAY=()
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    GPU_ARRAY+=("${gpu}")
  done
fi

if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "[manager] no GPUs resolved" >&2
  exit 1
fi

ckpt_abs="$(readlink -f "${CKPT}")"
ckpt_stem="$(basename "${ckpt_abs}")"
ckpt_stem="${ckpt_stem%.*}"
run_tag="manager_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}/task_logs" "${OUTPUT_DIR}/task_status" \
  "${OUTPUT_DIR}/robocasa"
: > "${OUTPUT_DIR}/failed_tasks.txt"
: > "${OUTPUT_DIR}/task_gpu_map.txt"

task_file="${OUTPUT_DIR}/tasks.txt"
if [[ -n "${TASK_FILE}" ]]; then
  cp "${TASK_FILE}" "${task_file}"
else
  python - "${task_file}" "${CONFIG}" "${TASK_IDS}" <<'PY'
import sys

from mmengine import Config


def get_eval_runner_cfg(cfg):
    if hasattr(cfg.eval, 'runner'):
        return cfg.eval.runner
    return cfg.eval


def parse_task_ids(raw, num_tasks):
    if raw is None or str(raw).strip() == '':
        return list(range(num_tasks))
    value = str(raw).strip()
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    value = value.replace(',', ' ')
    task_ids = [int(item) for item in value.split() if item]
    invalid = [task for task in task_ids if task < 0 or task >= num_tasks]
    if invalid:
        raise SystemExit(
            f'Invalid TASK_IDS {invalid}; expected range [0, {num_tasks - 1}]')
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit(f'Duplicate TASK_IDS are not supported: {task_ids}')
    return task_ids


output_file, config_path, raw_task_ids = sys.argv[1:4]
cfg = Config.fromfile(config_path)
runner_cfg = get_eval_runner_cfg(cfg)
num_tasks = len(runner_cfg.task_list)
with open(output_file, 'w', encoding='utf-8') as f:
    for task_id in parse_task_ids(raw_task_ids, num_tasks):
        f.write(f'{task_id}\n')
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
    echo "task${running_tasks[$idx]}:${running_gpus[$idx]}" >> "${OUTPUT_DIR}/task_gpu_map.txt"
  done
}

write_manager_config() {
  {
    echo "config: ${CONFIG}"
    echo "ckpt: ${ckpt_abs}"
    echo "task_ids: ${TASK_IDS:-all}"
    echo "gpus: ${GPU_ARRAY[*]}"
    echo "num_gpus_source: ${NUM_GPUS_SOURCE}"
    echo "max_tasks_per_gpu: ${MAX_TASKS_PER_GPU}"
    echo "max_tasks_per_gpu_source: ${MAX_TASKS_PER_GPU_SOURCE}"
    echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK}"
    echo "num_trials_per_task_source: ${NUM_TRIALS_PER_TASK_SOURCE}"
    echo "max_episode_steps: ${MAX_EPISODE_STEPS:-config default}"
    echo "eval_shard_strategy: ${EVAL_SHARD_STRATEGY}"
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

readarray -t TASK_QUEUE < "${task_file}"
total_tasks="${#TASK_QUEUE[@]}"
if [[ "${total_tasks}" -eq 0 ]]; then
  echo "[manager] no tasks found in ${task_file}" >&2
  exit 1
fi

next_task_idx=0
completed_tasks=0
failed_tasks=0
launch_count=0
overall_eval_successes=0
overall_eval_episodes=0

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

launch_task() {
  local task_id="$1"
  local gpu="$2"
  local port="$3"
  local suffix="${run_tag}_task${task_id}_gpu${gpu}"
  local log_file="${OUTPUT_DIR}/task_logs/task${task_id}_gpu${gpu}.log"
  local status_file="${OUTPUT_DIR}/task_status/task${task_id}.status"
  local result_file="${OUTPUT_DIR}/robocasa/gpu${gpu}_task${task_id}_results.json"

  rm -f "${status_file}" "${result_file}"
  GPU_LOAD["${gpu}"]=$((GPU_LOAD["${gpu}"] + 1))
  echo "[manager] launch task${task_id} -> GPU ${gpu} port ${port} load ${GPU_LOAD[${gpu}]}/${MAX_TASKS_PER_GPU}"
  write_gpu_load_file

  (
    set +e
    cfg_options=(
      "${EVAL_RUNNER_PREFIX}.task_ids=[${task_id}]"
      "${EVAL_RUNNER_PREFIX}.eval_shard_strategy=${EVAL_SHARD_STRATEGY}"
      "${EVAL_RUNNER_PREFIX}.run_id_suffix=${suffix}"
      "${EVAL_RUNNER_PREFIX}.result_output_dir=${OUTPUT_DIR}"
      "${EVAL_RUNNER_PREFIX}.result_gpu_id=${gpu}"
    )
    if [[ "${NUM_TRIALS_PER_TASK_SOURCE}" != "config" ]]; then
      cfg_options+=("${EVAL_RUNNER_PREFIX}.num_trials_per_task=${NUM_TRIALS_PER_TASK}")
    fi
    if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
      cfg_options+=("${EVAL_RUNNER_PREFIX}.max_episode_steps=${MAX_EPISODE_STEPS}")
    fi
    if [[ "${SAVE_VIDEO_SOURCE}" != "config" ]]; then
      cfg_options+=("${EVAL_RUNNER_PREFIX}.save_video=$(bool_cfg "${SAVE_VIDEO}")")
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" \
      NPROC_PER_NODE=1 \
      WORLD_SIZE=1 \
      RANK=0 \
      MASTER_ADDR="${MASTER_ADDR:-localhost}" \
      MASTER_PORT="${port}" \
      bash scripts/eval.sh "${CONFIG}" "${ckpt_abs}" \
        --cfg-options "${cfg_options[@]}" "${EXTRA_ARGS[@]}" \
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
  running_tasks+=("${task_id}")
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

record_eval_result() {
  local result_file="$1"
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
    local task_id="${running_tasks[$idx]}"
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
        echo "[manager] failed task${task_id}: ${status_line}" | tee -a "${OUTPUT_DIR}/failed_tasks.txt"
      else
        record_eval_result \
          "${OUTPUT_DIR}/robocasa/gpu${gpu}_task${task_id}_results.json"
        local task_success_rate
        local overall_success_rate
        task_success_rate="$(format_success_rate \
          "${RECORDED_TASK_SUCCESSES}" "${RECORDED_TASK_EPISODES}")"
        overall_success_rate="$(format_success_rate \
          "${overall_eval_successes}" "${overall_eval_episodes}")"
        echo "[manager] done task${task_id} on GPU ${gpu} (${completed_tasks}/${total_tasks}) task=${RECORDED_TASK_SUCCESSES}/${RECORDED_TASK_EPISODES} (${task_success_rate}%) overall=${overall_eval_successes}/${overall_eval_episodes} (${overall_success_rate}%)"
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
      failed_tasks=$((failed_tasks + 1))
      echo "[manager] failed task${task_id}: child exited before writing status (rc=${rc}, log=${log_file})" \
        | tee -a "${OUTPUT_DIR}/failed_tasks.txt"
    else
      new_pids+=("${pid}")
      new_gpus+=("${gpu}")
      new_status+=("${status_file}")
      new_tasks+=("${task_id}")
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
  success_rate="$(format_success_rate "${overall_eval_successes}" \
    "${overall_eval_episodes}")"
  echo "[manager] status completed=${completed_tasks}/${total_tasks} running=${running_count} pending=${pending_count} failed=${failed_tasks} overall=${overall_eval_successes}/${overall_eval_episodes} (${success_rate}%)"
  for gpu_id in "${GPU_ARRAY[@]}"; do
    echo "[manager]   GPU ${gpu_id}: ${GPU_LOAD[${gpu_id}]}/${MAX_TASKS_PER_GPU}"
  done
}

write_gpu_load_file
write_manager_config

echo "[manager] config=${CONFIG}"
echo "[manager] ckpt=${ckpt_abs}"
echo "[manager] tasks=${total_tasks} (${TASK_IDS:-all})"
echo "[manager] gpus=${GPU_ARRAY[*]}"
echo "[manager] max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
echo "[manager] num_trials_per_task=${NUM_TRIALS_PER_TASK}"
echo "[manager] eval_shard_strategy=${EVAL_SHARD_STRATEGY}"
echo "[manager] save_video=${SAVE_VIDEO}"
echo "[manager] output=${OUTPUT_DIR}"
echo "[manager] task_file=${task_file}"

last_status_time=0
while [[ "${completed_tasks}" -lt "${total_tasks}" ]]; do
  while [[ "${next_task_idx}" -lt "${total_tasks}" ]]; do
    gpu="$(find_least_loaded_gpu)"
    if [[ -z "${gpu}" ]]; then
      break
    fi
    task_id="${TASK_QUEUE[${next_task_idx}]}"
    port=$((MASTER_PORT_BASE + launch_count))
    launch_task "${task_id}" "${gpu}" "${port}"
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
  --config "${CONFIG}"
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

CKPT="${ckpt_abs}" CONFIG="${CONFIG}" python "${SUMMARY_TOOL}" \
  "${summary_args[@]}"

echo "[manager] summary: ${OUTPUT_DIR}/summary.csv"
