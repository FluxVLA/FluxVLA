#!/usr/bin/env bash
# Persistent, resumable LIBERO-Plus evaluation manager.
#
# Unlike scripts/eval_libero_manager.sh, each worker loads the model once and
# evaluates a shard of tasks. Completed tasks are atomically checkpointed so a
# stopped run can resume without re-evaluating them.
#
# Usage:
#   CONFIG=configs/model/libero_spatial.py CKPT=/path/to/checkpoint \
#   SUITE=libero_spatial OUTPUT_DIR=work_dirs/libero_plus/run \
#     bash scripts/eval_libero_plus_manager.sh
#
# Positional CONFIG and CKPT are also accepted:
#   bash scripts/eval_libero_plus_manager.sh CONFIG CKPT --cfg-options ...
#
# Important overrides:
#   SUITE=libero_spatial              One LIBERO suite per invocation.
#   TASK_IDS="0,1,2"                 Empty means every task in the suite.
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   WORKERS_PER_GPU=1                 Persistent model processes per GPU.
#   NUM_TRIALS_PER_TASK=1             LIBERO-Plus requires exactly one.
#   RESUME=1                          Reuse completed per-worker results.
#   SAVE_ROLLOUT_VIDEOS=False
#   SAVE_FAILED_ROLLOUT_VIDEOS=False
#   OUTPUT_DIR=work_dirs/.../run      Run root, not a per-suite directory.
#   TASK_CLASSIFICATION=/path/to/task_classification.json
#   DRY_RUN=1                         Build assignments without launching.
#
# OUTPUT_DIR layout:
#   summary.{csv,txt,json}, task_success_rates.csv,
#   libero_plus_summary.{csv,txt,json}, libero_plus_by_suite.csv,
#   libero_plus_missing_tasks.csv,
#   <suite>/worker_results/workerN_results.json,
#   <suite>/worker_logs/, <suite>/worker_status/, <suite>/worker_tasks/,
#   eval_runs/<checkpoint>/EVAL-*/rank*.txt
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export FLUXVLA_LIBERO_PACKAGE=libero_plus
DEFAULT_LIBERO_PLUS_CONFIG_PATH="$(python - <<'PY'
import importlib.util
from ctypes.util import find_library
from pathlib import Path

required = (
    ('libero_plus', 'libero-plus'),
    ('skimage', 'scikit-image'),
    ('wand', 'Wand'),
)
missing = [
    f'{distribution} (Python import: {module})'
    for module, distribution in required
    if importlib.util.find_spec(module) is None
]
assert not missing, (
    'LIBERO-Plus is missing its additional Python package(s):\n  - ' +
    '\n  - '.join(missing) +
    '\nUpdate the existing FluxVLA environment with:\n'
    '  bash scripts/update_env.sh --skip-pull')

magickwand_names = (
    'MagickWand',
    'MagickWand-7.Q16HDRI',
    'MagickWand-7.Q16',
    'MagickWand-6.Q16HDRI',
    'MagickWand-6.Q16',
)
assert any(find_library(name) for name in magickwand_names), (
    'LIBERO-Plus requires the ImageMagick MagickWand system library '
    '(libMagickWand). Install `libmagickwand-dev` first, then run:\n'
    '  bash scripts/update_env.sh --skip-pull')

spec = importlib.util.find_spec('libero_plus.libero')
assert spec is not None and spec.origin is not None, (
    'Cannot resolve the installed libero-plus package. Reinstall it with:\n'
    '  bash scripts/update_env.sh --skip-pull')
print(Path(spec.origin).resolve().parents[2] / '.libero')
PY
)"
LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-${DEFAULT_LIBERO_PLUS_CONFIG_PATH}}"
export LIBERO_CONFIG_PATH
[[ -n "${LIBERO_CONFIG_PATH}" \
    && -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || {
  echo "[libero-plus-manager] LIBERO-Plus is not configured; run: python scripts/download_libero_plus_assets.py" >&2
  exit 1
}

usage() {
  sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

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
SUITE="${SUITE:-${SUITES:-}}"
TASK_IDS="${TASK_IDS-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TASK_CLASSIFICATION="${TASK_CLASSIFICATION:-${CLASSIFICATION:-}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-}"
MODEL_BUILD_DEVICE="${MODEL_BUILD_DEVICE:-}"
MODEL_BUILD_DTYPE="${MODEL_BUILD_DTYPE:-}"
PREPROCESS_EVERY_STEP="${PREPROCESS_EVERY_STEP:-}"
SAVE_ROLLOUT_VIDEOS="${SAVE_ROLLOUT_VIDEOS:-}"
SAVE_FAILED_ROLLOUT_VIDEOS="${SAVE_FAILED_ROLLOUT_VIDEOS:-}"
SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS:-}"
ROLLOUT_DIR="${ROLLOUT_DIR:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
RESUME="${RESUME:-1}"
LAUNCH_DELAY="${LAUNCH_DELAY:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_TOOL="${SUMMARY_TOOL:-tools/summarize_libero_plus_results.py}"
USER_CFG_OPTIONS=()
WORKER_USER_CFG_OPTIONS=()
FORWARDED_EXTRA_ARGS=()

split_user_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
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
split_user_args "$@"

[[ -f "${CONFIG}" ]] || {
  echo "[libero-plus-manager] config does not exist: ${CONFIG}" >&2
  exit 1
}
[[ -f "${CKPT}" ]] || {
  echo "[libero-plus-manager] checkpoint does not exist: ${CKPT}" >&2
  exit 1
}

CFG_PARSE_ARGS=("${CONFIG}")
if [[ "${#USER_CFG_OPTIONS[@]}" -gt 0 ]]; then
  CFG_PARSE_ARGS+=(--cfg-options "${USER_CFG_OPTIONS[@]}")
fi

CFG_VALUES="$(python - "${CFG_PARSE_ARGS[@]}" <<'PY'
import argparse

from mmengine import Config, DictAction


def get_path(obj, path):
    current = obj
    for key in path.split('.'):
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        else:
            if not hasattr(current, key):
                return None
            current = getattr(current, key)
    return current


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
prefix = 'eval.runner' if hasattr(cfg.eval, 'runner') else 'eval'
fields = {
    'CFG_SUITE': ('eval.runner.task_suite_name', 'eval.task_suite_name'),
    'CFG_TASK_IDS': (
        'eval.manager.task_ids', 'eval.runner.task_ids', 'eval.task_ids'),
    'CFG_OUTPUT_DIR': (
        'eval.manager.output_dir', 'eval.runner.result_output_dir',
        'eval.result_output_dir'),
    'CFG_NUM_TRIALS': (
        'eval.runner.num_trials_per_task', 'eval.num_trials_per_task'),
    'CFG_MODEL_BUILD_DEVICE': (
        'eval.runner.model_build_device', 'eval.model_build_device'),
    'CFG_MODEL_BUILD_DTYPE': (
        'eval.runner.model_build_dtype', 'eval.model_build_dtype'),
    'CFG_PREPROCESS_EVERY_STEP': (
        'eval.runner.preprocess_every_step', 'eval.preprocess_every_step'),
    'CFG_SAVE_ROLLOUT_VIDEOS': (
        'eval.runner.save_rollout_videos', 'eval.save_rollout_videos'),
    'CFG_SAVE_FAILED_ROLLOUT_VIDEOS': (
        'eval.runner.save_failed_rollout_videos',
        'eval.save_failed_rollout_videos'),
    'CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS': (
        'eval.runner.save_multi_view_rollout_videos',
        'eval.save_multi_view_rollout_videos'),
    'CFG_ROLLOUT_DIR': ('eval.runner.rollout_dir', 'eval.rollout_dir'),
}
print(f'CFG_EVAL_PREFIX\t{prefix}')
for name, paths in fields.items():
    print(f'{name}\t{format_value(first_path(cfg, *paths))}')
PY
)"

CFG_EVAL_PREFIX=""
CFG_SUITE=""
CFG_TASK_IDS=""
CFG_OUTPUT_DIR=""
CFG_NUM_TRIALS=""
CFG_MODEL_BUILD_DEVICE=""
CFG_MODEL_BUILD_DTYPE=""
CFG_PREPROCESS_EVERY_STEP=""
CFG_SAVE_ROLLOUT_VIDEOS=""
CFG_SAVE_FAILED_ROLLOUT_VIDEOS=""
CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS=""
CFG_ROLLOUT_DIR=""
while IFS=$'\t' read -r key value; do
  case "${key}" in
    CFG_EVAL_PREFIX) CFG_EVAL_PREFIX="${value}" ;;
    CFG_SUITE) CFG_SUITE="${value}" ;;
    CFG_TASK_IDS) CFG_TASK_IDS="${value}" ;;
    CFG_OUTPUT_DIR) CFG_OUTPUT_DIR="${value}" ;;
    CFG_NUM_TRIALS) CFG_NUM_TRIALS="${value}" ;;
    CFG_MODEL_BUILD_DEVICE) CFG_MODEL_BUILD_DEVICE="${value}" ;;
    CFG_MODEL_BUILD_DTYPE) CFG_MODEL_BUILD_DTYPE="${value}" ;;
    CFG_PREPROCESS_EVERY_STEP) CFG_PREPROCESS_EVERY_STEP="${value}" ;;
    CFG_SAVE_ROLLOUT_VIDEOS) CFG_SAVE_ROLLOUT_VIDEOS="${value}" ;;
    CFG_SAVE_FAILED_ROLLOUT_VIDEOS) CFG_SAVE_FAILED_ROLLOUT_VIDEOS="${value}" ;;
    CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS) CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${value}" ;;
    CFG_ROLLOUT_DIR) CFG_ROLLOUT_DIR="${value}" ;;
  esac
done <<< "${CFG_VALUES}"

EVAL_PREFIX="${CFG_EVAL_PREFIX:-eval}"
SUITE="${SUITE:-${CFG_SUITE}}"
TASK_IDS="${TASK_IDS:-${CFG_TASK_IDS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUTPUT_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/libero_plus_eval_manager/$(date +%Y%m%d_%H%M%S)}"
# Standard LIBERO configs use 50 trials; the Plus protocol always overrides
# that config default to one unless the caller explicitly supplies an invalid
# value (which is rejected below).
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
MODEL_BUILD_DEVICE="${MODEL_BUILD_DEVICE:-${CFG_MODEL_BUILD_DEVICE:-cuda}}"
MODEL_BUILD_DTYPE="${MODEL_BUILD_DTYPE:-${CFG_MODEL_BUILD_DTYPE:-bf16}}"
PREPROCESS_EVERY_STEP="${PREPROCESS_EVERY_STEP:-${CFG_PREPROCESS_EVERY_STEP:-True}}"
SAVE_ROLLOUT_VIDEOS="${SAVE_ROLLOUT_VIDEOS:-${CFG_SAVE_ROLLOUT_VIDEOS:-False}}"
SAVE_FAILED_ROLLOUT_VIDEOS="${SAVE_FAILED_ROLLOUT_VIDEOS:-${CFG_SAVE_FAILED_ROLLOUT_VIDEOS:-False}}"
SAVE_MULTI_VIEW_ROLLOUT_VIDEOS="${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS:-${CFG_SAVE_MULTI_VIEW_ROLLOUT_VIDEOS:-False}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${CFG_ROLLOUT_DIR}}"

if [[ -z "${TASK_CLASSIFICATION}" ]]; then
  TASK_CLASSIFICATION="$(python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec('libero_plus.libero')
if spec is not None and spec.origin is not None:
    path = Path(spec.origin).resolve().parent / 'benchmark' / 'task_classification.json'
    if path.is_file():
        print(path)
PY
)"
fi
[[ -n "${TASK_CLASSIFICATION}" && -f "${TASK_CLASSIFICATION}" ]] || {
  echo "[libero-plus-manager] cannot find LIBERO-Plus task_classification.json; install requirements-sim.txt or set TASK_CLASSIFICATION" >&2
  exit 1
}

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *)
    echo "[libero-plus-manager] expected exactly one LIBERO-Plus suite; got: ${SUITE:-<empty>}" >&2
    exit 2
    ;;
esac
[[ "${NUM_TRIALS_PER_TASK}" == "1" ]] || {
  echo "[libero-plus-manager] LIBERO-Plus requires NUM_TRIALS_PER_TASK=1" >&2
  exit 2
}
[[ "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[libero-plus-manager] WORKERS_PER_GPU must be a positive integer: ${WORKERS_PER_GPU}" >&2
  exit 2
}
[[ "${MONITOR_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[libero-plus-manager] MONITOR_INTERVAL must be a positive integer: ${MONITOR_INTERVAL}" >&2
  exit 2
}

bool_cfg() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES) echo True ;;
    *) echo False ;;
  esac
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a RAW_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  GPU_ARRAY=()
  for gpu_id in "${RAW_GPU_ARRAY[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    [[ -n "${gpu_id}" ]] && GPU_ARRAY+=("${gpu_id}")
  done
else
  NUM_GPUS="${NUM_GPUS:-8}"
  GPU_ARRAY=()
  for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
    GPU_ARRAY+=("${gpu_index}")
  done
fi
[[ "${#GPU_ARRAY[@]}" -gt 0 ]] || {
  echo "[libero-plus-manager] no GPUs resolved" >&2
  exit 1
}
for gpu_id in "${GPU_ARRAY[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
    echo "[libero-plus-manager] GPU identifiers must be numeric: ${gpu_id}" >&2
    exit 2
  }
done

CONFIG="$(readlink -f "${CONFIG}")"
CKPT="$(readlink -f "${CKPT}")"
CKPT_TITLE="$(basename "${CKPT}")"
CKPT_TITLE="${CKPT_TITLE%.*}"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
TASK_CLASSIFICATION="$(readlink -f "${TASK_CLASSIFICATION}")"
touch "${OUTPUT_DIR}/failed_tasks.txt"
SUITE_DIR="${OUTPUT_DIR}/${SUITE}"
WORKER_RESULTS_DIR="${SUITE_DIR}/worker_results"
WORKER_LOG_DIR="${SUITE_DIR}/worker_logs"
WORKER_STATUS_DIR="${SUITE_DIR}/worker_status"
WORKER_TASK_DIR="${SUITE_DIR}/worker_tasks"
mkdir -p "${WORKER_RESULTS_DIR}" "${WORKER_LOG_DIR}" \
  "${WORKER_STATUS_DIR}" "${WORKER_TASK_DIR}"
if [[ -d "${SUITE_DIR}/${SUITE}/worker_results" ]]; then
  echo "[libero-plus-manager] nested legacy worker_results layout is not " \
    "supported: ${SUITE_DIR}/${SUITE}/worker_results" >&2
  exit 1
fi
if [[ "$(bool_cfg "${RESUME}")" == True ]]; then
  if compgen -G "${SUITE_DIR}/gpu*_task*_results.json" > /dev/null ||
     compgen -G "${WORKER_RESULTS_DIR}/worker*_results.jsonl" > /dev/null; then
    echo "[libero-plus-manager] incompatible per-task/JSONL results cannot " \
      "be resumed; use a new OUTPUT_DIR or set RESUME=0" >&2
    exit 1
  fi
else
  find "${SUITE_DIR}" -maxdepth 1 -type f \
    -name 'gpu*_task*_results.json' -delete
  find "${WORKER_RESULTS_DIR}" -maxdepth 1 -type f \
    \( -name 'worker*_results.json' -o \
       -name 'worker*_results.jsonl' \) -delete
fi

RUN_TAG="persistent_$(date +%Y%m%d_%H%M%S)"
ASSIGNMENT_FILE="${WORKER_TASK_DIR}/assignments.tsv"
REQUESTED_FILE="${WORKER_TASK_DIR}/requested_tasks.txt"
PENDING_FILE="${WORKER_TASK_DIR}/pending_tasks.txt"

python - "${ASSIGNMENT_FILE}" "${REQUESTED_FILE}" "${PENDING_FILE}" \
  "${OUTPUT_DIR}" "${SUITE}" "${TASK_IDS}" "${RESUME}" \
  "${WORKERS_PER_GPU}" "${GPU_ARRAY[@]}" <<'PY'
import contextlib
import glob
import io
import json
import os
import sys

from libero_plus.libero import benchmark


def parse_task_ids(raw_value, num_tasks):
    value = str(raw_value).strip()
    if value == '' or value.lower() == 'none':
        return list(range(num_tasks))
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    task_ids = [
        int(item) for item in value.replace(',', ' ').split() if item
    ]
    if not task_ids:
        raise SystemExit('TASK_IDS did not contain any valid task ids.')
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit(f'Duplicate TASK_IDS are not supported: {task_ids}')
    invalid = [task for task in task_ids if task < 0 or task >= num_tasks]
    if invalid:
        raise SystemExit(
            f'Invalid TASK_IDS {invalid}; expected range [0, {num_tasks - 1}].')
    return task_ids


(
    assignment_file,
    requested_file,
    pending_file,
    output_dir,
    suite_name,
    raw_task_ids,
    resume,
    workers_per_gpu,
    *gpus,
) = sys.argv[1:]
if not gpus:
    raise SystemExit('At least one GPU is required.')
with contextlib.redirect_stdout(io.StringIO()):
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
requested = parse_task_ids(raw_task_ids, int(task_suite.n_tasks))
requested_set = set(requested)

completed = set()
if str(resume).lower() not in {'0', 'false', 'no'}:
    pattern = os.path.join(
        output_dir, suite_name, 'worker_results',
        'worker*_results.json')
    completed_sources = {}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding='utf-8') as checkpoint_file:
                checkpoint = json.load(checkpoint_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
              f'Cannot read per-worker results {path}: {exc}')
        if checkpoint.get('task_suite') != suite_name:
            raise SystemExit(f'Per-worker result suite mismatch: {path}')
        for result_key, result in checkpoint.get('results', {}).items():
            source = f'{path}:results[{result_key!r}]'
            try:
                task_id = int(result['task_id'])
                episodes = int(result['total_episodes'])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f'Invalid task result {source}: {exc}')
            if task_id in completed_sources:
                raise SystemExit(
                    f'Duplicate task {task_id} in '
                    f'{completed_sources[task_id]} and '
                    f'{source}.')
            completed_sources[task_id] = source
            if task_id in requested_set and episodes >= 1:
                completed.add(task_id)

pending = [task_id for task_id in requested if task_id not in completed]
workers_per_gpu = int(workers_per_gpu)
workers = [
    (gpu, local_index, gpu_index * workers_per_gpu + local_index)
    for gpu_index, gpu in enumerate(gpus)
    for local_index in range(workers_per_gpu)
]
assignments = {worker_id: [] for _, _, worker_id in workers}
for index, task_id in enumerate(pending):
    assignments[workers[index % len(workers)][2]].append(task_id)

with open(requested_file, 'w', encoding='utf-8') as output:
    output.write('\n'.join(str(task_id) for task_id in requested) + '\n')
with open(pending_file, 'w', encoding='utf-8') as output:
    if pending:
        output.write('\n'.join(str(task_id) for task_id in pending) + '\n')
with open(assignment_file, 'w', encoding='utf-8') as output:
    for gpu, local_index, worker_id in workers:
        task_ids = assignments[worker_id]
        if not task_ids:
            continue
        output.write(
            f'{gpu}\t{local_index}\t{worker_id}\t'
            + ','.join(str(task_id) for task_id in task_ids)
            + '\n')
print(
    f'[libero-plus-manager] suite={suite_name} requested={len(requested)} '
    f'completed={len(completed)} pending={len(pending)} '
    f'workers={len(workers)} workers_per_gpu={workers_per_gpu}')
PY

cat > "${SUITE_DIR}/manager_config.yaml" <<EOF
config: ${CONFIG}
checkpoint: ${CKPT}
suite: ${SUITE}
output_dir: ${OUTPUT_DIR}
task_classification: ${TASK_CLASSIFICATION}
gpus: ${GPU_ARRAY[*]}
task_ids: ${TASK_IDS:-all}
num_trials_per_task: ${NUM_TRIALS_PER_TASK}
worker_mode: persistent-task-shards
result_storage_mode: per_worker
workers_per_gpu: ${WORKERS_PER_GPU}
resume: ${RESUME}
model_build_device: ${MODEL_BUILD_DEVICE}
model_build_dtype: ${MODEL_BUILD_DTYPE}
preprocess_every_step: ${PREPROCESS_EVERY_STEP:-config default}
save_rollout_videos: ${SAVE_ROLLOUT_VIDEOS}
save_failed_rollout_videos: ${SAVE_FAILED_ROLLOUT_VIDEOS}
save_multi_view_rollout_videos: ${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS}
rollout_dir: ${ROLLOUT_DIR:-default}
assignment_file: ${ASSIGNMENT_FILE}
EOF

if [[ "$(bool_cfg "${DRY_RUN}")" == True ]]; then
  echo "[libero-plus-manager] dry run assignments: ${ASSIGNMENT_FILE}"
  exit 0
fi

build_worker_cfg_options() {
  local task_ids="$1"
  local worker_id="$2"
  local suffix="$3"
  local cfg_options=(
    "${EVAL_PREFIX}.task_suite_name=${SUITE}"
    "${EVAL_PREFIX}.task_ids=[${task_ids}]"
    "${EVAL_PREFIX}.num_trials_per_task=1"
    "${EVAL_PREFIX}.eval_shard_strategy=task"
    "${EVAL_PREFIX}.model_build_device=${MODEL_BUILD_DEVICE}"
    "${EVAL_PREFIX}.model_build_dtype=${MODEL_BUILD_DTYPE}"
    "${EVAL_PREFIX}.save_rollout_videos=$(bool_cfg "${SAVE_ROLLOUT_VIDEOS}")"
    "${EVAL_PREFIX}.save_failed_rollout_videos=$(bool_cfg "${SAVE_FAILED_ROLLOUT_VIDEOS}")"
    "${EVAL_PREFIX}.save_multi_view_rollout_videos=$(bool_cfg "${SAVE_MULTI_VIEW_ROLLOUT_VIDEOS}")"
    "${EVAL_PREFIX}.run_id_suffix=${suffix}"
    "${EVAL_PREFIX}.result_output_dir=${OUTPUT_DIR}"
    "${EVAL_PREFIX}.result_gpu_id=${worker_id}"
    "${EVAL_PREFIX}.result_storage_mode=per_worker"
  )
  if [[ -n "${PREPROCESS_EVERY_STEP}" ]]; then
    cfg_options+=(
      "${EVAL_PREFIX}.preprocess_every_step=$(bool_cfg "${PREPROCESS_EVERY_STEP}")")
  fi
  if [[ -n "${ROLLOUT_DIR}" ]]; then
    cfg_options+=("${EVAL_PREFIX}.rollout_dir=${ROLLOUT_DIR}")
  fi
  WORKER_CFG_OPTIONS=("${WORKER_USER_CFG_OPTIONS[@]}" "${cfg_options[@]}")
}

PIDS=()
WORKER_LABELS=()
WORKER_STATUS_FILES=()
DONE=()

kill_process_tree() {
  local pid="$1"
  local child_pid
  for child_pid in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_process_tree "${child_pid}"
  done
  kill "${pid}" 2>/dev/null || true
}

cleanup_workers() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill_process_tree "${pid}"
  done
}
trap cleanup_workers INT TERM

process_finished() {
  local pid="$1"
  local process_state
  process_state="$(ps -p "${pid}" -o stat= 2>/dev/null || true)"
  [[ -z "${process_state}" || "${process_state}" == *Z* ]]
}

while IFS=$'\t' read -r gpu_id local_index worker_id task_ids; do
  [[ -n "${task_ids}" ]] || continue
  task_count=$(awk -F',' '{print NF}' <<< "${task_ids}")
  worker_label="gpu${gpu_id}_worker${local_index}"
  log_file="${WORKER_LOG_DIR}/${worker_label}.log"
  status_file="${WORKER_STATUS_DIR}/worker${worker_id}.status"
  rm -f "${status_file}"
  echo "[libero-plus-manager] launch ${worker_label}: ${task_count} tasks"
  {
    echo
    echo "===== ${RUN_TAG} start $(date '+%F %T') tasks=${task_ids} ====="
  } >> "${log_file}"
  (
    set +e
    build_worker_cfg_options \
      "${task_ids}" "${worker_id}" "${RUN_TAG}_${worker_label}"
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    FLUXVLA_ALLOW_LIBERO_TASK_SHARDING=1 \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      python scripts/eval.py \
        --config "${CONFIG}" \
        --ckpt-path "${CKPT}" \
        --cfg-options "${WORKER_CFG_OPTIONS[@]}" \
        "${FORWARDED_EXTRA_ARGS[@]}" \
        >> "${log_file}" 2>&1
    return_code=$?
    if [[ "${return_code}" -eq 0 ]]; then
      echo "SUCCESS|${return_code}|$(date +%s)|${log_file}" > "${status_file}"
    else
      echo "FAILED|${return_code}|$(date +%s)|${log_file}" > "${status_file}"
    fi
    exit "${return_code}"
  ) &
  PIDS+=("$!")
  WORKER_LABELS+=("${worker_label}")
  WORKER_STATUS_FILES+=("${status_file}")
  DONE+=(0)
  sleep "${LAUNCH_DELAY}"
done < "${ASSIGNMENT_FILE}"

progress_snapshot() {
  python - "${REQUESTED_FILE}" "${WORKER_RESULTS_DIR}" <<'PY'
import glob
import json
import os
import sys

requested = {
    int(line.strip())
    for line in open(sys.argv[1], encoding='utf-8')
    if line.strip()
}
results = {}
for path in sorted(glob.glob(
        os.path.join(sys.argv[2], 'worker*_results.json'))):
    try:
        with open(path, encoding='utf-8') as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
    except (OSError, json.JSONDecodeError):
        continue
    for result in checkpoint.get('results', {}).values():
        try:
            task_id = int(result['task_id'])
        except (KeyError, TypeError, ValueError):
            continue
        if task_id in requested:
            results[task_id] = result
completed = len(results)
successes = sum(int(result.get('successes', 0)) for result in results.values())
duration = sum(float(result.get('duration', 0.0)) for result in results.values())
rate = successes / completed * 100.0 if completed else 0.0
avg_time = duration / completed if completed else 0.0
print(f'{completed}\t{len(requested)}\t{successes}\t{rate:.2f}\t{avg_time:.2f}')
PY
}

remaining="${#PIDS[@]}"
last_status_time=0
failed=0
while [[ "${remaining}" -gt 0 ]]; do
  for index in "${!PIDS[@]}"; do
    [[ "${DONE[${index}]}" == 0 ]] || continue
    pid="${PIDS[${index}]}"
    if ! process_finished "${pid}"; then
      continue
    fi
    if wait "${pid}"; then
      echo "[libero-plus-manager] ${WORKER_LABELS[${index}]} completed"
    else
      return_code=$?
      status_line="missing status"
      if [[ -f "${WORKER_STATUS_FILES[${index}]}" ]]; then
        status_line="$(cat "${WORKER_STATUS_FILES[${index}]}")"
      fi
      echo "[libero-plus-manager] ${WORKER_LABELS[${index}]} failed (rc=${return_code}, ${status_line})" \
        | tee -a "${OUTPUT_DIR}/failed_tasks.txt" >&2
      failed=1
    fi
    DONE[${index}]=1
    remaining=$((remaining - 1))
  done

  if [[ "${failed}" -ne 0 ]]; then
    cleanup_workers
    for index in "${!PIDS[@]}"; do
      [[ "${DONE[${index}]}" == 1 ]] || wait "${PIDS[${index}]}" 2>/dev/null || true
    done
    echo "[libero-plus-manager] stopped after worker failure; rerun with RESUME=1" >&2
    exit 1
  fi

  now="$(date +%s)"
  if [[ $((now - last_status_time)) -ge "${MONITOR_INTERVAL}" ]]; then
    IFS=$'\t' read -r completed requested successes success_rate avg_time \
      <<< "$(progress_snapshot)"
    echo "[libero-plus-manager] status completed=${completed}/${requested} successes=${successes} (${success_rate}%) avg_task_time=${avg_time}s workers=${remaining}"
    last_status_time="${now}"
  fi
  [[ "${remaining}" -eq 0 ]] || sleep 2
done

python - "${REQUESTED_FILE}" "${WORKER_RESULTS_DIR}" "${SUITE}" <<'PY'
import glob
import json
import os
import sys

requested = [
    int(line.strip())
    for line in open(sys.argv[1], encoding='utf-8')
    if line.strip()
]
requested_set = set(requested)
seen = {}
for path in sorted(glob.glob(
        os.path.join(sys.argv[2], 'worker*_results.json'))):
    try:
        with open(path, encoding='utf-8') as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Cannot read per-worker results {path}: {exc}')
    if checkpoint.get('task_suite') != sys.argv[3]:
        raise SystemExit(f'Per-worker result suite mismatch: {path}')
    for result_key, result in checkpoint.get('results', {}).items():
        source = f'{path}:results[{result_key!r}]'
        if result.get('task_suite') != sys.argv[3]:
            raise SystemExit(f'Task result suite mismatch: {source}')
        task_id = int(result['task_id'])
        if task_id not in requested_set:
            continue
        if task_id in seen:
            raise SystemExit(
                f'Duplicate task {task_id} in {seen[task_id]} and '
                f'{source}.')
        if int(result.get('total_episodes', 0)) != 1:
            raise SystemExit(
                f'Task {task_id} in {source} does not contain one episode.')
        successes = int(result.get('successes', -1))
        if successes not in (0, 1):
            raise SystemExit(
                f'Task {task_id} in {source} has invalid '
                f'successes={successes}.')
        seen[task_id] = source
missing = [task_id for task_id in requested if task_id not in seen]
if missing:
    raise SystemExit(
        f'Expected {len(requested)} task results, found {len(seen)}; '
        f'missing task IDs: {missing[:20]}')
print(f'[libero-plus-manager] verified {len(seen)}/{len(requested)} task results')
PY

SUMMARY_ARGS=(
  --run-root "${OUTPUT_DIR}"
  --task-classification "${TASK_CLASSIFICATION}"
  --output-dir "${OUTPUT_DIR}"
  --title "${CKPT_TITLE}"
  --config "${CONFIG}"
  --ckpt "${CKPT}"
)
if [[ "$(bool_cfg "${REQUIRE_COMPLETE}")" != True ]]; then
  SUMMARY_ARGS+=(--allow-incomplete)
fi
CONFIG="${CONFIG}" CKPT="${CKPT}" \
  python "${SUMMARY_TOOL}" "${SUMMARY_ARGS[@]}"

echo "[libero-plus-manager] suite complete: ${SUITE_DIR}"
echo "[libero-plus-manager] summary: ${OUTPUT_DIR}/summary.csv"
echo "[libero-plus-manager] LIBERO-Plus summary: ${OUTPUT_DIR}/libero_plus_summary.csv"
