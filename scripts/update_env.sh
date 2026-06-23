#!/usr/bin/env bash
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

DRY_RUN=0
SKIP_PULL=0
SKIP_PROJECT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.3.0}"
DATASETS_VERSION="${DATASETS_VERSION:-4.0.0}"
MUJOCO_VERSION="${MUJOCO_VERSION:-3.2.6}"
BDDL_VERSION="${BDDL_VERSION:-1.0.1}"
HYDRA_CORE_VERSION="${HYDRA_CORE_VERSION:-1.2.0}"
ROBOMIMIC_VERSION="${ROBOMIMIC_VERSION:-0.2.0}"
LIBERO_REF="${LIBERO_REF:-058fda1ddebe92918af091cb6816759ca6d003f0}"
LIBERO_SPEC="${LIBERO_SPEC:-libero @ git+https://github.com/yinchimaoliang/LIBERO.git@${LIBERO_REF}}"
ROBOSUITE_REF="${ROBOSUITE_REF:-e293cc32ff3c48957a4ebcad09952432b0dc9049}"
ROBOSUITE_SPEC="${ROBOSUITE_SPEC:-robosuite @ git+https://github.com/yinchimaoliang/robosuite.git@${ROBOSUITE_REF}}"
PIP_INDEX_URLS="${PIP_INDEX_URLS:-}"
GIT_PULL_ARGS="${GIT_PULL_ARGS:---ff-only}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/update_env.sh [options]

Options:
  --dry-run       Print commands without executing them.
  --skip-pull     Do not run git pull.
  --skip-project  Do not reinstall FluxVLA in editable mode.
  -h, --help      Show this help.

Environment variables:
  PYTHON          Python executable to use. Default: $CONDA_PREFIX/bin/python
                  when available, otherwise python.
  PIP_INDEX_URLS  Optional space-separated pip indexes retried in order.
  GIT_PULL_ARGS   Arguments passed to git pull. Default: --ff-only.

This updater intentionally does not reinstall PyTorch or FlashAttention.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-pull)
      SKIP_PULL=1
      ;;
    --skip-project)
      SKIP_PROJECT=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

pip_install() {
  if [[ -z "${PIP_INDEX_URLS}" ]]; then
    run "${PYTHON_BIN}" -m pip install "$@"
    return
  fi

  local index_url
  for index_url in ${PIP_INDEX_URLS}; do
    echo "+ ${PYTHON_BIN} -m pip install --index-url ${index_url} $*"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      continue
    fi
    if "${PYTHON_BIN}" -m pip install --index-url "${index_url}" "$@"; then
      return
    fi
  done

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return
  fi
  return 1
}

cd "${PROJECT_ROOT}"

if [[ "${SKIP_PULL}" -eq 0 ]]; then
  # shellcheck disable=SC2086
  run git pull ${GIT_PULL_ARGS}
fi

pip_install --upgrade \
  "transformers==${TRANSFORMERS_VERSION}" \
  "datasets==${DATASETS_VERSION}"

pip_install \
  "mujoco==${MUJOCO_VERSION}" \
  gymnasium \
  lxml \
  "bddl==${BDDL_VERSION}" \
  "hydra-core==${HYDRA_CORE_VERSION}" \
  "robomimic==${ROBOMIMIC_VERSION}"

pip_install --force-reinstall --no-deps "${LIBERO_SPEC}"
pip_install --force-reinstall --no-deps "${ROBOSUITE_SPEC}"

if [[ "${SKIP_PROJECT}" -eq 0 ]]; then
  pip_install --no-build-isolation -e "${PROJECT_ROOT}"
fi

run "${PYTHON_BIN}" -c \
  'import transformers; print("transformers", transformers.__version__)'
