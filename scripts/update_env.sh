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

FLUXVLA_AV_VERSION="${FLUXVLA_AV_VERSION:-14.2.0}"
FLUXVLA_FFMPEG_VERSION="${FLUXVLA_FFMPEG_VERSION:-7}"
CONDA_INSTALL_TIMEOUT="${CONDA_INSTALL_TIMEOUT:-3600}"
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
  FLUXVLA_AV_VERSION
                   PyAV wheel version. Default: 14.2.0.
  FLUXVLA_FFMPEG_VERSION
                   conda-forge FFmpeg major version used by TorchCodec.
                   Default: 7.
  CONDA_INSTALL_TIMEOUT
                   Per conda command timeout in seconds. Default: 3600.

This updater installs the unified requirements-base.txt dependency set, but
intentionally does not reinstall PyTorch or FlashAttention.
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

run_conda_with_timeout() {
  echo "+ $*"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return
  fi
  if command -v timeout >/dev/null 2>&1; then
    timeout "${CONDA_INSTALL_TIMEOUT}" "$@"
  else
    "$@"
  fi
}

python_prefix() {
  "${PYTHON_BIN}" - <<'PY'
import sys
print(sys.prefix)
PY
}

conda_env_prefix() {
  local prefix=""
  prefix="$(python_prefix)"
  if [[ -n "${prefix}" && -d "${prefix}/conda-meta" ]]; then
    echo "${prefix}"
  elif [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/conda-meta" ]]; then
    echo "${CONDA_PREFIX}"
  fi
}

find_conda_bin() {
  local candidate prefix base_prefix
  local -a candidates=()

  if [[ -n "${CONDA_EXE:-}" ]]; then
    candidates+=("${CONDA_EXE}")
  fi
  if candidate="$(command -v conda 2>/dev/null)"; then
    candidates+=("${candidate}")
  fi

  prefix="$(conda_env_prefix)"
  if [[ -n "${prefix}" ]]; then
    if [[ "${prefix}" == */envs/* ]]; then
      base_prefix="${prefix%%/envs/*}"
      candidates+=("${base_prefix}/bin/conda")
    fi
    candidates+=("${prefix}/bin/conda")
  fi
  candidates+=("/root/miniconda3/bin/conda" "/opt/conda/bin/conda")

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
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

install_ffmpeg_runtime() {
  local machine conda_prefix="" conda_bin="" ffmpeg_spec
  machine="$(uname -m)"
  case "${machine}" in
    x86_64|amd64)
      ;;
    *)
      return
      ;;
  esac

  conda_prefix="$(conda_env_prefix)"
  conda_bin="$(find_conda_bin || true)"
  ffmpeg_spec="ffmpeg=${FLUXVLA_FFMPEG_VERSION}"
  if [[ -z "${conda_bin}" || -z "${conda_prefix}" ]]; then
    echo "Warning: no active conda environment was found; relying on system FFmpeg shared libraries for TorchCodec." >&2
    return
  fi

  echo "Installing TorchCodec FFmpeg runtime via conda-forge: ${ffmpeg_spec}"
  if "${conda_bin}" install --help 2>/dev/null | grep -q -- '--solver'; then
    if run_conda_with_timeout "${conda_bin}" install -y \
        -p "${conda_prefix}" -c conda-forge --solver=libmamba \
        "${ffmpeg_spec}"; then
      return
    fi
    echo "conda FFmpeg install with libmamba failed or timed out; trying default solver." >&2
  fi

  run_conda_with_timeout "${conda_bin}" install -y \
    -p "${conda_prefix}" -c conda-forge "${ffmpeg_spec}"
}

install_torchcodec() {
  local machine torch_version torchcodec_spec
  machine="$(uname -m)"
  case "${machine}" in
    x86_64|amd64)
      ;;
    *)
      echo "Skipping optional TorchCodec on ${machine}; video decoding will use PyAV."
      return
      ;;
  esac

  if ! torch_version="$("${PYTHON_BIN}" -c 'import torch; print(".".join(torch.__version__.split("+", 1)[0].split(".")[:2]))')"; then
    echo "Warning: unable to detect PyTorch; skipping optional TorchCodec." >&2
    return
  fi
  case "${torch_version}" in
    2.6)
      torchcodec_spec="torchcodec==0.2.1"
      ;;
    2.8)
      torchcodec_spec="torchcodec==0.7.0"
      ;;
    *)
      echo "Warning: torch ${torch_version} has no pinned TorchCodec version; keeping the PyAV fallback." >&2
      return
      ;;
  esac

  if ! pip_install "${torchcodec_spec}"; then
    echo "Warning: failed to install optional ${torchcodec_spec}; video decoding will use PyAV." >&2
    return
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "+ verify TorchCodec import"
    return
  fi
  if ! "${PYTHON_BIN}" -c \
      'from torchcodec.decoders import VideoDecoder; print("TorchCodec import verified")'; then
    echo "Warning: TorchCodec was installed but cannot be imported; video decoding will use PyAV unless torchcodec is requested explicitly." >&2
  fi
}

cd "${PROJECT_ROOT}"

if [[ "${SKIP_PULL}" -eq 0 ]]; then
  # shellcheck disable=SC2086
  run git pull ${GIT_PULL_ARGS}
fi

pip_install --upgrade -r "${PROJECT_ROOT}/requirements-base.txt"
pip_install --upgrade --only-binary=:all: "av==${FLUXVLA_AV_VERSION}"

pip_install \
  "mujoco==${MUJOCO_VERSION}" \
  gymnasium \
  lxml \
  "bddl==${BDDL_VERSION}" \
  "hydra-core==${HYDRA_CORE_VERSION}" \
  "robomimic==${ROBOMIMIC_VERSION}"

pip_install --force-reinstall --no-deps "${LIBERO_SPEC}"
pip_install --force-reinstall --no-deps "${ROBOSUITE_SPEC}"
install_ffmpeg_runtime
install_torchcodec

if [[ "${SKIP_PROJECT}" -eq 0 ]]; then
  pip_install --no-build-isolation -e "${PROJECT_ROOT}"
fi

run "${PYTHON_BIN}" -c \
  'import av, diffusers, peft, transformers; from diffusers import Cosmos2_5_PredictBasePipeline; print("av", av.__version__, "diffusers", diffusers.__version__, "peft", peft.__version__, "transformers", transformers.__version__)'
