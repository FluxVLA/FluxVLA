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

ENV_MODE="full"
PROFILE="auto"
DRY_RUN=0
SKIP_AV=0
SKIP_FLASH_ATTN=0
SKIP_PROJECT=0
SKIP_BUILD_TOOLS=0
SKIP_EGL_SETUP=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

DEFAULT_PIP_INDEX_URL="https://pypi.org/simple"
DEFAULT_PIP_MIRROR_URLS="https://mirrors.aliyun.com/pypi/simple https://mirrors.cloud.tencent.com/pypi/simple https://repo.huaweicloud.com/repository/pypi/simple https://pypi.mirrors.ustc.edu.cn/simple https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.bfsu.edu.cn/pypi/web/simple https://mirror.sjtu.edu.cn/pypi/web/simple https://mirror.nju.edu.cn/pypi/web/simple"
DEFAULT_PIP_INDEX_CANDIDATES="${PIP_INDEX_CANDIDATES:-${DEFAULT_PIP_INDEX_URL} ${DEFAULT_PIP_MIRROR_URLS}}"
DEFAULT_GH_PROXY_CANDIDATES="${GH_PROXY_CANDIDATES:-https://ghfast.top https://gh.llkk.cc https://gh.ddlc.top https://gh-proxy.com https://ghproxy.net https://gh-proxy.ygxz.in}"
USER_PIP_INDEX_URLS="${PIP_INDEX_URLS:-}"
PIP_INDEX_URLS="${PIP_INDEX_URLS:-}"
PIP_CONFIG_SENTINEL="__pip_config__"
PIP_INDEX_MODE="${PIP_INDEX_MODE:-auto}"
PIP_INDEX_PROBE_TIMEOUT="${PIP_INDEX_PROBE_TIMEOUT:-15}"
PIP_INSTALL_TIMEOUT="${PIP_INSTALL_TIMEOUT:-7200}"
PIP_NETWORK_TIMEOUT="${PIP_NETWORK_TIMEOUT:-900}"
PIP_RETRIES="${PIP_RETRIES:-2}"
PIP_EXISTS_ACTION="${PIP_EXISTS_ACTION:-i}"
CONDA_INSTALL_TIMEOUT="${CONDA_INSTALL_TIMEOUT:-3600}"
DOWNLOAD_CONNECT_TIMEOUT="${DOWNLOAD_CONNECT_TIMEOUT:-30}"
DOWNLOAD_SPEED_TIME="${DOWNLOAD_SPEED_TIME:-120}"
DOWNLOAD_SPEED_LIMIT="${DOWNLOAD_SPEED_LIMIT:-10240}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_CONNECTIONS="${DOWNLOAD_CONNECTIONS:-8}"
FLUXVLA_DOWNLOADER="${FLUXVLA_DOWNLOADER:-auto}"
FLUXVLA_AV_INSTALLER="${FLUXVLA_AV_INSTALLER:-auto}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
FLASH_ATTN_RELEASE_TAG="${FLASH_ATTN_RELEASE_TAG:-v${FLASH_ATTN_VERSION}}"
FLASH_ATTN_WHEEL_FILE="${FLASH_ATTN_WHEEL_FILE:-}"
FLASH_ATTN_WHEEL_DIRS="${FLASH_ATTN_WHEEL_DIRS:-${PROJECT_ROOT}/wheelhouse ${PROJECT_ROOT}/wheels ${HOME}/.cache/fluxvla/wheels}"
FLASH_ATTN_WHEEL_BASE_URLS="${FLASH_ATTN_WHEEL_BASE_URLS:-}"
FLUXVLA_EGL_SETUP="${FLUXVLA_EGL_SETUP:-auto}"
FLUXVLA_EGL_VENDOR_FILE="${FLUXVLA_EGL_VENDOR_FILE:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
FLUXVLA_EGL_APT_PACKAGES="${FLUXVLA_EGL_APT_PACKAGES:-libegl1 libglvnd0 libopengl0 libegl-dev libgl1-mesa-dev libx11-dev libglew-dev libosmesa6-dev}"
FLUXVLA_ROBOCASA_INSTALL="${FLUXVLA_ROBOCASA_INSTALL:-auto}"
FLUXVLA_ROBOCASA_SRC_ROOT="${FLUXVLA_ROBOCASA_SRC_ROOT:-${PROJECT_ROOT}/src}"
FLUXVLA_GROOT_REPO="${FLUXVLA_GROOT_REPO:-https://github.com/NVIDIA/Isaac-GR00T.git}"
FLUXVLA_GROOT_REF="${FLUXVLA_GROOT_REF:-4af2b622892f7dcb5aae5a3fb70bcb02dc217b96}"
FLUXVLA_GROOT_DIR="${FLUXVLA_GROOT_DIR:-${FLUXVLA_ROBOCASA_SRC_ROOT}/Isaac-GR00T}"
FLUXVLA_ROBOCASA_GR1_REPO="${FLUXVLA_ROBOCASA_GR1_REPO:-https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git}"
FLUXVLA_ROBOCASA_GR1_REF="${FLUXVLA_ROBOCASA_GR1_REF:-4840e671596f93ca03651524b9f72ffb1aadfeff}"
FLUXVLA_ROBOCASA_GR1_DIR="${FLUXVLA_ROBOCASA_GR1_DIR:-${FLUXVLA_ROBOCASA_SRC_ROOT}/robocasa-gr1-tabletop-tasks}"
FLUXVLA_ROBOCASA_ASSETS="${FLUXVLA_ROBOCASA_ASSETS:-always}"
FLUXVLA_ROBOCASA_ASSET_ENDPOINT="${FLUXVLA_ROBOCASA_ASSET_ENDPOINT:-${HF_ENDPOINT:-https://hf-mirror.com}}"
FLUXVLA_ROBOCASA_ASSET_CACHE="${FLUXVLA_ROBOCASA_ASSET_CACHE:-/tmp/robocasa-assets}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/install_env.sh [sim-only|real-only|full] [options]

Environment modes:
  sim-only   Install training + simulation dependencies.
  real-only  Install training + real-robot / remote inference dependencies.
  full       Install all dependencies. Default.

Aliases:
  sim        Same as sim-only.
  real       Same as real-only.

Options:
  --profile auto|cu124|cu128  GPU profile. auto prefers the current CUDA
                              toolkit / nvcc version: CUDA >= 12.8 selects
                              cu128, otherwise cu124. Driver CUDA and GPU
                              generation are fallback signals only.
  --dry-run                   Print commands without executing them.
  --skip-av                   Skip av installation.
  --skip-flash-attn           Skip FlashAttention wheel installation.
  --skip-project              Skip editable FluxVLA installation.
  --skip-build-tools          Skip cmake/ninja preflight check.
  --skip-egl-setup            Skip LIBERO / MuJoCo EGL system setup.
  --with-robocasa             Install RoboCasa source checkouts even in
                              real-only mode.
  --skip-robocasa             Skip Isaac-GR00T / RoboCasa GR1 source checkout
                              installation.
  --with-robocasa-assets      Force RoboCasa tabletop simulator asset download
                              when RoboCasa source checkouts are installed.
  --skip-robocasa-assets      Skip RoboCasa asset download. --skip-robocasa
                              also skips asset download.
  -h, --help                  Show this help.

Environment variables:
  PYTHON              Python executable to use. Default: $CONDA_PREFIX/bin/python
                      when available, otherwise python.
  PIP_INDEX_MODE      pip index selection mode when PIP_INDEX_URLS is not set.
                      auto: use your pip config if it defines an index;
                      otherwise probe PyPI and mirrors by response time.
                      config: never pass --index-url; use pip config/default.
                      official: use PyPI only. mirror: use mirrors only.
                      all: use PyPI and mirrors in the default order.
  PIP_EXISTS_ACTION  pip action for existing editable VCS checkouts.
                      Default: i (ignore), which avoids interactive prompts
                      when ./src already contains a checkout with a different
                      remote URL.
  PIP_INDEX_URLS      Space-separated pip indexes tried in order for normal
                      packages. If set, this overrides PIP_INDEX_MODE.
  PIP_INDEX_CANDIDATES
                      Space-separated pip index candidates for auto/all mode.
                      Default: PyPI plus Aliyun, Tencent, Huawei, USTC, and
                      Tsinghua mirrors. Auto mode sorts them by response time.
  PIP_INDEX_PROBE_TIMEOUT
                      Per-index connectivity probe timeout in seconds.
                      Default: 15.
  PIP_INSTALL_TIMEOUT Per pip command timeout in seconds before trying the
                      next index. Default: 7200.
  PIP_NETWORK_TIMEOUT pip --timeout value. Default: 900.
  PIP_RETRIES         pip --retries value. Default: 2.
  CONDA_INSTALL_TIMEOUT
                      Per conda command timeout in seconds. Default: 3600.
  DOWNLOAD_CONNECT_TIMEOUT
                      curl connect timeout for release assets. Default: 30.
  DOWNLOAD_SPEED_TIME curl low-speed timeout window. Default: 120.
  DOWNLOAD_SPEED_LIMIT
                      curl low-speed threshold in bytes/s. Default: 10240.
  DOWNLOAD_RETRIES    curl retry count for release assets. Default: 5.
  DOWNLOAD_CONNECTIONS
                      aria2c split / connection count for release assets.
                      Default: 8.
  FLUXVLA_DOWNLOADER  Release asset downloader: auto, aria2, or curl.
                      Default: auto. auto uses aria2c when available and falls
                      back to curl if aria2c fails.
  FLUXVLA_AV_INSTALLER
                      av installer: auto, pip, or conda. Default: auto
                      (pip wheel first, then conda fallback).
  TORCH_INDEX_URLS    Space-separated PyTorch wheel indexes. Defaults to the
                      official index for the selected CUDA profile.
  GH_PROXY            Override the GitHub release proxy used for the
                      FlashAttention wheel (e.g. https://ghfast.top).
                      Empty string disables proxying entirely.
  GH_PROXY_CANDIDATES Space-separated GitHub release proxy candidates tried
                      before direct GitHub when GH_PROXY is not set. Default:
                      ghfast, gh.llkk.cc, gh.ddlc.top, gh-proxy.com,
                      ghproxy.net, and gh-proxy.ygxz.in.
  FLASH_ATTN_VERSION  Prebuilt FlashAttention wheel version. Default:
                      2.8.3.post1.
  FLASH_ATTN_RELEASE_TAG
                      GitHub release tag. Default: v2.8.3.post1.
  FLASH_ATTN_WHEEL_URL
                      Override the exact prebuilt wheel URL.
  FLASH_ATTN_WHEEL_FILE
                      Use this local FlashAttention wheel file directly.
  FLASH_ATTN_WHEEL_DIRS
                      Space-separated directories searched before downloading.
                      Default: ./wheelhouse, ./wheels, then
                      ~/.cache/fluxvla/wheels.
  FLASH_ATTN_WHEEL_BASE_URLS
                      Space-separated mirror base URLs searched before GitHub
                      proxies. Each base URL should contain the wheel filename.
  FLASH_ATTN_TORCH_TAG
                      Override the auto-detected torch wheel tag, e.g. 2.8.
  FLASH_ATTN_CUDA_TAG Override the auto-detected CUDA wheel tag, e.g. cu12.
  FLUXVLA_WHEEL_CACHE Directory for cached release wheels. Default:
                      ~/.cache/fluxvla/wheels.
  FLUXVLA_EGL_SETUP   LIBERO / MuJoCo EGL setup mode for sim/full installs.
                      auto: probe EGL first, then try system setup if needed.
                      always: run the EGL check strictly and fail if broken.
                      never: skip EGL setup. Default: auto.
  FLUXVLA_EGL_VENDOR_FILE
                      NVIDIA GLVND EGL vendor JSON path. Default:
                      /usr/share/glvnd/egl_vendor.d/10_nvidia.json.
  FLUXVLA_EGL_APT_PACKAGES
                      Space-separated apt packages for EGL / OSMesa setup.
  FLUXVLA_ROBOCASA_INSTALL
                      RoboCasa source checkout mode: auto, always, or never.
                      auto installs for sim-only/full and skips real-only.
                      Default: auto.
  FLUXVLA_ROBOCASA_SRC_ROOT
                      Directory containing RoboCasa source checkouts. Default:
                      ./src under the FluxVLA repository.
  FLUXVLA_GROOT_REPO, FLUXVLA_GROOT_REF, FLUXVLA_GROOT_DIR
                      Isaac-GR00T repository, pinned ref, and checkout path.
  FLUXVLA_ROBOCASA_GR1_REPO, FLUXVLA_ROBOCASA_GR1_REF,
  FLUXVLA_ROBOCASA_GR1_DIR
                      RoboCasa GR1 task repository, pinned ref, and checkout
                      path.
  FLUXVLA_ROBOCASA_ASSETS
                      RoboCasa asset download mode: always or never. Default:
                      always. Assets are downloaded only when RoboCasa source
                      checkouts are installed.
  FLUXVLA_ROBOCASA_ASSET_ENDPOINT
                      Hugging Face endpoint for RoboCasa asset downloads.
                      Default: HF_ENDPOINT if set, otherwise
                      https://hf-mirror.com.
  FLUXVLA_ROBOCASA_ASSET_CACHE
                      Local archive cache for RoboCasa asset downloads.
                      Default: /tmp/robocasa-assets.

Examples:
  conda activate fluxvla
  bash scripts/install_env.sh sim-only
  bash scripts/install_env.sh real-only --profile cu128
  bash scripts/install_env.sh full --dry-run
  PIP_INDEX_CANDIDATES="https://mirrors.aliyun.com/pypi/simple https://mirrors.cloud.tencent.com/pypi/simple https://repo.huaweicloud.com/repository/pypi/simple https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.bfsu.edu.cn/pypi/web/simple https://pypi.org/simple" bash scripts/install_env.sh full
  FLUXVLA_EGL_SETUP=always bash scripts/install_env.sh sim-only
  FLUXVLA_ROBOCASA_SRC_ROOT=/data/src bash scripts/install_env.sh sim-only
  bash scripts/install_env.sh sim-only --skip-robocasa
  GH_PROXY_CANDIDATES="https://ghfast.top https://gh.llkk.cc https://gh-proxy.com" bash scripts/install_env.sh full
EOF
}

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

run_with_timeout() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if command -v timeout >/dev/null 2>&1; then
    PIP_EXISTS_ACTION="${PIP_EXISTS_ACTION}" timeout "${PIP_INSTALL_TIMEOUT}" "$@"
  else
    PIP_EXISTS_ACTION="${PIP_EXISTS_ACTION}" "$@"
  fi
}

run_conda_with_timeout() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if command -v timeout >/dev/null 2>&1; then
    timeout "${CONDA_INSTALL_TIMEOUT}" "$@"
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    sim|sim-only)
      ENV_MODE="sim-only"
      shift
      ;;
    real|real-only)
      ENV_MODE="real-only"
      shift
      ;;
    full)
      ENV_MODE="full"
      shift
      ;;
    --profile)
      if [[ $# -lt 2 ]]; then
        echo "--profile requires a value: auto, cu124, or cu128" >&2
        exit 1
      fi
      PROFILE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-av)
      SKIP_AV=1
      shift
      ;;
    --skip-flash-attn)
      SKIP_FLASH_ATTN=1
      shift
      ;;
    --skip-project)
      SKIP_PROJECT=1
      shift
      ;;
    --skip-build-tools)
      SKIP_BUILD_TOOLS=1
      shift
      ;;
    --skip-egl-setup)
      SKIP_EGL_SETUP=1
      shift
      ;;
    --with-robocasa)
      FLUXVLA_ROBOCASA_INSTALL="always"
      shift
      ;;
    --skip-robocasa)
      FLUXVLA_ROBOCASA_INSTALL="never"
      shift
      ;;
    --with-robocasa-assets)
      FLUXVLA_ROBOCASA_ASSETS="always"
      shift
      ;;
    --skip-robocasa-assets)
      FLUXVLA_ROBOCASA_ASSETS="never"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${PROFILE}" != "auto" && "${PROFILE}" != "cu124" && "${PROFILE}" != "cu128" ]]; then
  echo "--profile must be one of: auto, cu124, cu128" >&2
  exit 1
fi

case "${PIP_INDEX_MODE}" in
  auto|config|official|mirror|all)
    ;;
  *)
    echo "PIP_INDEX_MODE must be one of: auto, config, official, mirror, all" >&2
    exit 1
    ;;
esac

case "${FLUXVLA_AV_INSTALLER}" in
  pip|conda|auto)
    ;;
  *)
    echo "FLUXVLA_AV_INSTALLER must be one of: pip, conda, auto" >&2
    exit 1
    ;;
esac

case "${FLUXVLA_DOWNLOADER}" in
  auto|aria2|curl)
    ;;
  *)
    echo "FLUXVLA_DOWNLOADER must be one of: auto, aria2, curl" >&2
    exit 1
    ;;
esac

if [[ "${SKIP_EGL_SETUP}" == "1" ]]; then
  FLUXVLA_EGL_SETUP="never"
fi

case "${FLUXVLA_EGL_SETUP}" in
  auto|always|never)
    ;;
  *)
    echo "FLUXVLA_EGL_SETUP must be one of: auto, always, never" >&2
    exit 1
    ;;
esac

case "${FLUXVLA_ROBOCASA_INSTALL}" in
  auto|always|never)
    ;;
  *)
    echo "FLUXVLA_ROBOCASA_INSTALL must be one of: auto, always, never" >&2
    exit 1
    ;;
esac

case "${FLUXVLA_ROBOCASA_ASSETS}" in
  always|never)
    ;;
  *)
    echo "FLUXVLA_ROBOCASA_ASSETS must be one of: always, never" >&2
    exit 1
    ;;
esac

if [[ "${FLUXVLA_ROBOCASA_INSTALL}" == "never" ]]; then
  FLUXVLA_ROBOCASA_ASSETS="never"
fi

detect_gpu_names() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
  fi
}

detect_compute_caps() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | grep -E '^[0-9]+([.][0-9]+)?$' || true
  fi
}

detect_cuda_versions() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>/dev/null \
      | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' \
      | head -n 1 || true
  fi
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>/dev/null \
      | sed -n 's/.*release[[:space:]]*\([0-9][0-9.]*\),.*/\1/p' \
      | head -n 1 || true
  fi
  if [[ -f /usr/local/cuda/version.txt ]]; then
    sed -n 's/.*CUDA Version[[:space:]]*\([0-9][0-9.]*\).*/\1/p' \
      /usr/local/cuda/version.txt | head -n 1 || true
  fi
}

detect_driver_cuda_version() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>/dev/null \
      | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' \
      | head -n 1 || true
  fi
}

detect_nvcc_cuda_version() {
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>/dev/null \
      | sed -n 's/.*release[[:space:]]*\([0-9][0-9.]*\),.*/\1/p' \
      | head -n 1 || true
  fi
}

detect_toolkit_cuda_versions() {
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>/dev/null \
      | sed -n 's/.*release[[:space:]]*\([0-9][0-9.]*\),.*/\1/p' \
      | head -n 1 || true
  fi
  if [[ -f /usr/local/cuda/version.txt ]]; then
    sed -n 's/.*CUDA Version[[:space:]]*\([0-9][0-9.]*\).*/\1/p' \
      /usr/local/cuda/version.txt | head -n 1 || true
  fi
}

cuda_version_ge() {
  local version="$1"
  local minimum="$2"
  awk -v version="${version}" -v minimum="${minimum}" '
    BEGIN {
      split(version, v, ".")
      split(minimum, m, ".")
      major_v = v[1] + 0
      minor_v = v[2] + 0
      major_m = m[1] + 0
      minor_m = m[2] + 0
      exit !((major_v > major_m) || (major_v == major_m && minor_v >= minor_m))
    }'
}

has_cuda_at_least() {
  local minimum="$1"
  local version
  while IFS= read -r version; do
    if [[ -n "${version}" ]] && cuda_version_ge "${version}" "${minimum}"; then
      return 0
    fi
  done < <(detect_cuda_versions)
  return 1
}

has_toolkit_cuda_at_least() {
  local minimum="$1"
  local version
  while IFS= read -r version; do
    if [[ -n "${version}" ]] && cuda_version_ge "${version}" "${minimum}"; then
      return 0
    fi
  done < <(detect_toolkit_cuda_versions)
  return 1
}

profile_cuda_version() {
  local selected="$1"
  if [[ "${selected}" == "cu128" ]]; then
    echo "12.8"
  else
    echo "12.4"
  fi
}

profile_torch_version() {
  local selected="$1"
  if [[ "${selected}" == "cu128" ]]; then
    echo "2.8"
  else
    echo "2.6"
  fi
}

check_cuda_profile_compatibility() {
  local selected="$1"
  local expected_cuda driver_cuda toolkit_versions
  expected_cuda="$(profile_cuda_version "${selected}")"
  driver_cuda="$(detect_driver_cuda_version)"
  toolkit_versions="$(detect_toolkit_cuda_versions | sort -Vu | tr '\n' ' ')"

  echo "CUDA compatibility check:"
  echo "  selected torch profile: ${selected} (expects CUDA ${expected_cuda} runtime)"
  echo "  driver CUDA support: ${driver_cuda:-unknown}"
  echo "  local CUDA toolkit(s): ${toolkit_versions:-unknown}"

  if [[ -n "${driver_cuda}" ]] && ! cuda_version_ge "${driver_cuda}" "${expected_cuda}"; then
    echo "Error: selected profile ${selected} requires NVIDIA driver support for CUDA ${expected_cuda}+," >&2
    echo "       but nvidia-smi reports CUDA ${driver_cuda}." >&2
    echo "Fix one of these:" >&2
    echo "  1. Use a compatible wheel profile: bash scripts/install_env.sh ${ENV_MODE} --profile cu124" >&2
    echo "  2. Upgrade the NVIDIA driver, then rerun with --profile ${selected}" >&2
    echo "  3. If the wrong torch wheel is already installed, run:" >&2
    echo "       python -m pip uninstall -y torch torchvision torchaudio" >&2
    exit 1
  fi

  if [[ -z "${driver_cuda}" ]]; then
    echo "Warning: nvidia-smi was not found or did not report CUDA support." >&2
    echo "         PyTorch may install, but GPU availability cannot be verified here." >&2
  fi

  if [[ -n "${toolkit_versions}" ]] && ! has_toolkit_cuda_at_least "${expected_cuda}"; then
    echo "Warning: local CUDA toolkit appears older than CUDA ${expected_cuda}." >&2
    echo "         PyTorch wheels bundle their CUDA runtime, so this can still work." >&2
    echo "         If compiling CUDA extensions later fails, install a matching toolkit" >&2
    echo "         or choose a matching --profile." >&2
  fi
}

needs_sim_runtime() {
  [[ "${ENV_MODE}" == "sim-only" || "${ENV_MODE}" == "full" ]]
}

needs_robocasa_sources() {
  case "${FLUXVLA_ROBOCASA_INSTALL}" in
    always)
      return 0
      ;;
    never)
      return 1
      ;;
    auto)
      needs_sim_runtime
      ;;
  esac
}

ensure_git_available() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ command -v git"
    return 0
  fi

  if command -v git >/dev/null 2>&1; then
    return 0
  fi

  echo "Error: git is required to install RoboCasa source checkouts." >&2
  echo "       Install git or rerun with --skip-robocasa." >&2
  exit 1
}

git_checkout_repo() {
  local name="$1"
  local repo="$2"
  local ref="$3"
  local dir="$4"
  local parent
  parent="$(dirname "${dir}")"

  echo "Preparing ${name}: ${repo}@${ref}"
  echo "  checkout: ${dir}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ mkdir -p ${parent}"
    if [[ -d "${dir}/.git" ]]; then
      echo "+ git -C ${dir} fetch --tags origin"
    else
      echo "+ git clone ${repo} ${dir}"
      echo "+ git -C ${dir} fetch --tags origin"
    fi
    echo "+ git -C ${dir} checkout ${ref}"
    return 0
  fi

  mkdir -p "${parent}"
  if [[ -d "${dir}/.git" ]]; then
    :
  elif [[ -e "${dir}" ]]; then
    echo "Error: ${dir} exists but is not a git checkout." >&2
    echo "       Move it aside, set FLUXVLA_ROBOCASA_SRC_ROOT, or rerun with --skip-robocasa." >&2
    exit 1
  else
    run git clone "${repo}" "${dir}"
  fi

  run git -C "${dir}" fetch --tags origin
  run git -C "${dir}" checkout "${ref}"
}

run_privileged() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${EUID}" == "0" ]]; then
      echo "+ $*"
    else
      echo "+ sudo $*"
    fi
    return 0
  fi

  if [[ "${EUID}" == "0" ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    return 127
  fi
}

print_egl_manual_fix() {
  cat >&2 <<EOF
Fix LIBERO / MuJoCo EGL manually with:
  sudo apt-get update
  sudo apt-get install -y ${FLUXVLA_EGL_APT_PACKAGES}
  sudo mkdir -p $(dirname "${FLUXVLA_EGL_VENDOR_FILE}")
  printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' | sudo tee ${FLUXVLA_EGL_VENDOR_FILE}

Then launch eval with:
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export __EGL_VENDOR_LIBRARY_FILENAMES=${FLUXVLA_EGL_VENDOR_FILE}

If this is a Docker container, it must be started with NVIDIA graphics
capabilities, for example:
  docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
EOF
}

maybe_fail_egl_setup() {
  local message="$1"
  echo "Warning: ${message}" >&2
  print_egl_manual_fix
  if [[ "${FLUXVLA_EGL_SETUP}" == "always" ]]; then
    exit 1
  fi
}

install_egl_system_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    maybe_fail_egl_setup "apt-get is not available; cannot install EGL system packages automatically."
    return 1
  fi

  echo "Installing LIBERO / MuJoCo EGL system packages."
  if ! run_privileged apt-get update; then
    maybe_fail_egl_setup "failed to run apt-get update for EGL setup."
    return 1
  fi
  if ! run_privileged apt-get install -y ${FLUXVLA_EGL_APT_PACKAGES}; then
    maybe_fail_egl_setup "failed to install EGL system packages."
    return 1
  fi
}

write_nvidia_egl_vendor_file() {
  local vendor_dir
  vendor_dir="$(dirname "${FLUXVLA_EGL_VENDOR_FILE}")"

  if [[ -f "${FLUXVLA_EGL_VENDOR_FILE}" ]] \
      && grep -q 'libEGL_nvidia.so.0' "${FLUXVLA_EGL_VENDOR_FILE}" 2>/dev/null; then
    echo "NVIDIA EGL vendor file already exists: ${FLUXVLA_EGL_VENDOR_FILE}"
    return 0
  fi

  echo "Writing NVIDIA EGL vendor file: ${FLUXVLA_EGL_VENDOR_FILE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ mkdir -p ${vendor_dir}"
    echo "+ tee ${FLUXVLA_EGL_VENDOR_FILE} <<'EOF'"
    cat <<'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
EOF
    return 0
  fi

  if [[ "${EUID}" == "0" ]]; then
    mkdir -p "${vendor_dir}"
    cat > "${FLUXVLA_EGL_VENDOR_FILE}" <<'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
EOF
  elif command -v sudo >/dev/null 2>&1; then
    sudo mkdir -p "${vendor_dir}"
    cat <<'EOF' | sudo tee "${FLUXVLA_EGL_VENDOR_FILE}" >/dev/null
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
EOF
  else
    maybe_fail_egl_setup "sudo is not available; cannot write ${FLUXVLA_EGL_VENDOR_FILE}."
    return 1
  fi
}

has_nvidia_egl_library() {
  ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia\.so\.0' && return 0
  [[ -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]] && return 0
  [[ -e /usr/lib64/libEGL_nvidia.so.0 ]] && return 0
  [[ -e /usr/lib/aarch64-linux-gnu/libEGL_nvidia.so.0 ]] && return 0
  return 1
}

egl_device_count() {
  local vendor_file="${1:-${__EGL_VENDOR_LIBRARY_FILENAMES:-}}"

  if [[ "${vendor_file}" == "__unset__" ]]; then
    env -u __EGL_VENDOR_LIBRARY_FILENAMES \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${PYTHON_BIN}" - <<'PY'
import sys

try:
    from mujoco.egl import egl_ext as EGL
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

print(len(EGL.eglQueryDevicesEXT()))
PY
  elif [[ -n "${vendor_file}" ]]; then
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      __EGL_VENDOR_LIBRARY_FILENAMES="${vendor_file}" \
      "${PYTHON_BIN}" - <<'PY'
import sys

try:
    from mujoco.egl import egl_ext as EGL
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

print(len(EGL.eglQueryDevicesEXT()))
PY
  else
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${PYTHON_BIN}" - <<'PY'
import sys

try:
    from mujoco.egl import egl_ext as EGL
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

print(len(EGL.eglQueryDevicesEXT()))
PY
  fi
}

egl_probe_result() {
  local vendor_file="$1"
  local probe_output count

  if probe_output="$(egl_device_count "${vendor_file}" 2>&1)"; then
    count="$(printf '%s\n' "${probe_output}" | tail -n 1)"
    if [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]]; then
      printf '%s\t%s\n' "${vendor_file}" "${count}"
      return 0
    fi
  fi
  return 1
}

select_egl_vendor_file() {
  local result

  if [[ -n "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    if result="$(egl_probe_result "${__EGL_VENDOR_LIBRARY_FILENAMES}")"; then
      echo "${result}"
      return 0
    fi
  fi

  if [[ -f "${FLUXVLA_EGL_VENDOR_FILE}" ]]; then
    if result="$(egl_probe_result "${FLUXVLA_EGL_VENDOR_FILE}")"; then
      echo "${result}"
      return 0
    fi
  fi

  if result="$(egl_probe_result "__unset__")"; then
    echo "${result}"
    return 0
  fi

  return 1
}

write_conda_egl_hook() {
  local selected_vendor_file="${1:-}"
  if [[ -z "${CONDA_PREFIX:-}" || ! -d "${CONDA_PREFIX}" ]]; then
    echo "No active conda environment detected; add MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl to your eval launch env."
    return 0
  fi

  local hook_dir hook_file
  hook_dir="${CONDA_PREFIX}/etc/conda/activate.d"
  hook_file="${hook_dir}/fluxvla-libero-egl.sh"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ mkdir -p ${hook_dir}"
    echo "+ write ${hook_file}"
    return 0
  fi

  mkdir -p "${hook_dir}"
  cat > "${hook_file}" <<EOF
# Generated by FluxVLA install_env.sh for LIBERO / MuJoCo offscreen rendering.
export MUJOCO_GL="\${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="\${PYOPENGL_PLATFORM:-egl}"
EOF
  if [[ "${selected_vendor_file}" == "__unset__" ]]; then
    cat >> "${hook_file}" <<'EOF'
unset __EGL_VENDOR_LIBRARY_FILENAMES
EOF
  fi
  if [[ -n "${selected_vendor_file}" && "${selected_vendor_file}" != "__unset__" ]]; then
    cat >> "${hook_file}" <<EOF
if [ -f "${selected_vendor_file}" ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${selected_vendor_file}"
fi
EOF
  fi
  echo "Installed conda activation hook: ${hook_file}"
}

check_container_nvidia_capabilities() {
  local in_container=0 caps=""
  if [[ -f /.dockerenv || -f /run/.containerenv ]]; then
    in_container=1
  fi
  if [[ "${in_container}" != "1" ]]; then
    return 0
  fi

  if [[ -r /proc/1/environ ]]; then
    caps="$(tr '\0' '\n' < /proc/1/environ \
      | sed -n 's/^NVIDIA_DRIVER_CAPABILITIES=//p' | head -n 1)"
  fi
  caps="${caps:-${NVIDIA_DRIVER_CAPABILITIES:-}}"

  if [[ -z "${caps}" ]]; then
    echo "Warning: container NVIDIA_DRIVER_CAPABILITIES is not set; EGL may not see GPU devices." >&2
    return 0
  fi
  if [[ ",${caps}," != *",all,"* && ",${caps}," != *",graphics,"* ]]; then
    echo "Warning: container NVIDIA_DRIVER_CAPABILITIES=${caps}; LIBERO EGL needs graphics or all." >&2
  fi
}

configure_libero_egl_runtime() {
  if [[ "${FLUXVLA_EGL_SETUP}" == "never" ]]; then
    return
  fi
  if ! needs_sim_runtime; then
    return
  fi
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Skipping LIBERO EGL setup on non-Linux platform: $(uname -s)"
    return
  fi

  case "${MUJOCO_GL:-}" in
    osmesa|glfw|glx)
      echo "MUJOCO_GL=${MUJOCO_GL}; skipping EGL setup."
      return
      ;;
  esac

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ configure_libero_egl_runtime"
    echo "+ probe MuJoCo EGL devices"
    install_egl_system_packages || true
    write_nvidia_egl_vendor_file || true
    write_conda_egl_hook
    return
  fi

  if [[ "${FLUXVLA_EGL_SETUP}" == "auto" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Skipping LIBERO EGL setup because nvidia-smi is not available."
    return
  fi

  echo "Checking LIBERO / MuJoCo EGL runtime."
  check_container_nvidia_capabilities

  if has_nvidia_egl_library; then
    write_nvidia_egl_vendor_file || true
  fi

  local egl_selection selected_vendor_file count
  if egl_selection="$(select_egl_vendor_file)"; then
    selected_vendor_file="$(printf '%s' "${egl_selection}" | cut -f1)"
    count="$(printf '%s' "${egl_selection}" | cut -f2)"
    echo "MuJoCo EGL sees ${count} device(s)."
    write_conda_egl_hook "${selected_vendor_file}"
    return
  fi
  echo "MuJoCo EGL probe failed or saw no devices; applying system EGL setup."

  install_egl_system_packages || return 0
  write_nvidia_egl_vendor_file || return 0

  if egl_selection="$(select_egl_vendor_file)"; then
    selected_vendor_file="$(printf '%s' "${egl_selection}" | cut -f1)"
    count="$(printf '%s' "${egl_selection}" | cut -f2)"
    echo "MuJoCo EGL setup verified: ${count} device(s)."
    write_conda_egl_hook "${selected_vendor_file}"
    return
  fi

  maybe_fail_egl_setup "MuJoCo EGL still sees no devices after setup."
}

is_blackwell_gpu() {
  local names="$1"
  local caps="$2"

  if echo "${caps}" | awk -F. '{ if ($1 + 0 >= 12) found=1 } END { exit found ? 0 : 1 }'; then
    return 0
  fi

  echo "${names}" | grep -Eiq 'rtx[[:space:]]*(50|pro[[:space:]]*50)|geforce[[:space:]]*rtx[[:space:]]*50|blackwell|gb20|gb10'
}

profile_for_cuda_version() {
  local version="$1"

  if [[ -z "${version}" ]]; then
    return 1
  fi
  if cuda_version_ge "${version}" "12.8"; then
    echo "cu128"
  else
    echo "cu124"
  fi
}

resolve_profile() {
  if [[ "${PROFILE}" != "auto" ]]; then
    echo "${PROFILE}"
    return
  fi

  local nvcc_cuda toolkit_cuda driver_cuda names caps
  nvcc_cuda="$(detect_nvcc_cuda_version)"
  if [[ -n "${nvcc_cuda}" ]]; then
    profile_for_cuda_version "${nvcc_cuda}"
    return
  fi

  toolkit_cuda="$(detect_toolkit_cuda_versions | sort -Vu | tail -n 1)"
  if [[ -n "${toolkit_cuda}" ]]; then
    profile_for_cuda_version "${toolkit_cuda}"
    return
  fi

  driver_cuda="$(detect_driver_cuda_version)"
  if [[ -n "${driver_cuda}" ]]; then
    profile_for_cuda_version "${driver_cuda}"
    return
  fi

  names="$(detect_gpu_names)"
  caps="$(detect_compute_caps)"
  if is_blackwell_gpu "${names}" "${caps}"; then
    echo "cu128"
  else
    echo "cu124"
  fi
}

python_tag() {
  "${PYTHON_BIN}" - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
}

cxx11_abi() {
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || echo "FALSE"
import torch
print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())
PY
}

torch_major_minor() {
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import torch
version = torch.__version__.split('+', 1)[0]
parts = version.split('.')
if len(parts) >= 2:
    print(f'{parts[0]}.{parts[1]}')
PY
}

torch_cuda_major_tag() {
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import torch
cuda = torch.version.cuda or ''
major = cuda.split('.', 1)[0]
if major:
    print(f'cu{major}')
PY
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
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/conda-meta" ]]; then
    echo "${CONDA_PREFIX}"
  elif [[ -n "${prefix}" && -d "${prefix}/conda-meta" ]]; then
    echo "${prefix}"
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

ensure_pip() {
  if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    return
  fi

  local prefix="" conda_bin=""
  prefix="$(python_prefix)"
  conda_bin="$(find_conda_bin || true)"

  echo "pip is not installed for ${PYTHON_BIN}; bootstrapping pip."

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -n "${conda_bin}" && -n "${prefix}" && -d "${prefix}/conda-meta" ]]; then
      echo "+ ${conda_bin} install -y -p ${prefix} pip"
    else
      echo "+ ${PYTHON_BIN} -m ensurepip --upgrade"
    fi
    return
  fi

  if [[ -n "${conda_bin}" && -n "${prefix}" && -d "${prefix}/conda-meta" ]]; then
    if run_conda_with_timeout "${conda_bin}" install -y -p "${prefix}" pip; then
      :
    else
      echo "conda failed to install pip; trying ensurepip." >&2
    fi
  fi

  if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    if "${PYTHON_BIN}" -m ensurepip --version >/dev/null 2>&1; then
      run "${PYTHON_BIN}" -m ensurepip --upgrade
    else
      echo "Error: pip is unavailable for ${PYTHON_BIN}, and ensurepip is not available." >&2
      echo "Fix it manually, then rerun this script:" >&2
      if [[ -n "${prefix}" && -d "${prefix}/conda-meta" ]]; then
        echo "       conda install -y -p ${prefix} pip" >&2
      else
        echo "       ${PYTHON_BIN} -m ensurepip --upgrade" >&2
      fi
      exit 1
    fi
  fi

  if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    echo "Error: pip bootstrap finished, but ${PYTHON_BIN} -m pip still fails." >&2
    exit 1
  fi
}

torch_tag_for_profile() {
  local selected="$1"
  if [[ "${selected}" == "cu128" ]]; then
    echo "2.8"
  else
    echo "2.6"
  fi
}

cuda_tag_for_profile() {
  local selected="$1"
  case "${selected}" in
    cu124|cu128)
      echo "cu12"
      ;;
    *)
      echo "${selected%[0-9][0-9]}"
      ;;
  esac
}

platform_tag() {
  case "$(uname -m)" in
    x86_64|amd64)
      echo "linux_x86_64"
      ;;
    aarch64|arm64)
      echo "linux_aarch64"
      ;;
    *)
      echo "unsupported"
      ;;
  esac
}

need_cmake_install() {
  if ! command -v cmake >/dev/null 2>&1; then
    return 0
  fi
  local cmake_path major
  cmake_path="$(command -v cmake)"
  if head -n 1 "${cmake_path}" 2>/dev/null | grep -q "python"; then
    return 0
  fi
  major="$(cmake --version 2>/dev/null | head -n 1 \
    | awk '{print $3}' | cut -d. -f1)"
  if [[ -n "${major}" && "${major}" -ge 4 ]]; then
    return 0
  fi
  return 1
}

ensure_build_tools() {
  if [[ "${SKIP_BUILD_TOOLS}" == "1" ]]; then
    return
  fi

  local need_cmake=0 need_ninja=0
  if need_cmake_install; then
    need_cmake=1
  fi
  if ! command -v ninja >/dev/null 2>&1; then
    need_ninja=1
  fi

  if [[ "${need_cmake}" == "0" && "${need_ninja}" == "0" ]]; then
    return
  fi

  local pkgs=()
  if [[ "${need_cmake}" == "1" ]]; then
    pkgs+=("cmake<4")
  fi
  if [[ "${need_ninja}" == "1" ]]; then
    pkgs+=("ninja")
  fi

  local conda_prefix="" conda_bin=""
  conda_prefix="$(conda_env_prefix)"
  conda_bin="$(find_conda_bin || true)"
  if [[ -z "${conda_bin}" || -z "${conda_prefix}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_conda_with_timeout conda install -y -c conda-forge "${pkgs[@]}"
      return
    fi
    echo "Error: cmake<4 and ninja are required to build native deps" >&2
    echo "  Activate a conda environment and rerun, or install them yourself," >&2
    echo "  then pass --skip-build-tools." >&2
    exit 1
  fi

  echo "Installing build tools via conda-forge: ${pkgs[*]}"
  run_conda_with_timeout "${conda_bin}" install -y -p "${conda_prefix}" -c conda-forge "${pkgs[@]}"
}

probe_pip_index() {
  local index="$1"

  "${PYTHON_BIN}" - "${index}" "${PIP_INDEX_PROBE_TIMEOUT}" <<'PY'
import sys
import time
import urllib.request

url = sys.argv[1].rstrip("/") + "/"
timeout = float(sys.argv[2])
started = time.monotonic()

try:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FluxVLA install_env.sh"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        response.read(256)
    if status >= 400:
        raise RuntimeError(f"HTTP {status}")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"{time.monotonic() - started:.3f}")
PY
}

select_reachable_pip_indexes() {
  local indexes="$1"
  local index elapsed
  local -a scored=()

  for index in ${indexes}; do
    echo "  probing ${index}" >&2
    if elapsed="$(probe_pip_index "${index}" 2>/dev/null)"; then
      echo "    ok (${elapsed}s)" >&2
      scored+=("${elapsed} ${index}")
    else
      echo "    unavailable" >&2
    fi
  done

  if [[ "${#scored[@]}" == "0" ]]; then
    return 1
  fi

  printf '%s\n' "${scored[@]}" | sort -n | awk '{ print $2 }' | tr '\n' ' '
}

pip_has_index_config() {
  if [[ -n "${PIP_INDEX_URL:-}" || -n "${PIP_EXTRA_INDEX_URL:-}" || -n "${PIP_NO_INDEX:-}" ]]; then
    return 0
  fi

  local config
  config="$("${PYTHON_BIN}" -m pip config list 2>/dev/null || true)"
  grep -Eq '(^|[.])(index-url|extra-index-url|no-index)=' <<<"${config}"
}

resolve_pip_index_urls() {
  if [[ -n "${USER_PIP_INDEX_URLS}" ]]; then
    echo "Using user-provided PIP_INDEX_URLS; skipping automatic pip index probing." >&2
    echo "${USER_PIP_INDEX_URLS}"
    return
  fi

  if [[ "${PIP_INDEX_MODE}" == "config" ]]; then
    echo "PIP_INDEX_MODE=config; using pip config/default index." >&2
    echo "${PIP_CONFIG_SENTINEL}"
    return
  fi

  if [[ "${PIP_INDEX_MODE}" == "auto" ]] && pip_has_index_config; then
    echo "PIP_INDEX_MODE=auto; detected pip index config, so the installer will not override it." >&2
    echo "${PIP_CONFIG_SENTINEL}"
    return
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: skipping automatic pip index probing." >&2
    case "${PIP_INDEX_MODE}" in
      config)
        echo "${PIP_CONFIG_SENTINEL}"
        ;;
      official)
        echo "${DEFAULT_PIP_INDEX_URL}"
        ;;
      mirror)
        echo "${DEFAULT_PIP_MIRROR_URLS}"
        ;;
      auto|all)
        echo "${DEFAULT_PIP_INDEX_CANDIDATES}"
        ;;
    esac
    return
  fi

  case "${PIP_INDEX_MODE}" in
    official)
      echo "PIP_INDEX_MODE=official; using PyPI only." >&2
      echo "${DEFAULT_PIP_INDEX_URL}"
      ;;
    mirror)
      echo "PIP_INDEX_MODE=mirror; probing pip mirrors." >&2
      select_reachable_pip_indexes "${DEFAULT_PIP_MIRROR_URLS}" \
        || echo "${DEFAULT_PIP_MIRROR_URLS}"
      ;;
    all)
      echo "PIP_INDEX_MODE=all; using default pip candidate order." >&2
      echo "${DEFAULT_PIP_INDEX_CANDIDATES}"
      ;;
    auto)
      echo "PIP_INDEX_MODE=auto; probing PyPI and mirrors by response time." >&2
      local selected_indexes
      if selected_indexes="$(select_reachable_pip_indexes "${DEFAULT_PIP_INDEX_CANDIDATES}")"; then
        echo "${selected_indexes}"
      else
        echo "No pip index probe succeeded; falling back to default order." >&2
        echo "${DEFAULT_PIP_INDEX_CANDIDATES}"
      fi
      ;;
  esac
}

resolve_pip_fallback_index_urls() {
  if [[ "${PIP_INDEX_MODE}" == "config" || -n "${PIP_NO_INDEX:-}" ]]; then
    return 1
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${DEFAULT_PIP_INDEX_CANDIDATES}"
    return
  fi

  local selected_indexes
  echo "Retrying with probed pip index candidates." >&2
  if selected_indexes="$(select_reachable_pip_indexes "${DEFAULT_PIP_INDEX_CANDIDATES}")"; then
    echo "${selected_indexes}"
  else
    echo "No pip index probe succeeded; falling back to default candidate order." >&2
    echo "${DEFAULT_PIP_INDEX_CANDIDATES}"
  fi
}

pip_install_from_indexes() {
  local indexes="$1"
  shift

  if [[ -z "${indexes}" || "${indexes}" == "${PIP_CONFIG_SENTINEL}" ]]; then
    echo "Using pip config/default index."
    if run_with_timeout "${PYTHON_BIN}" -m pip install \
      --timeout "${PIP_NETWORK_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "$@"; then
      return 0
    fi

    local fallback_indexes
    if fallback_indexes="$(resolve_pip_fallback_index_urls)"; then
      pip_install_from_indexes "${fallback_indexes}" "$@"
      return
    fi
    return 1
  fi

  local index
  for index in ${indexes}; do
    echo "Using pip index: ${index}"
    if run_with_timeout "${PYTHON_BIN}" -m pip install \
        --timeout "${PIP_NETWORK_TIMEOUT}" \
        --retries "${PIP_RETRIES}" \
        --index-url "${index}" \
        "$@"; then
      return 0
    fi
    echo "pip install failed or timed out with ${index}; trying next index..." >&2
  done

  echo "pip install failed with all configured indexes." >&2
  return 1
}

pip_install_with_mirrors() {
  pip_install_from_indexes "${PIP_INDEX_URLS}" "$@"
}

pip_install_direct() {
  run_with_timeout "${PYTHON_BIN}" -m pip install \
    --timeout "${PIP_NETWORK_TIMEOUT}" \
    --retries "${PIP_RETRIES}" \
    "$@"
}

install_torch() {
  local selected="$1"
  local torch_indexes
  if [[ "${selected}" == "cu128" ]]; then
    torch_indexes="${TORCH_INDEX_URLS:-https://download.pytorch.org/whl/cu128}"
    pip_install_from_indexes "${torch_indexes}" \
      torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  else
    torch_indexes="${TORCH_INDEX_URLS:-https://download.pytorch.org/whl/cu124}"
    pip_install_from_indexes "${torch_indexes}" \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  fi
}

verify_torch_install() {
  local selected="$1"
  local expected_torch expected_cuda
  expected_torch="$(profile_torch_version "${selected}")"
  expected_cuda="$(profile_cuda_version "${selected}")"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ verify_torch_install ${selected}"
    return
  fi

  "${PYTHON_BIN}" - "${selected}" "${expected_torch}" "${expected_cuda}" <<'PY'
import sys

selected, expected_torch, expected_cuda = sys.argv[1:4]

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "Error: PyTorch import failed after installation.\n"
        f"       {type(exc).__name__}: {exc}\n"
        "Fix: rerun the installer, or clear the broken wheel first:\n"
        "       python -m pip uninstall -y torch torchvision torchaudio"
    ) from exc

torch_version = torch.__version__
torch_base = torch_version.split("+", 1)[0]
torch_major_minor = ".".join(torch_base.split(".")[:2])
torch_cuda = torch.version.cuda or "none"

print(
    "PyTorch installed:",
    f"torch={torch_version}",
    f"torch.version.cuda={torch_cuda}",
    f"cuda_available={torch.cuda.is_available()}",
)

errors = []
if torch_major_minor != expected_torch:
    errors.append(
        f"selected profile {selected} expects torch {expected_torch}.x, "
        f"but installed torch is {torch_version}"
    )
if not torch_cuda.startswith(expected_cuda):
    errors.append(
        f"selected profile {selected} expects a CUDA {expected_cuda} PyTorch wheel, "
        f"but installed torch.version.cuda is {torch_cuda}"
    )

if errors:
    raise SystemExit(
        "Error: installed PyTorch wheel does not match the selected CUDA profile.\n"
        + "\n".join(f"       - {err}" for err in errors)
        + "\nFix:\n"
        + "       python -m pip uninstall -y torch torchvision torchaudio\n"
        + f"       bash scripts/install_env.sh <mode> --profile {selected}\n"
        + "Or choose --profile cu124 if your driver only supports CUDA 12.4."
    )
PY
}

install_av() {
  if [[ "${SKIP_AV}" == "1" ]]; then
    return
  fi

  case "${FLUXVLA_AV_INSTALLER}" in
    pip)
      pip_install_with_mirrors --only-binary=:all: av==14.4.0
      ;;
    conda)
      install_av_with_conda
      ;;
    auto)
      if pip_install_with_mirrors --only-binary=:all: av==14.4.0; then
        return
      fi
      echo "pip av wheel install failed or timed out; falling back to conda." >&2
      install_av_with_conda
      ;;
  esac
}

install_av_with_conda() {
  local conda_prefix="" conda_bin=""
  conda_prefix="$(conda_env_prefix)"
  conda_bin="$(find_conda_bin || true)"
  if [[ -z "${conda_bin}" || -z "${conda_prefix}" ]]; then
    echo "Error: FLUXVLA_AV_INSTALLER=conda requires a conda environment." >&2
    echo "       Activate one, or set PYTHON to a conda-env python." >&2
    exit 1
  fi

  if "${conda_bin}" install --help 2>/dev/null | grep -q -- '--solver'; then
    if run_conda_with_timeout "${conda_bin}" install -y -p "${conda_prefix}" -c conda-forge --solver=libmamba av=14.4.0; then
      return
    fi
    echo "conda av install with libmamba failed or timed out; trying default solver." >&2
  fi

  run_conda_with_timeout "${conda_bin}" install -y -p "${conda_prefix}" -c conda-forge av=14.4.0
}

install_requirements() {
  pip_install_with_mirrors -r "${PROJECT_ROOT}/requirements-base.txt"

  case "${ENV_MODE}" in
    sim-only)
      pip_install_with_mirrors -r "${PROJECT_ROOT}/requirements-sim.txt"
      ;;
    real-only)
      pip_install_with_mirrors -r "${PROJECT_ROOT}/requirements-real.txt"
      check_ros_python_runtime
      ;;
    full)
      pip_install_with_mirrors -r "${PROJECT_ROOT}/requirements-sim.txt"
      pip_install_with_mirrors -r "${PROJECT_ROOT}/requirements-real.txt"
      check_ros_python_runtime
      ;;
  esac
}

download_robocasa_assets() {
  if [[ "${FLUXVLA_ROBOCASA_ASSETS}" != "always" ]]; then
    return
  fi

  local asset_downloader="${PROJECT_ROOT}/scripts/download_robocasa_assets.py"
  local asset_dir="${FLUXVLA_ROBOCASA_GR1_DIR}/robocasa/models/assets"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ ${PYTHON_BIN} ${asset_downloader} --assets-dir ${asset_dir} --cache-dir ${FLUXVLA_ROBOCASA_ASSET_CACHE} --endpoint ${FLUXVLA_ROBOCASA_ASSET_ENDPOINT}"
    return
  fi

  if [[ ! -f "${asset_downloader}" ]]; then
    echo "Error: FluxVLA RoboCasa asset downloader not found: ${asset_downloader}" >&2
    exit 1
  fi

  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
  then
    pip_install_with_mirrors "huggingface_hub>=0.23"
  fi

  echo "Downloading and normalizing RoboCasa tabletop simulator assets."
  "${PYTHON_BIN}" "${asset_downloader}" \
    --assets-dir "${asset_dir}" \
    --cache-dir "${FLUXVLA_ROBOCASA_ASSET_CACHE}" \
    --endpoint "${FLUXVLA_ROBOCASA_ASSET_ENDPOINT}"
}

install_robocasa_sources() {
  if ! needs_robocasa_sources; then
    return
  fi

  ensure_git_available
  echo "Installing RoboCasa source checkouts under: ${FLUXVLA_ROBOCASA_SRC_ROOT}"

  git_checkout_repo \
    "Isaac-GR00T" \
    "${FLUXVLA_GROOT_REPO}" \
    "${FLUXVLA_GROOT_REF}" \
    "${FLUXVLA_GROOT_DIR}"
  pip_install_direct --no-deps -e "${FLUXVLA_GROOT_DIR}"

  git_checkout_repo \
    "RoboCasa GR1 tasks" \
    "${FLUXVLA_ROBOCASA_GR1_REPO}" \
    "${FLUXVLA_ROBOCASA_GR1_REF}" \
    "${FLUXVLA_ROBOCASA_GR1_DIR}"
  pip_install_direct --no-deps -e "${FLUXVLA_ROBOCASA_GR1_DIR}"

  download_robocasa_assets
}

check_ros_python_runtime() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ check_ros_python_runtime"
    return
  fi

  "${PYTHON_BIN}" - <<'PY' || true
import importlib.util
import os

required = ("rospkg", "catkin_pkg", "netifaces")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(
        "Warning: missing ROS Python runtime package(s): "
        + ", ".join(missing)
    )
    print("         Re-run `bash scripts/install_env.sh real-only` or install them with pip.")

if importlib.util.find_spec("rospy") is None:
    setup = "/opt/ros/noetic/setup.bash"
    if os.path.exists(setup):
        print(
            "Warning: rospy is not visible in this Python environment. "
            f"Run `source {setup}` before launching real-robot inference."
        )
    else:
        print(
            "Warning: rospy is not installed or not visible. Install ROS Noetic "
            "and source its setup.bash before real-robot inference."
        )
PY
}

download_via_proxy() {
  local upstream="$1"
  local cache_dir="${FLUXVLA_WHEEL_CACHE:-${HOME}/.cache/fluxvla/wheels}"
  local fname="${upstream##*/}"
  local out="${cache_dir}/${fname}"

  mkdir -p "${cache_dir}"
  if [[ -s "${out}" ]]; then
    if cached_wheel_is_valid "${out}"; then
      echo "${out}"
      return 0
    fi
    echo "  removing incomplete cached wheel: ${out}" >&2
    rm -f "${out}"
  fi

  local urls=()
  local base
  if [[ -n "${FLASH_ATTN_WHEEL_BASE_URLS}" ]]; then
    for base in ${FLASH_ATTN_WHEEL_BASE_URLS}; do
      urls+=("${base%/}/${fname}")
    done
  fi

  if [[ "${upstream}" == *"github.com/"* ]]; then
    local proxies=()
    if [[ -v GH_PROXY ]]; then
      if [[ -n "${GH_PROXY}" ]]; then
        proxies+=("${GH_PROXY%/}")
        proxies+=("")
      else
        proxies+=("")
      fi
    else
      local proxy
      for proxy in ${DEFAULT_GH_PROXY_CANDIDATES}; do
        proxies+=("${proxy%/}")
      done
      proxies+=("")
    fi

    local p
    for p in "${proxies[@]}"; do
      if [[ -z "${p}" ]]; then
        urls+=("${upstream}")
      else
        urls+=("${p}/${upstream}")
      fi
    done
  else
    urls+=("${upstream}")
  fi

  local url
  for url in "${urls[@]}"; do
    echo "  fetching: ${url}" >&2
    if download_file_to_cache "${url}" "${out}"; then
      if ! downloaded_wheel_is_valid "${out}"; then
        echo "  downloaded file is not a valid wheel: ${out}" >&2
        rm -f "${out}" "${out}.aria2"
        continue
      fi
      echo "${out}"
      return 0
    fi
    rm -f "${out}" "${out}.aria2"
  done

  echo "${upstream}"
  return 1
}

download_file_to_cache() {
  local url="$1"
  local out="$2"
  local mode="${FLUXVLA_DOWNLOADER}"

  if [[ "${mode}" == "auto" && -n "$(command -v aria2c 2>/dev/null)" ]]; then
    if download_file_with_aria2 "${url}" "${out}"; then
      return 0
    fi
    echo "  aria2c failed; retrying the same URL with curl" >&2
    rm -f "${out}" "${out}.aria2"
  elif [[ "${mode}" == "aria2" ]]; then
    download_file_with_aria2 "${url}" "${out}"
    return $?
  fi

  download_file_with_curl "${url}" "${out}"
}

download_file_with_aria2() {
  local url="$1"
  local out="$2"
  local dir fname
  dir="$(dirname "${out}")"
  fname="$(basename "${out}")"

  if ! command -v aria2c >/dev/null 2>&1; then
    echo "  aria2c is not installed" >&2
    return 127
  fi

  aria2c \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --continue=true \
    --connect-timeout="${DOWNLOAD_CONNECT_TIMEOUT}" \
    --dir="${dir}" \
    --file-allocation=none \
    --lowest-speed-limit="${DOWNLOAD_SPEED_LIMIT}" \
    --max-connection-per-server="${DOWNLOAD_CONNECTIONS}" \
    --max-tries="${DOWNLOAD_RETRIES}" \
    --min-split-size=1M \
    --out="${fname}" \
    --retry-wait=2 \
    --split="${DOWNLOAD_CONNECTIONS}" \
    --summary-interval=0 \
    --timeout="${DOWNLOAD_SPEED_TIME}" \
    "${url}"
}

download_file_with_curl() {
  local url="$1"
  local out="$2"

  curl -L --fail --retry "${DOWNLOAD_RETRIES}" --retry-delay 2 \
    --connect-timeout "${DOWNLOAD_CONNECT_TIMEOUT}" \
    --speed-time "${DOWNLOAD_SPEED_TIME}" \
    --speed-limit "${DOWNLOAD_SPEED_LIMIT}" \
    -C - -o "${out}" "${url}"
}

cached_wheel_is_valid() {
  wheel_zip_is_readable "$1"
}

downloaded_wheel_is_valid() {
  wheel_zip_contents_are_valid "$1"
}

wheel_zip_is_readable() {
  local wheel="$1"
  [[ -s "${wheel}" ]] || return 1
  "${PYTHON_BIN}" - "${wheel}" >/dev/null 2>&1 <<'PY'
import sys
import zipfile

try:
    with zipfile.ZipFile(sys.argv[1]) as zf:
        raise SystemExit(0 if zf.infolist() else 1)
except Exception:
    raise SystemExit(1)
PY
}

wheel_zip_contents_are_valid() {
  local wheel="$1"
  [[ -s "${wheel}" ]] || return 1
  "${PYTHON_BIN}" - "${wheel}" >/dev/null 2>&1 <<'PY'
import sys
import zipfile

try:
    with zipfile.ZipFile(sys.argv[1]) as zf:
        raise SystemExit(0 if zf.testzip() is None else 1)
except Exception:
    raise SystemExit(1)
PY
}

find_local_flash_attn_wheel() {
  local fname="$1"
  local candidate dir

  if [[ -n "${FLASH_ATTN_WHEEL_FILE}" ]]; then
    if [[ -s "${FLASH_ATTN_WHEEL_FILE}" ]] \
        && cached_wheel_is_valid "${FLASH_ATTN_WHEEL_FILE}"; then
      echo "${FLASH_ATTN_WHEEL_FILE}"
      return 0
    fi
    echo "Configured FLASH_ATTN_WHEEL_FILE is not a valid wheel: ${FLASH_ATTN_WHEEL_FILE}" >&2
    return 1
  fi

  for dir in ${FLASH_ATTN_WHEEL_DIRS}; do
    candidate="${dir%/}/${fname}"
    if [[ -s "${candidate}" ]]; then
      if cached_wheel_is_valid "${candidate}"; then
        echo "${candidate}"
        return 0
      fi
      echo "  ignoring invalid local wheel: ${candidate}" >&2
    fi
  done

  return 1
}

install_flash_attn() {
  if [[ "${SKIP_FLASH_ATTN}" == "1" ]]; then
    return
  fi

  local selected="$1"
  local tag abi torch_tag cuda_tag detected_torch_tag detected_cuda_tag
  local torch_source cuda_source
  local platform wheel_url wheel_name wheel_src
  tag="$(python_tag)"
  abi="$(cxx11_abi)"
  detected_torch_tag="$(torch_major_minor)"
  detected_cuda_tag="$(torch_cuda_major_tag)"
  if [[ -n "${FLASH_ATTN_TORCH_TAG:-}" ]]; then
    torch_tag="${FLASH_ATTN_TORCH_TAG}"
    torch_source="override"
  elif [[ -n "${detected_torch_tag}" ]]; then
    torch_tag="${detected_torch_tag}"
    torch_source="detected from torch"
  else
    torch_tag="$(torch_tag_for_profile "${selected}")"
    torch_source="profile fallback"
  fi

  if [[ -n "${FLASH_ATTN_CUDA_TAG:-}" ]]; then
    cuda_tag="${FLASH_ATTN_CUDA_TAG}"
    cuda_source="override"
  elif [[ -n "${detected_cuda_tag}" ]]; then
    cuda_tag="${detected_cuda_tag}"
    cuda_source="detected from torch"
  else
    cuda_tag="$(cuda_tag_for_profile "${selected}")"
    cuda_source="profile fallback"
  fi
  platform="$(platform_tag)"

  if [[ "${platform}" == "unsupported" ]]; then
    echo "No bundled FlashAttention prebuilt wheel is configured for $(uname -m)." >&2
    echo "Set FLASH_ATTN_WHEEL_URL to a matching wheel or pass --skip-flash-attn." >&2
    exit 1
  fi

  wheel_url="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/${FLASH_ATTN_RELEASE_TAG}/flash_attn-${FLASH_ATTN_VERSION}+${cuda_tag}torch${torch_tag}cxx11abi${abi}-${tag}-${tag}-${platform}.whl}"
  wheel_name="${wheel_url##*/}"

  echo "FlashAttention wheel selection:"
  echo "  python tag: ${tag}"
  echo "  torch tag: torch${torch_tag} (${torch_source})"
  echo "  cuda tag: ${cuda_tag} (${cuda_source})"
  echo "  cxx11 ABI: ${abi}"
  echo "  platform: ${platform}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ flash_attn_installed_matches || find_local_flash_attn_wheel ${wheel_name} || download_via_proxy ${wheel_url}"
    echo "+ ${PYTHON_BIN} -m pip install --no-deps <cached_wheel>"
    echo "+ verify_flash_attn"
    return
  fi

  if flash_attn_installed_matches; then
    verify_flash_attn
    return
  fi

  if wheel_src="$(find_local_flash_attn_wheel "${wheel_name}")"; then
    echo "Using local FlashAttention wheel: ${wheel_src}"
  elif ! wheel_src="$(download_via_proxy "${wheel_url}")"; then
    echo "Failed to download a matching prebuilt FlashAttention wheel." >&2
    echo "Selected wheel URL: ${wheel_url}" >&2
    echo "Override with FLASH_ATTN_WHEEL_FILE, FLASH_ATTN_WHEEL_URL, FLASH_ATTN_WHEEL_BASE_URLS, or pass --skip-flash-attn." >&2
    exit 1
  fi
  pip_install_direct --no-deps "${wheel_src}"
  verify_flash_attn
}

flash_attn_installed_matches() {
  "${PYTHON_BIN}" - "${FLASH_ATTN_VERSION}" <<'PY'
import sys
from importlib import metadata

expected = sys.argv[1]
try:
    installed = metadata.version("flash-attn")
except metadata.PackageNotFoundError:
    raise SystemExit(1)

if installed != expected:
    raise SystemExit(1)

try:
    from flash_attn.flash_attn_interface import (  # noqa: F401
        flash_attn_func,
        flash_attn_varlen_func,
    )
except Exception:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

verify_flash_attn() {
  "${PYTHON_BIN}" - <<'PY'
from importlib import metadata

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

print(
    "FlashAttention installed:",
    metadata.version("flash-attn"),
    flash_attn_func.__name__,
    flash_attn_varlen_func.__name__,
)
PY
}

install_project() {
  if [[ "${SKIP_PROJECT}" == "1" ]]; then
    return
  fi
  pip_install_direct --no-build-isolation -e "${PROJECT_ROOT}"
}

verify_project_import() {
  if [[ "${SKIP_PROJECT}" == "1" || "${DRY_RUN}" == "1" ]]; then
    return
  fi

  "${PYTHON_BIN}" - <<'PY'
import fluxvla

print("FluxVLA installed:", fluxvla.__file__)
PY
}

print_wandb_guidance() {
  cat <<'EOF'

wandb setup:
  - FluxVLA installs wandb as a dependency but does not run `wandb login`.
  - For online logging, run `wandb login` and paste only the 40-character API key from https://wandb.ai/authorize.
  - Do not paste the authorize URL, shell commands, or any other text into the prompt.
  - To skip wandb entirely, run: export WANDB_MODE=disabled
EOF
}

main() {
  cd "${PROJECT_ROOT}"

  local selected names caps cuda_versions
  selected="$(resolve_profile)"
  names="$(detect_gpu_names)"
  caps="$(detect_compute_caps)"
  cuda_versions="$(detect_cuda_versions | sort -Vu | tr '\n' ' ')"
  PIP_INDEX_URLS="$(resolve_pip_index_urls)"

  echo "Environment mode: ${ENV_MODE}"
  echo "Detected GPUs: ${names:-unknown}"
  echo "Detected compute capabilities: ${caps:-unknown}"
  echo "Detected CUDA versions: ${cuda_versions:-unknown}"
  echo "Selected profile: ${selected}"
  if [[ "${PIP_INDEX_URLS}" == "${PIP_CONFIG_SENTINEL}" ]]; then
    echo "pip indexes: pip config/default"
  else
    echo "pip indexes: ${PIP_INDEX_URLS}"
  fi
  echo "pip command timeout: ${PIP_INSTALL_TIMEOUT}s"
  echo "pip network timeout: ${PIP_NETWORK_TIMEOUT}s"
  echo "pip index probe timeout: ${PIP_INDEX_PROBE_TIMEOUT}s"
  echo "conda command timeout: ${CONDA_INSTALL_TIMEOUT}s"
  echo "av installer: ${FLUXVLA_AV_INSTALLER}"
  echo "RoboCasa source install: ${FLUXVLA_ROBOCASA_INSTALL}"
  if needs_robocasa_sources; then
    echo "RoboCasa asset download: ${FLUXVLA_ROBOCASA_ASSETS}"
  else
    echo "RoboCasa asset download: never (RoboCasa source checkout skipped)"
  fi
  echo "RoboCasa source root: ${FLUXVLA_ROBOCASA_SRC_ROOT}"

  ensure_build_tools
  ensure_pip
  check_cuda_profile_compatibility "${selected}"
  install_torch "${selected}"
  verify_torch_install "${selected}"
  install_av
  install_requirements
  install_robocasa_sources
  configure_libero_egl_runtime
  install_flash_attn "${selected}"
  install_project
  verify_project_import
  print_wandb_guidance

  if ! command -v nvidia-smi >/dev/null 2>&1 \
      && ! command -v nvcc >/dev/null 2>&1 \
      && [[ ! -f /usr/local/cuda/version.txt ]]; then
    echo "Warning: no CUDA toolkit or nvidia-smi signal was found; auto detection uses cu124."
  fi
}

main
