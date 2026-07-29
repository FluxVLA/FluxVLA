#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_REPO="fluxvla"

usage() {
    cat <<'EOF'
Usage:
  docker/build_docker.sh [TARGET] [VERSION]
  docker/build_docker.sh [VERSION]

Targets:
  all         Build the recommended layered Orin stack: base, wheel, fa, ros, ros-fa. Default.
  base        Build fluxvla:orin-base.
  wheel       Build the flash-attn SM87 wheel under /mnt/nvme/fluxvla-wheels.
  fa          Build fluxvla:orin-fa from fluxvla:orin-base and the staged wheel.
  ros         Build fluxvla:orin-ros from fluxvla:orin-base.
  ros-fa      Build fluxvla:orin-ros-fa from fluxvla:orin-fa and fluxvla:orin-ros.

Examples:
  docker/build_docker.sh
  docker/build_docker.sh all 1.0.0
  docker/build_docker.sh fa
  docker/build_docker.sh ros 1.0.0
  FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh all

Environment:
  FLUXVLA_USE_CN_MIRRORS=0|1
  FLUXVLA_UBUNTU_PORTS_MIRROR=URL
  FLUXVLA_PIP_INDEX_URL=URL
  FLUXVLA_PIP_TRUSTED_HOST=HOST
  FLUXVLA_FLASH_ATTN_MAX_JOBS=N
    FLUXVLA_EXT_MAX_JOBS=N
  FLUXVLA_WHEEL_DIR=/mnt/nvme/fluxvla-wheels
  FLUXVLA_BASE_IMAGE=fluxvla:orin-base
  FLUXVLA_FLASH_ATTN_WHEEL=flash_attn-*.whl
  FLUXVLA_BUILD_DRY_RUN=1

Notes:
  - VERSION defaults to dev. When provided, image build steps also tag fluxvla:<variant>-<VERSION>.
  - ROS build steps use CN mirrors by default unless FLUXVLA_USE_CN_MIRRORS is already set.
EOF
}

is_version_arg() {
    [[ "${1:-}" =~ ^([0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?|dev)$ ]]
}

TARGET="${1:-all}"
VERSION="${2:-dev}"
DRY_RUN="${FLUXVLA_BUILD_DRY_RUN:-0}"

case "${TARGET}" in
    -h|--help|help)
        usage
        exit 0
        ;;
esac

if is_version_arg "${TARGET}"; then
    VERSION="${TARGET}"
    TARGET="all"
fi

case "${TARGET}" in
    all|base|wheel|fa|ros|ros-fa)
        ;;
    *)
        echo "Error: unknown target '${TARGET}'" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

case "${VERSION}" in
    "")
        VERSION="dev"
        ;;
esac

case "${DRY_RUN}" in
    0|1)
        ;;
    *)
        echo "Error: FLUXVLA_BUILD_DRY_RUN must be 0 or 1" >&2
        exit 1
        ;;
esac

run_step() {
    echo
    echo "=========================================="
    echo "Build step: $*"
    echo "=========================================="
    if [ "${DRY_RUN}" = "1" ]; then
        printf 'Dry run:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

prepare_build_env() {
    cd "${REPO_ROOT}"

    USE_CN_MIRRORS="${FLUXVLA_USE_CN_MIRRORS:-0}"
    case "${USE_CN_MIRRORS}" in
        0|1)
            ;;
        *)
            echo "Error: FLUXVLA_USE_CN_MIRRORS must be 0 or 1" >&2
            exit 1
            ;;
    esac

    if [ "${USE_CN_MIRRORS}" = "1" ]; then
        UBUNTU_PORTS_MIRROR="${FLUXVLA_UBUNTU_PORTS_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports}"
        PIP_INDEX_URL="${FLUXVLA_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
        PIP_TRUSTED_HOST="${FLUXVLA_PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
    else
        UBUNTU_PORTS_MIRROR="${FLUXVLA_UBUNTU_PORTS_MIRROR:-}"
        PIP_INDEX_URL="${FLUXVLA_PIP_INDEX_URL:-}"
        PIP_TRUSTED_HOST="${FLUXVLA_PIP_TRUSTED_HOST:-}"
    fi

    if [ "${DRY_RUN}" != "1" ]; then
        if ! command -v docker >/dev/null 2>&1; then
            echo "Error: Docker not installed" >&2
            exit 1
        fi
        if ! docker ps >/dev/null 2>&1; then
            echo "Error: Docker permission denied" >&2
            echo "Run: sudo usermod -aG docker ${USER:-<user>}" >&2
            echo "Then: newgrp docker" >&2
            exit 1
        fi
    fi

    if git rev-parse --git-dir >/dev/null 2>&1; then
        GIT_SHA="$(git rev-parse --short HEAD)"
        if ! git diff --quiet HEAD 2>/dev/null; then
            GIT_SHA="${GIT_SHA}-dirty"
        fi
    else
        GIT_SHA="nogit"
    fi
    BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

print_header() {
    local title="${1}"
    local variant="${2}"

    echo "=========================================="
    echo "${title}"
    echo "  version   : ${VERSION}"
    echo "  image     : ${IMAGE_REPO}:${variant}"
    if [ -n "${FLASH_ATTN_MAX_JOBS:-}" ]; then
        echo "  FA jobs   : ${FLASH_ATTN_MAX_JOBS}"
    fi
    echo "  CN mirrors: ${USE_CN_MIRRORS}"
    if [ -n "${UBUNTU_PORTS_MIRROR}" ]; then
        echo "  apt mirror: ${UBUNTU_PORTS_MIRROR}"
    fi
    if [ -n "${PIP_INDEX_URL}" ]; then
        echo "  pip index : ${PIP_INDEX_URL}"
    fi
    echo "Git SHA   : ${GIT_SHA}"
    echo "Build date: ${BUILD_DATE}"
    echo "=========================================="
}

build_image() {
    local dockerfile="${1}"
    local variant="${2}"
    shift 2

    local floating_tag="${IMAGE_REPO}:${variant}"
    local tag_args=(-t "${floating_tag}")
    if [ "${VERSION}" != "dev" ]; then
        tag_args+=(-t "${IMAGE_REPO}:${variant}-${VERSION}")
    fi

    run_step docker build --progress=plain \
        -f "${dockerfile}" \
        --build-arg "VERSION=${VERSION}" \
        --build-arg "GIT_SHA=${GIT_SHA}" \
        --build-arg "BUILD_DATE=${BUILD_DATE}" \
        --build-arg "UBUNTU_PORTS_MIRROR=${UBUNTU_PORTS_MIRROR}" \
        --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" \
        --build-arg "PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}" \
        --build-arg "FLUXVLA_EXT_MAX_JOBS=${FLUXVLA_EXT_MAX_JOBS:-2}" \
        "${tag_args[@]}" \
        "$@" \
        "${REPO_ROOT}"

    echo
    echo "Build complete: ${floating_tag}"
}

stage_wheelhouse() {
    local wheel_dir="${1:-${FLUXVLA_WHEEL_DIR:-/mnt/nvme/fluxvla-wheels}}"
    local stage_dir="${REPO_ROOT}/docker/.wheelhouse"

    if [ "${DRY_RUN}" = "1" ]; then
        echo "Dry run: stage wheels from ${wheel_dir} to ${stage_dir}"
        return 0
    fi

    rm -rf "${stage_dir}"
    mkdir -p "${stage_dir}"
    if [ -d "${wheel_dir}" ]; then
        find "${wheel_dir}" -maxdepth 1 -type f -name '*.whl' -exec cp -f {} "${stage_dir}/" \;
    fi
}

cleanup_wheelhouse() {
    rm -rf "${REPO_ROOT}/docker/.wheelhouse"
}

with_default_ros_mirror() {
    if [ -z "${FLUXVLA_USE_CN_MIRRORS+x}" ]; then
        export FLUXVLA_USE_CN_MIRRORS=1
    fi
}

build_base() {
    prepare_build_env
    print_header "Building FluxVLA Orin Base" "orin-base"
    build_image "${SCRIPT_DIR}/Dockerfile.orin" "orin-base" \
        --target "base"
}

build_wheel() {
    prepare_build_env
    FLASH_ATTN_MAX_JOBS="${FLUXVLA_FLASH_ATTN_MAX_JOBS:-1}"
    BASE_IMAGE="${FLUXVLA_BASE_IMAGE:-fluxvla:orin-base}"
    WHEEL_DIR="${FLUXVLA_WHEEL_DIR:-/mnt/nvme/fluxvla-wheels}"

    print_header "Building FluxVLA Orin FlashAttention Wheel" "flash-attn-wheel"

    if [ "${DRY_RUN}" != "1" ]; then
        mkdir -p "${WHEEL_DIR}"
    fi

    run_step docker run --rm \
        --runtime=nvidia \
        --ipc=host \
        --network=host \
        --shm-size=16g \
        -e "MAX_JOBS=${FLASH_ATTN_MAX_JOBS}" \
        -v "${WHEEL_DIR}:/wheelhouse" \
        "${BASE_IMAGE}" \
        bash -lc '
set -euo pipefail
cd /tmp
wget -q --tries=5 --waitretry=5 https://codeload.github.com/Dao-AILab/flash-attention/tar.gz/refs/tags/v2.5.5 -O flash-attention.tar.gz
mkdir -p flash-attention
tar -xzf flash-attention.tar.gz -C flash-attention --strip-components=1
rm -rf flash-attention/csrc/cutlass
mkdir -p flash-attention/csrc/cutlass
wget -q --tries=5 --waitretry=5 https://codeload.github.com/NVIDIA/cutlass/tar.gz/bbe579a9e3beb6ea6626d9227ec32d0dae119a49 -O cutlass.tar.gz
tar -xzf cutlass.tar.gz -C flash-attention/csrc/cutlass --strip-components=1
python3 - <<"PY"
from pathlib import Path
setup_py = Path("/tmp/flash-attention/setup.py")
s = setup_py.read_text()
old = """    cc_flag.append(\"-gencode\")
    cc_flag.append(\"arch=compute_80,code=sm_80\")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version(\"11.8\"):
            cc_flag.append(\"-gencode\")
            cc_flag.append(\"arch=compute_90,code=sm_90\")
"""
new = """    cc_flag.append(\"-gencode\")
    cc_flag.append(\"arch=compute_87,code=sm_87\")
"""
if old not in s:
    raise RuntimeError("flash-attn 2.5.5 setup.py arch block not found")
setup_py.write_text(s.replace(old, new))
PY
cd /tmp/flash-attention
python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheelhouse .
'

    if [ "${DRY_RUN}" != "1" ]; then
        echo "Built flash-attn wheel(s) under ${WHEEL_DIR}:"
        ls -1 "${WHEEL_DIR}"/*.whl
    fi
}

build_fa() {
    prepare_build_env
    FLASH_ATTN_MAX_JOBS="${FLUXVLA_FLASH_ATTN_MAX_JOBS:-1}"
    FLASH_ATTN_WHEEL="${FLUXVLA_FLASH_ATTN_WHEEL:-}"

    stage_wheelhouse
    trap cleanup_wheelhouse EXIT

    if [ -z "${FLASH_ATTN_WHEEL}" ] && [ -d "${REPO_ROOT}/docker/.wheelhouse" ]; then
        first_wheel="$(find "${REPO_ROOT}/docker/.wheelhouse" -maxdepth 1 -type f -name 'flash_attn-*.whl' | head -n 1 || true)"
        if [ -n "${first_wheel}" ]; then
            FLASH_ATTN_WHEEL="$(basename "${first_wheel}")"
        fi
    fi

    print_header "Building FluxVLA Orin FlashAttention" "orin-fa"
    build_image "${SCRIPT_DIR}/Dockerfile.orin" "orin-fa" \
        --target "fa" \
        --build-arg "FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS}" \
        --build-arg "FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL}"
}

build_ros() {
    with_default_ros_mirror
    prepare_build_env

    print_header "Building FluxVLA Orin ROS" "orin-ros"
    build_image "${SCRIPT_DIR}/Dockerfile.orin" "orin-ros" \
        --target "ros"
}

build_ros_fa() {
    prepare_build_env
    FLASH_ATTN_MAX_JOBS="${FLUXVLA_FLASH_ATTN_MAX_JOBS:-1}"
    FLASH_ATTN_WHEEL="${FLUXVLA_FLASH_ATTN_WHEEL:-}"

    stage_wheelhouse
    trap cleanup_wheelhouse EXIT

    if [ -z "${FLASH_ATTN_WHEEL}" ] && [ -d "${REPO_ROOT}/docker/.wheelhouse" ]; then
        first_wheel="$(find "${REPO_ROOT}/docker/.wheelhouse" -maxdepth 1 -type f -name 'flash_attn-*.whl' | head -n 1 || true)"
        if [ -n "${first_wheel}" ]; then
            FLASH_ATTN_WHEEL="$(basename "${first_wheel}")"
        fi
    fi

    print_header "Building FluxVLA Orin ROS + FlashAttention" "orin-ros-fa"
    build_image "${SCRIPT_DIR}/Dockerfile.orin" "orin-ros-fa" \
        --target "ros-fa" \
        --build-arg "FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS}" \
        --build-arg "FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL}"
}

cd "${REPO_ROOT}"

case "${TARGET}" in
    all)
        build_base
        build_wheel
        build_fa
        build_ros
        build_ros_fa
        ;;
    base)
        build_base
        ;;
    wheel)
        build_wheel
        ;;
    fa)
        build_fa
        ;;
    ros)
        build_ros
        ;;
    ros-fa)
        build_ros_fa
        ;;
esac

echo
echo "Requested Docker build target complete: ${TARGET} (${VERSION})"
