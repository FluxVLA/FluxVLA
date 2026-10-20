#!/usr/bin/env bash
# Source before launching Flux/RLinf workers; credentials are inherited only.
export FLUXVLA_ROOT=${FLUXVLA_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
export FLUX_ROBOTWIN_ROOT=${FLUX_ROBOTWIN_ROOT:-$FLUXVLA_ROOT/work_dirs/robotwin}
export RLINF_ROOT=${RLINF_ROOT:-$(dirname -- "$FLUXVLA_ROOT")/RLinf}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-$FLUX_ROBOTWIN_ROOT/src/RoboTwin}
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$FLUXVLA_ROOT:$RLINF_ROOT:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$FLUXVLA_ROOT/configs/rl/runtime/robotwin/nvidia_icd.json}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp}
export TMPDIR="$FLUX_ROBOTWIN_ROOT/cache/tmp"
# Unix-domain socket paths cannot use the long CPFS prefix. Ray's transient
# IPC/log directory stays short; model/trajectory/checkpoint artifacts use CPFS.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/flux-robotwin-ray}
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=17179869184
export TORCH_EXTENSIONS_DIR="$FLUX_ROBOTWIN_ROOT/cache/torch_extensions"
export HF_HOME="$FLUX_ROBOTWIN_ROOT/cache/huggingface"
export WANDB_PROJECT=fluxvla-rl-robotwin
export WANDB_MODE=online
export WANDB_DIR="$FLUX_ROBOTWIN_ROOT/results/wandb"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export JAX_PLATFORMS=cpu
export TORCHDYNAMO_DISABLE=1
export RLINF_EXT_MODULE=fluxvla.rl.rlinf_registry
export FLUX_RL_EXTERNAL_DISTRIBUTED=1
# SAPIEN ships OIDN 2.0.1, which predates Blackwell support. This is process
# local: never replace system libraries libraries in another environment.
FLUX_OIDN_LIB="$FLUX_ROBOTWIN_ROOT/src/oidn-2.3.3.x86_64.linux/lib"
if [[ -f "$FLUX_OIDN_LIB/libOpenImageDenoise.so.2.3.3" ]]; then
    export LD_LIBRARY_PATH="$FLUX_OIDN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_PRELOAD="$FLUX_OIDN_LIB/libOpenImageDenoise.so.2.3.3${LD_PRELOAD:+:$LD_PRELOAD}"
fi
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_DIR"
