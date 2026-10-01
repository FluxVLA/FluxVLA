#!/usr/bin/env bash
# Shared user entrypoint. Activate the FluxVLA RL environment before use.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash scripts/rl/run.sh train|eval CONFIG [Hydra overrides...]' \
        'Example: bash scripts/rl/run.sh train benchmarks/libero/smolvla/ppo_8gpu' \
        'Required: RLINF_ROOT points to an RLinf checkout.' \
        'Optional: FLUX_RL_MODEL_PATH, FLUX_RL_TOKENIZER_PATH, FLUX_RL_STATS_PATH,' \
        '          FLUX_RL_RESULTS_ROOT, FLUX_RL_RUN_NAME, FLUX_RL_PYTHON.' \
        'Run long jobs in tmux. Add --cfg job --resolve to inspect a config.'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 || ( $1 != train && $1 != eval ) ]]; then
    usage >&2
    exit 2
fi
rl_mode=$1
rl_config=$2
shift 2
rl_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $rl_config != benchmarks/* || $rl_config == *..* ||
      ! -f "$rl_root/configs/rl/$rl_config.yaml" ]]; then
    printf 'Unknown benchmark config: %s\n' "$rl_config" >&2
    exit 2
fi
: "${RLINF_ROOT:?Set RLINF_ROOT to the RLinf source checkout}"
if [[ ! -f "$RLINF_ROOT/rlinf/__init__.py" ]]; then
    printf 'RLINF_ROOT is not an RLinf checkout: %s\n' "$RLINF_ROOT" >&2
    exit 2
fi
RLINF_ROOT=$(cd -- "$RLINF_ROOT" && pwd)
export RLINF_ROOT
rl_inspect=false
for rl_arg in "$@"; do
    case "$rl_arg" in
        --cfg|--cfg=*|--help|-h|--info|--info=*) rl_inspect=true ;;
    esac
done
if [[ $rl_inspect == false && -z ${TMUX:-} ]]; then
    printf 'Start a tmux session first: tmux new -s fluxvla-rl\n' >&2
    exit 2
fi

export FLUXVLA_ROOT="$rl_root"
export PYTHONPATH="$rl_root:$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 FLUX_RL_EXTERNAL_DISTRIBUTED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
if [[ $rl_config == benchmarks/robotwin/* ]]; then
    source "$rl_root/scripts/rl/robotwin_env.sh"
fi

# Hydra needs quotes inside the argument for paths containing spaces or '='.
# Escape backslashes and quotes as data; never evaluate user-provided strings.
rl_overrides=()
add_path() {
    local rl_key=$1 rl_value=$2
    rl_value=${rl_value//\\/\\\\}
    rl_value=${rl_value//\"/\\\"}
    rl_overrides+=("$rl_key=\"$rl_value\"")
}
[[ -z ${FLUX_RL_MODEL_PATH:-} ]] || add_path actor.model.model_path "$FLUX_RL_MODEL_PATH"
[[ -z ${FLUX_RL_TOKENIZER_PATH:-} ]] || add_path actor.model.fluxvla.tokenizer_path "$FLUX_RL_TOKENIZER_PATH"
[[ -z ${FLUX_RL_STATS_PATH:-} ]] || add_path actor.model.fluxvla.norm_stats_path "$FLUX_RL_STATS_PATH"
[[ -z ${FLUX_RL_RESULTS_ROOT:-} ]] || add_path runner.logger.log_path "$FLUX_RL_RESULTS_ROOT"
rl_name=${FLUX_RL_RUN_NAME:-${rl_config//\//_}_${rl_mode}_$(date +%Y%m%d_%H%M%S)}
add_path runner.logger.experiment_name "$rl_name"
if [[ $rl_mode == eval ]]; then
    rl_overrides+=(runner.only_eval=true 'rollout.model=${actor.model}')
else
    rl_overrides+=(runner.only_eval=false)
fi

cd "$rl_root"
exec "${FLUX_RL_PYTHON:-python}" -u -m "fluxvla.rl.$rl_mode" \
    "--config-name=$rl_config" "${rl_overrides[@]}" "$@"
