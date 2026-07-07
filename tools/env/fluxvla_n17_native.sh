# Source this before working on the GR00T N1.7 native branch.
# It keeps the conda editable install untouched, but makes the n17 checkout win
# for the current shell.

export FLUXVLA_ACCEL_STUDY_ROOT=/mnt/workspace/mnt/data/yiming/fluxvla-upstream-accel-study
export FLUXVLA_N17_NATIVE_ROOT=/mnt/data/cpfs/mnt/data/yiming/fluxvla-n17-native-dev

_fluxvla_prepend_pythonpath() {
  local target="$1"
  local cleaned=""
  local entry
  local old_ifs="$IFS"
  IFS=':'
  for entry in ${PYTHONPATH:-}; do
    if [ -z "$entry" ]; then
      continue
    fi
    if [ "$entry" = "$FLUXVLA_ACCEL_STUDY_ROOT" ] || [ "$entry" = "$FLUXVLA_N17_NATIVE_ROOT" ]; then
      continue
    fi
    if [ -z "$cleaned" ]; then
      cleaned="$entry"
    else
      cleaned="$cleaned:$entry"
    fi
  done
  IFS="$old_ifs"
  if [ -z "$cleaned" ]; then
    export PYTHONPATH="$target"
  else
    export PYTHONPATH="$target:$cleaned"
  fi
}

_fluxvla_prepend_pythonpath "$FLUXVLA_N17_NATIVE_ROOT"

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR=/tmp/numba_cache

/root/miniconda3/envs/fluxvla/bin/python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("fluxvla")
print("[fluxvla-env] task=n17-native")
print("[fluxvla-env] fluxvla =", spec.origin if spec else None)
PY

unset -f _fluxvla_prepend_pythonpath
