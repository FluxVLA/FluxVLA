# GR00T N1.7 LIBERO

This guide describes the open-source setup for training and evaluating native
GR00T N1.7 on LIBERO in FluxVLA.

## What Is Included

- Native GR00T N1.7 model assembly in FluxVLA.
- Native Qwen3-VL/Cosmos backbone loading with the FluxVLA
  `transformers==5.3.0` environment.
- Native GR00T N1.7 processor and collator for LeRobot v2.1 parquet data.
- Four LIBERO full finetune configs, each containing both training and eval:
  - `configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py`
  - `configs/gr00t/gr00t_n17_native_libero_goal_full_finetune.py`
  - `configs/gr00t/gr00t_n17_native_libero_object_full_finetune.py`
  - `configs/gr00t/gr00t_n17_native_libero_spatial_full_finetune.py`

The configs use repository-relative defaults and can be overridden by
environment variables. They do not assume any private machine-specific mount.

## Environment

Use a FluxVLA environment installed from this repository version:

```bash
conda create -n fluxvla-n17 python=3.10 -y
conda activate fluxvla-n17
bash scripts/install_env.sh sim-only
```

`requirements-base.txt` already pins `transformers==5.3.0`; the native N1.7
code includes a compatibility shim for the official N1.7 checkpoint behavior
that was validated with the older Qwen3-VL runtime. `requirements-sim.txt`
already includes LIBERO/MuJoCo simulation dependencies.

Native N1.7 training with `assembly_runtime='native'` does not import the
official GR00T Python package. Native LIBERO simulation evaluation currently
uses the official GR00T N1.7 LIBERO Gymnasium wrapper, so evaluation requires
`gr00t` to be importable. Use one of these options:

```bash
# If Isaac-GR00T N1.7 is installed as a package, no extra variable is needed.
python -c "import gr00t; print(gr00t.__file__)"

# If you use a local checkout instead, point FluxVLA at it.
export FLUXVLA_GROOT_N17_PATH=/path/to/Isaac-GR00T
```

For headless LIBERO/MuJoCo evaluation, keep the standard simulation variables:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
```

## Assets

The default paths expected by the configs are:

```text
checkpoints/GR00T-N1.7-3B
checkpoints/GR00T-N1.7-LIBERO/libero_10
checkpoints/GR00T-N1.7-LIBERO/libero_goal
checkpoints/GR00T-N1.7-LIBERO/libero_object
checkpoints/GR00T-N1.7-LIBERO/libero_spatial
checkpoints/nvidia/Cosmos-Reason2-2B
datasets/libero_10_no_noops_lerobotv2.1
datasets/libero_goal_no_noops_lerobotv2.1
datasets/libero_object_no_noops_lerobotv2.1
datasets/libero_spatial_no_noops_lerobotv2.1
```

You can override these in every config:

```bash
export LIBERO_DATA_ROOT=/path/to/libero_10_no_noops_lerobotv2.1
export N17_INIT_CKPT=/path/to/GR00T-N1.7-3B
export N17_PROCESSOR_META=/path/to/GR00T-N1.7-LIBERO/libero_10
export N17_BACKBONE_MODEL_PATH=/path/to/Cosmos-Reason2-2B
```

Public assets can be downloaded with `huggingface-cli`:

```bash
huggingface-cli download nvidia/GR00T-N1.7-3B \
  --local-dir ./checkpoints/GR00T-N1.7-3B

huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --local-dir ./checkpoints/nvidia/Cosmos-Reason2-2B

huggingface-cli download nvidia/GR00T-N1.7-LIBERO \
  --include "libero_10/*" \
  --local-dir ./checkpoints/GR00T-N1.7-LIBERO

huggingface-cli download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include "libero_10_no_noops_lerobotv2.1/*" \
  --local-dir ./datasets
```

Repeat the last two commands for `libero_goal`, `libero_object`, and
`libero_spatial`.

If your terminal or network is unstable, download one asset at a time. If a
repository is gated or your organization requires authenticated downloads, run:

```bash
huggingface-cli login
```

## Preflight Script

Use the helper script before starting training:

```bash
python tools/prepare_groot_n17_libero.py --all-suites
```

It checks Python packages, optional `gr00t` availability for eval, checkpoint
metadata files, and LeRobot parquet dataset structure. To download public
assets first:

```bash
python tools/prepare_groot_n17_libero.py --all-suites --download
```

Use `--dry-run` to print download commands without running them:

```bash
python tools/prepare_groot_n17_libero.py --suite libero_10 --download --dry-run
```

## Training

The official N1.7 LIBERO recipe uses 20k steps, learning rate `1e-4`, weight
decay `1e-5`, warmup ratio `0.05`, global batch size 640, and state dropout
`0.2`. The configs expose these through environment variables.

Single-node 8-GPU example on H200:

```bash
export N17_INIT_CKPT=./checkpoints/GR00T-N1.7-3B
export N17_PROCESSOR_META=./checkpoints/GR00T-N1.7-LIBERO/libero_10
export N17_BACKBONE_MODEL_PATH=./checkpoints/nvidia/Cosmos-Reason2-2B
export LIBERO_DATA_ROOT=./datasets/libero_10_no_noops_lerobotv2.1
export N17_PER_DEVICE_BATCH_SIZE=10
export N17_GRAD_ACCUM_STEPS=8
export N17_MAX_STEPS=20000
export N17_LR=1e-4
export N17_WARMUP_RATIO=0.05
export N17_WEIGHT_DECAY=1e-5
export N17_SHARDING_STRATEGY=full-shard
export N17_ENABLE_GRAD_CKPT=1
export N17_ACTIVE_TRACKERS="('jsonl',)"

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --work-dir work_dirs/groot_n17_native_libero_10_full
```

For another suite, change the config, `LIBERO_DATA_ROOT`,
`N17_PROCESSOR_META`, and `--work-dir`.

## Evaluation

Evaluate with the same suite config:

```bash
export FLUXVLA_GROOT_N17_PATH=/path/to/Isaac-GR00T
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python scripts/eval.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --ckpt-path work_dirs/groot_n17_native_libero_10_full/checkpoints/<checkpoint>.safetensors
```

The config defaults to 50 trials per task for FluxVLA-style LIBERO reporting.
To reduce runtime for smoke tests:

```bash
python scripts/eval.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --ckpt-path work_dirs/groot_n17_native_libero_10_full/checkpoints/<checkpoint>.safetensors \
  --cfg-options eval.num_trials_per_task=1 eval.task_ids=[0]
```

## From-Zero H200 Validation Checklist

1. Clone this FluxVLA branch.
2. Create a fresh conda environment and run `bash scripts/install_env.sh sim-only`.
3. Install or expose Isaac-GR00T N1.7 so `import gr00t` works for eval.
4. Run `python tools/prepare_groot_n17_libero.py --all-suites --download`.
5. Run `python tools/prepare_groot_n17_libero.py --all-suites`.
6. Run a 1-2 step training smoke with `N17_MAX_STEPS=2`.
7. Run a 1-task, 1-trial evaluation smoke with `eval.task_ids=[0]`.
8. Start the full 20k-step training job.

Large checkpoint and dataset downloads are the most common failure point. Keep
the preflight output; it prints the exact missing path and the equivalent
download command so interrupted downloads can be resumed or repeated.
