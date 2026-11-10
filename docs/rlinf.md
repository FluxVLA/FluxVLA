# FluxVLA × RLinf

FluxVLA integrates RLinf as a synchronous PPO/GAE backend. Policies and
simulator adapters remain native FluxVLA components; RLinf provides Ray
placement, trajectories, PPO losses, FSDP2, checkpoints and weight sync.

## Installation

Use Python 3.10 and [RLinf](https://github.com/RLinf/RLinf) commit
`88988ad5999619953b18d10b15bea1d3a6e1cfce`. The RL installation profiles require
CUDA 12.8 and PyTorch 2.8; the installer rejects the incompatible `cu124`
profile before installing packages.

```bash
git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
export RLINF_ROOT=/path/to/RLinf
git -C "$RLINF_ROOT" checkout --detach 88988ad5999619953b18d10b15bea1d3a6e1cfce
python -m pip install --no-deps -e "$RLINF_ROOT"
bash scripts/install_env.sh fluxvla-all --profile cu128
# Backend and model dependencies without simulator installation:
# bash scripts/install_env.sh rlinf-only --profile cu128
```

Build FluxVLA's CUDA extensions for the installed PyTorch version. Simulator
execution additionally requires the corresponding simulator packages and
assets. RoboTwin uses the `RLinf_support` runtime interface; its Python
package and checkout-relative assets must be available to every environment
worker. `env.train.assets_path` is the asset root consumed as `ASSETS_PATH`
by that runtime.

## Architecture

```text
RLinf environment -> Flux observation adapter -> Flux RL policy
  -> action + transition log-probability + value -> trajectory/reward/GAE
  -> RLinf PPO/FSDP2 update -> weight sync -> rollout policy
```

Training uses RLinf's native `EmbodiedRunner` and `EmbodiedFSDPActor`;
standalone evaluation uses `EmbodiedEvalRunner`. The rollout adapter passes
the sampling mode explicitly and computes bootstrap values without sampling
another action. The RoboTwin environment adapter preserves complete worker
seed partitions, refreshes instructions on reset and selects the worker's
rendering device. Metrics and checkpoints use RLinf's native mechanisms.

`FluxPI05RLPolicy` and `FluxSmolVLARLPolicy` train their action expert,
action/time projections and value head while keeping the vision/language
prefix frozen. Training samples one Gaussian flow-SDE transition and stores
its FP32 latent chain; the actor recomputes that same transition without
resampling. These probabilities describe denoising transitions, not the
marginal density of the final physical action. Evaluation uses each model's
normal ODE path. PI0.5 supports LIBERO-10 and RoboTwin; SmolVLA supports
LIBERO-10. Unsupported combinations are rejected before workers launch.

Actor and rollout retain FP32 master weights, network computation may use
BF16, and latent/probability arithmetic stays FP32. `openpi_stem_fp32` enables
FP32 patch/position embeddings and installs OpenPI-compatible LayerNorms in
the vision encoder and output stage. These LayerNorms use the autocast dtype
while retaining the original parameters and checkpoint keys. The RoboTwin
FSDP2 configuration synchronizes each microbatch while retaining gradient
accumulation across the configured global batch.

## Configuration and commands

RL configuration combines `configs/rl/default.py`, one file from
`configs/rl/model/`, and one from `configs/rl/simulator/`.

```bash
# Resolve configuration without starting Ray.
python scripts/rl_train.py model=pi05 simulator=libero10 \
  actor.model.model_path=/path/to/sft --cfg job --resolve

python scripts/rl_train.py model=smolvla simulator=libero10 \
  actor.model.model_path=/path/to/smolvla_sft

python scripts/rl_train.py model=pi05 simulator=robotwin \
  actor.model.model_path=/path/to/robotwin_sft \
  env.train.assets_path=/path/to/robotwin_assets \
  env.train.task_config.task_name=beat_block_hammer \
  env.train.task_config.step_lim=400

python scripts/rl_eval.py model=pi05 simulator=robotwin \
  actor.model.model_path=/path/to/robotwin_sft \
  env.train.assets_path=/path/to/robotwin_assets \
  env.train.task_config.task_name=beat_block_hammer \
  env.train.task_config.step_lim=400 \
  env.eval.total_num_envs=8 env.eval.rollout_epoch=4 \
  runner.ckpt_path=/path/to/global_step_N
```

The shared RoboTwin configuration reuses RLinf's `robotwin_adjust_bottle`
environment template. Override `env.train.task_config.task_name` to select
another installed task; evaluation inherits the task and step limit. Supply
a matching SFT checkpoint with its tokenizer and normalization statistics.

RoboTwin defaults to RLinf's bundled training and evaluation seed files.
Override `env.train.seeds_path` and `env.eval.seeds_path` to use custom splits.
Each file maps task names to objects containing a `success_seeds` list. When
selecting checkpoints by validation performance, provide separate validation
seeds and reserve test seeds for final evaluation. Environment counts and
rollout rounds are controlled exclusively by configuration.

GPU placement, batch sizes, checkpoint paths and logger backends are Hydra
overrides. TensorBoard is the default logger; optional W&B logging follows
the normal W&B authentication and online/offline settings.

## Checkpoints

- `actor.model.model_path` identifies the initial SFT checkpoint.
- `runner.resume_dir` restores a complete RLinf training checkpoint.
- `runner.ckpt_path` selects weights for standalone evaluation; omit it to
  evaluate the initial SFT checkpoint.
- `runner.save_interval` and `runner.val_check_interval` control checkpoint
  saving and validation through RLinf's native runner.

### Export for standard FluxVLA evaluation

Export the consolidated actor checkpoint (not per-rank training shards):

```bash
python scripts/rl_export.py /path/to/global_step_N /path/to/policy.pt
# A direct actor/model_state_dict/full_weights.pt path is also accepted.
```

The export removes only `value_head.*`, preserving policy parameter names,
buffers and dtypes. It leaves the training checkpoint intact. Set the normal
FluxVLA config's `model.pretrained_name_or_path` to `policy.pt` and
`model.strict_mapping=True`, and explicitly set `model.name_mapping=None`
(or the equivalent field in the inference model config). Exports already
use native Flux names; do not retain the mapping for the upstream SFT
checkpoint. The shared loader keeps its legacy mapping semantics so existing
SFT experiments are unchanged. RL builders separately use `load_sft_weights`
to resolve native and upstream names and restore persistent buffers. Missing
weights and shape mismatches still fail strict native loading. An unexported RL actor includes the critic and is not a
strictly compatible checkpoint for a standard VLA model.

Use the model architecture, normalization statistics, tokenizer, action
horizon and inference precision/steps from the RL run. Exporting weights
does not transfer these settings into an evaluation config. For runners that
load a checkpoint separately, point that checkpoint option to the same
exported policy file.

For PI0.5 RoboTwin RL checkpoints, also set the following options in the
ordinary FluxVLA evaluation config **before constructing the model**:

```python
model = dict(
    # Keep the architecture/tokenizer/statistics from the training config.
    openpi_fp32_flow=True,
    openpi_rl_numerics=True,
    vision_backbone=dict(openpi_stem_fp32=True, openpi_rl_numerics=True),
)
```

Merge these fields into the existing model config; they are not a complete
model definition. The two `openpi_rl_numerics` flags preserve RL's RoPE,
AdaRMS, timestep embedding, vision LayerNorm and stem layout behavior.
They default to False for SFT and ordinary inference. Tensor-only exports
cannot encode these switches: successful strict weight loading alone does
not guarantee identical actions. Match explicit False overrides if the RL
run intentionally disabled either flag. Do not apply these RoboTwin settings
to unrelated PI0.5/SmolVLA checkpoints.

For the BF16 RL compute path with FP32 master weights, ordinary evaluation
also needs BF16 autocast, `enable_mixed_precision_training=True` and
`preserve_fp32_residuals=False` so Gemma receives the same BF16 inputs. For
FP32 RL compute, disable autocast and set
`enable_mixed_precision_training=False`. Match the runner's precision
settings as well as these model options.

The PI0.5 RL builder migrates older saved RL configs that lack the new flags:
it derives the model flag from `openpi_fp32_flow` and the vision flag from
`openpi_stem_fp32`, after applying `model_overrides`. This covers training
resume and `scripts/rl_eval.py`; explicit True/False flags always take
precedence. It does not change the defaults for existing SFT recipes.

## Conservative PI0.5 RoboTwin training

`tools/rl/robotwin_conservative/launch.py` packages the eight-GPU
`adjust_bottle` recipe used to avoid accumulating simulator memory across a
long training process. It uses 64 training environments, four rollout rounds,
actor/rollout microbatch 8, global batch 512, five PPO update epochs, policy
LR `5e-6` and critic LR `1e-4`. Both `env.train.enable_offload` and
`env.eval.enable_offload` are explicitly enabled; the top-level environment
offload setting alone does not enable them in the pinned RLinf revision.
Evaluation retains 128 fixed episodes.

Prepare the simulator dependencies and rendering environment as above, then:

```bash
export FLUXVLA_ROOT=/path/to/FluxVLA
export RLINF_ROOT=/path/to/RLinf
export ROBOTWIN_PATH=/path/to/RoboTwin
export PYTHONPATH="$FLUXVLA_ROOT:$RLINF_ROOT:$ROBOTWIN_PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline  # Set online after configuring W&B authentication.

python tools/rl/robotwin_conservative/launch.py \
  --run-dir /path/to/run \
  --assets-path /path/to/robotwin_assets \
  --checkpoint /path/to/robotwin_sft \
  --resolve
# Remove --resolve to start training, or resume the same run directory.

python tools/rl/robotwin_conservative/monitor.py \
  --run-dir /path/to/run --once
```

The first segment evaluates and saves at step 1; subsequent segments end at
steps 5, 10, 15, ..., 1000. After each successful segment, the supervisor
terminates remaining descendants of that training process and validates the
full weights and distributed checkpoint files. The next process restores
model, optimizer, scheduler and actor RNG through `runner.resume_dir`.
Environment seed cursors and rollout RNG restart, so the result is not
bitwise equivalent to uninterrupted training. Each training step samples 256
trajectories, versus 1024 in the official-sized configuration.

The launcher stops on worker communication failures or host RAM usage at
least 85% for three consecutive 15-second samples. It retains completed
checkpoints and never automatically deletes them. It requires 100 GiB free
before starting each segment. `psutil` is required by the supervisor;
TensorBoard is required by the optional monitor. The run directory contains
`status.json`, `checkpoints.json`, `commands/`, `logs/`, `memory.jsonl`, and
`results/`. The monitor writes `progress.json`; omit `--once` to refresh it
every minute. W&B records each segment as a separate run within one group
and defaults to offline mode; TensorBoard retains cumulative training steps.

Use a distinct run directory for a new experiment. To continue an existing
one, rerun the same command with the same SFT checkpoint and asset paths;
only checkpoints recorded after successful segment completion are resumed.
