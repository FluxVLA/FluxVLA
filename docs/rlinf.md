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
