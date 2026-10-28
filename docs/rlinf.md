# FluxVLA × RLinf

FluxVLA integrates RLinf as a synchronous PPO/GAE backend. Policies and
simulator adapters remain native FluxVLA registry components; RLinf provides
Ray placement, trajectories, PPO losses, FSDP2, checkpoints and weight sync.

## RLinf compatibility

The integration targets [RLinf](https://github.com/RLinf/RLinf) commit
`88988ad5999619953b18d10b15bea1d3a6e1cfce`. Use this exact revision rather
than an unpinned `main` checkout. Configuration composition has been checked
against this revision; distributed PPO and simulator execution require the
full runtime and are not claimed as validated by that check.

## Environments

Create a Python 3.10 conda environment, point `RLINF_ROOT` at an RLinf source
checkout, and select one install profile:

```bash
git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
export RLINF_ROOT=/path/to/RLinf
git -C "$RLINF_ROOT" checkout --detach 88988ad5999619953b18d10b15bea1d3a6e1cfce
python -m pip install --no-deps -e "$RLINF_ROOT"
bash scripts/install_env.sh fluxvla-all
# or, for backend development and unit tests without simulators:
bash scripts/install_env.sh rlinf-only
```

`fluxvla-all` is the only profile that promises end-to-end LIBERO and RoboTwin
execution. RLinf is installed editable from `RLINF_ROOT`; no overlay venv or
machine-specific site-packages path is used.

## Architecture

```text
RLinf environment -> registered Flux adapter -> registered Flux RL policy
  -> action + transition log-prob + value -> trajectory/reward/GAE
  -> RLinf PPO/FSDP2 update -> weight sync -> rollout policy
```

`FluxPI05RLPolicy` and `FluxSmolVLARLPolicy` train their action expert,
action/time projections and value head while keeping the vision/language
prefix frozen. Training samples one Gaussian flow-SDE transition and stores
its FP32 latent chain; the actor recomputes that same transition without
resampling. Evaluation uses each model's normal ODE path. PI0.5 supports
LIBERO-10 and RoboTwin `adjust_bottle`; SmolVLA currently supports LIBERO-10
only.

OpenPI precision behavior lives in the PI0.5, SigLIP and Gemma implementations,
not in an RL-only monkey patch. Actor and rollout retain FP32 master weights,
network computation may use BF16, and latent/probability arithmetic stays FP32.
`openpi_stem_fp32` enables FP32 patch/position embeddings and installs
OpenPI-compatible LayerNorms in the vision encoder and output stage. These
LayerNorms compute in the autocast dtype (BF16 during RL) while retaining FP32
master parameters and checkpoint keys; FP32 stem computation alone does not
provide this normalization behavior.

## Configuration and commands

RL configuration is composed from one model and one simulator:

```text
configs/rl/default.py
configs/rl/model/pi05.py
configs/rl/model/smolvla.py
configs/rl/simulator/libero10.py
configs/rl/simulator/robotwin_adjust_bottle.py
```

```bash
# Resolve configuration without starting Ray.
python scripts/rl_train.py model=pi05 simulator=libero10 \
  actor.model.model_path=/path/to/sft --cfg job --resolve

# SmolVLA uses the same runner and LIBERO adapter.
python scripts/rl_train.py model=smolvla simulator=libero10 \
  actor.model.model_path=/path/to/smolvla_sft

# Training and standalone held-out evaluation.
python scripts/rl_train.py model=pi05 simulator=robotwin_adjust_bottle
python scripts/rl_eval.py model=pi05 simulator=robotwin_adjust_bottle \
  runner.ckpt_path=/path/to/global_step_N
```

GPU placement, batch sizes, checkpoint paths and experiment names are ordinary
Hydra overrides. To add a model, add one registered RL-capable VLA builder and
one model config. To add a simulator, add registered observation/action
processor classes and one simulator config; do not create product configs for
every model/simulator/train/eval combination.

SmolVLA + RoboTwin is rejected during configuration validation. Its LeRobot
normalization, tokenizer, absolute-action and image preprocessing contract must
be implemented and tested before that combination is enabled.

## Checkpoints and audit rules

- `actor.model.model_path` is always the initial SFT checkpoint.
- `runner.resume_dir` restores a complete RLinf training checkpoint.
- `runner.ckpt_path` selects weights for standalone evaluation.
- Training checkpoint selection uses validation seeds only; equal scores choose
  the earlier step. Test seeds never reselect a checkpoint.
- RoboTwin records worker, slot, seed, checkpoint step, success and return for
  every completed episode. Missing, duplicate or unexpected seeds make
  `scripts/rl_report.py` fail.
- The audited actor stops on rollout/actor probability drift or non-finite
  gradients rather than silently skipping an update.

## Recorded RoboTwin result

For `adjust_bottle`, one completed local run evaluated 150 fixed test seeds:

| Model                       | Success |   Rate | Difference from SFT |
| --------------------------- | ------: | -----: | ------------------: |
| SFT                         | 129/150 | 86.00% |                   — |
| validation-selected step 30 | 135/150 | 90.00% |            +4.00 pp |
| final step 200              | 136/150 | 90.67% |            +4.67 pp |

This is evidence for one task and one training seed on the local simulator. It
does not establish multi-seed, cross-task or external-benchmark improvement and
does not reproduce the separately published RLinf 96.09% figure.

## Tests

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest test/test_rl -q
```

The suite covers registry/config composition, both adapters, fixed-noise ODE
parity, FP32/BF16 probability recomputation, PPO gradients and frozen weights,
FSDP wrapping, seed coverage, resume semantics and validation-only selection.
