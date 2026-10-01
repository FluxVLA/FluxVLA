# Extending the FluxVLA RLinf frontend

The integration separates backend orchestration, policy-family mathematics,
native-model execution, and environment conversion. Reuse is explicit: the
flow-SDE implementation is not a generic likelihood for every VLA.

## Components and boundaries

| Component                            | Shared responsibility                                         | Model-specific input                                                               |
| ------------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `rl/models/registry.py`              | Driver/worker declarations and frontend validation            | Native model type, lazy policy/builder paths, supported adapters, FSDP constraints |
| `rl/rlinf_registry.py`               | RLinf registration and declared-builder dispatch              | One `PolicySpec` entry                                                             |
| `rl/models/builder.py`               | Strict checkpoint loading and flow-policy construction        | Native config, horizon key, optional checkpoint resolver                           |
| `rl/benchmarks/registry.py`          | Environment adapter construction                              | Environment factory, native preprocessing config                                   |
| `rl/models/flow/policy.py`           | Initialization, freezing, critic, sampling, replay, bootstrap | Dimensions, module paths, prefix and velocity hooks                                |
| `rl/models/flow/sampler.py`          | FP32 flow-SDE transitions, Gaussian probability/entropy       | Denoising steps and noise level                                                    |
| `rl/train.py`, `rl/eval.py`, workers | Backend launch, mode propagation, synchronization             | Configuration, not per-model dispatch branches                                     |

PI0.5 and SmolVLA retain their native parameter names and model-specific
prefix/cache/velocity implementations. OpenPI checkpoint aliases live in
`models/pi05/checkpoint.py`, not the generic loader. Unrelated models do not inherit
these alias exceptions. Environment recipes remain limited to the tested
LIBERO-10 and RoboTwin pilot; declaring a new model does not unlock a new
environment or prove its compatibility.

## Naming and tensor contract

| Name                   | Meaning                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `model_action_horizon` | Full number of action timesteps predicted by the model                                              |
| `model_action_dim`     | Full normalized latent action width, including padding                                              |
| `action_chunk`         | Number of predicted timesteps executed in the environment                                           |
| `action_env_dim`       | Number of action coordinates accepted by the environment                                            |
| `critic_hidden_size`   | Feature width supplied to the value head                                                            |
| `rl_trainable_modules` | Native module paths to unfreeze (critic is added automatically)                                     |
| `rl_frozen_modules`    | Native paths explicitly frozen and kept in eval mode, including exceptions within trainable modules |

The native names `n_action_steps`, `chunk_size`, `max_action_dim`, and
`llm_expert` stay inside native/model-specific code. The shared sampler uses
the model-neutral properties above. Existing configuration fields and public
builder entrypoints are retained for compatibility. `action_dim` in the
RLinf config/configure method means environment width, not padded latent width.

Current flow-policy hooks are:

- `_prefix(obs) -> (hidden, valid_token_mask, cache)`, where hidden is
  `[B, L, critic_hidden_size]` and mask is `[B, L]`. The prefix is frozen;
  its cache is rebuilt during replay, never serialized into trajectories.
- `_velocity(obs, mask, cache, latent, timestep) -> velocity`, with latent
  and velocity `[B, model_action_horizon, model_action_dim]` and time `[B]`.
- Native `predict_action(**obs, noise=...)` supplies the ODE eval path.
- Optional `_configure_model_rl()` handles numerical alignment that cannot
  safely be shared with another architecture.

Adapters return a flat tensor dictionary: `images`, `img_masks`,
`lang_tokens`, `lang_masks`, `states`. All have batch dimension first.
`env_actions(model_actions, action_chunk, action_dim, *, env_obs)` performs
denormalization and embodiment conversion exactly once, with current-observation
context available for delta actions. Training adds FP32 `chains`
`[B, num_steps + 1, model_action_horizon, model_action_dim]` and integer
`denoise_inds` `[B, num_steps]` to replay inputs. These keys are stable
interfaces, not model names to rename during adaptation.

## Adding another flow-matching VLA

Place its implementation under `fluxvla/rl/models/<model>/`. Shared flow
mathematics stays in `models/flow/`; model code must not import a concrete
benchmark. Benchmark adapters live under `benchmarks/<benchmark>/adapter.py`,
with protocol validation and runtime extensions alongside them. Adapters must
not import another benchmark merely to reuse a utility.

`benchmarks/registry.py` selects adapter factories, validation, optional
environment/worker overrides and runner classes through lazy import paths.
Add new benchmark declarations there instead of adding benchmark-name branches
to the common train/eval/worker modules. Model compatibility constraints remain
in `models/registry.py`. New combinations require validation, not just a new
registration entry.

RoboTwin's recorder and protocol-aware runner belong to its benchmark package.
User launch and resource tools live under `scripts/rl/`; internal diagnostic
and historical experiment scripts are not shipped. Train/eval module names
and state-dict keys are unchanged, but old internal `bridge` imports and
flat config names are removed. See the main guide for current launch paths.

1. Add a thin policy inheriting `FlowPPOPolicyMixin`, its native Flux model,
   and RLinf `BasePolicy`. Implement the properties/hooks and declare module
   paths. Do not copy the shared sampler, PPO replay, critic or loader.
2. Add its immutable `PolicySpec` in `models/registry.py`, including safe FSDP
   layer/module targets. A source declaration is required in both driver and
   workers; driver-only mutation of a Python registry does not reach Ray.
   Native type and policy class are separate: the builder validates the
   former before substituting the latter. Imports are lazy.
3. Add a model config and recipe. Reuse an adapter only if cameras, tokenizer,
   state encoding, statistics and action semantics actually match. Otherwise
   add an adapter factory without embedding its transforms in the policy.
4. Add a checkpoint resolver only for documented model-family exceptions.
   Native names take precedence over mappings. The generic loader still
   enforces one candidate, matching shape, and complete effective parameters.
5. Require native-vs-adapter preprocessing/ODE parity, old-vs-replayed
   log-probability parity, real PPO finite gradients/freeze checks, trajectory
   alignment, and checkpoint restoration. Audit actual forward call paths
   before selecting FSDP wrapping; class-name similarity is not sufficient.
   Then run a real two-GPU update/restore probe and full-SFT environment gate.

No model-specific edits to train/eval/rollout dispatch are needed when adding
a compatible model within an existing environment recipe. FSDP targets,
native execution, checkpoint mappings and normalization are intentionally not
forced to match another model.

For an autoregressive, diffusion, or otherwise incompatible action policy,
provide a separate `PolicySpec.builder` and policy implementing RLinf's
predict/forward/value contract. Do not reuse flow-SDE Gaussian probabilities
merely to satisfy the interface. Current frontend validation still imposes
the tested synchronous PPO/GAE, FP32-master and other v1 constraints; new
algorithm support requires explicit implementation and verification.

## Extension regression tests

`test/test_rl/test_extensibility.py` exercises a third registration name
without modifying dispatch, lazy metadata imports, family-scoped checkpoint
aliases, and a toy flow model with unrelated module names and dimensions.
The toy test covers rollout/replay, native eval, gradients and freezing;
it is a contract test, not evidence of a third production model integration.
Run it together with all existing PI0.5/SmolVLA tests before publishing.

GPU and simulator validation must be repeated on the target environment;
passing interface tests is not evidence of improved benchmark performance.
