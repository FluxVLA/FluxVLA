# Changelog

Notable user-facing changes to FluxVLA are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Historical entries are reconstructed from repository tags and commit history.
Release publication dates have not been verified and are intentionally omitted;
the existing lightweight tags do not record publication dates, and commit dates
must not be substituted for them. Add verified dates in `YYYY-MM-DD` format when
release metadata is available. Unreleased entries track changes on `main` after
the latest tag, together with this changelog's documentation additions.

## [Unreleased]

### Added

- FluxVLA technical report.
- Deterministic CPU unit tests for language and vision backbones, VLA policies,
  action heads, projectors, and KV caches.
- CPU CI runs across three seeds with coverage and no-skip checks, pytest logs,
  and JUnit reports; full CUDA tests are gated on configured GPU runners.
- FluxVLA-specific GitHub issue forms for bug reports and feature requests,
  contribution and validation guidance, a code of conduct with private reporting
  contacts, and a versioned project changelog.

### Changed

- Rename the exported BF16 capability helper from `check_bloat16_supported` to
  `check_bf16_supported`, updating utility exports and training runner calls
  without changing hardware detection behavior. External callers must update
  imports of the old name.

## [0.1.5]

### Added

- Native GR00T N1.7 training and evaluation support for LIBERO, RoboCasa, and
  Franka, including Qwen3-VL integration and checkpoint loading.
- Cosmos3 Nano, Super, and Edge model integration with LIBERO and ALOHA recipes,
  sequence preprocessing, checkpoint conversion, and regression tests.
- FastWAM world-action models with unconditional, IDM, and Joint variants,
  LIBERO recipes, and full-data RoboCasa training and evaluation.
- Native DiT4DiT support with a Cosmos2.5 backbone, action head, LIBERO and
  RoboCasa recipes, and source-to-FluxVLA parity tests.
- RoboCasa training recipes for PI0.5, Cosmos3 Nano, and SmolVLA, with benchmark
  results and checkpoint references.
- Checkpoint-free, API-backed GPT-6 evaluation for LIBERO and RoboCasa, including
  multi-view inputs, native action control, and organized rollout artifacts.
- Jetson Orin Triton-accelerated GR00T N1.5 and PI0.5 inference, a Docker runtime,
  and deployment guides.
- FluxThemis ROS 1/2 evaluation serving with multi-GPU workers and result
  reporting.

### Changed

- Align PI0.5 training and evaluation with OpenPI/JAX preprocessing, prompts,
  flow matching, optimizer settings, EMA, and normalization statistics.
- Support transformed dataset statistics, balanced multi-source sampling,
  deterministic dataset sharding, and aligned terminal padding.
- Rename `DeltaActions` to `RelativeActions` while preserving the old name for
  compatibility, and load PaliGemma tokenizers from local checkpoints.
- Separate GR00T N1.5 and N1.7 recipe names and make RoboCasa action layouts
  explicit.
- Extend Oli inference with per-finger hand control, configurable prepare poses,
  keyboard pause, continuous inference, and configurable inference statistics.

### Fixed

- Derive PI0 and PI0.5 attention masks and action outputs from runtime horizons
  instead of hard-coded action lengths and attention-head widths.
- Restore missing Cosmos3 collator fields and correct padded state/action,
  checkpoint, and BF16 evaluation handling.
- Align DiT4DiT training and evaluation preprocessing, statistics, loss
  computation, and action denormalization with its source recipe.
- Preserve per-timestep RoboCasa action statistics and validate execution chunks
  against model and statistics horizons.
- Route FastWAM training evaluation and prediction through FSDP forward
  boundaries.
- Correct GR00T LIBERO and PI0.5 RoboCasa paths, remove stale config arguments,
  and fix concurrent API-backed LIBERO initialization and rank/device binding.

## [0.1.4]

### Added

- Franka single-arm and dual-arm operators, inference runners, PI0.5 recipes,
  and deployment documentation.
- Oli whole-body operator, inference runner, and GR00T recipe.
- Accelerated GR00T RTC inference with prefix conditioning, CUDA Graph support,
  and a UR3 RTC runner.
- Manager-based LIBERO and RoboCasa evaluation with task scheduling, resumable
  outputs, rollout videos, summaries, and Feishu Sheets reporting.
- Opt-in dataset version validation through `expected_dataset_version` and
  dataset metadata.

### Changed

- Add one-command environment install/update scripts, split base/simulation/
  real-robot requirements, and pin the simulation stack.
- Add TorchCodec video decoding with PyAV fallback.
- Unify DDP/FSDP optimizer configuration under `runner.optimizer` and move
  learning-rate schedules into registry-backed `lr_scheduler` policies.
- Unify the UR operator with `BaseOperator` and update UR3 configuration and
  deployment guidance.
- Align the GR00T RoboCasa 24-task, 30-demonstration recipe with a fixed-step
  training schedule, updated optimizer settings, and bounded checkpoint
  retention.

### Fixed

- Restore model weights and preserve tokenizer assets when resuming DDP
  checkpoints.
- Correct step-based epoch counting and resumed epoch calculation.
- Preserve full action chunk shapes in ZMQ serving.
- Correct count-aware dataset mean/variance aggregation and LIBERO distributed
  progress/output handling.
- Stabilize evaluation manager argument forwarding and RoboCasa environment and
  policy seeding.
- Restore ARM/SARM LeRobot v3 dataset setup and remove stale runner config
  fields.

## [0.1.3]

### Added

- ARM reward modeling, progress inference, and RA-BC/AW-BC sample weighting,
  with ALOHA training and setup documentation.
- GR00T/Eagle RoboCasa fine-tuning, preprocessing, and evaluation support.
- PI0.5 RTC Triton inference kernels and accelerated ALOHA/UR3 configurations.

### Changed

- Unify dataset loading around Parquet/LeRobot with shared image augmentation,
  per-epoch reshuffling, and dataset-statistics overrides.
- Expand ARM/SARM compatibility and document local dataset and CLIP checkpoint
  preparation.

### Removed

- Legacy RLDS dataset pipeline in favor of Parquet/LeRobot.

## [0.1.2]

### Changed

- No source changes relative to `v0.1.1`: both tags point to commit
  `ade7d545a640e868dfca7e59cf3571e228556b9f`. No separate implementation changes
  are attributed to this tag.

## [0.1.1]

### Added

- SmolVLA model support with LIBERO and ALOHA fine-tuning recipes.
- XVLA with a Florence2 backbone, flow-matching head, and configurable action
  spaces.
- SARM reward modeling, LeRobot annotation tools, progress weighting, inference,
  and visualization workflows.
- Qwen3-VL backbone integration and GR00T/Qwen3-VL LIBERO recipes.
- Tron2 operator, training and inference runners, multi-camera inputs, and RTC
  support.
- FluxBiSim training and inference configurations with AlohaSim integration.

### Changed

- Reuse DreamZero KV caches and recent observation context during closed-loop
  evaluation, with optional negative-prompt classifier-free guidance.
- Update the supported Transformers version to 5.3.0 and document PyTorch 2.8
  with CUDA 12.8 for Blackwell GPUs.
- Tune SmolVLA LIBERO fine-tuning settings and expand PI0/DreamZero recipes.

### Fixed

- Prevent duplicated data across DataLoader workers and support optional
  per-epoch reshuffling.
- Remove redundant LIBERO image resizing and duplicate training progress logs.

## [0.1.0]

### Added

- Unified VLA training, simulation evaluation, and real-robot inference
  framework with PI0, PI0.5, GR00T, OpenVLA, and LLaVA-based policies.
- DreamZero video-action model training and LIBERO inference with optional
  temporal KV caching.
- CUDA Graph, CUDA, and Triton inference acceleration and safetensors checkpoint
  support.
- ZMQ remote policy serving with ALOHA/UR clients and configurable serialization.
- PI0.5 training-time RTC and configurable delay sampling temperature.
- ALOHA HDF5-to-LeRobot data conversion tools, TensorBoard logging,
  English/Chinese/Japanese documentation, and changed-file pre-commit CI.

### Changed

- Accelerate PI0/PI0.5 training with SDPA attention, no-shard FSDP, BF16 storage,
  and improved DataLoader settings.

### Fixed

- Prevent eval-after-train memory exhaustion and stale distributed/CUDA state
  through checkpoint loading improvements, runner cleanup, and isolated
  evaluation launches.

[0.1.0]: https://github.com/FluxVLA/FluxVLA/tree/v0.1.0
[0.1.1]: https://github.com/FluxVLA/FluxVLA/compare/v0.1.0...v0.1.1
[0.1.2]: https://github.com/FluxVLA/FluxVLA/compare/v0.1.1...v0.1.2
[0.1.3]: https://github.com/FluxVLA/FluxVLA/compare/v0.1.2...v0.1.3
[0.1.4]: https://github.com/FluxVLA/FluxVLA/compare/v0.1.3...v0.1.4
[0.1.5]: https://github.com/FluxVLA/FluxVLA/compare/v0.1.4...v0.1.5
[unreleased]: https://github.com/FluxVLA/FluxVLA/compare/v0.1.5...main
