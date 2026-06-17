# ARM RA-BC / AW-BC Reweighting

Tools for turning a trained **ARM (Advantage Reward Modeling)** reward model into
per-sample behavior-cloning weights — the **RA-BC** (Reward-Aligned Behavior
Cloning) and **AW-BC** (Advantage-Weight Behavior Cloning) pipelines.

> **Official implementation of** [ARM: Advantage Reward Modeling for Long-Horizon Manipulation](https://arxiv.org/abs/2604.03037)

Unlike SARM's stage annotation pipeline ([`tools/sarm_annotate/`](../sarm_annotate/README.md)),
ARM does **not** annotate the dataset. It reconstructs a dense per-frame
`progress` curve directly from the reward model's two heads, writes it to a
sidecar parquet, and consumes that parquet as a sample-weight source while a
policy (SmolVLA / PI0.5 / GR00T / ...) trains.

This package has two complementary parts:

1. **Progress reconstruction** (`progress_reconstruction.py`) — rebuild dense
   `[0, 1]` progress from strided ARM inference (interval head + success head).
   Shared by [`scripts/infer_arm_progress.py`](../../scripts/infer_arm_progress.py)
   (episode visualization) and
   [`scripts/compute_arm_awbc_progress.py`](../../scripts/compute_arm_awbc_progress.py)
   (parquet export), so the visualized curve and the curve used for training are
   byte-identical.
2. **Sample weighters** (re-exported from [`fluxvla.weighters`](../../fluxvla/weighters/arm_rabc.py))
   — `ArmRABCWeighter` and `ArmAWBCWeighter` map the reconstructed progress into
   per-sample weights inside the policy dataloader.

For the full conceptual walkthrough see [`docs/arm.md`](../../docs/arm.md). For
SARM's stage-based equivalent see [`docs/sarm.md`](../../docs/sarm.md).

## Two dataset roles

ARM reweighting always involves **two** datasets, which may be different:

| Role                        | Needs a `progress` column? | Used for                                        |
| --------------------------- | -------------------------- | ----------------------------------------------- |
| **ARM training dataset**    | yes (GT per-frame labels)  | training the reward model only                  |
| **Policy / DAgger dataset** | no                         | the BC dataset you actually train the policy on |

The policy dataset is typically a DAgger rollout collection that only recorded
observations and actions. ARM fills in the missing `progress` for it by
inference, so no manual labeling is required. See
[`docs/arm.md` → Dataset Requirements](../../docs/arm.md#dataset-requirements)
for the exact column contract.

The progress-labeled ARM example dataset is
[`limxdynamics/FluxVLAData/ARM_manual_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/ARM_manual_test_10Episodes_lerobotv3.0).
Download it (and fetch the CLIP backbone via
[`scripts/setup_sarm_data_ckpts.sh`](../../scripts/setup_sarm_data_ckpts.sh)) with:

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset \
  --include "ARM_manual_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
```

## What gets produced

`scripts/compute_arm_awbc_progress.py` writes a single `arm_progress.parquet`
with **one row per frame** of the policy dataset:

| Column           | Meaning                                                       |
| ---------------- | ------------------------------------------------------------- |
| `index`          | global frame index in the concatenated dataset (RA/AW-BC key) |
| `dataset_index`  | which dataset root the frame came from                        |
| `episode_index`  | episode the frame belongs to                                  |
| `frame_index`    | local frame index inside the episode                          |
| `episode_length` | number of frames in the episode (AW-BC duration scaling)      |
| `progress`       | **reconstructed** dense completion in `[0, 1]`                |

The `progress` column is a reconstructed completion curve (interval
accumulation + success anchor + interpolation), **not** a raw success-head
probability. Parquet schema metadata also records `reward_model_path` and
`progress_source = interval_success_reconstruction`.

## End-to-end pipeline

```text
[1] Train ARM on a progress-labeled dataset
        configs/arm/arm_clip_aloha_example.py
        │
        ▼
[2] compute_arm_awbc_progress.py on the policy dataset (no progress column OK)
        interval head → accumulate relative-advantage deltas {-1, 0, +1}
        success head  → anchor task completion at progress = 1
        interpolate   → dense per-frame progress ∈ [0, 1]
        │
        ▼
    arm_progress.parquet
        │
        ▼
[3] RA-BC / AW-BC during policy training
        delta = progress[t + chunk_size] - progress[t]
        RA-BC: sample_weight = rabc_weight(delta)
        AW-BC: sample_weight = rabc_weight(delta) × (episode_length / mean_episode_length)
```

### Step 1 — train the ARM reward model

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --work-dir ./work_dirs/arm \
  --cfg-options \
    train_dataloader.dataset.data_root_path=/path/to/progress_labeled_dataset \
    inference_dataset.data_root_path=/path/to/progress_labeled_dataset
```

Skip this step if you already have an ARM checkpoint.

### Step 2 — reconstruct progress on the policy dataset

```bash
python scripts/compute_arm_awbc_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/arm_awbc/arm_progress.parquet \
  --stride 150 \
  --cfg-options \
    inference_dataset.data_root_path=/path/to/policy_dataset
```

- `--stride N` runs ARM every `N` frames per episode (`150` ≈ 5 s at 30 fps),
  then interpolates dense progress to **every** frame. Use `--stride 1` for
  maximum fidelity at higher compute cost. Dense per-frame `progress` is always
  exported regardless of stride.
- Always compute progress on the **same dataset root** used for policy training
  so the global `index` column aligns with the policy `ParquetDataset`.

Sanity-check one episode before training:

```bash
python scripts/infer_arm_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --episode-idx 0 \
  --inference-stride 150 \
  --output-dir ./work_dirs/arm/arm_viz \
  --cfg-options inference_dataset.data_root_path=/path/to/policy_dataset
```

### Step 3 — train the policy with RA-BC / AW-BC weights

RA-BC / AW-BC wires into any existing policy finetune config (SmolVLA, PI0.5,
GR00T, ...). Take your policy config and make three additions:

1. set `expose_index=True` on the inner `ParquetDataset`, so every sample
   carries its global frame `index`;
2. insert `AttachRABCWeight` **before** `ProcessParquetInputs` in the transform
   chain, so `sample_weight` is computed per sample and carried through to the
   collator;
3. add `sample_weight` to `DictCollator.keys`.

```python
# AW-BC by default; use type='ArmRABCWeighter' for plain RA-BC.
arm_weighter = dict(
    type='ArmAWBCWeighter',
    progress_path='./work_dirs/arm_awbc/arm_progress.parquet',
    chunk_size=50,  # match the policy action horizon
    index_key='index',
)

train_dataloader = dict(
    dataset=dict(
        type='ParquetDataset',
        data_root_path=['/path/to/policy_dataset'],
        expose_index=True,
        transforms=[
            dict(type='AttachRABCWeight', weighter=arm_weighter),
            dict(type='ProcessParquetInputs', ...),
            # ... remaining policy transforms ...
        ],
    ),
)

runner = dict(
    collator=dict(
        type='DictCollator',
        keys=[..., 'sample_weight'],
    ),
)
```

Then launch training as usual:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config /path/to/your_policy_config.py \
  --work-dir ./work_dirs/arm_awbc_policy
```

The policy model's `forward` reads `sample_weight` and passes it to
`fluxvla.engines.losses.reduce_action_bc_loss`, which scales each sample's BC
loss by the weight. Any policy whose `forward` forwards `sample_weight` into
`reduce_action_bc_loss` (SmolVLA, PI0.5, OpenVLA, DreamZero, ...) works the same
way.

## Progress reconstruction algorithm

`build_cumulative_progress` (in `progress_reconstruction.py`) works as follows:

1. At each inference keyframe, take the **last** interval prediction in the
   causal window as the step delta (`+1` / `0` / `-1`).
2. Accumulate deltas across keyframes into a raw cumulative score.
3. Find the first keyframe where the success head predicts **done**.
4. Normalize the accumulated scores to `[0, 1]` up to that done frame, then hold
   `1.0` afterward.
5. Linearly interpolate to every frame in the episode.

## RA-BC vs AW-BC

|                          | RA-BC                   | AW-BC                                                 |
| ------------------------ | ----------------------- | ----------------------------------------------------- |
| **Per-sample weight**    | `rabc_weight(delta)`    | `rabc_weight(delta) × (episode_length / mean_length)` |
| **Progress parquet**     | required                | required (same file)                                  |
| **Episode-length stats** | not required            | derived online from the progress parquet              |
| **Registry type**        | `ArmRABCWeighter`       | `ArmAWBCWeighter`                                     |
| **Best for**             | uniform episode lengths | heterogeneous DAgger rollouts                         |

`rabc_weight(delta)` (where `delta = progress[t + chunk_size] - progress[t]`):

| Condition                        | Weight                                                      |
| -------------------------------- | ----------------------------------------------------------- |
| `delta > kappa` (default `0.01`) | `1.0` — clear progress gain                                 |
| `0 <= delta <= kappa`            | soft weight (linear interpolation over dataset delta stats) |
| `delta < 0`                      | `0.0` — progress regresses, skip sample                     |
| invalid / missing progress       | `fallback_weight` (default `1.0`)                           |

Set `chunk_size` to the policy action horizon (for example `50` for SmolVLA /
PI0.5). In the example config the weight is attached per sample by
`AttachRABCWeight`, and `reduce_action_bc_loss` applies it as a weighted average
over valid action elements — so only the *relative* weights within a batch
matter. (The weighters' `compute_batch_weights` helper additionally renormalizes
a whole batch to sum to the batch size for callers that score batches directly.)

## Module reference

`tools.arm_awbc` re-exports everything needed to script the pipeline:

| Symbol                          | Source                                                                 | Purpose                                        |
| ------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| `run_strided_episode_inference` | `progress_reconstruction.py`                                           | strided per-episode ARM inference (both heads) |
| `build_cumulative_progress`     | `progress_reconstruction.py`                                           | dense `[0, 1]` progress from keyframe records  |
| `extract_last_interval_delta`   | `progress_reconstruction.py`                                           | last interval label in a causal window         |
| `ArmRABCWeighter`               | [`fluxvla/weighters/arm_rabc.py`](../../fluxvla/weighters/arm_rabc.py) | RA-BC per-sample / per-batch weights           |
| `ArmAWBCWeighter`               | [`fluxvla/weighters/arm_rabc.py`](../../fluxvla/weighters/arm_rabc.py) | AW-BC weights (RA-BC + duration scaling)       |
| `resolve_arm_progress_path`     | `fluxvla/weighters/utils.py`                                           | resolve an `arm_progress.parquet` path         |

`ArmRABCWeights` / `ArmAWBCWeights` are deprecated aliases of the weighter
classes, kept for backward-compatible imports. The transform that attaches the
weight to each sample is
[`fluxvla.transforms.attach_rabc_weight.AttachRABCWeight`](../../fluxvla/transforms/attach_rabc_weight.py).

## Origin

| Local file                   | Description                                                            |
| ---------------------------- | ---------------------------------------------------------------------- |
| `progress_reconstruction.py` | FluxVLA-native; shared by visualization and parquet export.            |
| `__init__.py`                | Re-exports the reconstruction helpers and `fluxvla.weighters` ARM API. |

All files retain their Apache 2.0 Limx Dynamics headers.
