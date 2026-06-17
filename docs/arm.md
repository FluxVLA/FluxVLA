# ARM

FluxVLA's integration of [ARM (Advantage Reward Modeling)](https://arxiv.org/abs/2604.03037) for long-horizon robot manipulation.

> **Official implementation of** [ARM: Advantage Reward Modeling for Long-Horizon Manipulation](https://arxiv.org/abs/2604.03037)

ARM estimates **relative advantage** (Progressive / Stagnant / Regressive) over adjacent frames instead of brittle absolute progress alone. A causal observation window is encoded with CLIP, and a shared temporal transformer predicts:

- **Interval head**: tri-state labels in `{-1, 0, +1}` between consecutive frames in the window.
- **Success head**: whether the current frame has completed the task (`progress ≈ 1`).

Training minimizes `lambda_interval * CE + lambda_cls * focal_loss`.

For stage-based reward modeling, see [docs/sarm.md](sarm.md).

## ARM Checkpoints

For ARM workflows, keep dependent models under `./checkpoints` and reference them with relative paths from config files.

FluxVLA training checkpoints are written to the training output directory, typically under the `checkpoints/` subdirectory of the `--work-dir` you pass to `scripts/train.py`, for example `./work_dirs/arm/checkpoints/latest-checkpoint.pt`.

Recommended local layout:

```text
checkpoints/
└── clip-vit-base-patch32
```

Reserved local names:

- `./checkpoints/clip-vit-base-patch32`: CLIP backbone and tokenizer used by FluxVLA ARM configs for **online** image/text encoding during training and inference.
- Training-generated FluxVLA checkpoints live under `--work-dir/checkpoints/` instead of the root `./checkpoints/` directory.

Download CLIP with:

```bash
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./checkpoints/clip-vit-base-patch32
```

Or reuse the SARM setup script, which also prepares CLIP:

```bash
bash scripts/setup_sarm_data_ckpts.sh
```

## ARM Usage

FluxVLA's ARM config uses relative checkpoint paths under `./checkpoints`, consistent with the rest of the project.

Current ARM example config:

- `configs/arm/arm_clip_aloha_example.py` — CLIP ViT-B/32 backbone, single
  `observation.images.cam_high` camera, ALOHA-style `observation.state`, and a
  LeRobot v3 smoke dataset path. Copy or override fields for your robot setup.

Default settings in that config:

| Setting                           | Value                                 | Meaning                                                  |
| --------------------------------- | ------------------------------------- | -------------------------------------------------------- |
| `pretrained_name_or_path`         | `./checkpoints/clip-vit-base-patch32` | CLIP weights for `ARMBackbone` and the text tokenizer    |
| `n_history_steps` / `n_obs_steps` | `4`                                   | Causal window: 4 history frames + current frame          |
| `frame_gap`                       | `30`                                  | 30 frames between adjacent observations (≈1 s at 30 fps) |
| `interval_eps`                    | `1e-3`                                | Minimum absolute progress delta for ±1 interval labels   |
| `video_keys`                      | `observation.images.cam_high`         | Single-camera decode from `videos/`                      |
| `state_key`                       | `observation.state`                   | Robot state vector in parquet rows                       |

These configs expect:

- CLIP at `pretrained_name_or_path` (shared by `model.llm_backbone` and
  `TokenizeText.tokenizer.model_path`)
- A LeRobot v2.1 or v3.x dataset root with a scalar **`progress`** column in parquet rows

Notes:

- ARM reads **`progress`** directly from parquet. It does **not** use SARM subtask annotation columns (`sparse_subtask_*`, `dense_subtask_*`).
- Episode videos are decoded by the `DecodeLeRobotVideoSequence` transform (first step in `current_transforms`) and encoded online by CLIP inside `ARMBackbone`. Parquet `observation.video_features` (if present) is **not** used.
- For LeRobot v2.1/v3.x style datasets, `task` can be stored as a task index. FluxVLA resolves it back to task text at read time from `tasks.jsonl` or `tasks.parquet`.
- For LeRobot v3.x style datasets, video paths may be described either by `videos/<key>/chunk_index` / `file_index` or equivalent chunk/file columns on episode metadata or parquet rows. FluxVLA accepts those variants without requiring dataset rewrites.
- If your dataset uses a camera key other than `observation.images.cam_high`, override `train_dataloader.dataset.video_keys` and `inference_dataset.video_keys` with `--cfg-options`.
- When overriding dataset paths with `--cfg-options`, set **`train_dataloader.dataset.data_root_path`** and **`inference_dataset.data_root_path`** explicitly. Updating only a top-level `data_root_path` field does not propagate into nested dataset configs that were already copied at load time.

## Dataset Requirements

ARM training and inference require a standard LeRobot dataset with one additional scalar field per frame:

### Column contract (what FluxVLA reads)

Every parquet row must contain:

- `progress` (float in `[0, 1]`): task completion at the current frame.
- `observation.state` (or the key configured in `state_key`): robot state vector.
- Episode videos reachable via `video_keys` (for example `observation.images.cam_high`).

At sample time, `ARMDataset` builds a **causal** frame window `[t-4s, t-3s, t-2s, t-1s, t]` (with clamping inside the episode), derives **`interval_targets`** from consecutive `progress` deltas, and labels success when `progress >= 1 - success_eps`.

Example progress semantics:

- `0.0` at episode start
- monotonically increasing values during successful task execution
- `1.0` when the task is done

Unlike SARM, ARM does **not** need `meta/episodes` subtask list columns. Any pipeline that writes per-frame `progress` into LeRobot parquet is compatible.

### Example dataset layout

The starter config defaults to `./datasets/ARM_manual_test_10Episodes_lerobotv3.0`, a 10-episode LeRobot v3.x dataset with per-frame `progress` labels. Download it under `./datasets` with:

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset \
  --include "ARM_manual_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
```

Published reference dataset: [`limxdynamics/FluxVLAData/ARM_manual_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/ARM_manual_test_10Episodes_lerobotv3.0).

To train on your own data, override both dataset roots:

```bash
# Override both train and inference dataset roots
--cfg-options \
  train_dataloader.dataset.data_root_path=/path/to/your_lerobot_dataset \
  inference_dataset.data_root_path=/path/to/your_lerobot_dataset
```

LeRobot v3.x video metadata sanity checks from [docs/sarm.md](sarm.md) also apply when decoding episode videos for ARM.

## Training

Example training command:

```bash
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --work-dir ./work_dirs/arm \
  --eval-after-train \
  --cfg-options \
    train_dataloader.per_device_batch_size=8 \
    train_dataloader.dataset.data_root_path=/path/to/your_lerobot_dataset \
    inference_dataset.data_root_path=/path/to/your_lerobot_dataset \
    runner.max_epochs=None \
    runner.max_steps=5000 \
    runner.save_iter_interval=500
```

Example minimal real-dataset smoke run:

```bash
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --work-dir ./work_dirs/arm_smoke \
  --cfg-options \
    train_dataloader.dataset.data_root_path=/path/to/your_lerobot_dataset \
    inference_dataset.data_root_path=/path/to/your_lerobot_dataset \
    train_dataloader.per_device_batch_size=1 \
    train_dataloader.per_device_num_workers=0 \
    runner.max_steps=1 \
    runner.max_epochs=None \
    runner.save_iter_interval=1
```

Checkpoints are saved under `./work_dirs/arm/checkpoints/` (or your chosen `--work-dir`).

## Inference

FluxVLA provides three ARM inference entry points:

| Script                                                       | Purpose                                                                          |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `scripts/infer_arm_progress.py`                              | Debug / visualize reconstructed progress on one or more episodes                 |
| `scripts/compute_arm_awbc_progress.py`                       | Rebuild dense `progress` for an entire policy dataset and write RA/AW-BC parquet |
| `scripts/train.py` + `configs/arm/arm_clip_aloha_example.py` | Train ARM on a dataset that already has GT `progress` labels                     |

For **RA-BC / AW-BC on DAgger datasets without a `progress` column**, use
`compute_arm_awbc_progress.py`. It shares the same progress reconstruction
algorithm as episode visualization (implemented in
`tools/arm_awbc/progress_reconstruction.py`).

`scripts/infer_arm_progress.py` supports two complementary modes:

1. **Episode visualization** — strided per-episode inference, cumulative progress reconstruction, side-by-side camera + progress chart, and MP4 output.
2. **JSONL batch export** — dataloader-based offline scoring over many frames.

Set `--episode-idx` for visualization and/or `--output-path` for JSONL. At least one must be provided.

### Episode visualization (recommended)

Runs inference every `--inference-stride` frames (default `150` = 5 s at 30 fps), accumulates interval deltas into dense per-frame progress, and renders one panel per second: **left = camera frame**, **right = progress chart** (predicted blue curve, GT green curve, red current-time marker, orange inference keyframes).

```bash
python scripts/infer_arm_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --episode-idx 0 \
  --inference-stride 150 \
  --fps 30 \
  --vis-fps 5 \
  --image-key observation.images.cam_high \
  --output-dir ./work_dirs/arm/arm_viz \
  --cfg-options \
    inference_dataset.data_root_path=/path/to/your_lerobot_dataset
```

Outputs under `--output-dir`:

```text
arm_viz/
├── episode_0000/
│   ├── frame-000000.png
│   ├── frame-000001.png
│   └── ...
├── episode_0000_visualization.mp4
└── episode_0000_results.json
```

Visualize multiple episodes by looping `--episode-idx`:

```bash
for EP in 0 1 2 3 4 5; do
  python scripts/infer_arm_progress.py \
    --config configs/arm/arm_clip_aloha_example.py \
    --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
    --episode-idx "${EP}" \
    --inference-stride 150 \
    --fps 30 --vis-fps 5 \
    --image-key observation.images.cam_high \
    --output-dir ./work_dirs/arm/arm_viz \
    --cfg-options inference_dataset.data_root_path=/path/to/your_lerobot_dataset
done
```

Key visualization arguments:

| Flag                 | Default                  | Description                                          |
| -------------------- | ------------------------ | ---------------------------------------------------- |
| `--episode-idx`      | —                        | Episode index to visualize (required for this mode)  |
| `--dataset-idx`      | `0`                      | Dataset index when `data_root_path` is a list        |
| `--inference-stride` | `150`                    | Run model every N frames along the episode           |
| `--fps`              | `30`                     | Source video FPS; one visualization frame per second |
| `--vis-fps`          | `5`                      | Output MP4 playback FPS                              |
| `--chart-width`      | `900`                    | Width of the progress chart panel                    |
| `--image-key`        | first `video_keys` entry | Camera stream used for raw frame decode              |

Progress reconstruction (`tools/arm_awbc/progress_reconstruction.py`,
also used by `compute_arm_awbc_progress.py`) works as follows:

1. At each inference step, take the **last** interval prediction in the causal window as the step delta (`+1` / `0` / `-1`).
2. Accumulate deltas across inference keyframes.
3. Find the first keyframe where the success head predicts **done**.
4. Normalize accumulated scores to `[0, 1]` up to that done frame, then hold `1.0` afterward.
5. Linearly interpolate to every frame in the episode.

The result is a dense per-frame `progress` curve in `[0, 1]` that can be written
to parquet for RA/AW-BC even when the underlying LeRobot dataset has **no**
`progress` column.

### JSONL batch export

For offline scoring over the full dataset (without video rendering). This mode
exports **raw head outputs per frame** (`pred_success_prob`, `pred_interval`) and
does **not** run cumulative progress reconstruction. For RA/AW-BC parquet export,
use `scripts/compute_arm_awbc_progress.py` instead.

```bash
python scripts/infer_arm_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/arm/arm_progress.jsonl \
  --batch-size 8 \
  --max-batches 10 \
  --cfg-options \
    inference_dataset.data_root_path=/path/to/your_lerobot_dataset
```

Each JSONL record includes `pred_success_prob`, `pred_interval`, `pred_interval_sequence`, and optional `gt_progress` when the dataset provides it.

You can combine both modes in one invocation:

```bash
python scripts/infer_arm_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/arm/arm_progress.jsonl \
  --episode-idx 0 \
  --inference-stride 150 \
  --output-dir ./work_dirs/arm/arm_viz \
  --cfg-options inference_dataset.data_root_path=/path/to/your_lerobot_dataset
```

## RA-BC and AW-BC (Policy Reweighting)

> **Official implementation of** [ARM: Advantage Reward Modeling for Long-Horizon Manipulation](https://arxiv.org/abs/2604.03037)

### End-to-end pipeline

ARM policy reweighting splits into two dataset roles:

1. **ARM training dataset** — must contain GT `progress` labels (used only to train the reward model).
2. **Policy / DAgger dataset** — the BC dataset you actually train on; may have **no** `progress` column.

```text
[1] Train ARM on progress-labeled dataset
         │
         ▼
[2] compute_arm_awbc_progress.py on policy dataset (no progress column OK)
         │  interval head  → accumulate relative advantage deltas
         │  success head   → anchor task completion at progress = 1
         │  interpolate    → dense per-frame progress ∈ [0, 1]
         ▼
    arm_progress.parquet
         │
         ▼
[3] RA-BC / AW-BC during policy training
         delta = progress[t + chunk_size] - progress[t]
         RA-BC:  sample_weight = rabc_weight(delta)
         AW-BC:  sample_weight = rabc_weight(delta) × (episode_length / mean_episode_length)
```

Step 2 uses the **same reconstruction logic** as episode visualization in
`scripts/infer_arm_progress.py` (shared implementation:
`tools/arm_awbc/progress_reconstruction.py`). The parquet `progress` column is
therefore a **reconstructed completion curve**, not a raw success-head probability.

SARM RA-BC uses separate `progress_sparse` / `progress_dense` columns instead;
see [docs/sarm.md](sarm.md).

### RA-BC (Reward-Aligned Behavior Cloning)

RA-BC assigns each sample a weight from the expected progress improvement over
the policy action horizon:

```text
delta = progress[t + chunk_size] - progress[t]
sample_weight = rabc_weight(delta)
```

`chunk_size` should match the policy action horizon (for example `50` for SmolVLA
/ PI0.5). Intuitively, frames whose progress is about to rise are up-weighted;
frames where progress stalls or regresses are down-weighted or dropped.

**`rabc_weight` mapping** (implemented in `fluxvla.weighters.ArmRABCWeighter`):

| Condition                        | Weight                                                         |
| -------------------------------- | -------------------------------------------------------------- |
| `delta > kappa` (default `0.01`) | `1.0` — clear progress gain                                    |
| `0 <= delta <= kappa`            | soft weight (linear interpolation between dataset delta stats) |
| `delta < 0`                      | `0.0` — progress regresses, skip sample                        |
| invalid / missing progress       | `fallback_weight` (default `1.0`)                              |

After per-sample weights are computed, batch weights are renormalized so they sum
to batch size (same path as AW-BC).

**When to use:** episode lengths are roughly uniform. RA-BC is the simpler default
when you only need progress-based filtering.

**Prerequisites:**

1. A trained **ARM checkpoint** (train with `configs/arm/arm_clip_aloha_example.py` or your fork)
2. An **ARM progress parquet** on the **same policy dataset** you will train on

Episode length stats are **not** required for RA-BC.

**Policy training config:**

See `tools/arm_awbc/README.md` for the full RA-BC / AW-BC tool guide. To wire
RA-BC into your policy config, on the inner `ParquetDataset`:

1. set `expose_index=True` so each sample carries the global frame index
2. insert `AttachRABCWeight` **before** `ProcessParquetInputs`
3. add `sample_weight` to `DictCollator.keys`

```python
rabc_weighter = dict(
    type='ArmRABCWeighter',
    progress_path='./work_dirs/arm_rabc/arm_progress.parquet',
    chunk_size=50,
    index_key='index',
)

train_dataloader = dict(
    dataset=dict(
        type='ParquetDataset',
        data_root_path=['/path/to/policy_dataset'],
        expose_index=True,
        transforms=[
            dict(type='AttachRABCWeight', weighter=rabc_weighter),
            dict(type='ProcessParquetInputs', ...),
            ...
        ],
    ),
)

runner = dict(
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'images', 'lang_tokens', 'lang_masks',
            'actions', 'action_masks', 'sample_weight',
        ],
    ),
)
```

Implementation helpers:

- `fluxvla.weighters.ArmRABCWeighter` — RA-BC weight computation (registry-backed)
- `fluxvla.transforms.attach_rabc_weight.AttachRABCWeight` — attach
  `sample_weight` in the dataset transform chain

The loss path in `fluxvla.engines.losses.reduce_action_bc_loss` multiplies BC
losses by `sample_weight` when present.

### RA-BC vs AW-BC

AW-BC extends RA-BC with an episode-length scaling term:

```text
sample_weight = rabc_weight * (episode_length / mean_episode_length)
```

|                               | RA-BC                   | AW-BC                                                  |
| ----------------------------- | ----------------------- | ------------------------------------------------------ |
| **Formula**                   | `rabc_weight`           | `rabc_weight × (episode_length / mean_episode_length)` |
| **Progress parquet**          | required                | required (same file)                                   |
| **Episode length stats JSON** | not required            | not required (derived online from progress parquet)    |
| **Weighter**                  | `ArmRABCWeighter`       | `ArmAWBCWeighter`                                      |
| **Best for**                  | uniform episode lengths | heterogeneous DAgger rollouts                          |

Both methods share the same `rabc_weight` rule on
`progress[t + chunk_size] - progress[t]`. AW-BC adds a duration factor so
**longer episodes are up-weighted** and **shorter episodes are down-weighted**,
which stabilizes BC training when rollout lengths vary widely.

Use RA-BC when episode lengths are similar. Switch to AW-BC when DAgger-style
collections mix short and long trajectories and you want duration-aware
rebalancing on top of progress filtering.

## AW-BC (Advantage Weight Behavior Cloning)

The steps below cover the full AW-BC pipeline. Steps 1–2 also produce the
progress parquet needed for RA-BC; for RA-BC alone, use `ArmRABCWeighter`
in Step 3.

### Prerequisites

AW-BC needs two artifacts:

1. A trained **ARM checkpoint** (train with `configs/arm/arm_clip_aloha_example.py` or your fork)
2. An **ARM progress parquet** on the **same policy dataset** you will train on

### Step 1: train ARM (skip if you already have a checkpoint)

ARM training requires a dataset with per-frame `progress` labels. Use a
progress-labeled LeRobot export for this step, then apply the checkpoint to any
target BC / DAgger dataset in Step 2.

```bash
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --work-dir ./work_dirs/arm \
  --cfg-options \
    train_dataloader.dataset.data_root_path=/path/to/progress_labeled_dataset \
    inference_dataset.data_root_path=/path/to/progress_labeled_dataset
```

### Step 2: precompute ARM progress on the policy dataset

Run `scripts/compute_arm_awbc_progress.py` on the dataset you will use for BC /
DAgger policy training. The script:

1. Iterates over every episode in the policy dataset.
2. Runs strided ARM inference (`--stride`, default `150`) using **both heads**.
3. Rebuilds dense `[0, 1]` progress for **every frame** via
   `tools/arm_awbc/progress_reconstruction.py` (interval accumulation + success
   anchor + linear interpolation).
4. Writes `arm_progress.parquet` with one row per frame.

This works on datasets with **no existing `progress` column** — typical for
DAgger rollouts where only actions and observations were recorded.

| Flag            | Default                  | Meaning                                                            |
| --------------- | ------------------------ | ------------------------------------------------------------------ |
| `--stride`      | `150`                    | Run ARM every N frames within each episode (matches visualization) |
| `--output-path` | `./arm_progress.parquet` | Output parquet for RA/AW-BC                                        |
| `--ckpt-path`   | required                 | Trained ARM checkpoint from Step 1                                 |

Use `--stride 1` for maximum fidelity (ARM runs on every frame as a keyframe) at
higher compute cost. Regardless of stride, the output parquet always contains a
`progress` value for every frame in every episode.

When attaching RA/AW-BC weights in policy training, set `chunk_size` to the
policy action horizon (for example `50` for SmolVLA / PI0.5). RA/AW-BC then
compares `progress[t + chunk_size] - progress[t]` on the reconstructed curve.

```bash
python scripts/compute_arm_awbc_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/arm_awbc/arm_progress.parquet \
  --stride 150 \
  --cfg-options \
    inference_dataset.data_root_path=/path/to/policy_dataset
```

Output columns: `index`, `dataset_index`, `episode_index`, `frame_index`,
`episode_length`, `progress`.

**Verify before policy training:** run visualization on one episode and confirm
the reconstructed curve looks reasonable:

```bash
python scripts/infer_arm_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --episode-idx 0 \
  --inference-stride 150 \
  --output-dir ./work_dirs/arm/arm_viz \
  --cfg-options inference_dataset.data_root_path=/path/to/policy_dataset
```

### Step 3: attach AW-BC weights in policy training

Follow the same `ParquetDataset` setup as [RA-BC](#ra-bc-reward-aligned-behavior-cloning)
(`expose_index=True`, `AttachRABCWeight` before `ProcessParquetInputs`,
`sample_weight` in `DictCollator.keys`). Use `ArmAWBCWeighter`:

```python
awbc_weighter = dict(
    type='ArmAWBCWeighter',
    progress_path='./work_dirs/arm_awbc/arm_progress.parquet',
    chunk_size=50,
    index_key='index',
)
```

`ArmAWBCWeighter` derives `episode_length` and `mean_episode_length` online from
the progress parquet (`episode_length` column, with a fallback to per-episode
frame counts in the same file). No separate stats JSON is required.

Implementation helpers:

- `fluxvla.weighters.ArmAWBCWeighter` — AW-BC weight computation
- `fluxvla.transforms.attach_rabc_weight.AttachRABCWeight` — attach
  `sample_weight` in the dataset transform chain

### End-to-end example: `./datasets/SARM_manual_test_10Episodes_lerobotv2.1`

This 10-episode LeRobot v2.1 dataset has heterogeneous lengths (1715–2689
frames, mean **2179.3**). It does **not** contain a `progress` column, so it
can serve as a **policy / DAgger BC dataset**, while ARM itself should be
trained on a separate progress-labeled dataset.

Assume you already have `./work_dirs/arm/checkpoints/latest-checkpoint.pt` from
Step 1:

```text
POLICY_DATA=./datasets/SARM_manual_test_10Episodes_lerobotv2.1
WORK=./work_dirs/sarm_manual_v21_awbc
```

**A. ARM progress on the policy dataset**

```bash
python scripts/compute_arm_awbc_progress.py \
  --config configs/arm/arm_clip_aloha_example.py \
  --ckpt-path ./work_dirs/arm/checkpoints/latest-checkpoint.pt \
  --output-path ${WORK}/arm_progress.parquet \
  --stride 150 \
  --cfg-options \
    inference_dataset.data_root_path=${POLICY_DATA} \
    inference_dataset.video_keys="['observation.images.cam_high']"
```

**B. Duration scaling (computed online at weighter init)**

When `ArmAWBCWeighter` loads `arm_progress.parquet`, it reads each
episode's `episode_length` and computes `mean_episode_length` automatically.
On the 10-episode example dataset (1715–2689 frames, mean **2179.3**):

| Episode | Length | `length / mean` |
| ------- | ------ | --------------- |
| 0       | 2689   | ≈ 1.23          |
| 5       | 1715   | ≈ 0.79          |

**C. Policy config snippet**

```python
awbc_weighter = dict(
    type='ArmAWBCWeighter',
    progress_path='./work_dirs/sarm_manual_v21_awbc/arm_progress.parquet',
    chunk_size=50,
    index_key='index',
)

# ParquetDataset for SARM_manual v2.1:
# - expose_index=True
# - action column is named "action" (map to "actions" in ProcessParquetInputs)
# - video key: observation.images.cam_high
```

Then launch your SmolVLA / PI0.5 / GR00T config with `data_root_path`
pointing at `${POLICY_DATA}`.

### AW-BC usage notes

- **Progress source**: `arm_progress.parquet` stores **reconstructed** dense
  progress (interval accumulation + success anchor + interpolation), not raw
  success-head probabilities. The reconstruction is shared with
  `infer_arm_progress.py` visualization via
  `tools/arm_awbc/progress_reconstruction.py`.
- **`--stride` in progress precompute** controls ARM inference keyframe spacing
  within each episode. Dense per-frame `progress` is always exported regardless
  of stride. Lower stride is more accurate but slower.
- **`chunk_size` must match** the policy action horizon used during training.
  RA-BC / AW-BC compares `progress[t]` with `progress[t + chunk_size]`.
- **`index` alignment**: the global `index` exposed by `ParquetDataset` must
  match the `index` column in `arm_progress.parquet`. Always compute progress
  on the **same dataset root** used for policy training.
- **Episode length prior**: derived online from `arm_progress.parquet` when
  `ArmAWBCWeighter` initializes. Ensure Step 2 ran on the full policy
  dataset so every episode appears in the parquet.
- **Batch normalization**: `ArmAWBCWeighter.compute_batch_weights` renormalizes
  weights to sum to batch size, same as RA-BC.

## Implementation Map

| Component                 | Location                                                    |
| ------------------------- | ----------------------------------------------------------- |
| Training config           | `configs/arm/arm_clip_aloha_example.py`                     |
| Dataset loader            | `fluxvla/datasets/arm_dataset.py`                           |
| CLIP + temporal heads     | `fluxvla/models/backbones/llms/arm.py`                      |
| Reward model + losses     | `fluxvla/models/vlas/arm_reward_model.py`                   |
| Inference + visualization | `scripts/infer_arm_progress.py`                             |
| ARM progress for RA/AW-BC | `scripts/compute_arm_awbc_progress.py`                      |
| Progress reconstruction   | `tools/arm_awbc/progress_reconstruction.py`                 |
| RA-BC / AW-BC weighters   | `fluxvla/weighters/` (`ArmRABCWeighter`, `ArmAWBCWeighter`) |
| RA-BC attach transform    | `fluxvla/transforms/attach_rabc_weight.py`                  |
| RA/AW-BC tool guide       | `tools/arm_awbc/README.md`                                  |

## ARM vs SARM at a Glance

|                             | SARM                                                         | ARM                                                                 |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- |
| Supervision                 | Sparse/dense subtask stages in episode metadata              | Scalar `progress` in parquet                                        |
| Model output                | Stage index + in-stage progress                              | Interval advantage + success probability                            |
| Typical visualization       | Matplotlib PNG (`infer_sarm_progress.py`)                    | Camera + PIL chart video (`infer_arm_progress.py`)                  |
| Policy reweighting progress | SARM progress parquet (`progress_sparse` / `progress_dense`) | ARM reconstructed progress parquet (`compute_arm_awbc_progress.py`) |
| Annotation tools            | `tools/sarm_annotate/`                                       | External progress labeling pipeline                                 |

Both share LeRobot dataset I/O (`DecodeLeRobotVideoSequence` / `fluxvla.datasets.utils.video_decode`, task text resolution, state normalization) through the same underlying dataset stack.
