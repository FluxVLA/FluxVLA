# LingBot-VLA-v2 training inputs

## External resources

Training never imports the original LingBot model implementation. Supply the
verified model/training YAML, checkpoint and tokenizer explicitly:

```bash
export LINGBOT_NATIVE_CONFIG=/path/to/verified/model_training.yaml
export LINGBOT_CHECKPOINT=/path/to/verified/lingbot_checkpoint
export LINGBOT_TOKENIZER_PATH=/path/to/matching/qwen_tokenizer
export LINGBOT_DATA_PATH=/path/to/robotwin_lerobot
export WORK_DIR=/path/to/separate/writable_output
```

The YAML defines model/training settings, not a Python source directory. Use
the exact YAML and checkpoint from the reference experiment. A missing input
must fail explicitly; there is no private machine path fallback. Historical
official-run YAML snapshots in this directory are evidence, not automatically
selected portable production configs. Their external resource paths must be
reviewed before explicitly using one.

## Online training versus diagnostic replay

`lingbot_vla_v2_online_flow_only_100.py` uses real online LeRobot reads, shared
FP32 normalization/schema/image/tokenizer transforms and optional index-keyed
flow targets. Auxiliary teacher losses are disabled in this recipe; it is not
the complete auxiliary-teacher production configuration. `flow_rng_seed=42`
is an explicit parity experiment contract, not a requirement to keep the same
noise for repeated samples in arbitrary production training.

```bash
export LINGBOT_NUM_WORKERS=2
bash scripts/train.sh \
  configs/lingbot_vla_v2/lingbot_vla_v2_online_flow_only_100.py "$WORK_DIR"
```

For MCore online training, also set the read-only Megatron-LM checkout:

```bash
export MEGATRON_ROOT=/path/to/Megatron-LM
NPROC_PER_NODE=4 TP=1 EP=1 MB=1 GB=64 STEPS=100 \
  bash scripts/train_mcore_lingbot_vla_v2.sh --lingbot-sampler-seed 42
```

Match Native gradient accumulation, LR/scheduler, precision, attention/MoE,
trainable scope, noise/time and sample stream before comparing losses. The
MCore launch script defaults to sampler seed 42 and a training/worker seed of
42; changing one must be reflected in the reference configuration.

The `*_parity.py` replay configs and `run_lingbot_gb64_4gpu_case.sh` instead
require `LINGBOT_MANIFEST` pointing to an existing captured-input manifest.
Input artifacts are never inferred from the output directory. The GB64 case
script explicitly selects replay for both backends. Do not present replay
throughput as online data-loading throughput.

## Worker policy

Replay diagnostics keep `num_workers=0` to reduce debugging variables. Online
Native worker count is configurable through `LINGBOT_NUM_WORKERS` or
`--cfg-options`; MCore accepts the same environment variable. Benchmark workers
0/2/4/8 with real video/processor workloads rather than assuming a larger value
is better.

LingBot datasets opt into logical-DP worker seeding. Native and MCore use a
dedicated loader generator and the same worker initializer; TP ranks sharing a
DP shard do not seed augmentation from different physical `LOCAL_RANK`s.
Other model datasets retain their existing worker policy. MCore reuses its
DataLoader across epoch rollover so persistent workers stay alive.

This does not certify arbitrary stochastic augmentation replay after resume,
or make augmentation independent of worker count. Current resume tests cover
deterministic preprocessing and explicit per-index flow. Sample-keyed random
augmentations or worker-state checkpointing need separate certification.

## One online Parquet entry point

Native and MCore both use `LingBotParquetDataset`. The former
`LingBotOnlineDataset` implementation has been removed without an alias or
re-export. Online is a loading mode, not an alternative storage format.

The dataset composes the existing v2 `ParquetDataset` or v3
`ParquetDatasetV3` metadata reader and the shared
`fluxvla.transforms.lingbot_online.LingBotSampleTransform`. Both versions use
the same base action-window assembler with action key `action`, window start
0, explicit row identity, preserved task text and terminal padding masks.
Existing unrelated datasets retain their default task resampling behavior.

Embedded Parquet RGB images and external video RGB features are supported.
For v3 shared videos, file coordinates and each camera's `from_timestamp`
come from the episode metadata; missing offsets fail instead of assuming 0.
Future images use the stored timestamp of the clamped future row, including
irregular timestamps. Numeric action windows do not decode embedded images.
Only requested current/future rows are decoded online; no replay is needed.

`LINGBOT_VIDEO_BACKEND` / `--lingbot-video-backend` selects the explicit video
backend (default `torchcodec`). `LINGBOT_RESIZE_DTYPE` /
`--lingbot-resize-dtype` selects `float32` (default) or `uint8` resizing.
The current embedded-image reference uses float32 images in [0,1] before
resize. For references that resize uint8 video frames, explicitly choose
`uint8`. Different backends/dtypes are not assumed numerically equivalent.
New 10/100-step optimizer and full-production certification remains separate
from field-level input equivalence.

## CPU input-equivalence validation

The following diagnostic compares the current reader with a pinned
pre-refactor implementation from this repository, using real online videos
and a real tokenizer. It does not load model-ready replay batches or a model.

```bash
PYTHONPATH=. python tools/validate_lingbot_input_refactor.py \
  --reference-ref 7c2e8cd \
  --data "$LINGBOT_DATA_PATH" --tokenizer "$LINGBOT_TOKENIZER_PATH" \
  --stats configs/lingbot_vla_v2/assets/robotwin_norm_stats.json \
  --video-backend torchcodec --resize-dtype float32 \
  --workers 0 2 4 --output "$WORK_DIR/input_equivalence.json"
```

The reference source is confined to this diagnostic tool. Production loaders
do not depend on Git or execute historical code. Input equivalence is a
prerequisite, not a substitute for 10/100-step parameter-update certification.
