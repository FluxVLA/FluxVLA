# Raw VLM data and VLA/VLM mixed training

## Scope and preparation

VLM SFT reads the original **LLaVA-format JSON/JSONL annotations and extracted
media files**. There is no catalog, offline conversion command or prepared
JSONL requirement. See the [README](../README.md) for LLaVA-665K annotation,
image and checkpoint downloads. Run from the repository root:

```text
datasets/llava665k/
├── llava_v1_5_mix665k.json
├── coco/train2017/
├── gqa/images/
├── ocr_vqa/images/
├── textvqa/train_images/
└── vg/
    ├── VG_100K/
    └── VG_100K_2/
```

The annotation download does not include the images. Preserve its relative
media paths. An existing local/OSS filesystem mount can be symlinked to this
directory; direct object-storage URLs and streaming archives are not supported.
Robot-action data still uses the existing LeRobot preparation and normalization
pipeline. "Raw data" here does not mean arbitrary robot HDF5 files or an
arbitrary annotation schema can be used without an adapter.

## Dataset and transform contract

```python
dataset = dict(
    type='LLaVAFormatDataset',
    anno_json_path='./datasets/llava665k/llava_v1_5_mix665k.json',
    data_root_path='./datasets/llava665k',
    index_cache_dir='./datasets/.cache/vlm',
    max_retry_times=0,
    seed=42,
    transforms=[
        dict(type='BuildVLMMessages'),
        dict(
            type='ProcessPi05VLMSFTInputs',
            tokenizer_path='./checkpoints/pi05_base',
            image_size=224,
            max_length=1024),
    ])
```

1. `LLaVAFormatDataset` reads one raw record and adds `data_root`. It does not
   instantiate a model or processor. Without transforms, this dict is its output.
2. `BuildVLMMessages` normalizes LLaVA `from/value` conversations or supported
   `role/content` messages. It resolves media paths and preserves turn order.
3. `ProcessPi05VLMSFTInputs` encodes Pi05 image/text inputs. For native Qwen VLMs,
   use `ProcessVLMSFTInputs` with the matching `processor_path` and
   `processor_type`, as in [the Qwen config](../configs/vlm/qwen3_vl_2b_sft.py).
4. Each task has its own collator. Metadata and raw conversations are not passed
   to the model. All assistant replies and their end tokens are supervised;
   user/system tokens and padding have label `-100`.

For Pi05, a sample has `input_ids`, `attention_mask`, `labels` of shape `[L]`
and `images` of shape `[3*N, H, W]`. The collated batch has:

| Key              | Shape                | Meaning                         |
| ---------------- | -------------------- | ------------------------------- |
| `input_ids`      | `[B, L_max]`         | Token IDs (`int64`)             |
| `attention_mask` | `[B, L_max]`         | Valid text positions            |
| `labels`         | `[B, L_max]`         | Assistant supervision (`int64`) |
| `images`         | `[B, 3*N_max, H, W]` | Images normalized to `[-1, 1]`  |
| `img_masks`      | `[B, N_max]`         | Valid image views               |

`H=W=224` in the examples. Text-only batches omit `images`/`img_masks`.
There are **no states, actions or action masks** in VLM batches. Qwen instead
uses the processor's packed `pixel_values` and `image_grid_thw` (and video
equivalents); those shapes depend on the visual patch count, not Pi05's layout.

## Indexing, multiple sources and errors

- JSON arrays/objects and JSONL/NDJSON are indexed by byte offsets on first use.
  The original file is not rewritten or loaded wholesale into worker memory.
  There is no 64 MiB **file-size** limit; an oversized individual JSON record
  still has a defensive buffer limit. The initial index requires a full scan.
- Index files use 16 bytes per row plus a fixed header and are memory-mapped.
  The writable `index_cache_dir` can be outside a read-only raw-data directory.
  Workers reuse the index; processor/tokenizer loading is lazy and process-local.
- Changing the source path, size or timestamps invalidates its index. Do not
  mutate annotations while training. Shared caches require filesystem locking
  and atomic-rename support; node-local caches are also usable.
- Multiple sources are configured with `anno_json_path=[dict(path=..., weight=...), ...]` and corresponding `data_root_path=[..., ...]`. `weight` is a **repeat
  factor**, not a sampling probability: effective length is
  `floor(weight)*N + round(frac(weight)*N)`. Fractional selection is seeded and
  fixed; source sizes affect the resulting mixture.
- Start new data checks with `max_retry_times=0`. Configured retries report and
  replace bad samples, changing the effective distribution; they do not repair
  missing media or validate the whole corpus.

## Training

Prepare the matching weights/tokenizer under `./checkpoints` as in the README.
These examples use two GPUs; full fine-tuning needs substantial GPU memory.

```bash
# Pure Pi05 VLM SFT, directly from raw LLaVA JSON and images.
NPROC_PER_NODE=2 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_llava665k_full_finetune.py \
  ./work_dirs/pi05_vlm_sft

# Pure native Qwen VLM SFT.
NPROC_PER_NODE=2 bash scripts/train.sh \
  configs/vlm/qwen3_vl_2b_sft.py ./work_dirs/qwen3_vl_2b_sft

# Pi05 robot-action and language training with separate data streams.
NPROC_PER_NODE=2 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py \
  ./work_dirs/pi05_robocasa_llava665k_mixed
```

The [RoboTwin mixed config](../configs/pi05/pi05_paligemma_robotwin_llava665k_mixed_full_finetune.py)
uses the same raw language pipeline with the existing RoboTwin robot dataset.

- Pure SFT uses `runner.train_task='vlm'`. Pi05 freezes unused action parameters
  but retains them in checkpoints for later policy use.
- Mixed training uses `runner.train_task='mixed'`, `train_dataloader` for VLA
  and `vlm_dataloader` for VLM. The default schedule is
  `('vla', 'vla', 'vla', 'vlm')`: **one task per optimizer update**, synchronized
  across ranks. It does not concatenate VLM fields into a robot-action batch.
- Each example uses per-device batch 1 and accumulation 8: global batch 16 on
  two GPUs for either task. VLM gradients are normalized by supervised token
  count; mixed VLA gradients by valid action-element count. Loss weights and
  update ratios are separate from dataset-source repeat factors.
- Pi05 is the supported mixed-training model. Models without this contract
  reject `train_task='mixed'` rather than silently ignoring the task.
- Full mixed `.pt` checkpoints retain optimizer/scheduler state, task counters,
  data cursors and per-rank RNG state. Resume requires matching data, schedule,
  batch sizes, trainable parameters and world size. `.safetensors` is weights
  only, not a full resume checkpoint.
- Optional held-out language validation uses `VLMEvaluator` and a separate
  dataset with `max_retry_times=0`. It reports teacher-forced loss and token
  accuracy, not benchmark generation scores. Mixed configs keep their existing
  robot evaluator; pure SFT does not require robot evaluation data.

## Supported boundaries and tests

The common message transform supports text, single/multiple images, multiple
turns and initial system messages. Pi05 accepts images only in the first user
turn and rejects videos. The Qwen processor path additionally supports local
videos; this does not imply every model or external message schema supports
every media combination. Raw OpenAI `image_url`, audio/tool messages,
HF Parquet embedded media, tar/WebDataset, sequence packing and dynamic
token-budget batching need dedicated adapters or further implementation.

Use held-out annotations for validation. Downloaded image archives may contain
multiple splits; neither indexing nor schema validation prevents data leakage
or verifies licensing, duplicates, or complete media integrity.

Offline regression tests (no pretrained weights downloaded):

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --cpu-model-tests \
  test/test_configs/test_vlm_raw_annotations_config.py \
  test/test_datasets/test_llava_format_dataset.py \
  test/test_datasets/test_vlm_annotations.py \
  test/test_transforms/test_vlm_transforms.py \
  test/test_models/test_vlm_training.py \
  test/test_runners/test_mixed_training.py -q
```

The CPU import-isolation mode is the repository's unit-test harness, not a
production runtime. Distributed integration must also be tested with the
normal training entry point, real weights and available media. Short training
smoke tests do not establish full-run stability, convergence or robot success.
