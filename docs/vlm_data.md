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
            type='ProcessPaliGemmaVLMSFTInputs',
            tokenizer_path='./checkpoints/pi05_base',
            image_size=224,
            max_length=1024),
    ])
```

1. `LLaVAFormatDataset` reads one raw record and adds `data_root`. It does not
   instantiate a model or processor. Without transforms, this dict is its output.
2. `BuildVLMMessages` normalizes LLaVA `from/value` conversations or supported
   `role/content` messages. It resolves media paths and preserves turn order.
3. `ProcessPaliGemmaVLMSFTInputs` in `paligemma_vlm_transforms.py` encodes
   PaliGemma-vocabulary conversations and normalized image views, without
   depending on a Pi05 model or robot inputs. It retains the causal-text,
   image-prefix contract used by the Pi05 examples; it does not switch to the
   native Hugging Face PaliGemma processor's bidirectional prompt convention.
   For native Qwen VLMs,
   use `ProcessVLMSFTInputs` with the matching `processor_path` and
   `processor_type`, as in [the Qwen config](../configs/vlm/qwen3vl_2b_llava665k_full_finetune.py).
4. `VLMSFTCollator` batches both Pi05 and native Qwen VLM inputs. Mixed training
   keeps a separate collator for the VLA stream. Metadata and raw conversations
   are not passed to the model. All assistant replies and their end tokens are
   supervised; user/system tokens and padding have label `-100`.

The former `ProcessPi05VLMSFTInputs` config type remains a registry alias for
loading saved configs; new configs use `ProcessPaliGemmaVLMSFTInputs`. When
resuming an existing mixed-training checkpoint, retain its saved config type
because the resume contract includes the original dataset configuration.
Exported preprocessing metadata is named `paligemma_vlm_preprocessing.json`.

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
Configure `VLMSFTCollator` with `image_layout='views'` for Pi05 image padding;
the default `image_layout='patches'` concatenates native Qwen patches. The
explicit layout avoids confusing raw `images` path metadata with encoded
tensors. A VLM batch must use one visual layout: conflicting tensor fields
or tensor images in patch mode are rejected rather than silently dropped.

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
  configs/vlm/qwen3vl_2b_llava665k_full_finetune.py ./work_dirs/qwen3vl_2b_llava665k_full_finetune

# Pi05 robot-action and language training with separate data streams.
NPROC_PER_NODE=2 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py \
  ./work_dirs/pi05_robocasa_llava665k_mixed
```

The [RoboTwin mixed config](../configs/pi05/pi05_paligemma_robotwin_llava665k_mixed_full_finetune.py)
uses the same raw language pipeline with the existing RoboTwin robot dataset.

- Pure SFT uses `runner.train_task='vlm'`. Pi05 freezes unused action parameters
  but retains them in checkpoints for later policy use.
- Both Qwen configs follow `model_llava665k_full_finetune.py` naming. The
  [configurable Qwen recipe](../configs/vlm/qwen_llava665k_full_finetune.py)
  retains the `FLUXVLA_VLM_*` environment overrides. Both Qwen recipes and
  pure Pi05 SFT run **one full epoch** (`max_epochs=1`, `max_steps=None`),
  saving at each epoch end and retaining at most two checkpoints. Their
  per-device batch **4** and accumulation **2** give global batch **16** on
  two GPUs; keep the GPU count fixed for comparison. The runner derives the
  optimizer/scheduler budget from the dataset size, rounded up to whole
  updates. Increase `runner.max_epochs` for longer runs. Epoch mode uses
  `save_epoch_interval`, not `save_iter_interval`.
  Compared with batch 1 / accumulation 8, this performs two rather than eight
  serial microbatches per update, but larger batches need more activation and
  padding memory. If memory is insufficient, use batch 2 / accumulation 4 or
  batch 1 / accumulation 8 to preserve the same effective batch size.
- Mixed training uses `runner.train_task='mixed'`, `train_dataloader` for VLA
  and `vlm_dataloader` for VLM. The default schedule is
  `('vla', 'vla', 'vla', 'vlm')`: **one task per optimizer update**, synchronized
  across ranks. It does not concatenate VLM fields into a robot-action batch.
  Mixed configs remain step-based (`max_steps` set, `max_epochs=None`) because
  the two streams have independent lengths and epoch counters. They retain
  per-device batch 1 and accumulation 8: global batch 16 on two GPUs for
  either task.
- VLM gradients are normalized by supervised token count; mixed VLA gradients
  by valid action-element count. Loss weights and update ratios are separate
  from dataset-source repeat factors.
- Mixed updates skip task-inactive parameters entirely (including Adam
  momentum, decay and step counts). A fully text-only VLM update also skips
  vision/projector parameters; image usage is combined across all ranks and
  accumulated microbatches before deciding which parameters to skip.
- All five raw-data examples use **4 workers per rank per training stream**,
  matching the existing robot configs. Mixed training on two GPUs therefore
  uses up to 16 data-loader workers (2 ranks × 2 streams × 4). Reduce workers
  for CPU/RAM-constrained machines, or set them to 0 when debugging transforms.
- Mixed streams restart and reseed workers at each task's epoch boundary.
  This adds worker-startup overhead but makes random crop/color augmentation
  reproducible on resume by replaying the saved epoch/offset. Ordinary VLA
  and pure SFT loader behavior is unchanged.
- Pi05 is the supported mixed-training model. Models without this contract
  reject `train_task='mixed'` rather than silently ignoring the task.
- Full mixed `.pt` checkpoints retain optimizer/scheduler state, task counters,
  data cursors and per-rank RNG state. Resume requires matching data, schedule,
  collator configs, worker counts, batch sizes, trainable parameters and world
  size. Mixed checkpoint contract **version 2** also fixes inactive language-tail
  Adam updates during VLA steps. Older mixed full-resume checkpoints are rejected:
  their optimizer/worker RNG trajectories are incompatible. Load their model
  weights as pretrained initialization for a **new run** instead. Do not edit
  the saved contract to bypass this check. Compatible version-2 resumes must
  still preserve worker counts and preprocessing configuration.
  `.safetensors` is weights only, not a full resume checkpoint.
- Optional held-out language validation uses `VLMEvaluator` and a separate
  dataset with `max_retry_times=0`. It reports teacher-forced loss and token
  accuracy, not benchmark generation scores. Mixed configs keep their existing
  robot evaluator; pure SFT does not require robot evaluation data.

## Mixed-training regression checks

No training data or pretrained checkpoints are needed for these checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest --cpu-model-tests @test/cpu-tests.txt -q

# Requires two CUDA GPUs and the training environment. Uses real tiny Pi05,
# DDP/FSDP, optional activation checkpointing and two-microbatch accumulation.
torchrun --standalone --nproc-per-node=2 \
  test/distributed/mixed_pi05_regression.py \
  --output ./work_dirs/mixed-regression.json
```

The CPU checks include actual random crop/color jitter replay across epochs
with 0/2/4 workers, map-style and repeating robot datasets, and worker cleanup.
The GPU check covers asymmetric image/text-only ranks, fully text-only updates,
and images present only in the first accumulated microbatch. Its reference
uses autograd's unused-parameter decisions independently of the mixed helper;
it checks weights, exact inactive non-updates and Adam step counts. FP32 weight
tolerances are `atol=3e-6, rtol=1e-4`. `--isolated-imports` is available only for
minimal test environments; the default exercises production package imports.

## TextVQA generation evaluation

The pure [Pi05 LLaVA config](../configs/pi05/pi05_paligemma_llava665k_full_finetune.py)
and both [Qwen configs](../configs/vlm/qwen3vl_2b_llava665k_full_finetune.py)
set `eval.type='TextVQAEvalRunner'`. This
is a **generation benchmark**, not the optional training-time `VLMEvaluator`
loss hook. All raw LLaVA training records remain in the SFT dataset: there is
no extra train/validation split of the training annotations.

Prepare the official TextVQA validation annotation separately:

```bash
mkdir -p ./datasets/textvqa
curl -fL --retry 3 \
  https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json \
  -o ./datasets/textvqa/TextVQA_0.5.1_val.json
```

Images are reused from `./datasets/llava665k/textvqa/train_images/`, extracted
from TextVQA's **train_val_images.zip** as described in the README. The JSON
contains questions and ten human answers per question; do **not** add it to
the SFT annotation sources. Missing annotations/images fail explicitly rather
than being replaced by training samples.

```bash
# Train, save the checkpoint, then launch a fresh distributed TextVQA job.
NPROC_PER_NODE=2 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_llava665k_full_finetune.py \
  ./work_dirs/pi05_vlm_sft --eval-after-train

# Qwen uses its own native processor/chat template with the same benchmark.
NPROC_PER_NODE=2 bash scripts/train.sh \
  configs/vlm/qwen3vl_2b_llava665k_full_finetune.py \
  ./work_dirs/qwen3vl_2b_llava665k_full_finetune --eval-after-train

# Short connectivity check on the full model and original raw data.
# Batch=1 / accumulation=1 is a low-memory check, not the default batch.
# Clear max_epochs when overriding the recipe to a step-based smoke run.
NPROC_PER_NODE=2 bash scripts/train.sh \
  configs/pi05/pi05_paligemma_llava665k_full_finetune.py \
  ./work_dirs/pi05_textvqa_smoke --eval-after-train --cfg-options \
  runner.max_epochs=None runner.max_steps=2 runner.save_iter_interval=2 \
  runner.grad_accumulation_steps=1 train_dataloader.per_device_batch_size=1 \
  train_dataloader.dataset.max_retry_times=0 \
  eval.max_samples=32 eval.run_dir=./work_dirs/pi05_textvqa_smoke/textvqa

# Independently evaluate a trained checkpoint (substitute its path).
NPROC_PER_NODE=2 bash scripts/eval.sh \
  configs/pi05/pi05_paligemma_llava665k_full_finetune.py \
  ./work_dirs/pi05_vlm_sft/checkpoints/latest-checkpoint.safetensors

# Pretrained baseline only, without needing a training checkpoint.
NPROC_PER_NODE=2 bash scripts/eval.sh \
  configs/pi05/pi05_paligemma_llava665k_full_finetune.py base

# Qwen base-only evaluation (use a saved checkpoint path for comparison).
NPROC_PER_NODE=2 bash scripts/eval.sh \
  configs/vlm/qwen3vl_2b_llava665k_full_finetune.py base
```

- Inputs are **image + question**, plus the configured short-answer suffix.
  Reference answers never enter the model input. There are no external OCR
  tokens. Greedy decoding uses a KV cache, with at most 32 new tokens.
- Scoring uses the LLaVA/EvalAI answer normalizer and leave-one-annotator-out
  TextVQA accuracy, not token accuracy, loss or perplexity. Scores of one to
  four matching annotators are 0.3, 0.6, 0.9 and 1.0 respectively. This is not
  the OCR-assisted LLaVA evaluation protocol.
- By default `eval.compare_base=True` evaluates both the source model weights
  and the trained checkpoint on the same questions. Set it to `False` to run
  only the trained model. Trained weights load strictly with FluxVLA keys,
  without reapplying the source OpenPI mapping. `.safetensors` and full `.pt`
  exports are supported; merge LoRA adapters before using this runner.
- Qwen selects `eval.preprocessing.type='ProcessVLMSFTInputs'`, matching its
  training processor, image pixel limit and maximum input length. Generation
  adds the native assistant header, disables thinking, and preserves native
  patch/grid shapes. Its pretrained baseline comes from
  `model.vlm_backbone.model_path`; Pi05 keeps its PaliGemma preprocessing.
- Each GPU holds one model replica and a disjoint set of questions. No
  duplicate padding questions are scored. `eval.max_samples` limits the
  **global** number of questions; `None` evaluates all 5,000 validation
  questions. A small subset is only a smoke test, not a benchmark result.
- `eval.run_dir` receives `base_predictions.jsonl`,
  `checkpoint_predictions.jsonl`, `summary.json`, `summary.md` and the config.
  Reports include predictions, per-question scores, dataset/tokenizer hashes,
  precision, decoding settings and the base/checkpoint accuracy difference.
  Choose a new output directory to preserve an earlier evaluation report.

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
  test/test_runners/test_textvqa_eval_runner.py \
  test/test_runners/test_mixed_training.py -q
```

The CPU import-isolation mode is the repository's unit-test harness, not a
production runtime. Distributed integration must also be tested with the
normal training entry point, real weights and available media. Short training
smoke tests do not establish full-run stability, convergence or robot success.
