# FastWAM

FluxVLA's FastWAM integration uses Wan2.2 video-generation-model components and an ActionDiT action expert for world-action modeling.

## FastWAM Checkpoints

Keep FastWAM dependency weights under `./checkpoints` and point the DiffSynth-style loader at that directory:

```bash
cd /path/to/FluxVLA

export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
```

Recommended local layout:

```text
checkpoints/
├── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
├── Wan-AI/
│   ├── Wan2.1-T2V-1.3B/
│   │   └── google/umt5-xxl/
│   └── Wan2.2-TI2V-5B/
│       ├── Wan2.2_VAE.pth
│       ├── diffusion_pytorch_model*.safetensors
│       └── models_t5_umt5-xxl-enc-bf16.pth
└── text_embeds_cache/  # optional; many configs use an external cache path
```

FastWAM configs use these model repositories:

- `Wan-AI/Wan2.2-TI2V-5B`: Wan2.2 video DiT, Wan2.2 VAE, and Wan text encoder weights.
- `Wan-AI/Wan2.1-T2V-1.3B`: tokenizer files used by online text-encoding eval configs.
- `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`: preprocessed ActionDiT backbone initialization.

### Wan2.2-TI2V-5B

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include "diffusion_pytorch_model*.safetensors" \
  --include "Wan2.2_VAE.pth" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" \
  --local-dir ./checkpoints/Wan-AI/Wan2.2-TI2V-5B
```

`redirect_common_files=False` is the default for FluxVLA FastWAM configs, so the
loader reads the Hugging Face `.pth` VAE and text encoder files from this local
model directory. The loader checks local files only and raises an error if any
required weight is missing.

### Wan2.1-T2V-1.3B

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local-dir ./checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

### ActionDiT Backbone Initialization

Generate the ActionDiT backbone payload expected by FastWAM configs:

```bash
python tools/fastwam/preprocess_action_dit_backbone.py \
  --config configs/fastwam/fastwam_libero_full_finetune.py \
  --output ./checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cpu \
  --dtype float32
```

You can reuse the same output path for the default FastWAM LIBERO configs because they share the 1024-hidden ActionDiT backbone recipe.

## Text Embedding Cache

FastWAM encodes text online by default. The data pipeline tokenizes each task,
and the frozen T5 encoder only runs when a prompt is not already cached. The
default LIBERO configs keep embeddings in a per-process CPU-memory LRU cache
and do not read or write cache files. No precompute command is required. The
cache is rebuilt after each process restart.

The relevant `vlm_backbone` options are:

- `load_text_encoder=True`
- `text_embed_cache_dir=None`: disable disk caching
- `text_embed_cache_device='cpu'`: keep cached embeddings in system RAM
- `text_embed_cache_size`: maximum number of in-memory prompt embeddings
- `text_embed_cache_context_len` and `text_embed_cache_enc_id`: disk-cache
  compatibility fields, used only if disk caching is enabled

One 128x4096 BF16 embedding uses roughly 1 MiB. A four-suite LIBERO run with
about 40 unique tasks therefore uses roughly 40 MiB of CPU RAM per rank.

To persist embeddings across restarts, set `text_embed_cache_dir` to a writable
directory. The old precompute tool can then optionally warm that disk cache:

```bash
python tools/fastwam/precompute_text_embeds.py \
  --dataset-dir ./datasets/libero_10_no_noops_lerobotv2.1 \
  --cache-dir ./checkpoints/text_embeds_cache/libero \
  --context-len 128
```

Repeat `--dataset-dir` for multiple dataset roots. Keep `cache_dir`,
`context_len`, and `enc_id` aligned with the model cache options. Existing
precomputed cache files remain compatible when disk caching is enabled.

Online encoding keeps the frozen T5 encoder resident on every training rank
(roughly 11 GiB of additional BF16 weights). If GPU memory is tighter than
startup convenience, keep using the optional precompute tool and the legacy
`LoadCachedTextEmbedding` path instead.

## Related Configs

Common FastWAM configs include:

- `configs/fastwam/fastwam_libero_full_finetune.py`
- `configs/fastwam/fastwam_joint_libero_full_finetune.py`
- `configs/fastwam/fastwam_idm_libero_full_finetune.py`

For RoboCasa or private-data configs, check the config-local `_ckpt_root`,
`text_embed_cache_*`, and `action_dit_pretrained_path` values before launching
training.
