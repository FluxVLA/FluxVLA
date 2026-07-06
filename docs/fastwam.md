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

FastWAM training configs usually set `load_text_encoder=False` and consume cached `context/context_mask` tensors through `LoadCachedTextEmbedding`. Create the cache before training:

```bash
python tools/fastwam/precompute_text_embeds.py \
  --dataset-dir ./datasets/libero_10_no_noops_lerobotv2.1 \
  --cache-dir ./checkpoints/text_embeds_cache/libero \
  --context-len 128
```

Repeat `--dataset-dir` for multiple dataset roots. Keep the `cache_dir`, `context_len`, and `enc_id` aligned with the `LoadCachedTextEmbedding` transform in your config.

## Related Configs

Common FastWAM configs include:

- `configs/fastwam/fastwam_libero_full_finetune.py`
- `configs/fastwam/fastwam_joint_libero_full_finetune.py`
- `configs/fastwam/fastwam_idm_libero_full_finetune.py`

For RoboCasa or private-data configs, check the config-local `_text_embed_cache_dir`, `_ckpt_root`, and `action_dit_pretrained_path` values before launching training.
