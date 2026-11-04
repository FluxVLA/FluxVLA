# VPP2 on RoboDojo

VPP2 runs through FluxVLA's model registry, input transforms, checkpoint loading
and ZMQ serving. FluxThemis prepares the simulator and schedules rollouts.

## Installation and weights

Install FluxVLA following the [installation instructions](../README.md).
The tested inference environment used Python 3.10, PyTorch 2.8.0+cu128,
Transformers 5.3.0, PEFT 0.19.1 and FlashAttention 2.8.3.post1. CUDA PyTorch,
FlashAttention and native extensions must match the runtime; use
`scripts/install_env.sh` for those components. The recorded single-input run
allocated approximately 52 GB of GPU memory; allow additional runtime headroom.

Download the released bundle and frozen Wan encoders, then convert once to
native sharded safetensors from the repository root:

```bash
hf download Haodong082399/VPP2 \
  --revision 04824ddb3aae11f13db50984ed7d540c3b77dd9f \
  --include 'checkpoints/joint2b_s100000/*' \
            'checkpoints/Wan2.1-I2V-14B-480P/*' \
  --local-dir ./checkpoints/vpp2_release

python -m tools.vpp2.convert_vpp2_checkpoint \
  --bundle ./checkpoints/vpp2_release/checkpoints/joint2b_s100000 \
  --out ./checkpoints/vpp2_joint2b_s100000_fluxvla

export VPP2_CHECKPOINT="$PWD/checkpoints/vpp2_joint2b_s100000_fluxvla"
export VPP2_WAN_DIR="$PWD/checkpoints/vpp2_release/checkpoints/Wan2.1-I2V-14B-480P"
```

Conversion validates every head weight name and shape and writes
`dataset_statistics.json` alongside the weights. `VPP2_STATS` can override this
statistics path. The native Python configuration controls runtime architecture
and inference settings. If direct Hugging Face access is unavailable, configure
`HF_ENDPOINT` before downloading and keep the same checkpoint revision.

## Inference service

With the weight variables above set, start the native server:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/distributed_zmq_server.py \
  --config configs/vpp2/vpp2_wan21_14b_robodojo_eval.py
```

The [configuration](../configs/vpp2/vpp2_wan21_14b_robodojo_eval.py) contains the
model, request preprocessing, action denormalization and server settings. Its
report metadata supplies expected episode quotas to the existing result service;
simulator task selection and rollout settings are configured in Themis.

Inputs contain three camera views, the episode anchor, eight history frames
spaced 25 simulator steps apart and the absolute EE16 state. VPP2 predicts 32
absolute EE16 actions using ten denoising steps and sigma shift 1. Action noise
uses `1 + replan_index`; video noise adds 1,000,003. These policy seeds are
independent of environment seeds. The VAE runs in FP32, CLIP in FP16, and the
text encoder and prediction head in BF16.

## Evaluation

Prepare RoboDojo and launch evaluation with Themis's
[`configs/robodojo/vpp2.py`](https://github.com/FluxVLA/FluxThemis/blob/main/configs/robodojo/vpp2.py).
See the [Themis VPP2 instructions](https://github.com/FluxVLA/FluxThemis/blob/main/docs/robodojo.md#vpp2)
for full and subset runs, environment seeds and output locations. The simulator
host does not need FluxVLA's config, weights or Python installation. Start a
fresh inference service for each run; the service exits after an accepted
`run_end`. For interrupted runs, follow the
[Themis resume requirements](https://github.com/FluxVLA/FluxThemis/blob/main/docs/robodojo.md#resume).

## Validation results

The three-seed mean success rate is **32.56% ± 0.32 percentage points**, compared
with the published VPP2 result of **32.26%** (+0.30 percentage points).

This evaluation used one released `joint2b_s100000` checkpoint and environment
seeds 0, 1 and 2, with 54 task variants and 2,100 cases per seed (6,300 total).
Following the [official protocol](https://robodojo-benchmark.com/leaderboard/protocol),
each seed's success rate is the equal-weight mean of the five dimension success rates;
the reported result is the mean and population standard deviation across seeds.
