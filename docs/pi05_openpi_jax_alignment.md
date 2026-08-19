# PI0.5 real-robot alignment with OpenPI/JAX

## Reference

- Repository: `Physical-Intelligence/openpi`
- Commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Local checkout: `/root/projects/openpi`
- Minimal JAX environment: `/root/projects/openpi/.venv-jax`
- Source model: `Pi0Config(pi05=True)`
- Source ALOHA fine-tune: `pi05_aloha_pen_uncap`

The commit is intentionally fixed here so future OpenPI changes do not move
the comparison target.

## Alignment matrix

| Layer                  | OpenPI/JAX reference                                                                 | FluxVLA result                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| ALOHA coordinates      | Joint sign changes and PI gripper coordinates                                        | Generic `JointSignTransform` plus thin `OpenPIAlohaGripperCoordinates` adapter            |
| ALOHA action target    | Joint deltas, absolute grippers                                                      | Generic `DeltaActions`/`AbsoluteActions` in both directions                               |
| ALOHA hardware gripper | Model contract is normalized `[0, 1]`                                                | Training-data and ROS ranges are converted at their respective boundaries                 |
| Normalization          | PI0.5 Trossen q01/q99 asset                                                          | Vendored exact asset and key-selectable statistics                                        |
| Images                 | RGB, PIL aspect-preserving resize/pad                                                | RGB ROS decode and exact PIL resize backend                                               |
| Augmentation           | Main-view crop/resize/rotate; all-view color jitter                                  | Same policy and camera scope                                                              |
| Prompt                 | Case-preserving PI0.5 state prompt, 200 tokens                                       | Existing prompt transform verified; every real config uses 200                            |
| Episode tail           | Repeat final action and supervise it                                                 | `supervise_terminal_padding=True`                                                         |
| Flow time              | `Beta(1.5, 1) * 0.999 + 0.001`                                                       | Existing beta sampler retained and regression-tested                                      |
| Flow loss              | MSE over all 32 action dimensions                                                    | Existing `loss_action_dim=32` retained                                                    |
| Precision              | FP32 master parameters, image stem, and external flow MLPs; BF16 transformer compute | FP32 runner parameters plus explicit `openpi_stem_fp32` and `openpi_fp32_flow` boundaries |
| Optimizer              | AdamW `(0.9, 0.95)`, eps `1e-8`, wd `1e-10`, clip `1`                                | Matched, including weight decay on every parameter                                        |
| Schedule               | warmup 1000, peak `2.5e-5`, cosine to `2.5e-6` at 30000                              | Matched OpenPI scheduler                                                                  |
| Training length        | 20000 optimizer steps                                                                | `max_steps=20_000`                                                                        |
| Batch                  | Global batch 64                                                                      | Explicit 4-GPU recipe: `8 x 4 x 2` accumulation                                           |
| Seed                   | Training seed 42                                                                     | Explicit runner seed 42, propagated by `scripts/train.py`                                 |
| Checkpoint policy      | EMA 0.99 is used for inference                                                       | EMA checkpoint weights plus resumable raw training weights                                |

The aligned settings are applied to the ALOHA, Franka, Tron2, and UR3
real-robot configs. The standalone ALOHA config additionally uses the released
Trossen statistics and exact robot-coordinate adapter. Other robots continue
to use statistics computed from their own datasets because OpenPI does not
publish compatible statistics for them.

The checked-in ALOHA recipe maps its existing `-0.01--0.08` HDF/ROS gripper
range to the model's 0--1 contract before applying OpenPI's ALOHA transform.
Set `gripper_input_range` and `gripper_output_range` on the inference
transforms to the actual ROS interface
units when a station uses different limits.

## Compute robot-specific normalization statistics

Use `tools/compute_pi05_norm_stats.py` whenever the training dataset, action
semantics, action horizon, or terminal-padding policy changes. The script reads
one or more local LeRobot dataset roots and prints a Python dictionary that can
be pasted directly into the config as `dataset_statistics`; JSON is not needed.
Run it inside the FluxVLA environment (`conda activate fluxvla`) so NumPy and
PyArrow are available.

The script follows the PI0.5 ordering: coordinate/sign conversion, then
absolute-to-delta conversion for selected joints, then statistics. It computes
both mean/std and exact q01/q99 values over the complete action chunks. PI0.5
configs use q01/q99 quantile normalization. Large datasets use temporary
memory-mapped files; pass `--temp-dir /path/with/free-space` when `/tmp` is too
small.

For example, recompute dual-Franka joint-position statistics:

```bash
python tools/compute_pi05_norm_stats.py \
  /path/to/dual_franka_lerobot \
  --profile franka-qpos \
  --action-horizon 50 \
  --variable-name _PI05_FRANKA_QPOS_STATS \
  --output /tmp/pi05_franka_qpos_stats.py
```

The corresponding robot profiles are:

```bash
# ALOHA: sign conversion, OpenPI gripper coordinates, joint deltas.
python tools/compute_pi05_norm_stats.py /path/to/aloha_lerobot \
  --profile aloha --gripper-input-range=-0.01,0.08 \
  --action-horizon 50 --variable-name _OPENPI_TROSSEN_STATS

# UR joint commands: six joint deltas and one absolute gripper.
python tools/compute_pi05_norm_stats.py /path/to/ur_lerobot \
  --profile ur3 --action-horizon 50 \
  --variable-name _PI05_UR3_STATS

# Dual Franka Cartesian poses remain absolute. Quaternion components must not
# be subtracted independently.
python tools/compute_pi05_norm_stats.py /path/to/dual_franka_lerobot \
  --profile franka-eepose --action-horizon 50 \
  --variable-name _PI05_FRANKA_EEPOSE_STATS

# Tron2: sixteen joint deltas and two absolute grippers.
python tools/compute_pi05_norm_stats.py /path/to/tron2_lerobot \
  --profile tron2 --action-horizon 50 \
  --variable-name _PI05_TRON2_STATS
```

For an absolute-action dataset without a built-in profile, specify its parquet
keys explicitly:

```bash
python tools/compute_pi05_norm_stats.py /path/to/lerobot_dataset \
  --state-key observation.state --action-key action --no-delta \
  --action-horizon 16 --statistic-name my_dataset \
  --variable-name _PI05_DATASET_STATS
```

By default, the last action is repeated and included in the statistics, which
matches configs with `supervise_terminal_padding=True`. Add
`--exclude-terminal-padding` when repeated terminal actions are masked from the
loss. The default `--window-start-index 0` matches OpenPI's action offsets;
set it explicitly to the config's `window_start_idx` if a different data
contract is intentional. Always run the command on the exact paths used by the
config; statistics
from another recording, robot calibration, gripper range, or placeholder
`*_example` directory are not interchangeable. After copying the generated
dictionary into the config, pass it to `DistributedRepeatingDataset` through
`dataset_statistics`. Joint-delta inference must use
`DenormalizeDeltaAction` with the same mask so the current raw state is added
back after denormalization.

The checked-in configs currently use the following sources:

- dual-Franka qpos/eepose: recomputed from the exact `20260519` data root and
  embedded in each config;
- RoboCasa: recomputed from all 24 configured full-data roots (24,000
  episodes) with a 16-step supervised horizon and embedded in the config;
- LIBERO 10/goal/object/spatial: recomputed from each configured local data
  root, excluding masked terminal padding, and embedded in the configs;
- ALOHA: keeps the released OpenPI Trossen q01/q99 values;
- UR3, Tron2, and single-Franka example configs: their checked-in dataset
  paths are placeholders and were not present on the validation machine. Run
  the matching command above on the actual training path before switching
  those recipes to delta actions. Never substitute a different local recording
  merely because its dimensions match.

## Verification

FluxVLA checks:

```bash
/root/miniconda3/envs/fluxvla/bin/pytest -q \
  test/test_engines/test_fsdp_checkpointing.py \
  test/test_datasets/test_openpi_pi05_alignment.py \
  test/test_models/test_pi0_time_sampling.py
```

This covers source ALOHA transform round trips, PI0.5 prompt formatting,
image shape/range and camera augmentation scope, FP32 flow boundaries, beta
time sampling, OpenPI schedule endpoints, EMA update/swap/checkpoint semantics,
and FSDP checkpoint safety.

A direct comparison against the fixed OpenPI source also passed:

- ALOHA state/action coordinate transforms agree within `2e-7` after the
  source output is cast to JAX's effective float32 dtype.
- PIL aspect-preserving resize/pad is bit-exact on a non-square uint8 image.

OpenPI/JAX smoke training:

```bash
cd /root/projects/openpi
PYTHONPATH=/root/projects/openpi/src:/root/projects/openpi/packages/openpi-client/src \
LD_LIBRARY_PATH=/root/miniconda3/envs/fluxvla/lib/python3.10/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/fluxvla/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  .venv-jax/bin/python scripts/train.py debug_pi05 \
  --batch-size 4 --num-workers 0 --num-train-steps 1 --fsdp-devices 4
```

The `debug_pi05` path is OpenPI's own PI0.5 JAX data, model, flow-loss,
gradient, AdamW, EMA, and checkpoint pipeline. It uses the source repository's
dummy Gemma widths so it is suitable as a repeatable installation smoke test.

Verified result on 2026-08-11:

- JAX/JAXlib `0.5.3`, Flax `0.10.2`, Orbax `0.11.13`
- GPU backend with four `NVIDIA RTX PRO 5000 72GB Blackwell` devices
- `loss=2.2711`, `grad_norm=0.9593`, `param_norm=477.8021`
- Orbax checkpoint finalized successfully at
  `/root/projects/openpi/checkpoints/debug_pi05/debug_pi05/0`
- OpenPI source tests: 4 PI0 configuration tests and 10 transform tests pass

The minimal environment reuses FluxVLA's Python 3.10 Torch/CUDA packages and
contains a local compatibility shim for two Python 3.11 standard-library APIs
used by this OpenPI/Orbax revision. OpenPI model, data, loss, optimizer, and
checkpoint source files are unmodified.

## Equivalence boundary

The deterministic resize, ALOHA coordinate formulas, normalization formulas,
prompt construction, flow objective, optimizer hyperparameters, schedule, and
EMA semantics are aligned. The following are intentionally not claimed as
bitwise equivalent:

- NumPy/OpenCV augmentation and JAX/augmax use different RNGs and resampling
  kernels, although their distributions and camera scope match.
- PyTorch and JAX transformer kernels are expected to have normal BF16
  numerical drift.
- `PI05FlowMatchingRTCInference` is a BF16 Triton deployment fast path with a
  48-token allocation; use the standard ALOHA inference model for parity
  investigations.
- A full same-checkpoint tensor-by-tensor comparison requires the converted
  FluxVLA `pi05_base` checkpoint. The current local checkpoint path is a broken
  symlink and the remote model host was unreachable, so that final numerical
  check still requires the file to be supplied locally.

For real-robot validation, compare OpenPI and FluxVLA with the same episodes,
task text, Trossen statistics, seeds, global batch 64, and optimizer-step
budget. Log normalized inputs, sampled time/noise, pre-reduction loss, gradient
norm, and denormalized action chunks before comparing success rate.
