# fluxvla-export

Standalone model frontend for FluxHermes. FluxVLA owns model computation and
artifact contracts; FluxHermes owns export, backend builds, and deployment.
Installing this package does not install the FluxVLA training stack.

## Strict artifacts and dispatch

`load_config` validates the complete versioned deployment schema without loading
Torch. `load_checkpoint` and `load_fixture` validate safetensors hashes, complete
keys, shapes, dtypes, and finite values before use. Checkpoints are never unpickled,
renamed, or silently cast. The initial storage contract requires FP32 parameters.

`prepare_tensorrt` requires a validated deployment configuration, checkpoint, input
fixture, recipe name, and explicit device. The `pi05-v1` recipe provides prefill and denoise graph boundaries; unknown
recipes raise an error.

`TensorSpec`, `PreparedGraph`, and `PreparedDeployment` preserve specification
ordering and validate tensor/module boundaries. A valid handoff does not establish
ONNX exportability or backend numerical parity.

## Development

```bash
python -m pip install './packages/fluxvla-export[test]'
python -m unittest discover -s packages/fluxvla-export/tests -v
python -m unittest discover -s packages/fluxvla-export/tests_torch -v
```

Public imports remain dependency-light even when Torch and safetensors are
installed. CI builds and tests the standalone wheel outside the source package.

## Shared components

Registry-free Gemma, SigLIP, linear projection, and tensor helpers are shared
with the training adapters. Training installation must install this local package
before the root project; the installer scripts and CI do so explicitly.

## Native PI0.5 reference

`fluxvla_export.recipes.pi05.reference.prepare_reference` builds the canonical
standalone model from a validated configuration, checkpoint, and fixture. Its
policy mixins and FP32-time Euler loop are shared with ordinary PI0/PI0.5 training
inference. Reference calls preserve caller fixtures by cloning the native mutable
noise input. The lazy `pi05-v1` recipe is registered without importing Torch.

## Graphs and executor

Prefill exposes the flat KV cache; denoise exposes one velocity step. The metadata
executor runs the same Euler loop over engines implementing the handoff protocol.
States values are unused by PI0.5 denoise, but their dtype/device metadata remains
part of the executor contract. Native graph parity does not prove exportability.

## Producing deployment artifacts

`fluxvla_export.recipes.pi05.artifacts.produce_pi05_artifacts` derives architecture
and precision from canonical live modules, validates the checkpoint and fixture,
and writes hash-bound configuration/fixture artifacts to a fresh directory.
Supply preprocessing provenance explicitly; Python training configurations and
business normalization statistics are not guessed. The checkpoint is authoritative
(including EMA checkpoints) and is never rewritten or required to equal live
weights. The producer does not build an engine or validate deployment accuracy.

With an existing canonical eval-mode model, six-input fixture, and complete
`PreprocessingConfig`:

```python
from fluxvla_export.recipes.pi05.artifacts import produce_pi05_artifacts

artifacts = produce_pi05_artifacts(
    model=model,
    inputs=fixture_tensors,
    checkpoint="policy.safetensors",
    preprocessing=preprocessing,
    output_dir="deployment-inputs",  # Must not already exist.
)
```

## Supported ONNX test profile

Install the `test-onnx` extra for the pinned Torch 2.8 exporter stack. FP32 uses
legacy opset18 with constant folding disabled; native BF16 uses Dynamo opset22
with optimization enabled. Explicit bool CumSum promotion and AdaRMS slices retain
native operation ordering and avoid unsupported ONNX types/sequences.

CI requires real export/check tests, native parity, and captured-program replay.
CPU ONNX Runtime lacks BF16 Conv(22); its explicit rejection is tested, not
reported as BF16 inference parity. Torch 2.6 remains a native-only test lane.
