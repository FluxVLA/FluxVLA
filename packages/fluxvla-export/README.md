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
fixture, recipe name, and explicit device. This stage provides recipe dispatch but
registers no built-in model; unknown recipes raise an error.

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
