# fluxvla-export

Independent tensor, graph and deployment handoff contracts for FluxHermes.
FluxVLA owns model-specific computation; FluxHermes owns ONNX export and TensorRT
builds. This package has no runtime dependencies and does not install FluxVLA's
training stack.

The current package provides `TensorSpec`, `PreparedGraph`, `PreparedDeployment`
and engine/executor protocols. Model implementations, configuration/artifact
readers and `prepare_tensorrt` are not included yet.

Input/output mappings follow specification order. Callers own module mode and
backend option validation.

## Development

From the repository root, using Python 3.10–3.12:

```bash
python -m pip install ./packages/fluxvla-export
python -m unittest discover -s packages/fluxvla-export/tests -v
```

Real CPU tensor/module handoff tests require a separately installed PyTorch:

```bash
python -m unittest discover -s packages/fluxvla-export/tests_torch -v
```
