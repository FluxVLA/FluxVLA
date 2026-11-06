# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Real PI0.5 prefill/denoise preparation for the generic export contract."""

from __future__ import annotations
import hashlib
import json
from functools import partial
from pathlib import Path

from ...config import DeploymentConfig, config_to_document
from ...contracts import PreparedDeployment, PreparedGraph, TensorSpec
from ...errors import ContractError
from .executor import _checked_tensors, build_executor, make_runtime_metadata
from .reference import prepare_reference_from_config


class PI05Recipe:
    """Strictly load PI0.5 once and expose its two real tensor-only graphs."""

    name = 'pi05-v1'

    def prepare(
        self,
        *,
        config: DeploymentConfig,
        checkpoint: Path,
        inputs: Path,
        device: str,
    ) -> PreparedDeployment:
        reference = prepare_reference_from_config(
            config=config, checkpoint=checkpoint, inputs=inputs, device=device)
        resolved = reference.config

        import torch

        from .graphs import PI05DenoiseGraph, PI05PrefillGraph

        prefill = PI05PrefillGraph(reference.model).eval()
        denoise = PI05DenoiseGraph(
            reference.model,
            states_dtype=reference.inputs['states'].dtype,
        ).eval()
        policy_specs = {spec.name: spec for spec in resolved.input_specs}
        prefill_inputs = {
            name: reference.inputs[name]
            for name in prefill.input_names
        }
        with torch.no_grad():
            prefix_tensors = prefill(*prefill_inputs.values())
            if not isinstance(prefix_tensors, tuple) or (
                    len(prefix_tensors) != len(prefill.output_names)):
                raise ContractError('PI0.5 prefill returned an invalid flat '
                                    'cache output tuple.')
            prefix = dict(zip(prefill.output_names, prefix_tensors))
            prefill_output_specs = tuple(
                TensorSpec(name,
                           str(value.dtype).removeprefix('torch.'),
                           tuple(value.shape))
                for name, value in prefix.items())
            _checked_tensors(
                prefix,
                prefill_output_specs,
                owner='PI0.5 prepared prefill',
                device=reference.device)
            batch = reference.inputs['noise'].shape[0]
            timestep = torch.tensor(
                1.0, dtype=torch.float32,
                device=reference.device).expand(batch)
            denoise_inputs = {
                'noisy_actions': reference.inputs['noise'].clone(),
                'timestep': timestep,
                **prefix,
            }
            if tuple(denoise_inputs) != denoise.input_names:
                raise ContractError('PI0.5 denoise graph input names do not '
                                    'match the flat cache contract.')
            velocity = denoise(*denoise_inputs.values())
            if not isinstance(velocity, torch.Tensor):
                raise ContractError(
                    'PI0.5 denoise must return a velocity tensor.')
            velocity_spec = TensorSpec(
                'velocity',
                str(velocity.dtype).removeprefix('torch.'),
                tuple(velocity.shape))
            _checked_tensors({'velocity': velocity}, (velocity_spec, ),
                             owner='PI0.5 prepared denoise',
                             device=reference.device)

        action_spec = TensorSpec('actions', 'float32',
                                 policy_specs['noise'].shape)
        runtime = make_runtime_metadata(
            num_steps=resolved.inference.num_steps,
            vocab_size=resolved.model.vocab_size,
            input_specs=resolved.input_specs,
            prefill_output_specs=prefill_output_specs,
            velocity_spec=velocity_spec,
            output_spec=action_spec,
        )
        denoise_input_specs = (
            TensorSpec('noisy_actions', 'float32',
                       policy_specs['noise'].shape),
            TensorSpec('timestep', 'float32', (batch, )),
            *prefill_output_specs,
        )
        # BF16 requires tensor-typed Dynamo lowering and Conv schema 22.
        # Legacy SDPA lowering produces invalid complex casts for BF16.
        # Standard optimization materializes constant reduction axes required
        # by TensorRT; the native dtype boundaries and FP32 masters
        # are retained.
        if resolved.precision.autocast_dtype == 'bfloat16':
            export_options = {
                'opset': 22,
                'dynamo': True,
                'optimize': True,
                'do_constant_folding': False,
            }
        else:
            export_options = {
                'opset': 18,
                'dynamo': False,
                'do_constant_folding': False,
            }
        graphs = {
            'prefill':
            PreparedGraph(
                name='prefill',
                module=prefill,
                inputs=prefill_inputs,
                input_specs=tuple(policy_specs[name]
                                  for name in prefill.input_names),
                output_specs=prefill_output_specs,
                export_options=export_options,
            ),
            'denoise':
            PreparedGraph(
                name='denoise',
                module=denoise,
                inputs=denoise_inputs,
                input_specs=denoise_input_specs,
                output_specs=(velocity_spec, ),
                export_options=export_options,
            ),
        }
        document = config_to_document(resolved)
        source_config_sha256 = hashlib.sha256(
            json.dumps(document, sort_keys=True,
                       separators=(',', ':')).encode('utf-8')).hexdigest()
        return PreparedDeployment(
            recipe=self.name,
            graphs=graphs,
            policy_inputs=reference.inputs,
            policy_input_specs=resolved.input_specs,
            output_specs=(action_spec, ),
            reference_runner=reference.run_reference,
            executor_factory=partial(build_executor, metadata=runtime),
            metadata={
                'config_schema_version': resolved.schema_version,
                'source_config_sha256': source_config_sha256,
                'checkpoint_sha256': resolved.checkpoint_sha256,
                'inputs_sha256': resolved.inputs_sha256,
                'precision_policy': document['precision'],
                'state_semantics': resolved.preprocessing.state_semantics,
                'action_semantics': resolved.preprocessing.action_semantics,
                'states_values':
                'unused by PI0.5; states dtype is baked into denoise',
                'runtime': runtime,
            },
        )
