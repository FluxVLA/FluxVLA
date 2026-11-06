# Copyright 2026 Limx Dynamics
"""Persistent current-training/factory regression with real tensors.

Run under --cpu-model-tests. These tests intentionally compare current classes;
independent ORIGINAL-revision evidence is produced separately by the capture /
compare CLI in export_policy_probe.py, not invented by this same-tree test.
"""

import pytest
import torch
from safetensors.torch import save_file

from fluxvla_export.checkpoint import load_checkpoint
from fluxvla_export.config import load_config
from fluxvla_export.recipes.pi05.model import build_pi05_model
from fluxvla_export.recipes.pi05.reference import prepare_reference
from test.export_policy_probe import (INPUT_NAMES, cached_evidence, case_specs,
                                      deployment_document, previous)


@pytest.mark.parametrize('spec', case_specs(), ids=lambda spec: spec['name'])
def test_real_training_checkpoint_factory_and_strict_reference(spec, tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(previous.SEED)
        training = previous.build_case_model(spec)
        state = previous._clone_tensors(training.state_dict())
        inputs = previous.fixture_inputs('policy')
        expected, expected_details = cached_evidence(training, spec, inputs)
        checkpoint = tmp_path / 'state.safetensors'
        fixture = tmp_path / 'inputs.safetensors'
        config_path = tmp_path / 'deployment.json'
        save_file(state, str(checkpoint))
        save_file({name: inputs[name] for name in INPUT_NAMES}, str(fixture))
        previous._write_json(
            config_path,
            deployment_document(training, inputs, checkpoint, fixture))
        config = load_config(config_path, recipe='pi05-v1')

        torch.manual_seed(previous.SEED)
        independent = build_pi05_model(config)
        assert list(independent.state_dict()) == list(state)
        previous._compare_tensors(independent.state_dict(), state,
                                  'native initialization', 0, 0)
        weights = load_checkpoint(
            checkpoint,
            expected_sha256=config.checkpoint_sha256,
            expected_state=independent.state_dict())
        independent.load_state_dict(weights, strict=True)
        actual, actual_details = cached_evidence(independent, spec, inputs)
        previous._compare_tensors(actual, expected, spec['name'], 0, 0)
        assert actual_details == expected_details

        rng = torch.random.get_rng_state().clone()
        reference = prepare_reference(
            config=config_path,
            checkpoint=checkpoint,
            inputs=fixture,
            device='cpu')
        expected_inputs = {name: inputs[name] for name in INPUT_NAMES}
        previous._compare_tensors(reference.model.state_dict(), state,
                                  'strict checkpoint load', 0, 0)
        previous._compare_tensors(reference.inputs, expected_inputs,
                                  'strict fixture load', 0, 0)
        for _ in range(2):
            previous._compare_tensors(reference.run_reference(),
                                      {'actions': expected['final_action']},
                                      'reference final action', 0, 0)
        previous._compare_tensors(reference.model.state_dict(), state,
                                  'reference state preservation', 0, 0)
        previous._compare_tensors(reference.inputs, expected_inputs,
                                  'reference fixture preservation', 0, 0)
        original_inputs = previous._clone_tensors(inputs)
        with torch.no_grad(), torch.autocast(
                'cpu', dtype=torch.bfloat16, enabled=spec['amp']):
            actions = independent(*(inputs[name] for name in INPUT_NAMES))
        torch.testing.assert_close(
            actions, expected['final_action'], rtol=0, atol=0)
        assert actions.data_ptr() != inputs['noise'].data_ptr()
        previous._compare_tensors(inputs, original_inputs,
                                  'six-input forward fixture preservation', 0,
                                  0)
        previous._compare_tensors(independent.state_dict(), state,
                                  'six-input forward state preservation', 0, 0)
        torch.testing.assert_close(
            torch.random.get_rng_state(), rng, rtol=0, atol=0)
