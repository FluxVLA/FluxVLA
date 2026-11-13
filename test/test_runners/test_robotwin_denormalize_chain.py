# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chained denormalization stages in the RoboTwin evaluation runner."""

import numpy as np
import pytest
import torch

from fluxvla.engines.runners.robotwin_eval_runner import _run_denormalize
from fluxvla.policies.eval_io import RobotwinInferenceIO


def test_array_stage_output_is_returned_or_rewrapped_for_later_stages():
    # A single array-returning stage (legacy configs) is returned unchanged.
    stage = lambda data: np.asarray(data['action']) * 2  # noqa: E731
    out = _run_denormalize([stage], dict(action=np.ones((2, 3))))
    np.testing.assert_array_equal(out, 2)

    # In a chain the array is written back under 'action' while every other
    # key stays available to the following dict stages.
    calls = []

    def denormalize(data):
        calls.append(sorted(data))
        return np.asarray(data['action']) + 1

    def add_state(data):
        calls.append(sorted(data))
        data['action'] = data['action'] + data['state']
        return data

    out = _run_denormalize([denormalize, add_state],
                           dict(action=np.zeros((2, 3)), state=np.full(3, 5)))
    np.testing.assert_array_equal(out, 6)
    assert calls == [['action', 'state'], ['action', 'state']]


def test_single_step_array_output_keeps_chunk_dimension():
    # DenormalizePrivateAction squeezes a [1, D] prediction to [D]; the chain
    # must hand later stages (and the action validator) a [1, D] chunk.
    seen = []

    def squeeze(data):
        return np.asarray(data['action'])[0]

    def record(data):
        seen.append(np.asarray(data['action']).shape)
        return data

    out = _run_denormalize([squeeze], dict(action=np.arange(3.0)[None]))
    assert out.shape == (1, 3)
    _run_denormalize([squeeze, record], dict(action=np.arange(3.0)[None]))
    assert seen == [(1, 3)]


@pytest.mark.parametrize('container', ['single', 'list', 'tuple'])
@pytest.mark.parametrize('horizon', [1, 2])
def test_predictor_preserves_denormalization_chain_and_prefix(
        container, horizon):
    seen = []
    state = np.ones(14, dtype=np.float32)

    def denormalize(data):
        seen.append(np.asarray(data['action']).shape)
        np.testing.assert_array_equal(data['state'], state)
        return np.asarray(data['action'])[0].squeeze() + 10

    def add_state(data):
        seen.append(np.asarray(data['action']).shape)
        data['action'] += data['state']
        return data

    stages = denormalize
    if container == 'list':
        stages = [denormalize, add_state]
    elif container == 'tuple':
        stages = (denormalize, add_state)
    io = RobotwinInferenceIO(
        vla=None,
        dataset=None,
        denormalize_action=stages,
        execute_horizon=horizon,
        unnorm_key='private',
        mixed_precision_dtype=torch.bfloat16,
        enable_mixed_precision=False,
        device=torch.device('cpu'))
    actions = torch.arange(4 * 14, dtype=torch.float32).reshape(1, 4, 14)
    native, _ = io.process_actions(actions, raw_state=state)
    offset = 10 if container == 'single' else 11
    np.testing.assert_array_equal(native,
                                  actions[0, :horizon].numpy() + offset)
    assert native.shape == (horizon, 14)
    assert native.dtype == np.float32
    assert seen[0] == (1, horizon, 14)
    if container != 'single':
        assert seen[1] == (horizon, 14)
