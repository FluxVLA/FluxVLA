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

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'test/data/models/vlas/cosmos3_nano'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints/Cosmos3-Nano'
METADATA_PATH = DATA_DIR / 'metadata.json'
EXPECTED_ACTIONS_PATH = DATA_DIR / 'expected_actions.json'


def _regression_enabled():
    return os.environ.get('RUN_COSMOS3_REGRESSION') == '1'


def _required_assets_exist():
    if not METADATA_PATH.exists() or not EXPECTED_ACTIONS_PATH.exists():
        return False
    metadata = json.loads(METADATA_PATH.read_text())
    return (CHECKPOINT_DIR / 'transformer' / 'config.json').exists() and (
        CHECKPOINT_DIR / metadata['vision_asset']).exists()


@pytest.mark.skipif(
    not _regression_enabled() or not torch.cuda.is_available()
    or not _required_assets_exist(),
    reason=('Set RUN_COSMOS3_REGRESSION=1 with CUDA and the Cosmos3-Nano '
            'checkpoint available to run this full-checkpoint regression.'),
)
def test_cosmos3_nano_inverse_dynamics_checkpoint_regression(tmp_path):
    metadata = json.loads(METADATA_PATH.read_text())
    expected_actions = np.asarray(
        json.loads(EXPECTED_ACTIONS_PATH.read_text()), dtype=np.float32)
    sample = {
        key: metadata[key]
        for key in (
            'name',
            'model_mode',
            'prompt',
            'embodiment_id',
            'raw_action_dim',
            'action_chunk_size',
            'width',
            'height',
            'fps',
            'num_steps',
            'shift',
            'seed',
        )
    }
    sample['vision_path'] = str(CHECKPOINT_DIR / metadata['vision_asset'])
    sample_path = tmp_path / 'sample.json'
    output_dir = tmp_path / 'outputs'
    sample_path.write_text(json.dumps(sample))

    subprocess.run(
        [
            sys.executable,
            'scripts/cosmos3_fluxvla_infer.py',
            '--input',
            str(sample_path),
            '--output-dir',
            str(output_dir),
            '--device',
            'cuda:0',
            '--dtype',
            metadata['dtype'],
            '--max-samples',
            '1',
        ],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=300,
    )

    output_path = output_dir / metadata['name'] / f"{metadata['name']}.json"
    actual_actions = np.asarray(
        json.loads(output_path.read_text()), dtype=np.float32)
    assert tuple(actual_actions.shape) == tuple(
        metadata['expected_action_shape'])
    np.testing.assert_allclose(
        actual_actions,
        expected_actions,
        atol=metadata['atol'],
        rtol=metadata['rtol'])
