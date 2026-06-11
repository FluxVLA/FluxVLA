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
from typing import Any

import numpy as np

from fluxvla.engines import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def save_dataset_statistics(dataset_statistics, run_dir):
    """Save dataset statistics to `dataset_statistics.json`."""
    out_path = os.path.join(run_dir, 'dataset_statistics.json')
    with open(out_path, 'w') as f_json:
        json.dump(_jsonable(dataset_statistics), f_json, indent=2)
    overwatch.info(f'Saved dataset statistics file at path {out_path}')


def save_grouped_dataset_statistics(grouped_dataset_statistics, run_dir):
    """Save grouped dataset statistics to one JSON file per group."""
    for group_name, group_stats in grouped_dataset_statistics.items():
        out_path = os.path.join(run_dir,
                                f'dataset_statistics_{group_name}.json')
        with open(out_path, 'w') as f_json:
            json.dump(_jsonable(group_stats), f_json, indent=2)
        overwatch.info(f'Saved dataset statistics for group {group_name} '
                       f'at path {out_path}')
