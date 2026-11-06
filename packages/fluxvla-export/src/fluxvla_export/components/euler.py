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
"""Original PI plain Euler loop, independent of models and Transformers."""

import torch


def plain_euler(x_t, denoise, bsize, dt, device):
    """Denoise in place with the policy's original FP32 time/termination rule.

    ``dt`` is the original FP32 tensor ``-1.0 / num_steps``. Callers that own
    reusable fixture noise must clone it before calling; this function retains
    the native policy's mutation and float32 accumulation semantics.
    """
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        t_batch = time.expand(bsize)
        v_t = denoise(x_t, t_batch)
        x_t += dt * v_t
        time += dt
    return x_t
