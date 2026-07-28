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
"""
Fused position embedding add inplacetriton kernel test on AGX jetson Orin platform,
align with pytorch result, making sure exactly the same.

"""

import gc
import unittest

import pytest
import torch
import torch.nn.functional as F

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from fluxvla.ops.triton.position_embedding import \
    fused_position_embedding_add_inplace


class TestTritonOrin(unittest.TestCase):

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not _TRITON_AVAILABLE, reason='Triton not installed')
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='No CUDA device available')
    def test_fused_position_embedding_add_inplace(self):
        """fused position embedding add inplace triton kernel test"""
        torch.manual_seed(0)
        B, T, D = 2, 16, 128
        device = 'cuda'
        dtype = torch.float32

        x = torch.randn(B, T, D, device=device, dtype=dtype)
        # max_seq_len >= T, token t use embedding t row
        emb_weight = torch.randn(T, D, device=device, dtype=dtype)

        pos_ids = torch.arange(T, device=device)
        ref = x + F.embedding(pos_ids, emb_weight).unsqueeze(0)

        x_triton = x.clone()
        out = fused_position_embedding_add_inplace(x_triton, emb_weight)

        self.assertIs(out, x_triton)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())


if __name__ == '__main__':
    unittest.main()
