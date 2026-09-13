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

from types import SimpleNamespace

import pytest
import torch

from fluxvla.models.vlas.dreamzero_vla import DreamZeroVLA


class _FakeBackbone:

    def __init__(self):
        self.condition_latent = torch.full((1, 16, 1, 2, 2), 7.0)
        self.encoded_video = torch.full((1, 16, 1, 2, 2), 3.0)
        self.encode_video_calls = 0
        self.last_video = None
        self.last_image = None

    def set_frozen_modules_to_eval_mode(self):
        pass

    def encode_prompt(self, tokens, masks):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    def encode_image(self, image, num_frames, height, width):
        self.last_image = image
        clip = torch.zeros(1, 4)
        image_cond = torch.zeros(1, 20, 5, 2, 2)
        return clip, image_cond, self.condition_latent

    def encode_video(self, video):
        self.last_video = video
        self.encode_video_calls += 1
        return self.encoded_video


class _FakeHead:

    def __init__(self):
        self.model = SimpleNamespace(local_attn_size=-1)
        self.current_start_frame = 0
        self.num_frame_per_block = 2
        self.num_state_per_block = 1
        self.max_state_dim = 64
        self.last_kwargs = None
        self.reset_calls = 0

    def reset_inference_state(self):
        self.current_start_frame = 0
        self.reset_calls += 1

    def predict_action(self, **kwargs):
        self.last_kwargs = kwargs
        self.current_start_frame = max(1, self.current_start_frame) + 2
        return torch.zeros(1, 10, 32)


def _make_vla():
    vla = DreamZeroVLA.__new__(DreamZeroVLA)
    torch.nn.Module.__init__(vla)
    vla.use_cache = True
    vla.frame_window_size = 17
    vla.vlm_backbone = _FakeBackbone()
    vla.vla_head = _FakeHead()
    return vla


def _predict_once(vla, frames=1):
    return vla.predict_action(
        images=torch.arange(frames, dtype=torch.float32).view(
            1, 1, frames, 1, 1).expand(1, 3, frames, 8, 8),
        lang_tokens=torch.ones(1, 4, dtype=torch.long),
        lang_masks=torch.ones(1, 4, dtype=torch.long),
        states=torch.zeros(1, 29),
    )


@pytest.mark.parametrize('frames', [1, 4, 9])
def test_cache_prefill_encodes_one_clean_conditioning_frame(frames):
    vla = _make_vla()
    actions = _predict_once(vla, frames)

    assert actions.shape == (1, 10, 32)
    assert vla.vlm_backbone.encode_video_calls == 1
    assert vla.vlm_backbone.last_video.shape[2] == 1
    assert torch.all(vla.vlm_backbone.last_video == frames - 1)
    assert torch.all(vla.vlm_backbone.last_image == frames - 1)
    expected = vla.vlm_backbone.encoded_video.transpose(1, 2)
    torch.testing.assert_close(vla.vla_head.last_kwargs['latents'], expected)
    assert vla.vla_head.last_kwargs['states'].shape == (1, 1, 64)


def test_cache_continuation_encodes_observations_and_single_frame_resets():
    vla = _make_vla()
    _predict_once(vla, frames=4)
    _predict_once(vla, frames=4)
    torch.testing.assert_close(
        vla.vlm_backbone.last_video[0, 0, :, 0, 0],
        torch.tensor([0., 0., 0., 1., 1., 2., 2., 3., 3.]))
    assert vla.vla_head.reset_calls == 0
    _predict_once(vla, frames=1)
    assert vla.vla_head.reset_calls == 1
    assert vla.vla_head.current_start_frame == 3
    assert vla.vla_head.last_kwargs['observed_latent_frames'] == 1
