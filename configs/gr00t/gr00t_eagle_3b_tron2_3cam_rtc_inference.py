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

_base_ = './gr00t_eagle_3b_tron2_3cam_full_finetune.py'

inference_model = dict(
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        dtype=None,
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        max_input_seq_len=900,
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None)))

inference = dict(
    type='Tron2RTCInferenceRunner',
    async_execution=True,
    execute_horizon=10,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=5,
    ))
