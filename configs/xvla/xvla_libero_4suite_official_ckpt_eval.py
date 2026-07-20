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
"""Eval config for the official X-VLA LIBERO checkpoint.

The official X-VLA LIBERO client rotates only `agentview_image` by 180 degrees
and leaves `robot0_eye_in_hand_image` unchanged. FluxVLA-trained XVLA configs
keep the default FluxVLA LIBERO image convention.
"""

_base_ = './xvla_libero_4suite_full_finetune.py'
_XVLA_OFFICIAL_RESOURCE_PATH = './checkpoints/X-VLA-Libero'

inference_model = dict(
    eval_closed_loop_state=True,
    vlm_backbone=dict(dtype='float32'),
)

eval = dict(
    _delete_=True,
    type='LiberoEvalRunner',
    task_suite_name='libero_10',
    model_family='xvla',
    num_trials_per_task=50,
    eval_chunk_size=30,
    resize_size=224,
    num_steps_wait=10,
    controller_use_delta=False,
    seed=42,
    mixed_precision_dtype='float32',
    enable_mixed_precision_training=False,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        allow_private_stats_fallback=True,
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                embodiment_id=3,
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                num_padding_imgs=1,
                rotate_180_keys=['agentview_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                interpolation='bicubic',
                input_sizes=[[3, 224, 224], [3, 224, 224], [3, 224, 224]],
                means=[[123.675, 116.28, 103.53], [123.675, 116.28, 103.53],
                       [123.675, 116.28, 103.53]],
                stds=[[58.395, 57.12, 57.375], [58.395, 57.12, 57.375],
                      [58.395, 57.12, 57.375]],
            ),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_XVLA_OFFICIAL_RESOURCE_PATH,
                    tokenizer_cls='BartTokenizerFast',
                ),
                max_len=50,
                pad_token_id=1,
                use_conversation=False,
            ),
            dict(
                type='LiberoEE6DProprioFromInputs',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                out_key='states',
                target_dim=20,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeXVLALiberoAction',
        gripper_binarize=True,
    ),
)
