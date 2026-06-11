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

_base_ = './pi05_paligemma_franka_dual_full_finetune.py'

model = dict(
    ori_action_dim=8,
    action_loss_weights=[1] * 8,
)

inference_model = dict(
    ori_action_dim=8,
    action_loss_weights=[1] * 8,
)

train_dataloader = dict(
    dataset=dict(
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    './datasets/RealRobot_franka_single_lerobotv2.1'
                ],
                action_key='observation.state',
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_front',
                            'observation.images.cam_wrist_left'
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        }),
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=32,
                        state_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='min_max'),
                    dict(type='PreparePromptWithState'),
                    dict[str, str | dict[str, str]](
                        type='ProcessPrompts',
                        max_len=200,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'checkpoints/pi05_base',
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(type='SimpleNormalizeImages'),
                ],
                action_window_size=50)
        ]))

inference = dict(
    _delete_=True,
    type='FrankaInferenceRunner',
    task_descriptions={
        '1': 'Use the left Franka arm to complete the requested task.'
    },
    seed=7,
    action_mode='joint',
    active_arms=('left', ),
    async_execution=False,
    execute_horizon=50,
    # Single-arm prepare pose: [joint1..joint7, gripper_width]
    prepare_pose=None,
    camera_names=['cam_front', 'cam_wrist_left'],
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_front', 'cam_wrist_left'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_dim=32,
                state_key='proprio',
                action_key='action',
                norm_type='min_max'),
            dict(type='PreparePromptWithState'),
            dict[str, str | dict[str, str]](
                type='ProcessPrompts',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path='checkpoints/pi05_base',
                )),
            dict(type='ResizeImages', height=224, width=224),
            dict(type='SimpleNormalizeImages'),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='min_max',
        action_dim=8,
    ),
    action_chunk=50,
    operator=dict(
        type='FrankaOperator',
        command_mode='joint',
        arm_ns='/left_arm',
        img_left_topic='/camera_left_wrist/color/image_raw',
        img_front_topic='/camera_front/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_franka_state_left_topic=(
            '/left_arm/franka_state_controller/franka_states'),
        sync_warning_enabled=True,
        joint_cmd_topic=(
            '/left_arm/ruckig_joint_impedance_controller/target_joint_state'),
        cartesian_cmd_topic=(
            '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
        gripper_left_topic='/left_arm/franka_gripper/move/goal',
    ))
