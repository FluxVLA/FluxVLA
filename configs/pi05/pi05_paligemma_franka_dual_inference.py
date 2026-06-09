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

# Import model and training config
_base_ = './pi05_paligemma_franka_dual_full_finetune.py'

# Completely override inference config (use _delete_=True to prevent merging)
inference = dict(
    _delete_=True,  # This prevents merging with base config
    type='FrankaInferenceRunner',
    task_descriptions={
        '1':
        'The right arm picks up the shuttlecock bucket, hands it to the left arm, and places it on the plate.'  # noqa: E501
    },
    seed=7,
    action_mode='joint',
    # Prepare joints: [left_arm_joints, right_arm_joints]
    # Each arm: [joint1..joint7, gripper_width]
    prepare_pose=None,  # Set to None to home, or provide joints to enable
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
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
        action_dim=16,
    ),
    action_chunk=50,
    operator=dict(
        type='FrankaDualOperator',
        command_mode='joint',
        img_left_topic='/camera_left_wrist/color/image_raw',
        img_right_topic='/camera_right_wrist/color/image_raw',
        img_front_topic='/camera_front/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_arm_right_topic='/right_arm/joint_states',
        cartesian_cmd_left_topic=(
            '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
        cartesian_cmd_right_topic=(
            '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
        joint_cmd_left_topic=(
            '/left_arm/joint_ruckig_smooth_position_controller/target_joint_state'),  # noqa: E501
        joint_cmd_right_topic=(
            '/right_arm/joint_ruckig_smooth_position_controller/target_joint_state'),  # noqa: E501
        gripper_left_topic='/left_arm/franka_gripper/move/goal',
        gripper_right_topic='/right_arm/franka_gripper/move/goal',
        home_service='/cmd/home',
        auto_switch_controller=True,
    ))
