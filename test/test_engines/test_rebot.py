from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'configs/pi05/pi05_rebot_dual_inference.py'


def test_rebot_components_registered():
    import fluxvla.engines.operators  # noqa: F401
    import fluxvla.engines.runners  # noqa: F401
    from fluxvla.engines.operators.aloha_operator import AlohaOperator
    from fluxvla.engines.operators.base_operator import BaseOperator
    from fluxvla.engines.operators.rebot_dual_operator import RebotDualOperator
    from fluxvla.engines.runners.aloha_inference_runner import \
        AlohaInferenceRunner
    from fluxvla.engines.runners.base_inference_runner import \
        BaseInferenceRunner
    from fluxvla.engines.runners.rebot_dual_inference_runner import \
        RebotDualInferenceRunner
    from fluxvla.engines.utils.root import OPERATORS, RUNNERS

    assert OPERATORS.get('RebotDualOperator') is not None
    assert RUNNERS.get('RebotDualInferenceRunner') is not None
    assert issubclass(RebotDualOperator, BaseOperator)
    assert not issubclass(RebotDualOperator, AlohaOperator)
    assert issubclass(RebotDualInferenceRunner, BaseInferenceRunner)
    assert not issubclass(RebotDualInferenceRunner, AlohaInferenceRunner)


def test_rebot_config_is_standalone_and_uses_14d_contract():
    source = CONFIG_PATH.read_text(encoding='utf-8')
    cfg = Config.fromfile(CONFIG_PATH)

    assert '_base_' not in source
    assert cfg.inference.type == 'RebotDualInferenceRunner'
    assert cfg.inference.operator.type == 'RebotDualOperator'
    assert cfg.inference_model.ori_action_dim == 14
    assert cfg.inference.state_dim == 14
    assert cfg.inference.dataset.img_keys == [
        'cam_front', 'cam_wrist_left', 'cam_wrist_right'
    ]
    assert cfg.inference.denormalize_action.type == 'DenormalizeDeltaAction'
    assert cfg.inference.dataset.transforms[0].state_key == 'proprio'
    assert 'norm_stats_path' not in cfg.inference
    assert 'norm_stats_key' not in cfg.inference.dataset
    assert 'action_stats_key' not in cfg.inference.denormalize_action
    assert 'pick_place' not in source
    expected_mask = [True] * 6 + [False] + [True] * 6 + [False]
    assert cfg.inference.denormalize_action.delta_action_mask == expected_mask


def test_rebot_runner_preserves_continuous_gripper_angles():
    from fluxvla.engines.runners.rebot_dual_inference_runner import \
        RebotDualInferenceRunner

    runner = object.__new__(RebotDualInferenceRunner)
    expected = np.zeros((2, 14), dtype=np.float32)
    expected[:, [6, 13]] = -4.7
    runner._use_remote = False
    runner.action_chunk = 2
    runner._action_ctx = SimpleNamespace(state=np.zeros(14))
    runner.denormalize_action = lambda _: expected.copy()

    actual = runner._postprocess_actions(torch.zeros((1, 2, 32)))

    assert np.array_equal(actual, expected)


def test_rebot_uses_standard_private_statistics_layout():
    from fluxvla.transforms.normalize import DenormalizeDeltaAction

    mask = [True] * 6 + [False] + [True] * 6 + [False]
    stats = {
        'private': {
            'action': {
                'q01': [-1.0] * 14,
                'q99': [1.0] * 14,
            }
        }
    }
    transform = DenormalizeDeltaAction(
        norm_stats=stats,
        norm_type='quantile',
        action_dim=14,
        delta_action_mask=mask)
    state = np.arange(14, dtype=np.float32)

    actions = transform({
        'action': np.zeros((1, 2, 32), dtype=np.float32),
        'state': state,
    })

    expected = np.broadcast_to(np.where(mask, state, 0.0), (2, 14))
    assert np.array_equal(actions, expected)


def test_rebot_operator_publishes_canonical_joint_names():
    from fluxvla.engines.operators.rebot_dual_operator import RebotDualOperator

    class JointState:

        def __init__(self):
            self.header = SimpleNamespace(stamp=None)
            self.name = []
            self.position = []

    class Publisher:

        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    operator = object.__new__(RebotDualOperator)
    operator.left_joint_pub = Publisher()
    operator.right_joint_pub = Publisher()
    rospy = SimpleNamespace(Time=SimpleNamespace(now=lambda: 123.0))
    sensor_msgs = SimpleNamespace(msg=SimpleNamespace(JointState=JointState))

    with patch.dict(
            'sys.modules', {
                'rospy': rospy,
                'sensor_msgs': sensor_msgs,
                'sensor_msgs.msg': sensor_msgs.msg,
            }):
        operator.send_joints({'left': range(7), 'right': range(7, 14)})

    expected_names = [
        'shoulder_pan',
        'shoulder_lift',
        'elbow_flex',
        'wrist_flex',
        'wrist_yaw',
        'wrist_roll',
        'gripper',
    ]
    left = operator.left_joint_pub.messages[0]
    right = operator.right_joint_pub.messages[0]
    assert left.header.stamp == right.header.stamp == 123.0
    assert left.name == right.name == expected_names
    assert left.position == list(range(7))
    assert right.position == list(range(7, 14))


def test_rebot_operator_reorders_joint_states_by_name():
    from fluxvla.engines.operators.rebot_dual_operator import (
        JOINT_NAMES, RebotDualOperator)

    message = SimpleNamespace(
        name=list(reversed(JOINT_NAMES)), position=list(range(7)))

    positions = RebotDualOperator.joint_positions(message)

    assert np.array_equal(positions, np.arange(6, -1, -1))


def test_rebot_operator_interpolates_explicit_prepare_pose():
    from fluxvla.engines.operators.rebot_dual_operator import (
        JOINT_NAMES, RebotDualOperator)

    operator = object.__new__(RebotDualOperator)
    operator.home_timeout = 1.0
    operator.publish_rate = 30
    operator.arm_steps_length = np.full(7, 0.5, dtype=np.float32)
    joint_state = SimpleNamespace(name=JOINT_NAMES, position=[0.0] * 7)
    operator.get_frame = Mock(return_value={
        'left_arm': joint_state,
        'right_arm': joint_state,
    })
    operator.clear_observation_queues = Mock()
    operator.execute_trajectory = Mock()
    rospy = SimpleNamespace(
        is_shutdown=lambda: False,
        Rate=lambda _: SimpleNamespace(sleep=lambda: None))
    target = np.ones((2, 7), dtype=np.float32)

    with patch.dict('sys.modules', {'rospy': rospy}):
        operator.gohome(target)

    kwargs = operator.execute_trajectory.call_args.kwargs
    assert kwargs['arm_trajectories']['left'].shape == (2, 7)
    assert np.array_equal(kwargs['arm_trajectories']['left'][-1], target[0])
    assert np.array_equal(kwargs['arm_trajectories']['right'][-1], target[1])
    assert kwargs['dt'] == 1.0 / 30.0


def test_rebot_runner_splits_dual_arm_actions():
    from fluxvla.engines.runners.rebot_dual_inference_runner import \
        RebotDualInferenceRunner

    runner = object.__new__(RebotDualInferenceRunner)
    runner.disable_puppet_arm = False
    runner.dt = 1.0 / 30.0
    runner.ros_operator = SimpleNamespace(execute_trajectory=Mock())
    actions = np.arange(42, dtype=np.float32).reshape(3, 14)

    runner._execute_actions(actions, rate=None)

    kwargs = runner.ros_operator.execute_trajectory.call_args.kwargs
    assert np.array_equal(kwargs['arm_trajectories']['left'], actions[:, :7])
    assert np.array_equal(kwargs['arm_trajectories']['right'], actions[:, 7:])
    assert kwargs['dt'] == runner.dt
