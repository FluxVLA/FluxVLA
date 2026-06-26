import numpy as np


def test_oli_operator_registered():
    import fluxvla.engines.operators  # noqa: F401
    from fluxvla.engines.utils.root import OPERATORS
    assert OPERATORS.get('OliOperator') is not None


def test_rot6d_identity_to_unit_quat():
    from fluxvla.engines.operators.oli_operator import _rot6d_to_quat_xyzw
    rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    quat = _rot6d_to_quat_xyzw(rot6d)
    # identity rotation -> [0, 0, 0, 1] (xyzw)
    assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_oli_runner_registered_and_subclass():
    import fluxvla.engines.runners  # noqa: F401
    from fluxvla.engines.runners.base_inference_runner import \
        BaseInferenceRunner
    from fluxvla.engines.runners.oli_inference_runner import OliInferenceRunner
    from fluxvla.engines.utils.root import RUNNERS
    assert RUNNERS.get('OliInferenceRunner') is not None
    assert issubclass(OliInferenceRunner, BaseInferenceRunner)


def test_oli_config_loads():
    import os

    from mmengine import Config
    path = os.path.join('configs', 'gr00t',
                        'gr00t_eagle_3b_oli_full_finetune.py')
    cfg = Config.fromfile(path)
    assert cfg.inference.type == 'OliInferenceRunner'
    assert cfg.inference.operator.type == 'OliOperator'
    assert cfg.inference.denormalize_action.action_dim == 42
    assert cfg.inference.publish_rate == 30


def test_is_degenerate_rot6d():
    from fluxvla.engines.operators.oli_operator import _is_degenerate_rot6d
    assert _is_degenerate_rot6d([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is True
    assert _is_degenerate_rot6d([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) is True
    assert _is_degenerate_rot6d([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) is False


def test_rot6d_quat_known_rotations():
    from fluxvla.engines.operators.oli_operator import _rot6d_to_quat_xyzw

    # 90 degrees about z
    q = _rot6d_to_quat_xyzw([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert np.allclose(
        np.abs(q), [0.0, 0.0, 0.70710678, 0.70710678], atol=1e-3)
    # 180 degrees about x (exercises non-trace branch)
    q = _rot6d_to_quat_xyzw([1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert np.allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-3)
