import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.data.schema.embodied_types import (PolicyOutput, Trajectory,
                                              convert_trajectories_to_batch)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from fluxvla.rl.rlinf_registry import register
from fluxvla.rl.train import prepare_environment, validate_frontend_cfg
from fluxvla.rl.workers.rollout import FluxRolloutWorker

ROOT = Path(__file__).resolve().parents[2]


def test_config_composes_and_backend_validation(tiny_assets, monkeypatch):
    import rlinf.config as backend

    monkeypatch.delenv('RLINF_EXT_MODULE', raising=False)
    prepare_environment()
    register()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(
            config_name='benchmarks/libero/pi05/ppo',
            overrides=[f'actor.model.model_path={tiny_assets[2]}'])
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg)
    assert cfg.actor.model.model_type == 'fluxvla_pi05'
    assert cfg.env.train.task_suite_name == 'libero_10'
    assert cfg.weight_syncer.type == 'bucket'
    assert cfg.actor.model.num_action_chunks == 10
    assert cfg.actor.fsdp_config.mixed_precision.cast_forward_inputs is False
    # Resource discovery is mocked; the real embodied schema validator runs.
    monkeypatch.setattr(backend, 'Cluster', lambda *a, **k: None)
    monkeypatch.setattr(
        backend, 'HybridComponentPlacement',
        lambda *a, **k: SimpleNamespace(get_world_size=lambda component: 1))
    validated = backend.validate_embodied_cfg(cfg)
    assert validated.runner.weight_sync_interval == 1


def test_driver_and_fresh_worker_extension_registration(model_cfg):
    register()
    from rlinf.models import get_model

    policy = get_model(model_cfg)
    assert type(policy).__name__ == 'FluxPI05RLPolicy'
    code = '''
from rlinf.scheduler.cluster.utils import load_user_extension_module
load_user_extension_module()
from rlinf.models import _MODEL_REGISTRY
from rlinf.config import SupportedModel, EMBODIED_MODEL
assert "fluxvla_pi05" in _MODEL_REGISTRY
assert SupportedModel("fluxvla_pi05") in EMBODIED_MODEL
from hydra.core.config_store import ConfigStore
store = ConfigStore.instance()
assert store.load('models/pi05.yaml').node.model_type == 'fluxvla_pi05'
assert store.load('models/smolvla.yaml').node.model_type == 'fluxvla_smolvla'
'''
    env = dict(os.environ, RLINF_EXT_MODULE='fluxvla.rl.rlinf_registry')
    result = subprocess.run([sys.executable, '-B', '-c', code],
                            cwd=ROOT,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=90,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_dry_run_without_cluster(tiny_assets):
    result = subprocess.run([
        sys.executable, '-B', '-m', 'fluxvla.rl.train', '--cfg', 'job',
        '--resolve', f'actor.model.model_path={tiny_assets[2]}'
    ],
                            cwd=ROOT,
                            env=dict(
                                os.environ,
                                RLINF_EXT_MODULE='fluxvla.rl.rlinf_registry'),
                            capture_output=True,
                            text=True,
                            timeout=90,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'model_type: fluxvla_pi05' in result.stdout
    assert 'cast_forward_inputs: false' in result.stdout


def test_robotwin_eval_cli_dry_run(tiny_assets):
    result = subprocess.run([
        sys.executable, '-B', '-m', 'fluxvla.rl.eval', '--cfg', 'job',
        '--resolve', f'actor.model.model_path={tiny_assets[2]}'
    ],
                            cwd=ROOT,
                            env=dict(
                                os.environ,
                                RLINF_EXT_MODULE='fluxvla.rl.rlinf_registry'),
                            capture_output=True,
                            text=True,
                            timeout=90,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'only_eval: true' in result.stdout
    assert 'total_num_envs: 10' in result.stdout
    assert 'rollout_epoch: 15' in result.stdout


def test_rl_tensorboard_honors_no_tensorflow():
    code = '''
import sys
import fluxvla.rl
from tensorboard.compat import tf
from tensorboard.compat import tensorflow_stub
assert tf.io is tensorflow_stub.io
assert "tensorflow" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-B', '-c', code],
                            cwd=ROOT,
                            env=dict(os.environ, USE_TF='0'),
                            capture_output=True,
                            text=True,
                            timeout=90,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_regular_flux_import_does_not_import_rlinf():
    code = '''
import sys
class BlockRLinf:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rlinf" or fullname.startswith("rlinf."):
            raise AssertionError("ordinary FluxVLA import tried to load RLinf")
sys.meta_path.insert(0, BlockRLinf())
import fluxvla
'''
    result = subprocess.run([sys.executable, '-B', '-c', code],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=90,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_external_worker_import_does_not_create_process_group():
    code = '''
import torch.distributed as dist
def forbidden(*args, **kwargs):
    raise AssertionError("Flux import initialized RLinf's process group")
dist.init_process_group = forbidden
import fluxvla
assert not dist.is_initialized()
'''
    result = subprocess.run([sys.executable, '-B', '-c', code],
                            cwd=ROOT,
                            env=dict(
                                os.environ,
                                FLUX_RL_EXTERNAL_DISTRIBUTED='1',
                                WORLD_SIZE='2',
                                RANK='0',
                                LOCAL_RANK='0'),
                            capture_output=True,
                            text=True,
                            timeout=90,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_explicit_mode_and_bootstrap(policy, env_obs):
    register()
    worker = object.__new__(FluxRolloutWorker)
    worker.cfg = OmegaConf.create({'rollout': {}, 'algorithm': {}})
    worker.algorithm_cfg = worker.cfg.algorithm
    worker.model_cfg = OmegaConf.create({'model_type': 'fluxvla_pi05'})
    worker.expert_model = None
    worker.enable_dagger = False
    worker.setup_sample_params()
    seen = []

    def predict_action_batch(env_obs, **kwargs):
        seen.append(kwargs['mode'])
        return torch.zeros(3, 2, 7), {}

    worker.hf_model = SimpleNamespace(
        predict_action_batch=predict_action_batch)
    predict = inspect.unwrap(MultiStepRolloutWorker.predict)
    predict(worker, env_obs, mode='train')
    predict(worker, env_obs, mode='eval')
    assert seen == ['train', 'eval']
    worker.hf_model = policy
    state = torch.random.get_rng_state().clone()
    actual = worker.get_bootstrap_values(env_obs)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(actual, policy.get_values(env_obs))
    assert worker.get_bootstrap_values(None) is None


def test_policy_output_split_merge_and_trajectory_alignment(policy, env_obs):
    actions, result = policy.predict_action_batch(env_obs, mode='train')
    output = PolicyOutput(actions=actions, **result)
    split = MultiStepRolloutWorker._split_policy_output(None, output, [1, 2])
    merged = PolicyOutput.merge(split)
    torch.testing.assert_close(merged.actions, output.actions)
    trajectories = []
    for part in split:
        recomputed = policy(forward_inputs=part.forward_inputs)
        torch.testing.assert_close(recomputed['logprobs'], part.prev_logprobs)
        trajectories.append(
            Trajectory(
                actions=part.actions.unsqueeze(0).repeat(2, 1, 1, 1),
                prev_logprobs=part.prev_logprobs.unsqueeze(0).repeat(
                    2, 1, 1, 1),
                prev_values=part.prev_values.unsqueeze(0).repeat(2, 1, 1),
                forward_inputs={
                    key: value.unsqueeze(0).repeat(2, *([1] * value.ndim))
                    for key, value in part.forward_inputs.items()
                }))
    batch = convert_trajectories_to_batch(trajectories)
    for key, value in output.forward_inputs.items():
        torch.testing.assert_close(merged.forward_inputs[key], value)
        torch.testing.assert_close(batch['forward_inputs'][key][0], value)
        torch.testing.assert_close(batch['forward_inputs'][key][1], value)


def test_real_actor_env_and_fsdp_imports_and_wrap_policy(policy):
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.workers.actor.embodied_fsdp_actor_worker import \
        EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker

    assert callable(EmbodiedFSDPActor.create_group)
    assert callable(EnvWorker.create_group)
    register()
    wrap = get_fsdp_wrap_policy(
        policy,
        config={
            'wrap_policy': {
                'transformer_layer_cls_to_wrap':
                ['GemmaDecoderLayer', 'SiglipEncoderLayer'],
                'module_classes_to_wrap': ['LinearProjector', 'ValueHead']
            }
        },
        is_lora=False,
        model_type='fluxvla_pi05')
    assert wrap is not None


def test_fsdp2_wrap_targets_without_distributed_runtime(policy, monkeypatch):
    from rlinf.hybrid_engines.fsdp import utils

    wrapped = []

    def record(module, **kwargs):
        wrapped.append(module)
        return module

    monkeypatch.setattr(utils, 'fully_shard', record)
    result = utils.apply_fsdp2_to_model(
        policy,
        config={
            'wrap_policy': {
                'transformer_layer_cls_to_wrap':
                ['GemmaDecoderLayer', 'SiglipEncoderLayer'],
                'module_classes_to_wrap': ['LinearProjector', 'ValueHead']
            }
        },
        device_mesh=None,
        mp_policy=None,
        offload_policy=None,
        reshard_after_forward=True)
    assert result is policy and wrapped[-1] is policy
    classes = {type(module).__name__ for module in wrapped}
    assert {
        'GemmaDecoderLayer', 'SiglipEncoderLayer', 'LinearProjector',
        'ValueHead'
    } <= classes


@pytest.mark.parametrize(('key', 'value'), [
    ('actor.model.joint_logprob', True),
    ('actor.model.is_lora', True),
    ('rollout.enable_torch_compile', True),
    ('actor.fsdp_config.mixed_precision.param_dtype', 'bf16'),
    ('actor.fsdp_config.mixed_precision.cast_forward_inputs', True),
    ('algorithm.entropy_bonus', 0.01),
])
def test_unsupported_configs_rejected_before_cluster(tiny_assets, key, value):
    prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(
            config_name='benchmarks/libero/pi05/ppo',
            overrides=[f'actor.model.model_path={tiny_assets[2]}'])
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        validate_frontend_cfg(cfg)


def test_real_gae_uses_bootstrap_and_stops_at_terminal(policy, env_obs):
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns

    bootstrap = policy.get_values(env_obs)[:, 0]
    values = torch.stack([bootstrap - 0.2, bootstrap])
    dones = torch.tensor([[False, False, False], [False, True, False]])
    rewards = torch.ones(1, 3)
    advantages, returns = compute_gae_advantages_and_returns(
        rewards,
        gamma=0.99,
        gae_lambda=0.95,
        values=values,
        dones=dones,
        normalize_advantages=False)
    expected = rewards[0] + 0.99 * bootstrap * (~dones[1])
    torch.testing.assert_close(returns[0], expected)
    torch.testing.assert_close(advantages[0], expected - values[0])
