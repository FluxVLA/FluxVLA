"""Nested Hydra recipes retain root namespaces and backend semantics."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from mmengine.config import Config
from omegaconf import OmegaConf

from fluxvla.rl.train import prepare_environment
from fluxvla.rl.utils.config import register_model_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs/rl'
RECIPES = sorted(
    str(path.relative_to(CONFIGS).with_suffix(''))
    for path in (CONFIGS / 'benchmarks').rglob('*.yaml')
    if not path.stem.startswith('base'))


def test_config_groups_are_explicit():
    assert not list(CONFIGS.glob('*.yaml'))
    assert not list(CONFIGS.glob('*.json'))
    assert len(RECIPES) == 8
    assert not list((CONFIGS / 'experiments').rglob('*.yaml'))
    assert not list((CONFIGS / 'diagnostics').rglob('*.yaml'))
    assert not list((CONFIGS / 'model').glob('*'))
    assert not list((CONFIGS / 'models').rglob('*.yaml'))
    assert (CONFIGS / 'models/pi05.py').is_file()
    assert (CONFIGS / 'models/smolvla.py').is_file()
    assert (CONFIGS / 'runtime/robotwin/nvidia_icd.json').is_file()
    common = OmegaConf.load(CONFIGS / 'base/ppo.yaml')
    assert 'model' not in common.actor
    assert 'wrap_policy' not in common.actor.fsdp_config
    assert 'max_episode_steps' not in common.env.train
    for benchmark in ('libero', 'robotwin'):
        other = 'robotwin' if benchmark == 'libero' else 'libero'
        for path in (CONFIGS / 'benchmarks' / benchmark).rglob('*.yaml'):
            assert f'/benchmarks/{other}/' not in path.read_text()
            assert '/experiments/' not in path.read_text()
            assert '/diagnostics/' not in path.read_text()


@pytest.mark.parametrize('name', RECIPES)
def test_all_nested_recipes_compose_at_root(name):
    prepare_environment()
    with initialize_config_dir(version_base='1.3', config_dir=str(CONFIGS)):
        cfg = compose(config_name=name)
    assert not {'benchmarks', 'experiments', 'diagnostics', 'base'} & set(cfg)
    assert cfg.algorithm.loss_type == 'actor_critic'
    assert cfg.weight_syncer.type == 'bucket'
    assert cfg.actor.fsdp_config.strategy == 'fsdp2'
    assert cfg.env.eval.is_eval
    assert cfg.env.eval.auto_reset and cfg.env.eval.ignore_terminations
    assert not cfg.env.train.video_cfg.save_video
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert raw['env']['train']['reward_coef'] == '${algorithm.reward_coef}'
    assert cfg.actor.fsdp_config.mixed_precision.cast_forward_inputs is False
    smolvla = 'smolvla' in name
    expected = ['SmolVLMEncoderLayer'
                ] if smolvla else ['GemmaDecoderLayer', 'SiglipEncoderLayer']
    assert list(cfg.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap
                ) == expected
    assert cfg.actor.model.model_type == ('fluxvla_smolvla'
                                          if smolvla else 'fluxvla_pi05')
    assert cfg.env.train.env_type == ('robotwin'
                                      if 'robotwin' in name else 'libero')
    assert cfg.runner.only_eval == (Path(name).stem in ('eval', 'eval_8gpu'))


def test_robotwin_mmengine_base_resolves_after_move():
    cfg = Config.fromfile(str(CONFIGS / 'models/pi05/robotwin.py'))
    assert cfg.model.n_action_steps == 50
    assert cfg.model.num_steps == 5
    assert cfg.model.openpi_fp32_flow
    assert cfg.model.vision_backbone.openpi_stem_fp32


@pytest.mark.parametrize('name', ['pi05', 'smolvla'])
def test_python_model_registration_preserves_defaults(name):
    register_model_configs()
    register_model_configs()
    # ConfigStore's virtual names use .yaml; there is no model YAML file.
    registered = ConfigStore.instance().load(f'models/{name}.yaml').node
    source = runpy.run_path(str(CONFIGS / 'models' / f'{name}.py'))['model']
    assert OmegaConf.to_container(registered, resolve=False) == source
    assert OmegaConf.is_missing(registered, 'model_path')
    assert registered.precision == 'fp32'
    assert registered.fluxvla.noise_level == 0.5


@pytest.mark.parametrize('name', ['pi05', 'smolvla'])
def test_python_model_cli_overrides_do_not_mutate_defaults(name):
    prepare_environment()
    with initialize_config_dir(version_base='1.3', config_dir=str(CONFIGS)):
        modified = compose(
            config_name=f'benchmarks/libero/{name}/ppo',
            overrides=[
                'actor.model.model_path=/tmp/custom.pt',
                'actor.model.fluxvla.noise_level=0.25',
                'actor.micro_batch_size=4'
            ])
        defaults = compose(config_name=f'benchmarks/libero/{name}/ppo')
    assert modified.actor.model.model_path == '/tmp/custom.pt'
    assert modified.actor.model.fluxvla.noise_level == 0.25
    assert defaults.actor.model.fluxvla.noise_level == 0.5
    assert OmegaConf.is_missing(defaults.actor.model, 'model_path')
    if name == 'smolvla':
        assert modified.actor.model.fluxvla.rollout_micro_batch_size == 4
        assert defaults.actor.model.fluxvla.rollout_micro_batch_size == 2


def test_python_config_registration_without_backend_import():
    code = '''
import sys
class BlockRLinf:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rlinf' or fullname.startswith('rlinf.'):
            raise AssertionError('config registration imported RLinf')
sys.meta_path.insert(0, BlockRLinf())
from fluxvla.rl.utils.config import register_model_configs
from hydra.core.config_store import ConfigStore
register_model_configs()
store = ConfigStore.instance()
assert store.load('models/pi05.yaml').node.model_type == 'fluxvla_pi05'
assert store.load('models/smolvla.yaml').node.model_type == 'fluxvla_smolvla'
assert 'fluxvla.rl.models.pi05.policy' not in sys.modules
assert 'fluxvla.rl.models.smolvla.policy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
