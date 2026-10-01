"""OOM continuation keeps batch contracts and the original validation best."""

from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.rlinf_registry import register
from fluxvla.rl.train import prepare_environment, validate_frontend_cfg

ROOT = Path(__file__).resolve().parents[2]


def test_resume_16_env_global_batch_contract(tiny_assets, monkeypatch):
    import rlinf.config as backend
    prepare_environment()
    register()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(
            config_name='benchmarks/robotwin/pi05/ppo',
            overrides=[
                f'actor.model.model_path={tiny_assets[2]}',
                'env.train.total_num_envs=16'
            ])
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg)
    monkeypatch.setattr(backend, 'Cluster', lambda *a, **k: None)
    monkeypatch.setattr(
        backend, 'HybridComponentPlacement',
        lambda *a, **k: SimpleNamespace(get_world_size=lambda name: 2))
    backend.validate_embodied_cfg(cfg)
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.micro_batch_size == 2
    assert cfg.env.train.total_num_envs * (200 // 50) == 64
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 32
    assert cfg.actor.optim.lr == 5e-6
    assert cfg.algorithm.update_epoch == 5
