"""Load Python model defaults into Hydra without importing native models."""

import runpy
from pathlib import Path


def register_model_configs():
    """Refresh process-local model configs before Hydra composition.

    Only top-level Python files declare RL model defaults. Nested files,
    such as models/pi05/robotwin.py, remain native MMEngine configurations.
    Hydra and OmegaConf are imported only when the RL entrypoint is used.
    """
    from hydra.core.config_store import ConfigStore

    config_dir = Path(__file__).resolve().parents[3] / 'configs/rl/models'
    store = ConfigStore.instance()
    for path in sorted(config_dir.glob('*.py')):
        if path.name.startswith('_'):
            continue
        model = runpy.run_path(str(path)).get('model')
        if not isinstance(model, dict):
            raise ValueError(f'{path} must define a model dictionary')
        store.store(
            group='models',
            name=path.stem,
            node=model,
            package='actor.model',
            provider='fluxvla')
