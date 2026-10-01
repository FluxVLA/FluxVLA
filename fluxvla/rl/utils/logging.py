"""Online logging authentication shared by all policies and benchmarks."""

import os


def require_wandb():
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError(
            'WANDB_API_KEY is missing; formal PPO must use online W&B')
    if os.environ.get('WANDB_MODE', 'online') != 'online':
        raise RuntimeError('Formal PPO requires WANDB_MODE=online')
    import wandb

    # Authenticate without wandb.login(), which can persist a key to .netrc.
    # MetricLogger owns the run, and the key stays only in the environment.
    if not wandb.Api(api_key=os.environ['WANDB_API_KEY'], timeout=30).viewer:
        raise RuntimeError('W&B authentication failed')
