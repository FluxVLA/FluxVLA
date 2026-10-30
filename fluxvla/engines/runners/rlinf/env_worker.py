"""Use FluxVLA's RoboTwin runtime adapter with RLinf environment workers."""

from rlinf.workers.env.env_worker import EnvWorker


class FluxEnvWorker(EnvWorker):

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        if env_cfg.env_type == 'robotwin':
            from .robotwin_env import FluxRoboTwinEnv
            env_cls = FluxRoboTwinEnv
        return super()._setup_env_and_wrappers(env_cls, env_cfg,
                                               num_envs_per_stage)
