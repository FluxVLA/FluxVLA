"""Shared environment orchestration with benchmark-selected extensions."""

from rlinf.workers.env.env_worker import EnvWorker

from fluxvla.rl.benchmarks.registry import get_benchmark_spec


class FluxEnvWorker(EnvWorker):

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        env_cls = get_benchmark_spec(
            env_cfg.env_type).get_environment_class(env_cls)
        return super()._setup_env_and_wrappers(env_cls, env_cfg,
                                               num_envs_per_stage)
