"""Keep RLinf environment orchestration, add RoboTwin protocol auditing."""

from rlinf.workers.env.env_worker import EnvWorker


class FluxEnvWorker(EnvWorker):

    def memory_report(self):
        import os
        import resource

        import psutil
        return {
            'rank':
            self._rank,
            'rss_gib':
            psutil.Process(os.getpid()).memory_info().rss / 2**30,
            'peak_rss_gib':
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            'train_live_subenvs':
            sum(
                len(env.unwrapped.venv.envs)
                for env in getattr(self, 'env_list', [])),
        }

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        if env_cfg.env_type == 'robotwin':
            from .robotwin_env import AuditedRoboTwinEnv
            env_cls = AuditedRoboTwinEnv
        return super()._setup_env_and_wrappers(env_cls, env_cfg,
                                               num_envs_per_stage)

    def set_experiment_step(self, step):
        for env in getattr(self, 'eval_env_list', []):
            env.unwrapped.experiment_step = int(step)
