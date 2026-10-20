"""RoboTwin-only environment diagnostics and episode step annotations."""

from fluxvla.rl.workers.env import FluxEnvWorker


class RoboTwinEnvWorker(FluxEnvWorker):

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

    def set_experiment_step(self, step):
        for env in getattr(self, 'eval_env_list', []):
            env.unwrapped.experiment_step = int(step)
