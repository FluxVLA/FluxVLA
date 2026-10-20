"""RoboTwin seed coverage and episode audit, without editing RLinf sources."""

import json
from pathlib import Path

import torch
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

from fluxvla.rl.benchmarks.robotwin.protocol import partition_seeds


class AuditedRoboTwinEnv(RoboTwinEnv):

    def _init_reset_state_ids(self):
        path = Path(self.cfg.seeds_path)
        seeds = json.loads(path.read_text())[self.task_name]['success_seeds']
        self.success_seeds = partition_seeds(
            seeds,
            seed=self.base_seed,
            rank=self.seed_offset,
            world_size=self.total_num_processes)
        self._current_seed_index = 0
        self._generator = torch.Generator().manual_seed(self.seed)
        self.experiment_step = 0
        self.update_reset_state_ids()

    def _init_env(self):
        from fluxvla.rl.benchmarks.robotwin.runtime import \
            install_strict_seed_runtime
        install_strict_seed_runtime()
        if self.cfg.get('full_video_dir'):
            from fluxvla.rl.benchmarks.robotwin.video import \
                install_demo_runtime
            install_demo_runtime(
                Path(self.cfg.full_video_dir) / f'worker_{self.seed_offset}')
        super()._init_env()

    def _record_metrics(self, step_reward, infos):
        infos = super()._record_metrics(step_reward, infos)
        finished = self._elapsed_steps >= self.cfg.max_episode_steps
        if finished.any() and self.cfg.get('episode_log_dir'):
            folder = Path(self.cfg.episode_log_dir)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder /
                  f'worker_{self.seed_offset}.jsonl').open('a') as stream:
                for index in finished.nonzero().flatten().tolist():
                    stream.write(
                        json.dumps({
                            'worker': self.seed_offset,
                            'slot': index,
                            'seed': int(self.reset_state_ids[index]),
                            'checkpoint_step': self.experiment_step,
                            'success': bool(self.success_once[index]),
                            'steps': int(self._elapsed_steps[index]),
                            'episode_return': float(self.returns[index])
                        }) + '\n')
        return infos
