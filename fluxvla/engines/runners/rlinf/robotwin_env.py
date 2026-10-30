"""RoboTwin runtime compatibility and complete worker seed partitions."""

import json
from pathlib import Path

import torch
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


class FluxRoboTwinEnv(RoboTwinEnv):

    def _init_reset_state_ids(self):
        seeds = json.loads(Path(
            self.cfg.seeds_path).read_text())[self.task_name]['success_seeds']
        values = torch.as_tensor(seeds, dtype=torch.long)
        if values.numel() != torch.unique(values).numel():
            raise ValueError('Duplicate seeds are not allowed')
        indices = torch.randperm(
            len(values),
            generator=torch.Generator().manual_seed(self.base_seed))
        # Keep all seeds, including a remainder smaller than a worker batch.
        partitions = torch.tensor_split(values[indices],
                                        self.total_num_processes)
        if not 0 <= self.seed_offset < len(partitions) or not len(
                partitions[self.seed_offset]):
            raise ValueError('Empty/invalid seed partition')
        self.success_seeds = partitions[self.seed_offset]
        self._current_seed_index = 0
        self._generator = torch.Generator().manual_seed(self.seed)
        self.update_reset_state_ids()

    def _init_env(self):
        from .robotwin_runtime import install_strict_seed_runtime
        install_strict_seed_runtime()
        super()._init_env()
