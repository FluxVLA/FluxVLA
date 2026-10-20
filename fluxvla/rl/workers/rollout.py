"""Minimal compatibility glue around the stock RLinf rollout worker."""

import torch
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from fluxvla.rl.benchmarks.registry import get_benchmark_spec


class FluxRolloutWorker(MultiStepRolloutWorker):

    async def evaluate(self, input_channel, output_channel):
        paired_seed = self.cfg.get('experiment', {}).get('paired_eval_seed')
        benchmark = get_benchmark_spec(
            self.model_cfg.fluxvla.get('observation_adapter', 'libero'))
        if not benchmark.pair_eval_rng and paired_seed is None:
            return await super().evaluate(input_channel, output_channel)
        # Pair SFT/best/last and repeated validation under identical noise;
        # do not let evaluation advance the training RNG stream.
        with torch.random.fork_rng():
            seed = int(self.cfg.actor.seed
                       if paired_seed is None else paired_seed) + self._rank
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            return await super().evaluate(input_channel, output_channel)

    def setup_sample_params(self):
        super().setup_sample_params()
        # RLinf's predict() passes mode explicitly only for its built-in types.
        self._train_sampling_params['mode'] = 'train'
        self._eval_sampling_params['mode'] = 'eval'

    @torch.no_grad()
    def get_bootstrap_values(self, final_obs):
        if final_obs is None:
            return None
        return self.hf_model.get_values(final_obs).cpu().contiguous()
