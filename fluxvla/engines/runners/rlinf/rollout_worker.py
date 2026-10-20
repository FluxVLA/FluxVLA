"""Minimal compatibility glue around the stock RLinf rollout worker."""

import torch
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class FluxRolloutWorker(MultiStepRolloutWorker):

    async def evaluate(self, input_channel, output_channel):
        if self.model_cfg.fluxvla.get('observation_adapter',
                                      'libero') != 'robotwin':
            return await super().evaluate(input_channel, output_channel)
        # Pair SFT/best/last and repeated validation under identical noise;
        # do not let evaluation advance the training RNG stream.
        with torch.random.fork_rng():
            torch.manual_seed(int(self.cfg.actor.seed) + self._rank)
            torch.cuda.manual_seed_all(int(self.cfg.actor.seed) + self._rank)
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
