# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-weighted validation through the distributed model entry point."""

import math

import torch
import torch.distributed as dist

from fluxvla.engines.utils.root import EVALUATORS
from .evaluator_policies import BaseEvaluator


@EVALUATORS.register_module()
class VLMEvaluator(BaseEvaluator):
    """Evaluate held-out labels without counting distributed padding twice.

    All ranks execute the same number of forwards (required by FSDP).
    Padding samples retain valid processor inputs but have ignored labels.
    ``max_batches`` limits batches per rank; None evaluates the full dataset.
    Reports teacher-forced loss, perplexity and token accuracy, not generation.
    """

    def __init__(self, eval_every=100, batch_size=1, max_batches=None):
        if eval_every < 1 or batch_size < 1:
            raise ValueError('eval_every and batch_size must be positive.')
        if max_batches is not None and max_batches < 1:
            raise ValueError('max_batches must be positive or None.')
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.max_batches = max_batches

    def should_run(self, runner):
        return (runner.metric.global_step > 0
                and runner.metric.global_step % self.eval_every == 0)

    @torch.no_grad()
    def run(self, runner, dataset):
        if runner.train_task not in ('vlm', 'mixed'):
            raise ValueError('VLMEvaluator requires train_task=vlm or mixed.')
        if len(dataset) == 0:
            raise ValueError('VLM validation dataset is empty.')
        if getattr(dataset, 'max_retry_times', 0):
            raise ValueError('VLM validation requires max_retry_times=0; '
                             'random replacement changes the evaluated set.')
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        local_size = math.ceil(len(dataset) / world_size)
        if self.max_batches is not None:
            local_size = min(local_size, self.max_batches * self.batch_size)
        totals = None
        was_training = runner.vla.training
        runner.vla.eval()
        try:
            for start in range(0, local_size, self.batch_size):
                indices = [
                    rank + i * world_size for i in range(
                        start, min(start + self.batch_size, local_size))
                ]
                collator = (
                    runner._mixed_training.vlm_collator
                    if runner.train_task == 'mixed' else runner.collator)
                batch = collator([dataset[i % len(dataset)] for i in indices])
                ignore_index = runner._unwrap_vla().ignore_index
                for row, index in enumerate(indices):
                    if index >= len(dataset):
                        batch['labels'][row].fill_(ignore_index)
                dtype = (
                    runner.mixed_precision_dtype
                    if runner.enable_mixed_precision_training
                    and not runner.keep_params_fp32 else None)
                batch = runner._prepare_batch(batch, runner.device_id, dtype)
                device_type = ('cuda' if isinstance(runner.device_id, int) else
                               torch.device(runner.device_id).type)
                with torch.autocast(
                        device_type,
                        dtype=runner.mixed_precision_dtype,
                        enabled=runner.enable_mixed_precision_training):
                    output = runner.vla(task='vlm', **batch)
                values = torch.stack([
                    output[key].detach().double()
                    for key in ('loss_sum', 'num_tokens', 'num_correct')
                ])
                totals = values if totals is None else totals + values
            if distributed:
                dist.all_reduce(totals)
            if totals[1].item() == 0:
                raise ValueError('VLM validation has no supervised tokens.')
            loss = (totals[0] / totals[1]).item()
            metrics = {
                'eval/vlm_loss': loss,
                'eval/perplexity': math.exp(loss) if loss < 709 else math.inf,
                'eval/token_accuracy': (totals[2] / totals[1]).item(),
                'eval/supervised_tokens': totals[1].item(),
            }
            runner.metric.log(runner.metric.global_step, metrics)
            return metrics
        finally:
            runner.vla.train(was_training)
