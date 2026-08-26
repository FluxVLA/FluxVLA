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

import os
from typing import Dict

import torch
import torch.distributed as dist

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.root import EVALUATORS
from fluxvla.engines.utils.video_metrics import save_video_frames

overwatch = initialize_overwatch(__name__)


class BaseEvaluator:
    """Base class for registry-backed training-time evaluators.

    Evaluators encapsulate the optional in-training evaluation hook that the
    train runners invoke after each optimizer step. Subclasses decide when an
    evaluation should run (``should_run``) and what it does (``run``).
    """

    def __init__(self, **kwargs) -> None:
        if kwargs:
            fields = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected evaluator config field(s) for '
                            f'{self.__class__.__name__}: {fields}')

    def should_run(self, runner) -> bool:
        """Return whether evaluation should run at the current step.

        Args:
            runner: The training runner driving the evaluation.

        Returns:
            bool: ``True`` if ``run`` should be invoked now.
        """
        return False

    def run(self, runner, dataset) -> None:
        """Run one evaluation pass.

        Args:
            runner: The training runner driving the evaluation.
            dataset: The dataset evaluation samples are drawn from.
        """
        raise NotImplementedError


@EVALUATORS.register_module(name=['training-eval', 'TrainingEvalEvaluator'])
class TrainingEvalEvaluator(BaseEvaluator):
    """In-training evaluation driven by a model ``compute_training_eval`` hook.

    Every ``eval_every`` optimizer steps, this samples one batch, runs the
    model's ``compute_training_eval`` hook under inference mode, reduces the
    returned scalar metrics across ranks, logs them with an ``eval/`` prefix,
    and saves a per-rank eval video when the hook returns ``video_frames``.
    When ``eval_every`` is ``0`` evaluation is disabled.

    The hook returns ``{'metrics': Dict[str, float], 'video_frames':
    Optional[Sequence[PIL.Image]]}``. Models without video generation return
    ``video_frames=None`` and nothing is saved.

    Args:
        eval_every (int): Run evaluation every N optimizer steps. ``0``
            disables evaluation. Defaults to 0.
        num_inference_steps (int): Inference steps forwarded to the model's
            eval hook. Defaults to 10.
        seed (int): Seed forwarded to the model's eval hook. Defaults to 42.
        save_video (bool): Whether to save eval videos when the hook returns
            frames. Defaults to True.
        video_fps (int): Frame rate for saved eval videos. Defaults to 8.
    """

    def __init__(self,
                 eval_every: int = 0,
                 num_inference_steps: int = 10,
                 seed: int = 42,
                 save_video: bool = True,
                 video_fps: int = 8,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.eval_every = int(eval_every)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        self.save_video = bool(save_video)
        self.video_fps = int(video_fps)
        self._unavailable_warned = False

    def _get_eval_item(self, dataset, index: int):
        if hasattr(dataset, '_get_item_from_global_idx'):
            return dataset._get_item_from_global_idx(index)
        try:
            return dataset[index]
        except (TypeError, AttributeError, NotImplementedError):
            return None

    def _sample_eval_batch(self, runner, dataset):
        try:
            dataset_len = len(dataset)
        except (TypeError, AttributeError):
            return None
        if dataset_len <= 0:
            return None

        generator = torch.Generator(device='cpu').manual_seed(
            int(runner.metric.global_step) + int(overwatch.rank()))
        index = torch.randint(
            0, dataset_len, (1, ), generator=generator).item()
        sample = self._get_eval_item(dataset, int(index))
        if sample is None:
            return None
        return runner.collator([sample])

    def _reduce_metrics(self, metrics: Dict) -> Dict[str, float]:
        scalar_items = []
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = float(value.detach().float().cpu().item())
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                continue
            scalar_items.append((key, value))
        if len(scalar_items) == 0:
            return {}

        keys = [key for key, _ in scalar_items]
        device = (
            torch.device('cuda', torch.cuda.current_device())
            if torch.cuda.is_available() else torch.device('cpu'))
        values = torch.tensor([value for _, value in scalar_items],
                              dtype=torch.float32,
                              device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= float(dist.get_world_size())
        return {
            key: float(value)
            for key, value in zip(keys,
                                  values.detach().cpu().tolist())
        }

    def _save_eval_video(self, runner, step: int, frames) -> None:
        if not (self.save_video and frames):
            return
        eval_dir = os.path.join(runner.metric.run_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        video_path = os.path.join(
            eval_dir,
            f'step_{int(step):06d}_rank_{int(overwatch.rank()):03d}.mp4')
        save_video_frames(frames, video_path, fps=self.video_fps)

    def should_run(self, runner) -> bool:
        if self.eval_every <= 0:
            return False
        step = int(runner.metric.global_step)
        return step > 0 and step % self.eval_every == 0

    def run(self, runner, dataset) -> None:
        step = int(runner.metric.global_step)
        if not runner.can_run_training_eval():
            if (overwatch.is_rank_zero() and not self._unavailable_warned):
                overwatch.warning(
                    'Training eval requested, but the model does not '
                    'implement `compute_training_eval`; skipping.')
                self._unavailable_warned = True
            return

        batch = self._sample_eval_batch(runner, dataset)
        if batch is None:
            if (overwatch.is_rank_zero() and not self._unavailable_warned):
                overwatch.warning(
                    'Training eval requested, but no indexable training '
                    'sample could be built; skipping.')
                self._unavailable_warned = True
            return

        was_training = runner.vla.training
        runner.vla.eval()
        try:
            with torch.no_grad():
                with torch.autocast(
                        'cuda',
                        dtype=runner.mixed_precision_dtype,
                        enabled=runner.enable_mixed_precision_training):
                    output = runner.run_training_eval(
                        batch=batch,
                        num_inference_steps=self.num_inference_steps,
                        seed=self.seed,
                    )
        finally:
            if was_training:
                runner.vla.train()

        output = output or {}
        self._save_eval_video(runner, step, output.get('video_frames'))

        metrics = self._reduce_metrics(output.get('metrics') or {})
        if not metrics:
            return
        payload = {f'eval/{key}': value for key, value in metrics.items()}
        if overwatch.is_rank_zero():
            runner.metric.log(step, payload)
            summary = ' '.join(f'{key}={value:.4f}'
                               for key, value in sorted(metrics.items()))
            overwatch.info(f'[eval] step={step} {summary}')
