"""RoboTwin fail-closed checks around RLinf's unchanged PPO implementation."""

import torch
import torch.distributed as dist
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from torch.distributed.tensor import DTensor


class AuditedEmbodiedActor(EmbodiedFSDPActor):

    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        # Check the actual differentiable forward before the first optimizer
        # update, including every accumulated microbatch. A preflight-only
        # forward cannot catch lifecycle-dependent FSDP2 probability drift.
        if getattr(self, '_rollout_optimizer_start',
                   None) != self.optimizer_steps:
            return super().train_micro_batch(
                micro_batch, metrics, is_last=is_last)

        def verify_output(module, args, kwargs, output):
            old = micro_batch['prev_logprobs'].to(output['logprobs'].device)
            drift = (output['logprobs'].detach() - old).abs().max()
            drift = torch.nan_to_num(
                drift, nan=float('inf'), posinf=float('inf'))
            dist.all_reduce(drift, op=dist.ReduceOp.MAX)
            self._loss_input_max_drift = max(self._loss_input_max_drift,
                                             float(drift))
            tolerance = float(
                self.cfg.experiment.get('probability_tolerance', 1e-4))
            if float(drift) > tolerance:
                raise FloatingPointError(
                    f'Actual pre-update loss input logprob drift '
                    f'{float(drift)} > {tolerance}')

        handle = self.model.register_forward_hook(
            verify_output, with_kwargs=True)
        try:
            return super().train_micro_batch(
                micro_batch, metrics, is_last=is_last)
        finally:
            handle.remove()

    def run_training(self):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        # Recompute one actual rollout microbatch before ANY optimizer update.
        # This catches actor/rollout precision or weight-version drift without
        # conflating it with the intentional PPO policy change after updating.
        count = int(self.cfg.actor.micro_batch_size)
        inputs = {
            key: value.flatten(0, 1)[:count].to(self.device).contiguous()
            for key, value in self.rollout_batch['forward_inputs'].items()
        }
        old = self.rollout_batch['prev_logprobs'].flatten(0, 1)[:count].to(
            self.device)
        with torch.no_grad():
            actual = self.model(
                forward_inputs=inputs, compute_values=False)['logprobs']
        drift = (actual - old).abs().max()
        drift = torch.nan_to_num(drift, nan=float('inf'), posinf=float('inf'))
        dist.all_reduce(drift, op=dist.ReduceOp.MAX)
        tolerance = float(
            self.cfg.experiment.get('probability_tolerance', 1e-4))
        if float(drift) > tolerance:
            raise FloatingPointError(f'Pre-update rollout/actor logprob drift '
                                     f'{float(drift)} > {tolerance}')
        self._rollout_optimizer_start = self.optimizer_steps
        self._loss_input_max_drift = 0.0
        metrics = super().run_training()
        metrics['actor/preupdate_logprob_max_abs_drift'] = float(drift)
        metrics['actor/loss_input_max_abs_drift'] = self._loss_input_max_drift
        return metrics

    def optimizer_step(self):
        # Stock RLinf skips non-finite gradients. This pilot must stop instead.
        flags = []
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad
                if isinstance(gradient, DTensor):
                    gradient = gradient.to_local()
                flags.append(torch.isfinite(gradient).all())
        finite = torch.stack(flags).all().to(
            torch.int32) if flags else torch.zeros(
                (), device=self.device, dtype=torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError('Missing or non-finite PPO gradients; '
                                     'pilot stopped before optimizer step')
        return super().optimizer_step()
