"""Experiment bookkeeping around the unmodified synchronous RLinf runner."""

import json
import math
import os
import time
from pathlib import Path

from rlinf.runners.embodied_runner import EmbodiedRunner

from fluxvla.rl.benchmarks.robotwin.protocol import choose_best


class BudgetStop(Exception):
    pass


class RoboTwinRunner(EmbodiedRunner):

    @property
    def artifact_dir(self):
        return Path(self.cfg.runner.logger.log_path
                    ) / self.cfg.runner.logger.experiment_name

    def run(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if 'wandb' in self.cfg.runner.logger.logger_backends:
            import wandb
            if wandb.run is None or wandb.run.offline:
                raise RuntimeError('Online W&B run did not initialize')
            (self.artifact_dir / 'wandb_run.json').write_text(
                json.dumps(
                    {
                        'id': wandb.run.id,
                        'url': wandb.run.url,
                        'project': wandb.run.project
                    },
                    indent=2) + '\n')
        stop_reason = 'max_steps'
        try:
            super().run()
        except BudgetStop:
            stop_reason = 'budget_boundary'
            self.logger.info('Stopped at a checkpoint boundary '
                             'before the six-hour deadline.')
            self._finish_run()
        (self.artifact_dir / 'runner_completion.json').write_text(
            json.dumps(
                {
                    'step': self.global_step,
                    'reason': stop_reason,
                    'completed_at': time.time()
                },
                indent=2) + '\n')

    def evaluate(self):
        self.env.set_experiment_step(self.global_step).wait()
        return super().evaluate()

    def _save_checkpoint(self):
        super()._save_checkpoint()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / 'last_checkpoint.json').write_text(
            json.dumps(
                {
                    'step':
                    self.global_step,
                    'path':
                    str(self.artifact_dir /
                        f'checkpoints/global_step_{self.global_step}')
                },
                indent=2) + '\n')

    def _maybe_eval_and_checkpoint(self, step):
        metrics = super()._maybe_eval_and_checkpoint(step)
        budget_path = os.environ.get('FLUX_RL_BUDGET_FILE')
        budget_due = False
        extra_save = False
        if budget_path:
            deadline = json.loads(
                Path(budget_path).read_text())['training_deadline']
            reserve = float(
                self.cfg.experiment.get('budget_reserve_seconds', 300))
            budget_due = time.time() >= deadline - reserve
        if budget_due and not metrics:
            # A short, budget-limited run still needs a validation-selected
            # checkpoint. Do this cooperatively, never via an async watchdog.
            self.update_rollout_weights()
            metrics = {
                f'eval/{key}': value
                for key, value in self.evaluate().items()
            }
            self.metric_logger.log(data=metrics, step=step)
            self._save_checkpoint()
            extra_save = True
        if metrics:
            if self.cfg.experiment.split != 'validation':
                raise ValueError(
                    'Training-time checkpoint selection cannot use test seeds')
            score = float(metrics['eval/success_once'])
            if not math.isfinite(score):
                raise FloatingPointError('Non-finite validation success rate')
            path = self.artifact_dir / 'checkpoint_selection.json'
            records = json.loads(
                path.read_text())['records'] if path.exists() else []
            records.append({
                'step': self.global_step,
                'split': 'validation',
                'success_rate': score
            })
            path.write_text(
                json.dumps({
                    'records': records,
                    'best': choose_best(records)
                },
                           indent=2) + '\n')
        if self.cfg.experiment.get('memory_audit', False):
            memory = self.env.memory_report().wait()
            with (self.artifact_dir /
                  'environment_memory.jsonl').open('a') as stream:
                stream.write(
                    json.dumps({
                        'step': self.global_step,
                        'workers': memory
                    }) + '\n')
            self.metric_logger.log(
                data={
                    f"memory/env_rank_{row['rank']}_rss_gib": row['rss_gib']
                    for row in memory
                },
                step=step)
        if budget_due:
            if (not extra_save
                    and self.global_step % self.cfg.runner.save_interval):
                self._save_checkpoint()
            self._budget_stop_requested = True
        return metrics

    def _log_step_metrics(self, *args, **kwargs):
        # Preserve the final round's PPO/throughput metrics before leaving the
        # backend loop; raising from _maybe_eval_and_checkpoint loses them.
        super()._log_step_metrics(*args, **kwargs)
        if getattr(self, '_budget_stop_requested', False):
            raise BudgetStop()
