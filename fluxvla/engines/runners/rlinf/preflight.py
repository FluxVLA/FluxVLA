"""Real-worker integration gate; never a scored training run."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from rlinf.runners.embodied_runner import EmbodiedRunner

from .actor_worker import AuditedEmbodiedActor
from .rollout_worker import FluxRolloutWorker


def gate_dir(cfg):
    return Path(cfg.experiment.root) / 'preflight' / cfg.experiment.get(
        'gate_name', '')


def fingerprint(state):
    return {
        name: hashlib.sha256(
            t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, t in state.items()
    }


class ProbeActor(AuditedEmbodiedActor):

    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        if not self.cfg.experiment.get('trace_ppo_inputs', False):
            return super().train_micro_batch(
                micro_batch, metrics, is_last=is_last)
        from rlinf.workers.actor import embodied_fsdp_actor_worker as backend
        original = backend.policy_loss
        trace_index = getattr(self, '_trace_micro_count', 0)
        self._trace_micro_count = trace_index + 1
        if self.cfg.experiment.get('trace_weights', False):
            torch.save(
                micro_batch,
                gate_dir(self.cfg) /
                'microbatch_{:03d}_rank_{}.pt'.format(trace_index, self._rank))

        def audited_loss(**kwargs):
            raw_diff = kwargs['logprobs'].detach() - kwargs['old_logprobs']
            loss, result = original(**kwargs)
            row = {
                'optimizer_steps': self.optimizer_steps,
                'raw_max_abs_drift': raw_diff.abs().max().item(),
                'chunk_logratio': raw_diff.sum(dim=(1, 2)).tolist(),
                'ratio': result['actor/ratio'],
                'clip_fraction': result['actor/clip_fraction']
            }
            with (gate_dir(self.cfg) /
                  f'loss_trace_rank_{self._rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            if row['raw_max_abs_drift'] > 1e-4 and self.optimizer_steps == 0:
                torch.save(
                    {
                        'forward_inputs': {
                            key: value.cpu()
                            for key, value in
                            micro_batch['forward_inputs'].items()
                        },
                        'prev_logprobs': kwargs['old_logprobs'].cpu(),
                        'actual_logprobs': kwargs['logprobs'].detach().cpu()
                    },
                    gate_dir(self.cfg) /
                    f'loss_failed_sample_rank_{self._rank}.pt')
                raise AssertionError(
                    'Actual first-update loss input probability mismatch')
            return loss, result

        backend.policy_loss = audited_loss
        try:
            result = super().train_micro_batch(
                micro_batch, metrics, is_last=is_last)
            if self.cfg.experiment.get('trace_weights', False):
                from torch.distributed.tensor import DTensor
                changed = []
                for name, parameter in self.model.named_parameters():
                    tensor = parameter.detach()
                    tensor = tensor.to_local() if isinstance(
                        tensor, DTensor) else tensor
                    if not torch.equal(tensor,
                                       self._weights_before_trace[name]):
                        changed.append(name)
                with (gate_dir(self.cfg) /
                      f'weight_trace_rank_{self._rank}.jsonl'
                      ).open('a') as stream:
                    stream.write(
                        json.dumps({
                            'microbatch': trace_index,
                            'changed': changed
                        }) + '\n')
            return result
        finally:
            backend.policy_loss = original

    def audit_state(self):
        state = self.get_model_state_dict(
            cpu_offload=True, full_state_dict=True)
        return {
            'rank':
            self._rank,
            'hashes':
            fingerprint(state),
            'frozen': [
                n for n, p in self.model.named_parameters()
                if not p.requires_grad
            ]
        }

    def audit_sample(self):
        # All actor ranks participate in FSDP forwards, including when their
        # world size differs from the number of rollout workers.
        source_rank = self._rank % self._component_placement.get_world_size(
            'rollout')
        path = gate_dir(self.cfg) / f'ray_sample_{source_rank}.pt'
        from rlinf.utils.nested_dict_process import put_tensor_device
        sample = put_tensor_device(
            torch.load(path, weights_only=True), self.device)
        self.model.eval()
        outputs = []
        micro = int(self.cfg.actor.micro_batch_size)
        with torch.no_grad():
            for start in range(0, len(sample['prev_logprobs']), micro):
                inputs = {
                    key: value[start:start + micro]
                    for key, value in sample['forward_inputs'].items()
                }
                outputs.append(self.model(forward_inputs=inputs))
        actual = {
            key: torch.cat([out[key] for out in outputs])
            for key in ('logprobs', 'values')
        }
        torch.testing.assert_close(
            actual['logprobs'], sample['prev_logprobs'], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            actual['values'].reshape(-1),
            sample['prev_values'].reshape(-1),
            atol=1e-5,
            rtol=1e-5)
        return {
            'rank':
            self._rank,
            'logprob_max_abs_drift':
            float((actual['logprobs'] - sample['prev_logprobs']).abs().max())
        }

    def perturb_for_restore(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.add_(.125)

    def run_training(self):
        if self.cfg.experiment.get('trace_weights', False):
            from torch.distributed.tensor import DTensor
            self._weights_before_trace = {
                name: (parameter.to_local() if isinstance(parameter, DTensor)
                       else parameter).detach().clone()
                for name, parameter in self.model.named_parameters()
            }
        if self.cfg.experiment.get('full_probability_audit', False):
            size = self.rollout_batch['prev_logprobs'].shape[
                0] * self.rollout_batch['prev_logprobs'].shape[1]
            order = torch.randperm(
                size,
                generator=torch.Generator().manual_seed(self.cfg.actor.seed +
                                                        self._rank))
            inputs = {
                key: value.flatten(0, 1)[order]
                for key, value in self.rollout_batch['forward_inputs'].items()
            }
            old = self.rollout_batch['prev_logprobs'].flatten(0, 1)[order]
            rows = []
            self.model.train()
            micro = int(self.cfg.actor.micro_batch_size)
            for start in range(0, size, micro):
                batch = {
                    key:
                    value[start:start + micro].to(self.device).contiguous()
                    for key, value in inputs.items()
                }
                # Use the actual differentiable training-forward path, not
                # only the original ordered no_grad sampling microbatch.
                output = self.model(forward_inputs=batch, compute_values=False)
                diff = output['logprobs'].detach() - old[start:start +
                                                         micro].to(self.device)
                row = {
                    'start': start,
                    'element_max_abs_drift': diff.abs().max().item(),
                    'chunk_logratio': diff.sum(dim=(1, 2)).tolist()
                }
                rows.append(row)
                if row['element_max_abs_drift'] > 1e-4:
                    torch.save(
                        {
                            'forward_inputs':
                            {k: v.cpu()
                             for k, v in batch.items()},
                            'prev_logprobs': old[start:start + micro].cpu()
                        },
                        gate_dir(self.cfg) /
                        f'failed_sample_rank_{self._rank}.pt')
                del output
            report = {'rank': self._rank, 'samples': size, 'checks': rows}
            (gate_dir(self.cfg) / f'full_probability_rank_{self._rank}.json'
             ).write_text(json.dumps(report, indent=2) + '\n')
            assert max(
                row['element_max_abs_drift'] for row in rows
            ) <= 1e-4, 'Full shuffled pre-update probability audit failed'
        metrics = super().run_training()
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise FloatingPointError(
                    f'Non-finite integration-gate metric: {key}')
        return metrics


class ProbeRollout(FluxRolloutWorker):

    def audit_state(self):
        return {
            'rank': self._rank,
            'hashes': fingerprint(self.hf_model.state_dict())
        }

    def audit_sample(self):
        root = Path(self.cfg.experiment.root) / 'preflight'
        obs = torch.load(root / 'real_observation.pt', weights_only=True)
        batch = int(self.cfg.env.train.total_num_envs) // self._world_size
        obs = {
            k: v * batch if isinstance(v, list) else v.repeat(
                batch, *([1] * (v.ndim - 1)))
            for k, v in obs.items()
        }
        self.hf_model.eval()
        _, sample = self.hf_model.predict_action(obs, mode='train')
        sample = {
            k: {n: t.cpu()
                for n, t in v.items()} if isinstance(v, dict) else v.cpu()
            for k, v in sample.items()
        }
        torch.save(sample, gate_dir(self.cfg) / f'ray_sample_{self._rank}.pt')


def run_gate(cfg):
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    from ..rlinf_train_runner import RLinfTrainRunner
    from .env_worker import FluxEnvWorker
    from .registry import register

    cfg = copy.deepcopy(cfg)
    if cfg.runner.get('resume_dir'):
        raise ValueError(
            'Preflight requires a fresh run, not runner.resume_dir')
    # Run exactly one PPO round and save it without consuming evaluation seeds.
    cfg.runner.max_epochs = 1
    cfg.runner.max_steps = 1
    cfg.runner.val_check_interval = -1
    cfg.runner.save_interval = 1

    dest = gate_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    register()
    RLinfTrainRunner.validate_frontend_cfg(cfg)
    cfg = validate_cfg(cfg)
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    groups = {
        name: worker.create_group(cfg).launch(
            cluster,
            name=cfg[name].group_name,
            placement_strategy=placement.get_strategy(name))
        for name, worker in (('actor', ProbeActor), ('rollout', ProbeRollout),
                             ('env', FluxEnvWorker))
    }
    runner = EmbodiedRunner(cfg, **groups)
    started = time.monotonic()
    runner.init_workers()
    memory_before = groups['env'].memory_report().wait()
    runner.update_rollout_weights()
    before = groups['actor'].audit_state().wait()[0]
    for result in groups['rollout'].audit_state().wait():
        assert result['hashes'] == before[
            'hashes'], 'Initial cross-group state mismatch'
    groups['rollout'].audit_sample().wait()
    drift = groups['actor'].audit_sample().wait()
    runner.run()
    memory_after_rollout = groups['env'].memory_report().wait()
    if cfg.env.train.get('enable_offload', False):
        assert all(row['train_live_subenvs'] == 0
                   for row in memory_after_rollout
                   ), 'Environment offload did not release SubEnv objects'
    after = groups['actor'].audit_state().wait()[0]
    changed = [
        n for n in before['hashes']
        if before['hashes'][n] != after['hashes'][n]
    ]
    assert not set(changed).intersection(
        before['frozen']), 'Frozen parameter changed'
    assert any(n.startswith('llm_expert.')
               for n in changed), 'Expert did not update'
    assert any(n.startswith('value_head.')
               for n in changed), 'Critic did not update'
    checkpoint = Path(
        cfg.runner.logger.log_path
    ) / cfg.runner.logger.experiment_name / 'checkpoints/global_step_1/actor'
    groups['actor'].perturb_for_restore().wait()
    groups['actor'].load_checkpoint(str(checkpoint)).wait()
    restored = groups['actor'].audit_state().wait()[0]
    assert restored['hashes'] == after[
        'hashes'], 'Actual actor checkpoint restore mismatch'
    runner.update_rollout_weights()
    for result in groups['rollout'].audit_state().wait():
        assert result['hashes'] == after[
            'hashes'], 'Post-update/restored cross-group mismatch'
    groups['rollout'].audit_sample().wait()
    restored_drift = groups['actor'].audit_sample().wait()
    report = {
        'passed': True,
        'elapsed_seconds': time.monotonic() - started,
        'initial_probability': drift,
        'restored_probability': restored_drift,
        'changed_parameter_count': len(changed),
        'frozen_unchanged': True,
        'cross_group_sync': True,
        'actor_checkpoint_restored': True,
        'real_environment_ppo_rounds': 1,
        'env_memory_before': memory_before,
        'env_memory_after_rollout': memory_after_rollout,
        'actor_world_size': placement.get_world_size('actor'),
        'rollout_world_size': placement.get_world_size('rollout'),
        'checkpoint': str(checkpoint)
    }
    (dest / 'ray_gate.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report
